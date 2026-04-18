import math

import torch
import triton
import triton.language as tl

class TorchFlashAttention(torch.autograd.Function):
    # Q, K, V: [B, N, d]
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        TILE_SIZE = 16

        (B, N, d) = Q.size()
        num_tiles: int = N // TILE_SIZE

        # Qs, Ks, Vs: [B, TILE_SIZE, d]
        Qs: tuple[torch.Tensor, ...] = Q.split(TILE_SIZE, dim=1)
        Ks: tuple[torch.Tensor, ...] = K.split(TILE_SIZE, dim=1)
        Vs: tuple[torch.Tensor, ...] = V.split(TILE_SIZE, dim=1)

        Os: list[torch.Tensor] = []
        Ls: list[torch.Tensor] = []

        # compute:
        # S = (Q @ K.T) / sqrt(d)
        # M = max_reduce(S, dim=2)
        # exp_P = exp(S - M[:, :, None])
        # L = sum(exp_P, dim=1)
        # O = exp_P / L
        for i in range(num_tiles):
            m_ij = torch.full((B, TILE_SIZE,), -torch.inf, device=Q.device)
            l_ij = torch.zeros((B, TILE_SIZE), device=Q.device)
            O_ij = torch.zeros((B, TILE_SIZE, d), device=Q.device)

            for j in range(num_tiles):
                # S_ij: [B, TILE_SIZE, TILE_SIZE]
                S_ij = (Qs[i] @ Ks[j].transpose(1,2)) / math.sqrt(d)

                if is_causal:
                    q_idx = torch.arange(0, TILE_SIZE, device=Q.device) + (i * TILE_SIZE)
                    k_idx = torch.arange(0, TILE_SIZE, device=Q.device) + (j * TILE_SIZE)
                    causal_mask = q_idx[:, None] >= k_idx[None, :]
                    S_ij = torch.where(causal_mask[None, :, :], S_ij, -torch.inf)

                # rowsum_Sij: [B, TILE_SIZE]
                rowsum_Sij = S_ij.sum(2)
                prev_m_ij = m_ij
                m_ij = torch.maximum(prev_m_ij, rowsum_Sij)

                # exp_Pij: [B, TILE_SIZE, TILE_SIZE]
                exp_P_ij = torch.exp(S_ij - m_ij[:, :, None])

                # exp_m_sub: [B, TILE_SIZE]
                exp_m_sub = torch.exp(prev_m_ij - m_ij)

                # l_ij: [B, TILE_SIZE]
                l_ij = exp_m_sub * l_ij + exp_P_ij.sum(dim=2)

                # O_ij: [B, TILE_SIZE, d]
                O_ij = (exp_m_sub[:, :, None] * O_ij) + (exp_P_ij @ Vs[j])

            # O_i: [B, TILE_SIZE, d]
            O_i = O_ij / l_ij[:, :, None]

            # L_i: [B, TILE_SIZE]
            L_i = m_ij + torch.log(l_ij)

            Os.append(O_i)
            Ls.append(L_i)

        # O: [B, N, d]
        O = torch.cat(Os, dim=1)

        # L: [B, N]
        L = torch.cat(Ls, dim=1)

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, *grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        dO = grad_output[0]

        scale = Q.size(2)

        # S: [B, N, N]
        S = Q @ K.transpose(1,2) / math.sqrt(scale)

        # P: [B, N, N]
        P = torch.exp(S - L[:,:,None])

        # dV: [B, N, D]
        dV = P.transpose(1,2) @ dO

        # dP: [B, N, N]
        dP = dO @ V.transpose(1,2)

        # D: [B, N]
        D = (P * dP).sum(dim=2)

        # dS: [B, N, N]
        dS = P * (dP - D[:, :, None])

        # dQ: [B, N, D]
        dQ = dS @ K / math.sqrt(scale)

        # dK: [B, N, D]
        dK = dS.transpose(1,2) @ Q / math.sqrt(scale)

        return dQ, dK, dV, None

# this is the naive attention kernel that flashattention improves upon
@triton.jit
def naive_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal,
):
    # Program indices
    # launch grid is configured such that
    # 1 block = 1 Q / O tile in 1 element of the batch
    qtile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(qtile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(qtile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(qtile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    Q_block: tl.tensor = tl.load(Q_block_ptr).to(tl.float32)

    # compute:
    # S: [Q_TILE_SIZE, N]
    # S = (Q_block @ K.T) / sqrt(d)
    Slist = []
    for j in range(Tk):
        K_block: tl.tensor = tl.load(K_block_ptr)
        S_ij: tl.tensor = tl.dot(Q_block, K_block.trans(1,0)) / scale
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        Slist.append(S)

    S = tl.cat(Slist)

    ### PEG
    ### K_block_ptr = theta_kblk(K_block_ptr, tl.advance(theta_kblk, (K_TILE_SIZE, 0)))
    ### K_block = tl.load(K_block_ptr)
    ### S_ij = tl.dot(Q_block, K_block.trans(1,0)) / scale

    # M: [Q_TILE_SIZE]
    # M = max_reduce(S, dim=D)
    M: tl.tensor = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    for S_ij in Slist:
        rowmax_Sij = tl.max(S_ij, dim=1)
        M = tl.maximum(M, rowmax_Sij)

    ### PEG
    ### rowmax_Sij = tl.max(S_ij, dim=1)

    # exp_P: [Q_TILE_SIZE, N]
    # exp_P = exp(S - M[:, None])
    exp_P = exp(S - M[:, None])

    ### PEG
    ### exp_P = tl.exp(S - tl.broadcast_to(M, (Q_TILE_SIZE, N)))
    ### exp_P must be stored/loaded from GMEM because it is indexed below

    # L : [Q_TILE_SIZE]
    # L = sum(exp_P, dim=1)
    L: tl.tensor = tl.full((Q_TILE_SIZE,), 0.0, dtype=tl.float32)
    for j in range(Tk):
        L += exp_P[:, j*K_TILE_SIZE:(j+1)*K_TILE_SIZE]

    ### PEG
    ### L = theta_L(tl.full((Q_TILE_SIZE,), 0.0), exp_P[:, j*K_TILE_SIZE:(j+1)*K_TILE_SIZE])

    # softmax: [Q_TILE_SIZE, N]
    softmax = exp_P / L

    ### PEG
    ### softmax = exp_P / L
    ### softmax must be stored/loaded from GMEM because it is indexed below
    
    # O: [Q_TILE_SIZE, D]
    Olist = []
    for i in range(Tk):
        V_block: tl.tensor = tl.load(V_block_ptr)
        O_ij: tl.tensor = tl.dot(softmax[:, i*K_TILE_SIZE:(i+1)*K_TILE_SIZE], V_block.trans(1,0)) / scale
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        Olist.append(O)

    O = tl.cat(Olist)

    ### PEG
    ### V_block_ptr = theta_vblk(V_block_ptr, tl.advance(V_block_ptr, (K_TILE_SIZE, 0)))
    ### V_block = tl.load(V_block_ptr)
    ### O_ij = tl.dot(softmax[:, i*K_TILE_SIZE:(i+1)*K_TILE_SIZE], V_block.trans(1,0)) / scale
    ### O = tl.cat(O_ij)

    tl.store(O_block_ptr, O)
    tl.store(L_block_ptr, L)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal,
):
    # Program indices
    # launch grid is configured such that
    # 1 block = 1 Q / O tile in 1 element of the batch
    qtile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(qtile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(qtile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(qtile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    Q_block: tl.tensor = tl.load(Q_block_ptr).to(tl.float32)
    m_ij: tl.tensor = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    O_ij: tl.tensor = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_ij: tl.tensor = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    q_idx = (qtile_index * Q_TILE_SIZE) + tl.arange(0, Q_TILE_SIZE)
    for j in range(Tk):
        K_block: tl.tensor = tl.load(K_block_ptr)
        V_block: tl.tensor = tl.load(V_block_ptr)

        S_ij: tl.tensor = tl.dot(Q_block, K_block.trans(1,0)) / scale

        if is_causal:
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(causal_mask, S_ij, float("-inf"))

        rowmax_Sij = tl.max(S_ij, 1)
        prev_m_ij = m_ij
        m_ij = tl.maximum(prev_m_ij, rowmax_Sij)

        exp_P_ij = tl.exp(S_ij - m_ij[:, None])
        exp_m_sub = tl.exp(prev_m_ij - m_ij)
        l_ij = exp_m_sub * l_ij + tl.sum(exp_P_ij, 1)

        O_ij = (exp_m_sub[:, None] * O_ij) + tl.dot(exp_P_ij, V_block)

        # advance block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_ij / l_ij[:, None]
    L_i = m_ij + tl.log(l_ij)

    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)

### FLASHATTENTION IN PEG
###
### Tk = tl.cdiv(N_KEY, K_TILE_SIZE)
###
### Q_block_ptr = tl.make_block_ptr(Q_ptr ...)
### K_block_ptr = theta_Kblk(tl.make_block_ptr(K_ptr ...), tl.advance(theta_Kblk, (K_TILE_SIZE, 0)))
### V_block_ptr = theta_Vblk(tl.make_block_ptr(V_ptr ...), tl.advance(theta_Vblk, (K_TILE_SIZE, 0)))
### O_block_ptr = tl.make_block_ptr(O_ptr ...)
### L_block_ptr = tl.make_block_ptr(L_ptr ...)
### 
### Q_block = tl.load(Q_block_ptr)
###
### K_block = tl.load(K_block_ptr)
### V_block = tl.load(V_block_ptr)
### S_ij = tl.dot(Q_block, K_block.trans(1,0)) / scale
### rowmax_Sij = tl.max(S_ij, 1)
### m_ij = theta_mij(tl.full((Q_TILE_SIZE,), -inf), tl.maximum(theta_mij, rowmax_Sij))
### exp_P_ij = tl.exp(S_ij - m_ij[:, None])
### exp_m_sub = tl.exp(theta_mij - tl.maximum(theta_mij, rowmax_Sij))
### l_ij = theta_lij(tl.zeros((Q_TILE_SIZE,)), exp_m_sub * l_ij + tl.sum(exp_P_ij, 1))
### O_ij = theta_Oij(tl.zeros((Q_TILE_SIZE, D)), (exp_m_sub[:, None] * O_ij) + tl.dot(exp_P_ij, V_block))
###
### O_i = O_ij / eval(Tk, l_ij) # eval(Tk, l_ij) returns the value of l_ij at index Tk-1
### L_i = eval(Tk, m_ij) + tl.log(l_ij)
###
### store1 = tl.store(O_block_ptr, O_i)
### store2 = tl.store(L_block_ptr, L_i)

class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        BATCH_SIZE, NQ, NK, d = Q.size(0), Q.size(1), K.size(1), Q.size(2)

        Q_TILE_SIZE, K_TILE_SIZE = 16, 16
        scale = math.sqrt(d)

        O = torch.empty((BATCH_SIZE, NQ, d), device=Q.device, dtype=torch.float32)
        L = torch.empty((BATCH_SIZE, NQ), device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(NQ // Q_TILE_SIZE, BATCH_SIZE)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            NQ, NK,
            scale,
            d,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O

def benchmark_attention():
    results = []
    for ctx_length in [128, 256, 512, 1028, 2048, 4096, 8192]:
        for embed_dim in [16, 32, 64, 128]:
            print(f"running benchmark for ctx_length {ctx_length}, embed_dim {embed_dim}")

            Q = torch.rand((1, ctx_length, 16), device="cuda")
            K = torch.rand((1, ctx_length, 16), device="cuda")
            V = torch.rand((1, ctx_length, 16), device="cuda")

            triton_callable = lambda: TritonFlashAttention.apply(Q, K, V, True)
            triton_res = triton.testing.do_bench(triton_callable, warmup=25, rep=100, quantiles=[0.25, 0.50, 0.75], return_mode="median")

            torch.cuda.synchronize()

            torch_callable = lambda: TorchFlashAttention.apply(Q, K, V, True)
            torch_res = triton.testing.do_bench(torch_callable, warmup=25, rep=100, quantiles=[0.25, 0.50, 0.75], return_mode="median")

            torch.cuda.synchronize()

            print(f"ctx_length {ctx_length}, embed_dim {embed_dim}")
            print(f"triton: {triton_res}")
            print(f"torch: {torch_res}")

if __name__ == "__main__":
    benchmark_attention()