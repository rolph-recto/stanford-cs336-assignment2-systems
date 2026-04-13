import math

import torch
import triton
import triton.language as tl

class TorchFlashAttention(torch.autograd.Function):
    # Q, K, V: [B, N, d]
    # TODO: implement causal masking
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
            m_ij = torch.full((B, TILE_SIZE,), -torch.inf)
            l_ij = torch.zeros(B, TILE_SIZE)
            O_ij = torch.zeros((B, TILE_SIZE, d))

            for j in range(num_tiles):
                # S_ij: [B, TILE_SIZE, TILE_SIZE]
                S_ij = (Qs[i] @ Ks[j].transpose(1,2)) / math.sqrt(d)

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
        strides=(stride_qq, stride_qd),
        offsets=(qtile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(qtile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, D),
        strides=(stride_lq,),
        offsets=(qtile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE),
        order=(0,),
    )

    Tk = N_KEYS // K_TILE_SIZE

    m_ij: tl.tensor = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    O_ij: tl.tensor = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_ij: tl.tensor = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    for j in range(Tk):
        Q_block: tl.tensor = tl.load(Q_block_ptr)
        K_block: tl.tensor = tl.load(K_block_ptr)
        V_block: tl.tensor = tl.load(V_block_ptr)

        S_ij: tl.tensor = tl.dot(Q_block, K_block.trans((1,0))) / scale

        rowsum_Sij = S_ij.sum(1)
        prev_m_ij = m_ij
        m_ij = tl.maximum(prev_m_ij, rowsum_Sij)

        exp_P_ij = tl.exp(S_ij - m_ij[:, None])
        exp_m_sub = tl.exp(prev_m_ij - m_ij)
        l_ij = exp_m_sub * l_ij + exp_P_ij.sum(1)

        O_ij = (exp_m_sub[:, None] * O_ij) + tl.dot(exp_P_ij, V_block)

        # advance block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(O_block_ptr, O_ij)
    tl.store(L_block_ptr, l_ij)

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
            Q.stride(0), Q.stride(1), Q.stride(2),
            L.stride(0), L.stride(1),
            NQ, NK,
            scale,
            tl.constexpr(d),
            tl.constexpr(Q_TILE_SIZE),
            tl.constexpr(K_TILE_SIZE),
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return 