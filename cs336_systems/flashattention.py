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
            

class TritonFlashAttention(torch.autograd.Function):
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

        for j in range(N_KEYS // K_TILE_SIZE):
            pass