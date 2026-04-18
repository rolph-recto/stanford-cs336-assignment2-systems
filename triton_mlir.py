import torch
import einops
import triton
import triton.compiler as triton_compiler
import triton.language as tl
import math
from triton.backends.compiler import AttrsDescriptor, GPUTarget
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction


def cdiv(x, y):
  return (x + y - 1) // y


def get_default_triton_target():
  try:
    return driver.active.get_current_target()
  except RuntimeError:
    # Fall back to a reasonable default when no GPU driver is active.
    return GPUTarget("cuda", 80, 32)


def jit_compile_to_triton_ir(
  kernel,
  signature,
  constants=None,
  *,
  num_warps=4,
  num_stages=3,
  target=None,
):
  target = target or get_default_triton_target()
  backend = triton_compiler.make_backend(target)
  options = backend.parse_options({
    "num_warps": num_warps,
    "num_stages": num_stages,
  })
  context = triton_compiler.compiler.ir.context()
  triton_compiler.compiler.ir.load_dialects(context)
  backend.load_dialects(context)

  src = triton_compiler.ASTSource(
    fn=kernel,
    signature=signature,
    constants=constants or {},
    attrs=AttrsDescriptor(),
  )
  module = src.make_ir(
    options,
    backend.get_codegen_implementation(),
    backend.get_module_map(),
    context,
  )
  return str(module)


def _vector_add_kernel(
  x_ptr,
  y_ptr,
  out_ptr,
  n_elements,
  BLOCK_SIZE: tl.constexpr,
):
  pid = tl.program_id(0)
  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  tl.store(out_ptr + offsets, x + y, mask=mask)


vector_add_kernel = JITFunction(_vector_add_kernel)

def _flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal,
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

flash_fwd_kernel = JITFunction(_flash_fwd_kernel)

def demo_vector_add_triton_ir():
  mod_str = jit_compile_to_triton_ir(
    vector_add_kernel,
    signature={
      "x_ptr": "*fp32",
      "y_ptr": "*fp32",
      "out_ptr": "*fp32",
      "n_elements": "i32",
    },
    constants={"BLOCK_SIZE": 1024},
  )
  print(mod_str)

def demo_flash_attention_ir():
  mod_str = jit_compile_to_triton_ir(
    flash_fwd_kernel,
    signature={
      "Q_ptr": "*fp32",
      "K_ptr": "*fp32",
      "V_ptr": "*fp32",
      "O_ptr": "*fp32",
      "L_ptr": "*fp32",
      "stride_qb": "i32",
      "stride_qq": "i32",
      "stride_qd": "i32",
      "stride_kb": "i32",
      "stride_kk": "i32",
      "stride_kd": "i32",
      "stride_vb": "i32",
      "stride_vk": "i32",
      "stride_vd": "i32",
      "stride_ob": "i32",
      "stride_oq": "i32",
      "stride_od": "i32",
      "stride_lb": "i32",
      "stride_lq": "i32",
      "N_QUERIES": "i32",
      "N_KEYS": "i32",
      "scale": "fp32",
      "is_causal": "B",
    },
    constants={
      "D": 128,
      "Q_TILE_SIZE": 16,
      "K_TILE_SIZE": 16,
    }
  )
  print(mod_str)

demo_flash_attention_ir()