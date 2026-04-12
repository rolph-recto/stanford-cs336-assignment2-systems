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

demo_vector_add_triton_ir()