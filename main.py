import argparse
import statistics
import timeit
from typing import Literal
from typing import TypedDict

import modal
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from tqdm import tqdm

BenchmarkMode = Literal["forward", "both"]
NGC_PYTORCH_IMAGE = "nvcr.io/nvidia/pytorch:26.02-py3"


class BenchmarkResult(TypedDict):
    mean: float
    std_dev: float
    device: str
    torch_version: str
    image_tag: str


app = modal.App("cs336-assignment2-benchmark")
image = (
    modal.Image.from_registry(NGC_PYTORCH_IMAGE)
    .uv_pip_install("tqdm", "einops", "einx", "jaxtyping")
    .add_local_python_source("cs336_basics")
)


def normalize_mode(mode: str) -> BenchmarkMode:
    if mode not in ("forward", "both"):
        raise ValueError("mode must be one of: forward, both")
    return mode

def benchmark(*,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    d_ff: int,
    rope_theta: int,
    warmup: int,
    runs: int,
    mode: BenchmarkMode,
) -> BenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = \
        BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta
        )
    model.to(device)

    def run_step():
        input = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        target = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        output = model(input)

        if mode == "both":
            loss = cross_entropy(output, target)
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        run_step()

    ts = []
    for _ in tqdm(range(runs)):
        t = timeit.timeit(stmt="run_step()", number=10, globals=locals())
        ts.append(t)

    mean = statistics.mean(ts)
    sd = statistics.stdev(ts)
    return {
        "mean": mean,
        "std_dev": sd,
        "device": device.type,
        "torch_version": torch.__version__,
        "image_tag": "local",
    }


def format_result(result: BenchmarkResult) -> str:
    return (
        f"mean: {result['mean']:.2f}\n"
        f"std dev: {result['std_dev']:.2f}\n"
        f"device: {result['device']}\n"
        f"torch version: {result['torch_version']}\n"
        f"image: {result['image_tag']}"
    )


def benchmark_kwargs(*,
    dmodel: int,
    dff: int,
    layers: int,
    heads: int,
    batch: int,
    vocab: int,
    context: int,
    theta: int,
    warmup: int,
    runs: int,
    mode: str,
) -> dict[str, int | BenchmarkMode]:
    return {
        "vocab_size": vocab,
        "context_length": context,
        "d_model": dmodel,
        "num_layers": layers,
        "num_heads": heads,
        "batch_size": batch,
        "d_ff": dff,
        "rope_theta": theta,
        "warmup": warmup,
        "runs": runs,
        "mode": normalize_mode(mode),
    }


@app.function(image=image, gpu="L4", timeout=3600, cpu=4, memory=16384)
def run_benchmark(
    dmodel: int = 768,
    dff: int = 3072,
    layers: int = 12,
    heads: int = 12,
    batch: int = 16,
    vocab: int = 10_000,
    context: int = 256,
    theta: int = 10_000,
    warmup: int = 5,
    runs: int = 10,
    mode: str = "both",
) -> BenchmarkResult:
    result = benchmark(**benchmark_kwargs(
        dmodel=dmodel,
        dff=dff,
        layers=layers,
        heads=heads,
        batch=batch,
        vocab=vocab,
        context=context,
        theta=theta,
        warmup=warmup,
        runs=runs,
        mode=mode,
    ))
    result["image_tag"] = NGC_PYTORCH_IMAGE
    return result


@app.local_entrypoint()
def modal_entrypoint(
    dmodel: int = 768,
    dff: int = 3072,
    layers: int = 12,
    heads: int = 12,
    batch: int = 16,
    vocab: int = 10_000,
    context: int = 256,
    theta: int = 10_000,
    warmup: int = 5,
    runs: int = 10,
    mode: str = "both",
):
    result = run_benchmark.remote(
        dmodel=dmodel,
        dff=dff,
        layers=layers,
        heads=heads,
        batch=batch,
        vocab=vocab,
        context=context,
        theta=theta,
        warmup=warmup,
        runs=runs,
        mode=mode,
    )
    print(format_result(result))

def main():
    parser = argparse.ArgumentParser("cs336-assignment2")
    parser.add_argument("--dmodel", type=int, default=768, help="d_model")
    parser.add_argument("--dff", type=int, default=3072, help="d_ff")
    parser.add_argument("--layers", type=int, default=12, help="num layers")
    parser.add_argument("--heads", type=int, default=12, help="num heads")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--vocab", type=int, default=10_000, help="vocab size")
    parser.add_argument("--context", type=int, default=256, help="context length")
    parser.add_argument("--theta", type=int, default=10_000, help="RoPE theta")
    parser.add_argument("--warmup", type=int, default=5, help="RoPE theta")
    parser.add_argument("--runs", type=int, default=10, help="number of runs")
    parser.add_argument("--mode", type=str, default="both", choices=["both","forward"], help="measure both forward/backward passes or forward only")
    args = parser.parse_args()

    result = benchmark(**benchmark_kwargs(
        dmodel=args.dmodel,
        dff=args.dff,
        layers=args.layers,
        heads=args.heads,
        batch=args.batch,
        vocab=args.vocab,
        context=args.context,
        theta=args.theta,
        warmup=args.warmup,
        runs=args.runs,
        mode=args.mode,
    ))
    print(format_result(result))

if __name__ == "__main__":
    main()
