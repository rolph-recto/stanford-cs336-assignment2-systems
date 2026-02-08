import statistics
import sys
import argparse
from typing import Literal

from tqdm import tqdm
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
import timeit
import torch

import numpy as np
import numpy.typing as npt

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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
    mode: Literal["forward"] | Literal["both"],
):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

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
    return mean, sd

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

    mean, sd = benchmark(
        vocab_size=args.vocab,
        context_length=args.context,
        d_model=args.dmodel,
        num_layers=args.layers,
        num_heads=args.heads,
        batch_size=args.batch,
        d_ff=args.dff,
        rope_theta=args.theta,
        warmup=args.warmup,
        runs=args.runs,
        mode=args.mode
    )
    print(f"mean: {mean:.2f}\nstd dev: {sd:.2f}")

if __name__ == "__main__":
    main()