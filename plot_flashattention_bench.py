import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HEADER_RE = re.compile(r"ctx_length (?P<ctx_length>\d+), embed_dim (?P<embed_dim>\d+)")
RUNTIME_RE = re.compile(r"(?P<impl>triton|torch): (?P<values>\[.*\])")


def parse_benchmark(path: Path) -> dict[int, dict[int, dict[str, list[float]]]]:
    results: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    current: tuple[int, int] | None = None

    for line in path.read_text().splitlines():
        header_match = HEADER_RE.fullmatch(line.strip())
        if header_match:
            current = (int(header_match.group("ctx_length")), int(header_match.group("embed_dim")))
            continue

        runtime_match = RUNTIME_RE.fullmatch(line.strip())
        if runtime_match and current is not None:
            ctx_length, embed_dim = current
            impl = runtime_match.group("impl")
            results[embed_dim][ctx_length][impl] = ast.literal_eval(runtime_match.group("values"))

    return results


def plot_benchmark(data: dict[int, dict[int, dict[str, list[float]]]], output: Path) -> None:
    embed_dims = sorted(data)
    if not embed_dims:
        raise ValueError("No benchmark records found.")

    fig, (upper_axis, lower_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06},
    )

    impl_labels = {
        "torch": "torch.compile",
        "triton": "Triton",
    }
    impl_axes = {
        "torch": upper_axis,
        "triton": lower_axis,
    }
    impl_styles = {
        "torch": {"linestyle": "-", "marker": "o"},
        "triton": {"linestyle": "--", "marker": "s"},
    }
    colors = {embed_dim: f"C{index}" for index, embed_dim in enumerate(embed_dims)}
    all_ctx_lengths = sorted({ctx_length for by_ctx_length in data.values() for ctx_length in by_ctx_length})
    ctx_positions = {ctx_length: index for index, ctx_length in enumerate(all_ctx_lengths)}
    offset_step = 0.12
    midpoint = (len(embed_dims) - 1) / 2
    dim_offsets = {
        embed_dim: (index - midpoint) * offset_step
        for index, embed_dim in enumerate(embed_dims)
    }

    runtime_ranges: dict[str, list[float]] = {"torch": [], "triton": []}

    for embed_dim in embed_dims:
        ctx_lengths = sorted(data[embed_dim])
        x_positions = [ctx_positions[ctx_length] + dim_offsets[embed_dim] for ctx_length in ctx_lengths]
        for impl, style in impl_styles.items():
            q25 = [data[embed_dim][ctx_length][impl][0] for ctx_length in ctx_lengths]
            medians = [data[embed_dim][ctx_length][impl][1] for ctx_length in ctx_lengths]
            q75 = [data[embed_dim][ctx_length][impl][2] for ctx_length in ctx_lengths]
            runtime_ranges[impl].extend([min(q25), max(q75)])
            yerr = [
                [median - lower for lower, median in zip(q25, medians, strict=True)],
                [upper - median for upper, median in zip(q75, medians, strict=True)],
            ]
            impl_axes[impl].errorbar(
                x_positions,
                medians,
                yerr=yerr,
                capsize=4,
                linewidth=2,
                markersize=6,
                color=colors[embed_dim],
                label=f"{impl_labels[impl]}, d={embed_dim}",
                **style,
            )

    upper_axis.set_ylim(min(runtime_ranges["torch"]) * 0.85, max(runtime_ranges["torch"]) * 1.15)
    lower_axis.set_ylim(min(runtime_ranges["triton"]) * 0.85, max(runtime_ranges["triton"]) * 1.15)

    for axis in (upper_axis, lower_axis):
        axis.set_yscale("log")
        axis.grid(True, which="both", axis="y", alpha=0.25)
        axis.grid(True, which="major", axis="x", alpha=0.15)

    upper_axis.spines["bottom"].set_visible(False)
    lower_axis.spines["top"].set_visible(False)
    upper_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    lower_axis.tick_params(axis="x", which="both", top=False)

    break_size = 0.012
    break_kwargs = {"color": "black", "clip_on": False, "linewidth": 1}
    upper_axis.plot((-break_size, +break_size), (-break_size, +break_size), transform=upper_axis.transAxes, **break_kwargs)
    upper_axis.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, +break_size),
        transform=upper_axis.transAxes,
        **break_kwargs,
    )
    lower_axis.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), transform=lower_axis.transAxes, **break_kwargs)
    lower_axis.plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        transform=lower_axis.transAxes,
        **break_kwargs,
    )

    dim_handles = [
        Line2D([0], [0], color=colors[embed_dim], linewidth=2, label=f"d={embed_dim}") for embed_dim in embed_dims
    ]
    impl_handles = [
        Line2D([0], [0], color="black", linewidth=2, label=impl_labels[impl], **style)
        for impl, style in impl_styles.items()
    ]
    upper_axis.legend(handles=dim_handles + impl_handles, ncols=3, frameon=False, loc="upper left")

    upper_axis.set_title("FlashAttention Benchmark: torch.compile vs Triton")
    lower_axis.set_xlabel("Context length")
    lower_axis.set_xticks(list(ctx_positions.values()), [str(ctx_length) for ctx_length in all_ctx_lengths])
    lower_axis.set_xlim(-0.5, len(all_ctx_lengths) - 0.5)
    fig.supylabel("Runtime (ms, log scale)")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.08, hspace=0.06)
    fig.savefig(output, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FlashAttention benchmark runtimes.")
    parser.add_argument("input", nargs="?", type=Path, default=Path("flashattention_bench.txt"))
    parser.add_argument("-o", "--output", type=Path, default=Path("flashattention_bench.png"))
    args = parser.parse_args()

    data = parse_benchmark(args.input)
    plot_benchmark(data, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
