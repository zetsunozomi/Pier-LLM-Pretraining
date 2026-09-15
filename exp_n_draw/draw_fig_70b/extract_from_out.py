from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "out.txt"
GPU_LABELS = (64, 128, 256)

HEADER_RE = re.compile(r"^(\d+)gpu:\s*$")
ITERATION_RE = re.compile(r"iteration\s+(\d+)/\s*220")
ELAPSED_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")


def parse_groups() -> dict[int, list[str]]:
    groups = {gpu: [] for gpu in GPU_LABELS}
    current_gpu = None

    for raw_line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        header = HEADER_RE.match(line)
        if header:
            current_gpu = int(header.group(1))
            if current_gpu not in groups:
                groups[current_gpu] = []
            continue

        if current_gpu is not None and line:
            groups[current_gpu].append(line)

    return groups


def elapsed_by_iteration(lines: list[str]) -> dict[int, float]:
    values = {}
    for line in lines:
        iteration = ITERATION_RE.search(line)
        elapsed = ELAPSED_RE.search(line)
        if iteration and elapsed:
            values[int(iteration.group(1))] = float(elapsed.group(1))
    return values


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    groups = parse_groups()
    summary_lines = [
        "gpu,ddp_ms,malt_ms,ddp_iterations,malt_iterations",
    ]

    for gpu in GPU_LABELS:
        lines = groups[gpu]
        split_path = BASE_DIR / f"out_{gpu}gpu.txt"
        split_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        elapsed = elapsed_by_iteration(lines)
        # Full DDP is represented by the warmup/initial section: iterations 10, 20, and 30.
        ddp_iterations = [10, 20, 30]
        # MALT is represented by the later section: iterations 30, 40, ..., 220.
        malt_iterations = [iteration for iteration in sorted(elapsed) if iteration >= 30]
        ddp_ms = average([elapsed[iteration] for iteration in ddp_iterations])
        malt_ms = average([elapsed[iteration] for iteration in malt_iterations])

        summary_lines.append(
            f"{gpu},{ddp_ms:.10f},{malt_ms:.10f},"
            f"{' '.join(map(str, ddp_iterations))},"
            f"{' '.join(map(str, malt_iterations))}"
        )

    summary_path = BASE_DIR / "out_summary.csv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    for gpu in GPU_LABELS:
        print(f"wrote {BASE_DIR / f'out_{gpu}gpu.txt'}")


if __name__ == "__main__":
    main()
