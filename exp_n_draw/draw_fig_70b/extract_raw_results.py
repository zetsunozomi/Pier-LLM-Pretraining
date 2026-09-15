from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "70B_project"
OUTPUT_PATH = BASE_DIR / "out.txt"

LOG_PATTERN = "70B_*gpu_*.out"
ITERATION_LINE = re.compile(
    r"^\s*\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] "
    r"iteration\s+\d+/\s+\d+\s+\| consumed samples:"
)


def main() -> None:
    logs = sorted(LOG_DIR.glob(LOG_PATTERN))
    extracted_lines = []

    for log_path in logs:
        with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                if ITERATION_LINE.match(line):
                    extracted_lines.append(line.strip())

    OUTPUT_PATH.write_text("\n".join(extracted_lines) + "\n", encoding="utf-8")
    print(f"read {len(logs)} logs")
    print(f"wrote {len(extracted_lines)} lines to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
