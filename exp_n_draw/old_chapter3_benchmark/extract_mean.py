import re
import statistics
import sys

def parse_times(log_file, min_iter, max_iter):
    """
    Parse the log file and extract 'elapsed time per iteration (ms)' values
    only for iterations in the given range [min_iter, max_iter].
    """
    times = []
    pattern_iter = re.compile(r"iteration\s+(\d+)/")
    pattern_time = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")

    with open(log_file, 'r') as f:
        for line in f:
            m_iter = pattern_iter.search(line)
            if m_iter:
                iter_num = int(m_iter.group(1))
                if iter_num < min_iter:
                    continue
                if iter_num > max_iter:
                    break
                m_time = pattern_time.search(line)
                if m_time:
                    times.append(float(m_time.group(1)))
    return times


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <log_file_path>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    min_iter, max_iter = 200, 500

    times = parse_times(log_file, min_iter, max_iter)
    print(f"Extracted times (iterations {min_iter}-{max_iter}):", times)

    if times:
        # 跳过第一个值，和你原本保持一致
        print("Mean elapsed time (ms):", statistics.mean(times[1:]))
    else:
        print(f"No times found in iterations {min_iter}-{max_iter}")


if __name__ == "__main__":
    main()
