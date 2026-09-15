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
    min_iter, max_iter = 100, 500

    times = parse_times(log_file, min_iter, max_iter)[1:]
    print(f"Extracted times (iterations {min_iter}-{max_iter}):", times)

    big_nums = times[0::5]  # elements at index 0,5,10,...
    small_nums = [t for i, t in enumerate(times) if i % 5 != 0]
    print(f"bigs: {big_nums}")
    print(f"len big {len(big_nums)}")
    print(f"smalls: {small_nums}")
    print(f"len small {len(small_nums)}")
    small_avg = statistics.mean(small_nums)
    big_avg = statistics.mean(big_nums)
    if times:
        print("Mean elapsed time (ms):", statistics.mean(times))
        print(f"big average: {big_avg}")
        print(f"small average: {small_avg}")
        print(f"calculated 50: {(small_avg* 4 + big_avg)/5}")
        print(f"calcualte 200: {(small_avg* 19 + big_avg)/20}")
        print(f"calcualte 500: {(small_avg* 49 + big_avg)/50}")
    else:
        print(f"No times found in iterations {min_iter}-{max_iter}")


if __name__ == "__main__":
    main()
