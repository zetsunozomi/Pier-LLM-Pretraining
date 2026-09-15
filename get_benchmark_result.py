import re
import argparse
import os

def extract_elapsed_times(log_file_path):
    """
    从日志文件中提取 elapsed time per iteration (ms) 的数值并存入列表
    """
    elapsed_times = []
    
    # 正则表达式说明：
    # elapsed time per iteration \(ms\):  匹配固定文本，注意转义括号
    # \s* 匹配可能存在的空格
    # ([\d.]+)                             捕获组：匹配数字或小数点
    pattern = re.compile(r'elapsed time per iteration \(ms\):\s*([\d.]+)')

    if not os.path.exists(log_file_path):
        print(f"错误: 找不到文件 {log_file_path}")
        return []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 提取第一个捕获组并转为 float
                value = float(match.group(1))
                elapsed_times.append(value)
    
    return elapsed_times

if __name__ == "__main__":
    # 参数化 log 路径
    parser = argparse.ArgumentParser(description="从训练日志中提取迭代耗时")
    parser.add_argument("--path", type=str, required=True, help="日志文件的路径")
    
    args = parser.parse_args()

    # 执行提取
    times_list = extract_elapsed_times(args.path)

    # 打印结果示例
    print(f"总共提取到 {len(times_list)} 条迭代记录")
    if times_list:
        print(f"平均迭代耗时: {sum(times_list) / len(times_list):.2f} ms")