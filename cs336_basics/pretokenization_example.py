import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件分块，每个块可以独立计数。
    如果边界最终重叠，可能返回少于期望数量的块。
    """
    assert isinstance(split_special_token, bytes), "特殊令牌必须表示为字节字符串"

    # 获取文件总大小（字节）
    file.seek(0, os.SEEK_END)  # 移动到文件末尾
    file_size = file.tell()    # 获取当前位置（即文件大小）
    file.seek(0)               # 回到文件开头

    chunk_size = file_size // desired_num_chunks  # 计算每个块的期望大小

    # 初始猜测的块边界位置，均匀分布
    # 块从前一个索引开始，不包含最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 最后一个边界设为文件大小

    mini_chunk_size = 4096  # 每次向前读取4KB字节

    # 调整边界，使其落在特殊令牌之后
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]  # 初始边界位置
        file.seek(initial_position)  # 移动到边界猜测位置
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块

            # 如果到达文件末尾，将此边界设为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊令牌
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:  # 如果找到特殊令牌
                # 将边界设置在特殊令牌之后的位置
                chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                break
            initial_position += mini_chunk_size  # 没找到，继续向前搜索

    # 确保所有边界都是唯一的，但可能少于期望的块数
    return sorted(set(chunk_boundaries))


# ## 使用示例
# with open(..., "rb") as f:  # 以二进制模式打开文件
#     num_processes = 4  # 进程数量
#     # 查找分块边界，使用"<|endoftext|>"作为分隔符
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # 以下是串行实现，但你可以通过将每个开始/结束对发送到一组进程来并行化
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)  # 移动到块开始位置
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")  # 读取块内容并解码
#         # 在你的块上运行预分词，并存储每个预分词的计数