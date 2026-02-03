"""
txt2memmap.py
把任意大小的纯文本文件转成 uint16 的 memmap，供后续训练直接切片。
"""

import numpy as np
from .tokenizer import Tokenizer

CHUNK_BYTES = 50 * 1024 * 1024  # 每次读 50 MB，可调
DTYPE = np.uint16                # 词汇表 < 65535 时用 uint16 省空间

def count_tokens(text_path, tokenizer):
    """第一遍：只统计总 token 数，不保存。"""
    total = 0
    with open(text_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(tokenizer.encode(chunk))
    return total

def write_memmap(text_path, tokenizer, mmap_path, total_tokens):
    """第二遍：正式写 memmap。"""
    mm = np.memmap(mmap_path, dtype=DTYPE, mode="w+", shape=(total_tokens,))
    idx = 0
    with open(text_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            ids = tokenizer.encode(chunk)
            mm[idx : idx + len(ids)] = ids
            idx += len(ids)
    mm.flush()
    print(f"finished. shape={mm.shape}, file={mmap_path}")

def main():
    text_file = "./cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
    mmap_file = "./cs336_basics/data/TinyStoriesV2-GPT4-train.memmap"
    
    special_tokens = ["<|endoftext|>"]
    model_prefix="bpe"
    vocab_file = f"./cs336_basics/TinyStories_bpe_results/{model_prefix}.vocab"
    merges_filepath = f"./cs336_basics/TinyStories_bpe_results/{model_prefix}.merges"
    tokenizer = Tokenizer.from_files(vocab_file, merges_filepath, special_tokens=special_tokens)


    print("counting tokens ...")
    total = count_tokens(text_file, tokenizer)
    print(f"total tokens = {total:,}")

    print("writing memmap ...")
    write_memmap(text_file, tokenizer, mmap_file, total)

if __name__ == "__main__":
    main()