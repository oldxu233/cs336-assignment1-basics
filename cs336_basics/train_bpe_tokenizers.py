# train_bpe_tokenizer.py
import os
import time
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex
from typing import List, Tuple, Dict, Set
import json
from pathlib import Path

def bytes_to_escaped(b: bytes) -> str:
    out = []
    for byte in b:
        if 32 <= byte <= 126 and byte != 92:
            out.append(chr(byte))
        else:
            out.append(f"\\x{byte:02x}")
    return "".join(out)

def escaped_to_bytes(s: str) -> bytes:
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 3 < len(s) and s[i+1] == "x":
            hex_str = s[i+2:i+4]
            try:
                val = int(hex_str, 16)
                out.append(val)
                i += 4
                continue
            except ValueError:
                pass
        out.append(ord(s[i]))
        i += 1
    return bytes(out)

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    t0 = time.time()
    # 过滤掉空字符串特殊 token（tokenizers 不允许）
    special_tokens = [t for t in special_tokens if t]

    # 1. 初始化 tokenizer
    tokenizer = Tokenizer(models.BPE())

    # 2. 设置预分词器（与你原 regex 一致）
    pre_tokenizer_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(pre_tokenizer_regex),
        behavior="isolated"
    )

    # 3. 训练
    t_read_start = time.time()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train(files=[input_path], trainer=trainer)
    t_read_end = time.time()

    # 4. 保存为临时 JSON 以提取 vocab 和 merges
    t_bpe_start = time.time()
    temp_file = "__temp_bpe_model.json"
    tokenizer.save(temp_file)

    with open(temp_file, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    os.remove(temp_file)

    # --- 构建 vocab: Dict[int, bytes] ---
    vocab_dict_str_to_id = model_data["model"]["vocab"]  # Dict[str, int]
    vocab: Dict[int, bytes] = {}
    for token_str, token_id in vocab_dict_str_to_id.items():
        vocab[token_id] = token_str.encode("utf-8")

    # --- 构建 merges: List[Tuple[bytes, bytes]] ---
    merges_list = model_data["model"]["merges"]  # List[str], e.g., ["a b", "ab c"]
    merges: List[Tuple[bytes, bytes]] = []
    # print("\n merges_list: ", merges_list) # [[' ', 't'], ['h', 'e'], [' ', 'a'], [' ', 's'],]
    # print("\n type of merges_list: ", type(merges_list)) # type of merges_list:  <class 'list'>
    for merge_pair in merges_list:
        if len(merge_pair) == 2:
            a_str, b_str = merge_pair
            merges.append((a_str.encode("utf-8"), b_str.encode("utf-8")))
        else:
            raise ValueError(f"Unexpected merge format: {merge_pair}")

    t_bpe_end = time.time()
    t1 = time.time()

    print(f"总耗时: {t1 - t0:.4f} 秒")
    print(f"  读文件耗时: {t_read_end - t_read_start:.4f} 秒")
    print(f"  BPE 循环耗时: {t_bpe_end - t_bpe_start:.4f} 秒")
    print(f"  合并次数: {len(merges)}")
    print(f"  最终词表大小: {len(vocab)}")

    return vocab, merges

def bytes_to_escaped(b: bytes) -> str:
    """
    把任意 bytes 转成可打印字符串：
    - 可见 ASCII 直接输出
    - 其它字节用 \\xNN 形式
    """
    out = []
    for byte in b:
        # 可见 ASCII，且不是反斜杠
        if 32 <= byte <= 126 and byte != 92:  # 92 是 '\'
            out.append(chr(byte))
        else:
            out.append(f"\\x{byte:02x}")
    return "".join(out)


def escaped_to_bytes(s: str) -> bytes:
    """
    反向解析 bytes_to_escaped 的输出：
    解析 \\xNN 序列，还原为原始 bytes。
    """
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 3 < len(s) and s[i+1] == "x":
            # 形如 \xNN
            hex_str = s[i+2:i+4]
            try:
                val = int(hex_str, 16)
                out.append(val)
                i += 4
                continue
            except ValueError:
                # 解析失败，就按普通字符处理
                pass
        # 普通字符
        out.append(ord(s[i]))
        i += 1
    return bytes(out)

# 将vocab和merges保存为.vocab和.merges文件(.vocab: dict[int, bytes], .merges: list[tuple[bytes, bytes]])
def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_file, merges_file):
    
    with open(vocab_file, "w", encoding="utf-8") as f:
        for id_val in sorted(vocab.keys()):
            bytes_val = vocab[id_val]
            
            escaped = bytes_to_escaped(bytes_val)
            f.write(f"{id_val}\t{escaped}\n")
    print(f"词表已保存到: {vocab_file}")

    
    with open(merges_file, 'w', encoding='utf-8') as f:
        for a, b in merges:
            try:
                a_escaped = bytes_to_escaped(a)
                b_escaped = bytes_to_escaped(b)
                f.write(f"{a_escaped}\t{b_escaped}\n")
            except:
                print("合并规则文件出错了奥")
    print(f"合并规则已保存到: {merges_file}")
    
    return vocab_file, merges_file

def load_vocab(path: str) -> Dict[int, bytes]:
    vocab = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        _id, _esc = ln.split("\t", 1)
        vocab[int(_id)] = escaped_to_bytes(_esc)
    return vocab

def load_vocab_reverse(path: str) -> Dict[bytes, int]:
    """返回  bytes->id  的映射，方便做集合运算"""
    vocab = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        _id, _esc = ln.split("\t", 1)
        vocab[escaped_to_bytes(_esc)] = int(_id)
    return vocab

def load_merges(path: str) -> List[Tuple[bytes, bytes]]:
    merges = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        a_esc, b_esc = ln.split("\t", 1)
        merges.append((escaped_to_bytes(a_esc), escaped_to_bytes(b_esc)))
    return merges

def compare_bpe_models(vocab1: str, merges1: str,
                       vocab2: str, merges2: str):
    v1, m1 = load_vocab(vocab1), load_merges(merges1)
    v2, m2 = load_vocab(vocab2), load_merges(merges2)

    # --- vocab 对比 ---
    ids1, ids2 = set(v1.keys()), set(v2.keys())
    print("=== Vocab ID 差异 ===")
    print("只在模型1里的ID:", sorted(ids1 - ids2))
    print("只在模型2里的ID:", sorted(ids2 - ids1))

    # 对共同 ID，看 bytes 是否相同
    common_ids = ids1 & ids2
    diff_tokens = {i: (v1[i], v2[i]) for i in common_ids if v1[i] != v2[i]}
    if diff_tokens:
        print("--- 共同 ID 但 token 不同 ---")
        for _id, (b1, b2) in diff_tokens.items():
            print(f"ID {_id}: 模型1={bytes_to_escaped(b1)} 模型2={bytes_to_escaped(b2)}")
    else:
        print("共同 ID 的 token 完全一致。")

    # --- merges 对比 ---
    set1, set2 = set(m1), set(m2)
    print("\n=== Merge 规则差异 ===")
    print("只在模型1里的 merges:", len(set1 - set2))
    for a, b in sorted(set1 - set2):
        print("  ", bytes_to_escaped(a), "+", bytes_to_escaped(b))
    print("只在模型2里的 merges:", len(set2 - set1))
    for a, b in sorted(set2 - set1):
        print("  ", bytes_to_escaped(a), "+", bytes_to_escaped(b))
def diff_vocab_bytes(vocab_file1: str, vocab_file2: str):
    v1 = load_vocab_reverse(vocab_file1)   # bytes -> id
    v2 = load_vocab_reverse(vocab_file2)

    set1: Set[bytes] = set(v1.keys())
    set2: Set[bytes] = set(v2.keys())

    only1 = set1 - set2
    only2 = set2 - set1

    print("=== 只出现在模型1的 token ===")
    for tok in sorted(only1):
        print(bytes_to_escaped(tok))

    print("\n=== 只出现在模型2的 token ===")
    for tok in sorted(only2):
        print(bytes_to_escaped(tok))

    print(f"\n共 {len(only1)} 个 token 仅模型1有，{len(only2)} 个 token 仅模型2有。")

if __name__ == '__main__':
    special_tokens = ["<|endoftext|>"]
    model_prefix = "bpe"

    vocab, merges = train_bpe("data/owt_train.txt", 32000, special_tokens)
    vocab_file = f"./OpenWebText_bpe_results/{model_prefix}.vocab"
    merges_file = f"./OpenWebText_bpe_results/{model_prefix}.merges"
    save_bpe_model(vocab, merges, vocab_file=vocab_file, merges_file=merges_file)

    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)
    vocab_file_ = f"./TinyStories_bpe_results/{model_prefix}.vocab"
    merges_file_ = f"./TinyStories_bpe_results/{model_prefix}.merges"
    save_bpe_model(vocab, merges, vocab_file=vocab_file_, merges_file=merges_file_)
    # compare_bpe_models(
    #     vocab_file, merges_file,
    #     vocab_file_, merges_file_
    # )
    # diff_vocab_bytes(vocab_file, vocab_file_)