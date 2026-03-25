import os
from typing import BinaryIO, Tuple, Dict, List
from collections import defaultdict, Counter
from multiprocessing import Pool
import regex as re
import time
import heapq
import logging
logger = logging.getLogger(__name__)
from cs336_basics.pretokenization_example import find_chunk_boundaries
# pytest -v tests/test_train_bpe.py

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


# 预分词
GPT2_PATTERN = None
SPECIAL_TOKEN_PATTERN = None
def init_worker(special_tokens):
    global GPT2_PATTERN, SPECIAL_TOKEN_PATTERN
    SPECIAL_TOKEN_PATTERN = re.compile("|".join(map(re.escape, special_tokens)))
    
    GPT2_PATTERN_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    GPT2_PATTERN = re.compile(GPT2_PATTERN_STR)
    

def count_one_chunk_(args):
    start, end, path = args
    c = Counter()
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    chunk = data.decode("utf-8", errors="ignore")
    for raw in SPECIAL_TOKEN_PATTERN.split(chunk):
        for m in GPT2_PATTERN.finditer(raw):
            tok = m.group(0)
            c[tok] += 1
    return c

Stats = Dict[Tuple[int, int], int]
WordDict = Dict[Tuple[int, ...], int]
def init_word_dict(token_freqs: Dict[str, int]):
    """
    总时间复杂度: 3*O(T) = O(T)
    设: 
        - N = 不同 token 的数量
        - L = 平均 token 长度（字节数）
        - T = 所有 token 的总字节数 = O(N * L)
    构建 word2freq	    O(T)	遍历所有 token 并UTF-8 编码
    构建 pair2freq	    O(T)	每个词扫一遍找相邻 pair
    构建 pair_to_words	O(T)	同样扫描所有 pair
    """
    word2freq: WordDict = {}
    pair2freq: Stats = defaultdict(int)
    for tok, f in token_freqs.items():
        seq = tuple(tok.encode("utf-8"))
        word2freq[seq] = word2freq.get(seq, 0) + f
    # 第一次完整扫描，建立pair统计
    for seq, f in word2freq.items():
        for a, b in zip(seq, seq[1:]):
            pair2freq[(a, b)] += f

    
    pair_to_words = defaultdict(set)
    for seq in word2freq:
        for a, b in zip(seq, seq[1:]):
            pair_to_words[(a, b)].add(seq)
    """
    pair_to_words = {
        (115, 104): {(115, 104, 101)},                    # "sh" 只在 "she" 中
        (104, 101): {(115, 104, 101), (116, 104, 101), (104, 101, 108, 108, 111)},  # "he" 在三个词中
        (116, 104): {(116, 104, 101)},                    # "th" 只在 "the" 中
        (101, 108): {(104, 101, 108, 108, 111)},          # "el" 只在 "hello" 中
        (108, 108): {(104, 101, 108, 108, 111)},          # "ll" 只在 "hello" 中
        (108, 111): {(104, 101, 108, 108, 111)},          # "lo" 只在 "hello" 中
    }
    """
    return word2freq, pair2freq, pair_to_words


# 单次合并
def merge_one(pair: Tuple[int, int], word2freq: WordDict, pair2freq: Stats, new_id: int, pair_to_words):
    """
    总时间复杂度 = O(k*m)
    设:
        - k = 包含该 pair 的词数量 = len(pair_to_words[pair])
        - m = 这些词的平均长度
    获取待处理序列   O(k)   list(pair_to_words[pair])
    扫描找合并位置   O(m)   while 循环遍历
    构建新序列       O(m)   最坏情况复制所有字节
    删除旧 pair_to_words关联  O(m)
    添加新 pair_to_words关联  O(m)
    更新word2freq            O(m)
    更新pair2freq, 移除旧pair, 添加新pair 2*O(m)
    """
    
    left, right = pair
    seqs_to_process = list(pair_to_words[pair])
    for old_seq in seqs_to_process:
        if old_seq not in word2freq:
            continue
        freq = word2freq[old_seq]
        new_seq = []
        i = 0
        n = len(old_seq)
        changed = False
        while i < n:
            if i + 1 < n and old_seq[i] == left and old_seq[i + 1] == right:
                new_seq.append(new_id)
                i += 2
                changed = True
            else:
                new_seq.append(old_seq[i])
                i += 1
        if not changed:
            continue
        new_seq_t = tuple(new_seq)

        # 删除旧 seq 的所有 pair 关联
        for a, b in zip(old_seq, old_seq[1:]):
            pair_to_words[(a, b)].discard(old_seq)
            # if not pair_to_words[(a, b)]:
            #     del pair_to_words[(a, b)]

        # 添加新 seq 的 pair 关联
        for a, b in zip(new_seq_t, new_seq_t[1:]):
            pair_to_words[(a, b)].add(new_seq_t)

        # 更新word2freq
        word2freq[old_seq] -= freq
        if word2freq[old_seq] == 0:
            del word2freq[old_seq]
        word2freq[new_seq_t] = word2freq.get(new_seq_t, 0) + freq

        # 更新pair2freq, 移除旧pair, 添加新pair
        for a, b in zip(old_seq, old_seq[1:]):
            pair2freq[(a, b)] -= freq
            if pair2freq[(a, b)] == 0:
                del pair2freq[(a, b)]
        for a, b in zip(new_seq_t, new_seq_t[1:]):
            pair2freq[(a, b)] += freq


def fast_bpe_loop(token_freqs, vocab: dict[int, bytes], num_merges: int):
    word2freq, pair2freq, pair2words = init_word_dict(token_freqs)
    merges = []
    next_id = max(vocab.keys()) + 1

    # 统计用
    per_merge_times = []        # 记录每次 merge 的耗时
    window_times = []           # 用来做滑动窗口统计（比如每1000次）
    window_size = 1000

    print(f"[BPE] 初始 pair2freq 大小: {len(pair2freq)}")

    for i in range(num_merges):
        if not pair2freq:
            print(f"[BPE] pair2freq 为空，提前在第 {i} 次 merge 结束")
            break

        t_merge_start = time.time()

        def merge_key(p):
            return (pair2freq[p], vocab[p[0]], vocab[p[1]])
            
        best_pair = max(pair2freq, key=merge_key)
        """
        pair2freq = {
            (97, 98): 100,   # 'a','b' 出现100次
            (98, 99): 100,   # 'b','c' 出现100次
            (99, 100): 50,   # 'c','d' 出现50次
        }
        vocab = {
            0: b'\x00',
            1: b'\x01',
            # ...
            65: b'A',
            97: b'a',
            98: b'b',
            99: b'c',
            # ...
            255: b'\xff'
        }
        vocab[97] = b'a'
        
        def merge_key(p):
            return (pair2freq[p], vocab[p[0]], vocab[p[1]])

        # 计算每个对的比较键
        merge_key((97,98)) = (100, 1, 2)
        merge_key((98,99)) = (100, 2, 3)
        merge_key((99,100)) = (50, 3, 4)

        # 比较过程：
        # (100,1,2) vs (100,2,3) → 第一个元素相等(100)，比较第二个：1 < 2
        # 所以 (100,1,2) < (100,2,3)
        # 因此 max() 会选 (98,99) 因为 (100,2,3) 更大
        """
        best_freq = pair2freq[best_pair]
        if best_freq == 0:
            print(f"[BPE] 最佳pair频率为0. 在第 {i} 次 merge 停止")
            break

        # 构造新 token
        left_id, right_id = best_pair
        vocab[next_id] = vocab[left_id] + vocab[right_id]
        merges.append((vocab[left_id], vocab[right_id]))

        # 做实际的 merge
        merge_one(best_pair, word2freq, pair2freq, next_id, pair2words)

        t_merge_end = time.time()
        dt = t_merge_end - t_merge_start
        per_merge_times.append(dt)
        window_times.append(dt)

        # 每 window_size 次打印一次窗口平均耗时和当前 pair2freq 大小
        if (i + 1) % window_size == 0:
            avg_window = sum(window_times) / len(window_times)
            print(
                f"[BPE] merge #{i + 1}: "
                f"最近 {len(window_times)} 次平均耗时 = {avg_window:.6f} 秒; "
                f"当前 pair2freq 大小 = {len(pair2freq)}"
            )
            window_times.clear()

        next_id += 1

    # 循环结束后，打印整体统计
    if per_merge_times:
        avg_all = sum(per_merge_times) / len(per_merge_times)
        print(
            f"[BPE] 总共执行 merge 次数: {len(per_merge_times)}, "
            f"单次平均耗时 = {avg_all:.6f} 秒"
        )
    else:
        print("[BPE] 未执行任何 merge。")

    return vocab, merges


def train_bpe(input_path, vocab_size, special_tokens: list[str]):
    t0 = time.time()
    
    # 初始化vocab
    vocab = {i: bytes([i]) for i in range(256)} # dict[int, bytes]
    next_id = 256
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[next_id] = token_bytes
        next_id += 1
    num_merges = vocab_size - len(vocab)

    # 读文件阶段：并行实现
    t_read_start = time.time()
    t_chunking_start = time.time()
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    t_chunking = time.time() - t_chunking_start

    tasks = [(s, e, input_path) for s, e in zip(boundaries[:-1], boundaries[1:])]
    t_parallel_start = time.time()
    with Pool(num_processes, initializer=init_worker, initargs=(special_tokens,)) as p:
        results = p.map(count_one_chunk_, tasks)
    """
    results = [
        Counter({"the": 100, "she": 50, ...}),  # 第1个块的统计
        Counter({"the": 80, "she": 30, ...}),   # 第2个块的统计
        Counter({"the": 120, "she": 40, ...}),  # 第3个块的统计
        # ...
    ]
    """
    t_parallel = time.time() - t_parallel_start

    t_merge_start = time.time()
    token_freqs = Counter()
    for c in results:
        token_freqs.update(c)
    t_merge = time.time() - t_merge_start

    t_read_end = time.time()
    t_read_total = t_read_end - t_read_start

    # 优化的BPE训练阶段
    t_bpe_start = time.time()
    vocab, merges = fast_bpe_loop(token_freqs, vocab, num_merges)
    # vocab, merges = fast_bpe_loop_optimized(
    #     token_freqs, vocab, num_merges
    # )
    t_bpe_end = time.time()
    
    t1 = time.time()
    
    print(f"总耗时: {t1 - t0:.4f} 秒")
    print(f"读文件总耗时: {t_read_total:.4f} 秒")
    print(f"  - 文件分块: {t_chunking:.4f} 秒")
    print(f"  - 并行处理: {t_parallel:.4f} 秒")
    print(f"  - 结果合并: {t_merge:.4f} 秒")
    print(f"  BPE 循环耗时: {t_bpe_end - t_bpe_start:.4f} 秒")
    print(f"  合并次数: {len(merges)}")
    print(f"  最终词表大小: {len(vocab)}")
    return vocab, merges



def test01():
    special_tokens = ["<|endoftext|>"]
    model_prefix="bpe"
    # vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 500, special_tokens)
    # print(f"vocab具体是： {vocab}, \n merges具体是：{merges}")
    
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)
    vocab_filepath = f"./TinyStories_bpe_results/{model_prefix}.vocab"
    merges_filepath = f"./TinyStories_bpe_results/{model_prefix}.merges"
    save_bpe_model(vocab, merges, vocab_file=vocab_filepath, merges_file=merges_filepath)

    # vocab, merges = train_bpe("data/owt_train.txt", 32000, special_tokens)
    # vocab_file_ = f"./OpenWebText_bpe_results/{model_prefix}.vocab"
    # merges_file_ = f"./OpenWebText_bpe_results/{model_prefix}.merges"
    # save_bpe_model(vocab, merges, vocab_file=vocab_file_, merges_file=merges_file_)


if __name__ == "__main__":
    test01()