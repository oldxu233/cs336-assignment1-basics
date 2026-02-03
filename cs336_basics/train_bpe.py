import os
from typing import BinaryIO, Tuple, Dict, List
from collections import defaultdict, Counter
from multiprocessing import Pool
import regex as re
import time
import heapq
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

    mini_chunk_size = 4096 * 2560  # 每次向前读取4KB * 2560 = 10240KB 字节

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



# 预分词
GPT2_PATTERN = None
SPECIAL_TOKEN_PATTERN = None
def init_worker(special_tokens):
    global GPT2_PATTERN, SPECIAL_TOKEN_PATTERN, SPECIAL_TOKENS_SET
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
    return word2freq, pair2freq, pair_to_words

def merge_one(pair: Tuple[int, int], word2freq: WordDict, pair2freq: Stats, new_id: int, pair_to_words):
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
            if i+1<n and old_seq[i] == left and old_seq[i+1] == right:
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


def fast_bpe_loop(token_freqs, vocab, num_merges):
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
        best_freq = pair2freq[best_pair]
        if best_freq == 0:
            print(f"[BPE] 最佳pair频率为0，在第 {i} 次 merge 停止")
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
    vocab = {i: bytes([i]) for i in range(256)}
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
    # vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 270, special_tokens)
    # print(f"vocab具体是： {vocab}, \n merges具体是：{merges}")
    
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)
    vocab_filepath = f"./TinyStories_bpe_results/{model_prefix}.vocab"
    merges_filepath = f"./TinyStories_bpe_results/{model_prefix}.merges"
    save_bpe_model(vocab, merges, vocab_file=vocab_filepath, merges_file=merges_filepath)

    # vocab, merges = train_bpe("data/owt_train.txt", 32000, special_tokens)
    # vocab_file_ = f"./OpenWebText_bpe_results/{model_prefix}.vocab"
    # merges_file_ = f"./OpenWebText_bpe_results/{model_prefix}.merges"
    # save_bpe_model(vocab, merges, vocab_file=vocab_file_, merges_file=merges_file_)

"""[BPE] 初始 pair2freq 大小: 2108
[BPE] merge #1000: 最近 1000 次平均耗时 = 0.004981 秒; 当前 pair2freq 大小 = 23860
[BPE] merge #2000: 最近 1000 次平均耗时 = 0.006334 秒; 当前 pair2freq 大小 = 32216
[BPE] merge #3000: 最近 1000 次平均耗时 = 0.007671 秒; 当前 pair2freq 大小 = 36765
[BPE] merge #4000: 最近 1000 次平均耗时 = 0.008062 秒; 当前 pair2freq 大小 = 39011
[BPE] merge #5000: 最近 1000 次平均耗时 = 0.008578 秒; 当前 pair2freq 大小 = 41152
[BPE] merge #6000: 最近 1000 次平均耗时 = 0.009102 秒; 当前 pair2freq 大小 = 43084
[BPE] merge #7000: 最近 1000 次平均耗时 = 0.009431 秒; 当前 pair2freq 大小 = 44621
[BPE] merge #8000: 最近 1000 次平均耗时 = 0.009759 秒; 当前 pair2freq 大小 = 45759
[BPE] merge #9000: 最近 1000 次平均耗时 = 0.010034 秒; 当前 pair2freq 大小 = 46739
[BPE] 总共执行 merge 次数: 9743, 单次平均耗时 = 0.008370 秒
总耗时: 201.2589 秒
读文件总耗时: 119.4693 秒
  - 文件分块: 0.0186 秒
  - 并行处理: 119.4226 秒
  - 结果合并: 0.0280 秒
  BPE 循环耗时: 81.7896 秒
  合并次数: 9743
  最终词表大小: 10000
词表已保存到: ./TinyStories_bpe_results/bpe.vocab
合并规则已保存到: ./TinyStories_bpe_results/bpe.merges"""



def update_data_structures(old_seq, new_seq, freq, word2freq, pair2freq, pair2words):
    """批量更新数据结构（版本稍作防御性修改）"""
    # 移除旧序列对应的所有 pair 统计
    for a, b in zip(old_seq, old_seq[1:]):
        key = (a, b)
        if key in pair2freq:
            pair2words[key].discard(old_seq)
            pair2freq[key] -= freq
            if pair2freq[key] <= 0:
                del pair2freq[key]
            if not pair2words[key]:
                del pair2words[key]

    # 删除旧序列频率
    word2freq.pop(old_seq, None)

    # 添加新序列
    word2freq[new_seq] = word2freq.get(new_seq, 0) + freq
    for a, b in zip(new_seq, new_seq[1:]):
        key = (a, b)
        pair2words[key].add(new_seq)
        pair2freq[key] = pair2freq.get(key, 0) + freq


def fast_bpe_loop_optimized(token_freqs, vocab, num_merges):
    word2freq, pair2freq, pair2words = init_word_dict(token_freqs)
    merges = []
    next_id = max(vocab.keys()) + 1

    # 初始化堆: (-freq, vocab[left], vocab[right], pair)
    heap = []
    for (left_id, right_id), freq in pair2freq.items():
        heapq.heappush(heap, (-freq, vocab[left_id], vocab[right_id], (left_id, right_id)))

    for i in range(num_merges):
        if not heap:
            break

        # 弹出有效pair
        while heap:
            neg_freq, _, _, best_pair = heapq.heappop(heap)
            current_freq = -neg_freq
            if best_pair in pair2freq and pair2freq[best_pair] == current_freq:
                break
        else:
            break

        if current_freq == 0:
            break

        left_id, right_id = best_pair
        # 创建新token
        new_bytes = vocab[left_id] + vocab[right_id]
        vocab[next_id] = new_bytes
        merges.append((vocab[left_id], vocab[right_id]))

        # 处理受影响的序列
        affected_seqs = list(pair2words[best_pair])
        for old_seq in affected_seqs:
            if old_seq not in word2freq:
                continue
            freq = word2freq[old_seq]

            # 生成新序列
            new_seq, changed = [], False
            j, n = 0, len(old_seq)
            while j < n:
                if j + 1 < n and old_seq[j] == left_id and old_seq[j + 1] == right_id:
                    new_seq.append(next_id)
                    j += 2
                    changed = True
                else:
                    new_seq.append(old_seq[j])
                    j += 1
            if not changed:
                continue

            new_seq_t = tuple(new_seq)
            update_data_structures(
                old_seq, new_seq_t, freq,
                word2freq, pair2freq, pair2words
            )

            # 将新序列产生的新pair加入堆
            for a, b in zip(new_seq_t, new_seq_t[1:]):
                key = (a, b)
                if key not in pair2freq:  # 可能已被移除
                    continue
                # 🔑 修复：用 a, b 而不是 left_id, right_id
                heapq.heappush(
                    heap,
                    (-pair2freq[key], vocab[a], vocab[b], key)
                )
        
        next_id += 1

    return vocab, merges

if __name__ == "__main__":
    test01()