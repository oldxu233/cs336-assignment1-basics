# 参考https://github.com/karpathy/nanochat 重写了train_bpe.py 获得了更好的时间复杂度
import os
from collections import defaultdict, Counter
from multiprocessing import Pool
import regex as re
import time
import heapq
import logging
logger = logging.getLogger(__name__)
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.rust_bpe import Tokenizer, Word
from cs336_basics.train_bpe import save_bpe_model

GPT2_PATTERN_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_GPT2_PATTERN = None
_SPECIAL_TOKEN_PATTERN = None
def init_worker(pattern_str, special_tokens):
    """进程初始化函数，只执行一次"""
    global _GPT2_PATTERN, _SPECIAL_TOKEN_PATTERN
    _GPT2_PATTERN = re.compile(pattern_str)
    _SPECIAL_TOKEN_PATTERN = re.compile("|".join(map(re.escape, special_tokens)))


def process_chunk(args):
    start, end, file_path = args
    local_counts = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        for raw in _SPECIAL_TOKEN_PATTERN.split(chunk):
            for m in _GPT2_PATTERN.finditer(raw):
                local_counts[m.group(0)] += 1
    return local_counts


def train_bpe_new(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    tokenizer = Tokenizer()
    # step 1: Vocabulary initialization. Done in tokenizer initialization

    # step2 : Pre-tokenization
    t_pre_token_start = time.time()
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # if numprocesses = 4: 
        #    boundaries: [0, 5625771, 11252572, 16877385, 22502601]
        
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunks.append((start, end, input_path))
        with Pool(
                num_processes, 
                initializer=init_worker, 
                initargs=(GPT2_PATTERN_STR, special_tokens)) as p:
            results = p.map(process_chunk, chunks)
    token_counts = Counter()  # 全局词频字典
    # {
    # "the": 15234,      # "the" 出现了15234次
    # "she": 8921,       # "she" 出现了8921次
    # ...
    # }
    for r in results:
        token_counts.update(r)
    t_pre_token = time.time() - t_pre_token_start
    
    # step3: bpe merges
    t_merge_start = time.time()    
    words = []
    counts = []
    for token, cnt in token_counts.items():
        ids = list(token.encode("utf-8"))
        words.append(Word(ids))
        counts.append(cnt)

    # Train to vocabulary size
    tokenizer.train_core_incremental(words, counts, vocab_size, 256 + len(special_tokens))

    # # Encode some text
    # encoded = tokenizer.encode("abc")
    # print(f"Encoded: {encoded}")
    
    # # Decode back
    # decoded = tokenizer.decode(encoded)
    # print(f"Decoded: {decoded}")

    merges = [
        (tokenizer.vocab[left], tokenizer.vocab[right]) 
        for (left, right) in tokenizer.merges.keys()
    ]
    t_merge = time.time() - t_merge_start
    
    print(f"总耗时: {time.time() - t0:.4f} 秒")
    print(f"--预分词耗时: {t_pre_token:.4f} 秒")
    print(f"--BPE merges耗时: {t_merge:.4f} 秒")

    next_id = len(tokenizer.vocab)
    for token in special_tokens:
        tokenizer.vocab[next_id] = token.encode('utf-8')
        next_id += 1
    return tokenizer.vocab, merges

def test01():
    special_tokens = ["<|endoftext|>"]
    model_prefix="bpe"
    # vocab, merges = train_bpe_new("data/TinyStoriesV2-GPT4-valid.txt", 270, special_tokens)
    # print(f"vocab具体是： {vocab}, \n merges具体是：{merges}")
    
    # vocab, merges = train_bpe_new("data/TinyStoriesV2-GPT4-valid.txt", 500, special_tokens)
    # vocab_file_ = f"./TinyStories_bpe_results/{model_prefix}_valid.vocab"
    # merges_file_ = f"./TinyStories_bpe_results/{model_prefix}_valid.merges"
    # save_bpe_model(vocab, merges, vocab_file=vocab_file_, merges_file=merges_file_)

    # vocab, merges = train_bpe_new("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)
    # vocab_filepath = f"./TinyStories_bpe_results/{model_prefix}_heap.vocab"
    # merges_filepath = f"./TinyStories_bpe_results/{model_prefix}_heap.merges"
    # save_bpe_model(vocab, merges, vocab_file=vocab_filepath, merges_file=merges_filepath)

    vocab, merges = train_bpe_new("data/owt_train.txt", 32000, special_tokens)
    vocab_filepath = f"./OpenWebText_bpe_results/{model_prefix}_heap.vocab"
    merges_filepath = f"./OpenWebText_bpe_results/{model_prefix}_heap.merges"
    save_bpe_model(vocab, merges, vocab_file=vocab_filepath, merges_file=merges_filepath)

if __name__ == "__main__":
    test01()