from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries
from multiprocessing import Pool
from functools import partial
import time

def process_chunk_for_ratio(args, vocab_filepath, merges_filepath, special_tokens):
    """处理单个文本块并返回字节数和token数"""
    start, end, input_path = args
    
    # 在每个进程中初始化分词器（避免序列化问题）
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=special_tokens)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    
    text = data.decode("utf-8", errors="ignore")
    encoded_ids = tokenizer.encode(text)
    
    return len(data), len(encoded_ids)

def compress_ratio_parallel(input_file, vocab_filepath, merges_filepath, num_processes, max_chunks=None):
    """并行计算压缩比"""
    special_tokens = ["<|endoftext|>"]
    
    # 找到文件分块边界
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 准备任务
    tasks = [(s, e, input_file) for s, e in zip(boundaries[:-1], boundaries[1:])]
    # print(f"tasks: {tasks}")
    # 如果指定了最大块数，限制任务数量
    if max_chunks and len(tasks) > max_chunks:
        tasks = tasks[:max_chunks]
    
    print(f"处理 {len(tasks)} 个数据块...")
    
    # 使用多进程并行处理
    worker = partial(process_chunk_for_ratio, 
                    vocab_filepath=vocab_filepath, 
                    merges_filepath=merges_filepath, 
                    special_tokens=special_tokens)
    
    total_bytes = 0
    total_tokens = 0
    
    with Pool(num_processes) as p:
        results = p.map(worker, tasks)
    
    # 汇总结果
    for bytes_count, tokens_count in results:
        total_bytes += bytes_count
        total_tokens += tokens_count
    
    # 计算压缩比
    if total_tokens > 0:
        ratio = total_bytes / total_tokens
        print(f"总字节数: {total_bytes:,}")
        print(f"总token数: {total_tokens:,}")
        print(f"压缩比: {ratio:.2f} (平均每个token对应 {ratio:.2f} 字节)")
        return ratio
    else:
        print("错误: 没有处理到任何token")
        return 0
    

def estimate_tokenizer_throughput(input_file, vocab_filepath, merges_filepath, special_tokens):
    # 初始化分词器
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=special_tokens)
    
    # 读取测试数据
    with open(input_file, "rb") as f:
        data = f.read()
    
    text = data.decode("utf-8", errors="ignore")
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行分词
    encoded_ids = tokenizer.encode(text)
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算处理时间
    processing_time = end_time - start_time
    
    # 计算吞吐量（每秒字节数）
    throughput_bytes_per_second = len(data) / processing_time
    
    print(f"Processed {len(data)} bytes in {processing_time:.2f} seconds")
    print(f"Tokenizer throughput: {throughput_bytes_per_second:.2f} bytes/second")
    
   
    

if __name__ == '__main__':
    model_prefix = "bpe"
    vocab_filepath = f"./TinyStories_bpe_results/{model_prefix}.vocab"
    merges_filepath = f"./TinyStories_bpe_results/{model_prefix}.merges"
    TinyStories = "./data/TinyStoriesV2-GPT4-valid.txt"

    vocab_file_ = f"./OpenWebText_bpe_results/{model_prefix}.vocab"
    merges_file_ = f"./OpenWebText_bpe_results/{model_prefix}.merges"
    OpenWebText = "./data/owt_valid.txt"
    

    # print("TinyStories压缩比计算开始")
    # start_time = time.time()
    # ratio1 = compress_ratio_parallel(TinyStories, vocab_filepath, merges_filepath, 
    #                                num_processes=10, max_chunks=10)
    # print(f"耗时: {time.time() - start_time:.2f}秒\n")
    
    # print("OpenWebText压缩比计算开始")
    # start_time = time.time()
    # ratio2 = compress_ratio_parallel(OpenWebText, vocab_file_, merges_file_, 
    #                             num_processes=10, max_chunks=10)
    # print(f"耗时: {time.time() - start_time:.2f}秒\n")

    # # 分词器互换
    # print("TinyStories分词器在OpenWebText的压缩比计算")
    # start_time = time.time()
    # ratio1 = compress_ratio_parallel(OpenWebText, vocab_filepath, merges_filepath, 
    #                                num_processes=10, max_chunks=10)
    # print(f"耗时: {time.time() - start_time:.2f}秒\n")
    
    # print("OpenWebText分词器在TinyStories的压缩比计算")
    # start_time = time.time()
    # ratio2 = compress_ratio_parallel(TinyStories, vocab_file_, merges_file_, 
    #                             num_processes=10, max_chunks=10)
    # print(f"耗时: {time.time() - start_time:.2f}秒\n")
    special_tokens = ["<|endoftext|>"]
    estimate_tokenizer_throughput(TinyStories, vocab_filepath, merges_filepath, special_tokens)