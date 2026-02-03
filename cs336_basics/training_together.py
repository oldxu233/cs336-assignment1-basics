import numpy as np
from .train_bpe import train_bpe, save_bpe_model
from .tokenizer import Tokenizer
from .data import get_batch
from .nn_utils import cross_entropy, perplexity, gradient_clipping
from .optimizer import adamw, get_lr_cosine_schedule
from .model import TransformerLM
from .serialization import save_checkpoint, load_checkpoint

from .nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init

import os
import time
from contextlib import nullcontext
import wandb
import torch

# 1. train bpe and save
special_tokens = ["<|endoftext|>"]
model_prefix="bpe"
train_file_path = "./cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
# vocab, merges = train_bpe(train_file_path, 10000, special_tokens)
# vocab_filepath = f"./TinyStories_bpe_results/{model_prefix}.vocab"
# merges_filepath = f"./TinyStories_bpe_results/{model_prefix}.merges"
# save_bpe_model(vocab, merges, vocab_file=vocab_filepath, merges_file=merges_filepath)

# User settings
run = "cs336-assignment1-OpenWebText_" + time.strftime('%Y%m%d_%H%M%S') # wandb 运行名称（"dummy" 表示不使用 wandb 日志）
# # run = "dummy"

# run = "layer_norm_ablation" + time.strftime('%Y%m%d_%H%M%S')
# run = "pre_norm_ablation" + time.strftime('%Y%m%d_%H%M%S')
# run = "no_pos_emb" + time.strftime('%Y%m%d_%H%M%S')
# run = "swiglu_ablation" + time.strftime('%Y%m%d_%H%M%S')

device_type = ""   # 设备类型：cuda / cpu / mps（空值则自动检测）
depth = 4          # Transformer 模型深度
max_seq_len = 256  # 最大上下文长度（序列长度）

# 优化器相关
device_batch_size = 32        # 单设备批次大小（避免 OOM）
val_batch_size = 2048         # 验证集批次大小
total_batch_size = 8192 * 4   # 总批次大小（以 token 数计）
lr = 1.2e-3                   # 初始学习率
weight_decay = 0.01           # 权重衰减
grad_clip = 1.0               # 梯度裁剪阈值（L2 范数上限）

# 训练总步数设置（按优先级使用其中一项）
total_tokens = 327680000
num_iterations = total_tokens // device_batch_size // max_seq_len # 显式指定训练步数（-1 表示禁用））; num_iterations = 总tokens / device_batch_size / max_seq_len
target_flops = -1.0           # 目标总 FLOPs（用于缩放定律实验）
target_param_data_ratio = 20  # 目标数据量与参数量之比（Chinchilla 推荐值为 20）
# Learning rate scheduling
max_learning_rate = 2e-3
min_learning_rate = max_learning_rate / 20
warmup_iters = 2000           # 学习率预热步数
cosine_cycle_iters = num_iterations - warmup_iters # 余弦退火周期长度 40,000 - 2000
# torch.optim.lr_scheduler.CosineAnnealingLR

# 评估
eval_every = 250                 # 每多少步评估一次验证集 bpb（bits per byte）
sample_every = 20000             # 每多少步采样生成文本
save_every = -1                  # 每多少步保存检查点（-1 表示只在训练结束时保存）

# 现在允许通过配置器在命令行界面（CLI）中覆盖设置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging


# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="cs336-chat", name=run, config=user_config)

# 2. Tokenizer will be useful for evaluation, also we need the vocab size
vocab_file = f"./cs336_basics/TinyStories_bpe_results/{model_prefix}.vocab"
merges_filepath = f"./cs336_basics/TinyStories_bpe_results/{model_prefix}.merges"
tokenizer = Tokenizer.from_files(vocab_file, merges_filepath, special_tokens=special_tokens)
vocab_size = 10000
eos = tokenizer.encode(special_tokens[0])[-1]

# Model kwargs are derived from the desired depth of the model
num_layers = depth
d_model = 512 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = 16
# d_ff = 1344
d_ff = 4*d_model # swiglu_ablation
# num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)

# Optimizer / data / training length related hyperparameters
# 算一下需要多少梯度累积步数才能达到目标总批次大小。
tokens_per_fwdbwd = device_batch_size * max_seq_len          # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks, 单卡ddp_world_size=1
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model
model = TransformerLM(vocab_size=vocab_size, context_length=max_seq_len, d_model=d_model, 
                      num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=10000.0)
model.to(device=device)


orig_model = model # 原始、未编译的模型，用于 1) 保存原始模型的 state_dict 2) 推理/评估阶段使用（编译后可能改变张量形状，用原始模型更稳妥）
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Initialize the Optimizer  AdamW for all layers
adamw_optimizer = adamw(model.parameters(), lr=lr, weight_decay=weight_decay, 
                        betas=(0.9, 0.95), eps=1e-8)

# If we are resuming, overwrite the model parameters with those of the checkpoint
model_tag = ""                  # 检查点目录名称（可选）
datset_name = "OpenWebText"
base_dir = "/home/xqzzz1/codes/cs336/assignment1-basics-main/cs336_basics/.cache"
if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)
output_dirname = model_tag if model_tag else f"lr{lr}" # e.g. lr0.0001
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", datset_name, output_dirname)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
resume_from_step = -1         # 从某一步恢复训练（-1 表示不恢复）
resuming = resume_from_step != -1
if resuming:
    resume_from_step = load_checkpoint(checkpoint_dir, model, adamw_optimizer)
    print(f"从第 {resume_from_step} 步恢复训练")

# Calculate number of iterations. 按以下**优先顺序**决定：  
# 1. 如果直接给出了迭代次数，就用这个值；  
# 2. 没给次数但给了“目标 FLOPs”，则用 FLOPs 反推迭代次数；  
# 3. 两者都没给，就按“目标 data:param 比例”来计算迭代次数。
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
# total_tokens = total_batch_size * num_iterations
print(f"Total number of training tokens: {total_tokens:,}")
print(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Initialize the DataLoaders for train/val
mmap_file_train = "./cs336_basics/data/TinyStoriesV2-GPT4-train.memmap"
train_data = np.memmap(mmap_file_train, dtype=np.uint16, mode="r")
mmap_file_valid = "./cs336_basics/data/TinyStoriesV2-GPT4-valid.memmap"
val_data = np.memmap(mmap_file_valid, dtype=np.uint16, mode="r")

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    avg_val_loss = 0
    avg_perplexity = 0
# else:
#     step = meta_data["step"]
#     loop_state = meta_data["loop_state"]
#     min_val_bpb = loop_state["min_val_bpb"]
#     smooth_train_loss = loop_state["smooth_train_loss"]
#     total_training_time = loop_state["total_training_time"]


# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",           # 地理知识
            "The chemical symbol of gold is",     # 化学知识
            "If yesterday was Friday, then tomorrow will be",  # 逻辑推理
            "The opposite of hot is",             # 反义词
            "The planets of the solar system are:",  # 科学知识
            "My favorite color is",               # 主观偏好
            "If 5*x + 3 = 13, then x is",        # 数学推理
        ]
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            with torch.no_grad(), autocast_ctx:
                out_tokenids = orig_model.generate(tokens, max_tokens=256, temperature=1.0,
                                top_p = 0.9, seed = 42, eos_token_id=eos)
            out_text = tokenizer.decode(out_tokenids)
            print(out_text)
        model.train()

    # 保存检查点（训练结束或每 save_every 步）
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        ckpt_filename = f"step_{step}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
        save_checkpoint(
            model,
            adamw_optimizer,
            step,
            ckpt_path
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break
    
    val_loss = None
    if master_process and (last_step or (step > 0 and step % eval_every == 0)):
        model.eval()
        val_losses = []
        perplexity_list = []
        with torch.no_grad():
            x_val, y_val = get_batch(val_data, val_batch_size, max_seq_len, device)
            with autocast_ctx:
                y_val_pred = model(x_val)
            loss_val = cross_entropy(y_val_pred, y_val)
            perplexity_ = perplexity(loss_val)
            val_losses.append(loss_val.item())
            perplexity_list.append(perplexity_)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_perplexity = sum(perplexity_list) / len(perplexity_list)
        model.train()

    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    model.train()
    for micro_step in range(grad_accum_steps):
        x, y = get_batch(train_data, device_batch_size, max_seq_len, device)
        with autocast_ctx:
            y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
    if grad_clip_enabled:
        grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for param in adamw_optimizer.param_groups:
        param["lr"] = lrm
    adamw_optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if avg_val_loss is not None:
            log_data["valid/loss"] = avg_val_loss
            log_data["valid/perplexity"] = avg_perplexity
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print(f"Total training time: {total_training_time/60:.2f}m")
print(f"Minimum validation bpb: {min_val_bpb:.4f}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()