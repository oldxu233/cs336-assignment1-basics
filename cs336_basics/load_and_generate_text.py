from .serialization import load_checkpoint
from .model import TransformerLM
from .optimizer import adamw
from .tokenizer import Tokenizer
from .nanochat.common import autodetect_device_type, compute_init
import os
import torch

depth = 4          # Transformer 模型深度
max_seq_len = 256  # 最大上下文长度（序列长度）
lr = 1.2e-3                     # 初始学习率
weight_decay = 0.01           # 权重衰减
step=40000
# Model kwargs are derived from the desired depth of the model
num_layers = depth
d_model = 512
num_heads = 16
d_ff = 1344
vocab_size = 10000

model = TransformerLM(vocab_size=vocab_size, context_length=max_seq_len, d_model=d_model, 
                      num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=10000.0)
device_type = ""   # 设备类型：cuda / cpu / mps（空值则自动检测）
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model.to(device=device)
model = torch.compile(model, dynamic=False)

base_dir = "/home/xqzzz1/codes/cs336/assignment1-basics-main/cs336_basics/.cache"
output_dirname = f"lr{lr}" # e.g. lr0.0001
datset_name = "TinyStories"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", datset_name, output_dirname)
ckpt_filename = f"step_{step}.pt"
ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)


adamw_optimizer = adamw(model.parameters(), lr=lr, weight_decay=weight_decay, 
                        betas=(0.9, 0.95), eps=1e-8)
resume_from_step = load_checkpoint(ckpt_path, model, adamw_optimizer)
print(f"从第 {resume_from_step} 步恢复训练")

special_tokens = ["<|endoftext|>"]
model_prefix="bpe"
vocab_file = f"./cs336_basics/TinyStories_bpe_results/{model_prefix}.vocab"
merges_filepath = f"./cs336_basics/TinyStories_bpe_results/{model_prefix}.merges"
tokenizer = Tokenizer.from_files(vocab_file, merges_filepath, special_tokens=special_tokens)
eos = tokenizer.encode(special_tokens[0])[-1]

prompt = "Once upon a time, "
tokens = tokenizer.encode(prompt)
out_tokenids = model.generate(tokens, max_tokens=256, temperature=1.0,
                              top_p = 0.9, seed = 42, eos_token_id=eos)
out_text = tokenizer.decode(out_tokenids)
print(out_text)
