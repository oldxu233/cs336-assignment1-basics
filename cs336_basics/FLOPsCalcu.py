from typing import Dict, Tuple

def flops_block(batch_size: int, seq_len: int, d_model: int, 
                num_heads: int, d_ff: int, 
                include_softmax: bool = True, 
                include_res: bool = True,
                include_rotary: bool = True):
    
    flops_rmsnorm = 8 * batch_size * seq_len * d_model + 4 * batch_size * seq_len
    
    # MHA 层
    flops_mha_linear = 8 * batch_size * seq_len * d_model * d_model
    flops_rotary = 0
    if include_rotary:
        flops_rotary = 2 * (3 * batch_size * seq_len * d_model)
    # 注意力矩阵乘 (QK^T + Attn·V)
    # d_v = d_k = d_model // num_heads
    # flops_attention = 2 * batch_size * num_heads * seq_len * seq_len * (d_v + d_k)
    flops_attention = 4 * batch_size * seq_len * seq_len * d_model
    flops_softmax = batch_size * num_heads * seq_len * (4*seq_len -1)
    if include_softmax:
        flops_attention += flops_softmax
    flops_mha = flops_mha_linear + flops_rotary + flops_attention

    # 残差连接 (2个): 2 * batch * seq * d_model
    flops_residual = 0
    if include_res:
        flops_residual = 2 * batch_size * seq_len * d_model
    
    # FFN (SwiGLU)
    flops_ffn = 6 * batch_size * seq_len * d_model * d_ff + 4 * batch_size * seq_len * d_ff
    return flops_rmsnorm + flops_mha + flops_ffn + flops_residual

def flops_calculator(batch_size: int, seq_len: int, d_model: int, 
                    num_layers: int, num_heads: int, d_ff: int, 
                    vocab_size: int, include_softmax: bool = True, 
                    include_res: bool = True,
                    include_rotary: bool = True):
    """
    Transformer FLOPs计算器
    """   
    # 单个Transformer块总FLOPs
    flops_per_block = flops_block(batch_size, seq_len, d_model, 
                num_heads, d_ff, include_softmax, include_res, include_rotary)
    
    # 2. 所有Transformer层
    flops_transformer_layers = num_layers * flops_per_block
    
    # 3. 输出层: 2 * batch * seq * d_model * vocab_size
    flops_rmsnorm = 4 * batch_size * seq_len * d_model + 2 * batch_size * seq_len
    flops_linear_layer = 2 * batch_size * seq_len * d_model * vocab_size
    flops_output_layer = flops_rmsnorm + flops_linear_layer
    
    # 4. 总FLOPs
    total_flops = flops_transformer_layers + flops_output_layer
    
    return total_flops


# 对比两种计算
def flops_GPT2_XL():
    params = {
        'batch_size': 1,
        'seq_len': 1024,
        'd_model': 1600,
        'num_layers': 48,
        'num_heads': 25,
        'd_ff': 6400,
        'vocab_size': 50257,
        'include_softmax': False,
        'include_res': False,
        'include_rotary': False
    }
    
    corrected = flops_calculator(**params)
    print(f"计算结果: {corrected/1e12:.4f} TFLOPs")


def flops_breakdown(batch_size: int, seq_len: int, d_model: int,
                    num_layers: int, num_heads: int, d_ff: int,
                    vocab_size: int,
                    include_softmax: bool = True,
                    include_res: bool = True,
                    include_rotary: bool = True) -> Tuple[float, Dict[str, float]]:
    """
    返回 (总FLOPs, 各组件占比dict)
    所有数值都以‘GFLOPs = 1e9’为单位
    """
    B, s, d, h, d_ff, V = batch_size, seq_len, d_model, num_heads, d_ff, vocab_size
    giga = 1e9

    # ---------- 单层 ----------
    # 1. MHA 投影
    proj = 8 * B * s * d * d
    # 2. rotary
    rotary = 2 * (3 * B * s * d) if include_rotary else 0
    # 3. 注意力矩阵乘
    attn = 4 * B * s * s * d
    # 4. softmax
    softmax = B * h * s * (4 * s - 1) if include_softmax else 0
    mha_total = proj + rotary + attn + softmax

    # 5. FFN (SwiGLU)
    ffn = 6 * B * s * d * d_ff + 4 * B * s * d_ff

    # 6. 两个 RMSNorm
    norm = 8 * B * s * d + 4 * B * s
    # 7. 残差加法
    res = 2 * B * s * d if include_res else 0
    block = mha_total + ffn + norm + res

    # ---------- 多层 ----------
    flops_attn_matmul = num_layers * attn
    flops_ffn = num_layers * ffn
    flops_mha_proj = num_layers * proj
    flops_softmax = num_layers * softmax
    flops_norm = num_layers * norm
    flops_res = num_layers * res

    # ---------- 输出层 ----------
    out_norm = 4 * B * s * d + 2 * B * s
    out_linear = 2 * B * s * d * V
    flops_output = out_norm + out_linear

    # ---------- 总计 ----------
    total = num_layers * block + flops_output

    # ---------- 百分比 ----------
    report = {
        "attention_matmul": flops_attn_matmul / giga,
        "ffn_total": flops_ffn / giga,
        "mha_proj_linear": flops_mha_proj / giga,
        "softmax": flops_softmax / giga,
        "layer_norm": (flops_norm + out_norm) / giga,
        "residual_add": flops_res / giga,
        "output_linear": out_linear / giga,
    }
    total_g = total / giga
    percent = {k: v / total_g * 100 for k, v in report.items()}
    return total_g, percent


# ----------------- DEMO -----------------
if __name__ == "__main__":
    cfg = dict(
        batch_size=1,
        seq_len=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        vocab_size=50257,
        include_softmax=True,
        include_res=True,
        include_rotary=True
    )
    total, pct = flops_breakdown(**cfg)
    print(f"Total : {total:.2f} GFLOPs")
    for k, v in pct.items():
        print(f"{k:20s}: {v:5.1f} %")