import torch
import torch.nn as nn
import math
from cs336_basics.nn_utils import softmax_

class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(2. / (in_features + out_features)))
                        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)

class TokenEmbedding(nn.Module):
    """
    创建一个嵌入层：词汇表有vocab_size个词 每个词用d_model维向量表示
    embedding = TokenEmbedding(vocab_size, d_model)
    
    例如，输入：2个样本，每个样本4个词ID
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

    输出：每个词ID都被替换成了对应的d_model维向量
    output = embedding(input)
    形状：(2, 4, d_model) - [批次大小, 序列长度, 词向量维度]"""
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(2. / (vocab_size + d_model)))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.functional.F.embedding(token_ids, self.weight)
    
class RMSNorm(nn.Module):
    """
    输入：
    x: (batch_size, sequence_length, d_model)

    输出：
    result: (batch_size, sequence_length, d_model)"""
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.gain
        return result.to(in_dtype)

def SiLU(x: torch.Tensor, beta: float = 1.0):
    return x * torch.sigmoid(beta * x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 / 3
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = LinearLayer(d_model, d_ff)
        self.w2 = LinearLayer(d_ff, d_model)
        self.w3 = LinearLayer(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model],
        # gate: torch.matmul(x, self.weight.T) = (..., d_model) x (d_model, d_ff) -> (..., d_ff)
        # value: (..., d_model) x (d_model, d_ff) -> (..., d_ff)
        # gate * value: (..., d_ff) x (..., d_ff) -> (..., d_ff)
        # self.w2(gate * value): (..., d_ff) x (d_ff, d_model) -> (..., d_model)
        gate = SiLU(self.w1(x)) 
        value = self.w3(x)
        return self.w2(gate * value)
    
class SiLU_(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = LinearLayer(d_model, d_ff)
        self.w2 = LinearLayer(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.w1(x)
        return self.w2(SiLU(x_))

    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, dim: int, seq_len: int, device=None):
        super().__init__()
        freqs_cis = self.precompute_freqs_cis(dim, seq_len * 2, theta, device)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
    
    @staticmethod
    def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device=None):
        """
        假设 dim=4, seq_len=6, theta=10000.0
        freqs = 1.0 / (10000^([0, 2]/4)) = [1.0, 0.01]
        t = [0, 1, 2, 3, 4, 5]
        freqs = 
        [[0.0000, 0.0000],
        [1.0000, 0.0100],
        [2.0000, 0.0200],
        [3.0000, 0.0300],
        [4.0000, 0.0400],
        [5.0000, 0.0500]]
        
        freqs_cis = [
            [cos(0)+isin(0), cos(0)+isin(0)],            # 位置0
            [cos(1.0)+isin(1.0), cos(0.01)+isin(0.01)],  # 位置1
            [cos(2.0)+isin(2.0), cos(0.02)+isin(0.02)],  # 位置2
            # ... 以此类推
        ]
        """
        # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # 生成token 序列索引 t = [0, 1, ..., seq_len - 1]
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        # 计算结果是个复数向量
        # 假设 freqs = [x, y]
        # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
        return freqs_cis
    
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
        # [x1, x2, x3, x4] -> [(x1,x2), (x3,x4)]
        x_shape = x.shape
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        # 把(x,y)看成 x + yi
        x_ = torch.view_as_complex(x_)

        # 乘以旋转因子（也是复数形式）
        # 旋转因子 = cosθ + sinθ i
        x_out = torch.view_as_real(x_ * freqs_cis).reshape(*x_shape)
        return x_out.type_as(x)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: Float[Tensor, " ... sequence_length d_k"]
        token_positions: Int[Tensor, " ... sequence_length"]
        """
        x = self.apply_rotary_emb(x, freqs_cis=self.freqs_cis[token_positions])
        return x
    




class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask: torch.Tensor = None):
        super().__init__()
        self.mask = mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q : (batch_size, num_heads, seq_len, d_k) 
            k : (batch_size, num_heads, seq_len, d_k) 
            v : (batch_size, num_heads, seq_len, d_v) 
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            output: (batch_size, ..., d_v)
        """
        d_k = q.shape[-1]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # score.shape: [batch, num_heads, seq_len, seq_len]
        # 矩阵乘法规则：[batch, m, n] @ [batch, n, p] = [batch, m, p]
        # print(f"score: {score.shape},  mask: {self.mask.shape}")
        # FLOPs = 2 × batch_size × num_heads × seq_len² × d_k

        # print(f"mask: {self.mask}")
        if self.mask is not None:
            mask = self.mask.to(score.device)
            score = score.masked_fill(~mask, float('-inf'))
        score = softmax_(score, dim=-1) #FLOPs = 5 × batch_size × num_heads × seq_len², score.shape: [batch, num_heads, seq_len, seq_len]
        return torch.matmul(score, v)   #FLOPs = 2 × batch_size × num_heads × seq_len² × d_v; shape: [batch, num_heads, seq_len, d_v]


class MultiheadSelfAttention(nn.Module):
    """
    Args：
        d_model (int)：前馈网络输入和输出的维度。
        num_heads (int)：多头注意力机制中使用的头数。
        max_seq_len (int)：如果你的实现进行预缓存，这里指要预缓存的最大序列长度。
        q_proj_weight (Float[Tensor, "d_k d_in"])：Q投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"])：K投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"])：V投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"])：输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"])：用于运行你实现的张量。
    返回值：
        Float[Tensor, " ... sequence_length d_out"]：使用给定的QKV投影权重和输入特征，运行你优化后的批量多头注意力实现所得到的输出张量。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_v = self.d_k = d_model // num_heads

        self.w_q = LinearLayer(d_model, d_model)
        self.w_k = LinearLayer(d_model, d_model)
        self.w_v = LinearLayer(d_model, d_model)
        self.w_o = LinearLayer(d_model, d_model)

    def _generate_causal_mask(self, seq_len: int):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.bool()

    def forward(self, input: torch.Tensor):
        batch_size, seq_len, d_model = input.shape
        # print(f"input.shape: {input.shape}") # torch.Size([4, 12, 64])

        casual_mask = self._generate_causal_mask(seq_len)
        casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)

        q = self.w_q(input) # batch_size, seq_len, d_model
        k = self.w_k(input)
        v = self.w_v(input)

        # batch_size, seq_len, self.num_heads, self.d_k -> batch_size, self.num_heads, seq_len, self.d_k
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        atten = ScaledDotProductAttention(mask=casual_mask)
        attention_output = atten(q, k, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(attention_output)
        return output
    

class MultiheadSelfAttentionWithRope(nn.Module):
    """
    参数：
        d_model (int)：前馈网络输入和输出的维度。
        num_heads (int)：多头注意力机制中使用的头数。
        max_seq_len (int)：如果实现包含预缓存机制，此处指定要预缓存的最大序列长度。
        theta (float)：RoPE（旋转位置编码）参数。
        q_proj_weight (Float[Tensor, "d_k d_in"])：查询(Q)投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"])：键(K)投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"])：值(V)投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"])：输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"])：用于运行实现的输入特征张量
        token_positions (Int[Tensor, " ... sequence_length"] | None)：可选的令牌位置张量

    返回值：
        Float[Tensor, " ... sequence_length d_out"]：经过优化的批处理多头注意力实现的输出张量，使用给定的QKV投影权重和输入特征进行计算。
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_v = self.d_k = d_model // num_heads

        self.w_q = LinearLayer(d_model, d_model)
        self.w_k = LinearLayer(d_model, d_model)
        self.w_v = LinearLayer(d_model, d_model)
        self.w_o = LinearLayer(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def _generate_causal_mask(self, seq_len: int):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.bool()

    def forward(self, input: torch.Tensor, token_positions: torch.Tensor = None):
        batch_size, seq_len, d_model = input.shape
        # print(f"input.shape: {input.shape}") # torch.Size([4, 12, 64])

        q = self.w_q(input) # batch_size, seq_len, d_model
        k = self.w_k(input)
        v = self.w_v(input)

        # batch_size, seq_len, self.num_heads, self.d_k -> batch_size, self.num_heads, seq_len, self.d_k
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        # print(f"q.shape before rope: {q.shape}")

        # print(f"token_positions.shape: {token_positions.shape}") # torch.Size([1, 12])
        if token_positions is None:
            token_positions = torch.arange(seq_len)
            token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1) #  torch.Size([1, 4, 12])
        
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        # print(f"q.shape after rope: {q.shape}")

        casual_mask = self._generate_causal_mask(seq_len)   # (L, L)
        casual_mask = casual_mask.unsqueeze(0).unsqueeze(0) # (1, 1, L, L)

        atten = ScaledDotProductAttention(mask=casual_mask)
        attention_output = atten(q, k, v)   # [batch, num_heads, seq_len, d_v]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)        # [batch, seq_len, d_model]
        output = self.w_o(attention_output) # [batch, seq_len, d_model] x [d_model x d_model]
        return output


class TransformerBlock(nn.Module):
    """
    参数：
        d_model (int)：Transformer块输入的维度。
        num_heads (int)：多头注意力中使用的头数。`d_model`必须能被`num_heads`整除。
        d_ff (int)：前馈网络内层的维度。
        max_seq_len (int)：如果你的实现包含预缓存机制，此处指定要预缓存的最大序列长度。
        theta (float)：RoPE参数。
        weights (dict[str, Tensor])：
        in_features (Float[Tensor, "batch sequence_length d_model"])：用于运行你实现的张量。
    返回：
        Float[Tensor, "batch sequence_length d_model"]：在输入特征上运行Transformer块（使用RoPE）后的输出张量。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.mha = MultiheadSelfAttentionWithRope(d_model, num_heads, max_seq_len, theta)
        # self.mha = MultiheadSelfAttention(d_model, num_heads)   # no_pos_emb
        self.ff = SwiGLU(d_model, d_ff)
        # self.ff = SiLU_(d_model, d_ff)                          # swiglu_ablation
        self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, input: torch.Tensor):
        # 有rmsnorm， pre-norm 结构
        x_normed = self.rmsnorm1(input)
        input = self.mha(x_normed) + input

        x_normed = self.rmsnorm2(input)
        input = self.ff(x_normed) + input
        
        # # 无 rmsnorm 结构
        # input = self.mha(input) + input
        # input = self.ff(input) + input

        # # 有rmsnorm， post-norm 结构
        # input = self.rmsnorm1(input + self.mha(input))
        # input = self.rmsnorm2(input + self.ff(input))
        return input


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = LinearLayer(d_model, vocab_size)
    
    
    def forward(self, input: torch.Tensor):
        output = self.embedding(input)
        for layer in self.layers:
            output = layer(output)
        output = self.norm(output)
        output = self.linear(output)
        return output
    
    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.embedding.weight.numel()
        l, h, q, t = self.num_layers, self.num_heads, self.vocab_size // self.num_heads, self.context_length
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
    
    def get_device(self):
        return self.embedding.weight.device

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_tokens,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int = 42,
        eos_token_id: int | None = None,
    ):
        """
        朴素自回归流式推理。为简化实现，我们做出如下假设：
        - 批次大小（batch size）为1
        - 标识序列（ids）及生成的令牌（yielded tokens）分别为普通Python列表和整数类型

        参数说明：
        - temperature: 温度，>0 时进行随机采样，=0 时退化为贪心解码
        - top_k: 若不为 None，启用 top-k 截断采样
        - top_p: 若不为 None，启用 nucleus / top-p 采样
        - seed: 随机种子，保证可复现
        - eos_token_id: 终止符 id，若生成到该 token 则提前结束
        """
        assert isinstance(tokens, list)
        device = self.get_device()

        # 随机数生成器：仅在需要随机采样时创建
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # 初始化输入 ids
        ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        for _ in range(max_tokens):
            # 1. 前向传播，获取最后一个位置的 logits
            logits = self.forward(ids)          # (1, T, vocab_size)
            logits = logits[:, -1, :]           # (1, vocab_size)

            # 2. top-k 截断（可选）
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                # 小于第 k 大值的 logits 置为 -inf
                logits[logits < v[:, [-1]]] = -float('inf')

            # 3. 温度缩放 + softmax 得到基础概率分布
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)  # (1, vocab_size)
            else:
                # temperature == 0：完全贪心，直接选 argmax
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
                ids = torch.cat([ids, next_ids], dim=1)
                token = next_ids.item()
                if eos_token_id is not None and token == eos_token_id:
                    break
                yield token
                continue

            # 4. top-p / nucleus 采样（可选）
            if top_p is not None and 0 < top_p < 1.0:
                # 降序排序
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # 找到累积概率 > top_p 的位置
                # keep mask: 累积概率 <= top_p
                keep_mask = cumulative_probs <= top_p
                # 为了至少保留一个 token，把第一个也保留
                keep_mask[..., 0] = True

                # 构造一个与 probs 同形状的 mask
                # 先全 0，再把 top-p 里对应的位置置为 1
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask.scatter_(dim=-1, index=sorted_indices, src=keep_mask)

                # 把不在 nucleus 里的概率置 0，再归一化
                probs = probs.masked_fill(~mask, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # 5. 根据最终概率分布随机采样下一个 token
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)  # (1, 1)

            # 6. 将采样到的 token 追加到序列中，并流式输出
            ids = torch.cat([ids, next_ids], dim=1)
            token = next_ids.item()

            # 可选：遇到 eos 提前结束
            if eos_token_id is not None and token == eos_token_id:
                break

            yield token