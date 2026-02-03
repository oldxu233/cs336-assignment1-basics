import torch
from collections.abc import Iterable


def softmax_(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    # print(f"x_max: {x_max}")
    x = torch.exp(x - x_max)
    # print(f"x: {x}")
    return x / torch.sum(x, dim=dim, keepdim=True)


def cross_entropy(inputs, targets):
    """
    inputs: [batch_size, num_classes] inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
    targets: [batch_size] Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    """
    x_max = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - x_max), dim=-1, keepdim=True)) + x_max
    log_prob = inputs - log_sum_exp  # 这就是 log_softmax
    # log_prob.shape: torch.Size([32, 256, 10000])
    # targets.shape: torch.Size([32, 256])

    # 把每个样本对应类别的 log-prob 拿出来
    if targets.dim() == 1:
        # 2D 情况: [batch_size, num_classes] 和 [batch_size]
        log_prob_true = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
    else:
        # 3D 情况: [batch_size, seq_len, num_classes] 和 [batch_size, seq_len]
        log_prob_true = log_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -log_prob_true.mean() # 平均交叉熵


def perplexity(cross_entropy_losses):
    return torch.exp(cross_entropy_losses)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_L2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(g.data.norm(2).pow(2) for g in grads))

    if total_norm > max_L2_norm:
        scale = max_L2_norm / (total_norm + eps)
        for g in grads:
            g.data.mul_(scale)

    return total_norm