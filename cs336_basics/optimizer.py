from collections.abc import Callable
from typing import Optional
import torch
from torch import Tensor
import math
from typing import Optional, Tuple, Union

class adamw(torch.optim.Optimizer):
    def __init__(
            self, 
            params, 
            lr: Union[float, Tensor] = 1e-3, 
            betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            alpha=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:    
            alpha = group["alpha"]
            beta1, beta2 = group["betas"]
            lambda_ = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['s'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state['s'] + 1
                
                m.mul_(beta1).add_(grad, alpha = 1 - beta1) # 假设每个参数张量p（大小为n），grad大小也是n。3n FLOPs
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2) # 4n FLOPs

                # 偏差校正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                step_size = alpha / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps) # 3n FLOPs

                p.data.mul_(1 - alpha * lambda_) # n FLOPs
                p.data.addcdiv_(m, denom, value=-step_size) # 3n FLOPs

                state['s'] = t
                
        return loss
    

def get_lr_cosine_schedule(iteration, max_lr, min_lr, warmup_iter, cosine_cycle_iters):
    if iteration < warmup_iter:
        return max_lr * iteration / warmup_iter
    elif iteration >= warmup_iter and iteration <= cosine_cycle_iters:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (iteration - warmup_iter) / (cosine_cycle_iters - warmup_iter)))
    else:
        return min_lr