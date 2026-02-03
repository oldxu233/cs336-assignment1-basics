import torch
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    n = len(dataset) - context_length
    ix = np.random.randint(0, n, size=batch_size)

    # 分配一次大缓冲（最快、最干净）
    x = np.empty((batch_size, context_length), dtype=np.uint16)
    y = np.empty((batch_size, context_length), dtype=np.uint16)

    for k, i in enumerate(ix):
        x[k] = dataset[i:i+context_length]
        y[k] = dataset[i+1:i+context_length+1]

    # 直接转 tensor 并放到 device
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    return x, y
