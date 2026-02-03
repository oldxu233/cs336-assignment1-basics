# pytest -k test_checkpointing
import torch 


def save_checkpoint(model, optimizer, iteration, out):
    """
    将模型、优化器状态和迭代次数保存到目标位置
    参数：
        model: torch.nn.Module - 待保存的模型
        optimizer: torch.optim.Optimizer - 待保存的优化器
        iteration: int - 当前迭代次数
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] - 保存路径或类文件对象
    """
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }, out)


def load_checkpoint(src, model, optimizer):
    """
    从源位置加载检查点，恢复模型和优化器状态
    参数：
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] - 加载路径或类文件对象
        model: torch.nn.Module - 待恢复的模型
        optimizer: torch.optim.Optimizer - 待恢复的优化器
    返回：
        int: 保存的迭代次数
    """   
    checkpoint = torch.load(src)
    # print(f"checkpoint.keys(): {checkpoint.keys()}")

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']