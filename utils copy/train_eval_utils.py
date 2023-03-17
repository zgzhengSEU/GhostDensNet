import sys

from tqdm import tqdm
import torch

from .distributed_utils import reduce_value, is_main_process


def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    device,
                    epoch):
    model.train()

    criterion = torch.nn.MSELoss(reduction='sum').to(device)

    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        train_loader = tqdm(train_loader, file=sys.stdout)

    for step, (img, gt_dmap) in enumerate(train_loader):
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap = model(img)
        # calculate loss
        loss = criterion(et_dmap, gt_dmap)
        loss.backward()
        loss = reduce_value(loss, average=True)
        # update mean losses
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # 在进程0中打印平均loss
        if is_main_process():
            train_loader.desc = "[epoch {}] mean loss {}".format(
                epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model,
             test_loader,
             device):
    model.eval()

    mae = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        test_loader = tqdm(test_loader, file=sys.stdout)

    for step, (img, gt_dmap) in enumerate(test_loader):
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap = model(img)
        mae += torch.abs(et_dmap.data.sum() - gt_dmap.data.sum())
        del img, gt_dmap, et_dmap

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    mae_sum = reduce_value(mae, average=False)

    return mae_sum.item()
