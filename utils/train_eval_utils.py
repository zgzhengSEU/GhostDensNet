import sys

from tqdm import tqdm
import torch
from torch import nn
import random
from matplotlib import pyplot as plt
import cv2
from .distributed_utils import reduce_value, is_main_process
import wandb
from torchvision import transforms
import math

def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    device,
                    epoch,
                    lr_scheduler,
                    warmup_scheduler):
    model.train()

    criterion = nn.MSELoss(reduction='sum').to(device)

    mean_loss = torch.zeros(1).to(device)

    # 在进程0中打印训练进度
    if is_main_process():
        train_loader = tqdm(train_loader, file=sys.stdout)

    for step, (img, gt_dmap) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap = model(img)
        # calculate loss
        # print(f'et_dmap.shape: {et_dmap.shape}')
        # print(f'gt_dmap.shape: {gt_dmap.shape}')
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
        with warmup_scheduler.dampening():
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()

def train_one_epoch_single_gpu(model,
                    optimizer,
                    train_loader,
                    device,
                    epoch,
                    lr_scheduler,
                    warmup_scheduler):
    model.train()

    criterion = nn.MSELoss(reduction='sum').to(device)

    mean_loss = torch.zeros(1).to(device)

    # 打印训练进度
    train_loader = tqdm(train_loader, file=sys.stdout)

    for step, (img, gt_dmap) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap = model(img)
        # calculate loss
        # print(f'et_dmap.shape: {et_dmap.shape}')
        # print(f'gt_dmap.shape: {gt_dmap.shape}')
        loss = criterion(et_dmap, gt_dmap)
        loss.backward()
        # update mean losses
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # 在进程0中打印平均loss

        train_loader.desc = "[epoch {}] mean loss {}".format(
            epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()
            
    return mean_loss.item()

@torch.no_grad()
def evaluate(model,
             test_loader,
             device,
             epoch,
             show_images=False,
             use_wandb=False):
    model.eval()

    mae = torch.zeros(1).to(device)
    mse = torch.zeros(1).to(device)
    # 在进程0中打印验证进度
    if is_main_process():
        test_loader = tqdm(test_loader, file=sys.stdout)

    index = random.randint(0, len(test_loader)-1)

    for step, (img, gt_dmap) in enumerate(test_loader):
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap = model(img)
        mae += abs(et_dmap.data.sum() - gt_dmap.data.sum())
        mse += (et_dmap.data.sum() - gt_dmap.data.sum()) * (et_dmap.data.sum() - gt_dmap.data.sum())
        if is_main_process():
            test_loader.desc = f"[epoch {epoch}]"

            if step == index and show_images:
                images = []
                # ============= img ==========
                _, h, w = img[0].shape
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                    std=[1/0.229, 1/0.224, 1/0.255]
                )
                img_np = inv_normalize(img[0]).permute(1, 2, 0).cpu().numpy()
                # ============= gt ===========
                fig1 = plt.figure()
                plt.axis('off')
                plt.imshow(img_np)
                gt_dmap_np = gt_dmap[0].permute(1, 2, 0).cpu().numpy()
                gt_dmap_np = cv2.resize(gt_dmap_np, dsize=(w, h))
                plt.imshow(gt_dmap_np, alpha=0.5, cmap='turbo')
                fig1.savefig(f"checkpoints/temp/temp_gt_{epoch}.png",
                             bbox_inches='tight', pad_inches=0)
                fig1.clear()
                plt.close('all')
                # ============ et ==============
                fig2 = plt.figure()
                plt.axis('off')
                plt.imshow(img_np)
                et_dmap_np = et_dmap[0].permute(1, 2, 0).cpu().numpy()
                et_dmap_np = cv2.resize(et_dmap_np, dsize=(w, h))
                plt.imshow(et_dmap_np, alpha=0.5, cmap='turbo')
                fig2.savefig(f"checkpoints/temp/temp_et_{epoch}.png",
                             bbox_inches='tight', pad_inches=0)
                fig2.clear()
                plt.close('all')
                # =========== upload ============
                if use_wandb:
                    images.append(wandb.Image(
                        img_np, caption=f"image {epoch}"))
                    images.append(wandb.Image(plt.imread(
                        f"checkpoints/temp/temp_gt_{epoch}.png"), caption=f"gt_density {epoch}"))
                    images.append(wandb.Image(plt.imread(
                        f"checkpoints/temp/temp_et_{epoch}.png"), caption=f"et_density {epoch}"))
                    wandb.log({"examples": [wandb.Image(im) for im in images]})
                    print(f"[epoch {epoch}] wandb upload img done!")

        del img, gt_dmap, et_dmap

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    mae_sum = reduce_value(mae, average=False)
    mse_sum = reduce_value(mse, average=False)
    return mae_sum.item(), mse_sum.item()

@torch.no_grad()
def evaluate_single_gpu(model,
             test_loader,
             device,
             epoch,
             show_images=False,
             use_wandb=False):
    model.eval()

    mae = torch.zeros(1).to(device)
    mse = torch.zeros(1).to(device)
    # 在进程中打印验证进度

    test_loader = tqdm(test_loader, file=sys.stdout)

    index = random.randint(0, len(test_loader)-1)

    for step, (img, gt_dmap) in enumerate(test_loader):
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap = model(img)
        mae += abs(et_dmap.data.sum() - gt_dmap.data.sum())
        mse += (et_dmap.data.sum() - gt_dmap.data.sum()) * (et_dmap.data.sum() - gt_dmap.data.sum())
        test_loader.desc = f"[epoch {epoch}]"

        if step == index and show_images:
            images = []
            # ============= img ==========
            _, h, w = img[0].shape
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
            img_np = inv_normalize(img[0]).permute(1, 2, 0).cpu().numpy()
            # ============= gt ===========
            fig1 = plt.figure()
            plt.axis('off')
            plt.imshow(img_np)
            gt_dmap_np = gt_dmap[0].permute(1, 2, 0).cpu().numpy()
            gt_dmap_np = cv2.resize(gt_dmap_np, dsize=(w, h))
            plt.imshow(gt_dmap_np, alpha=0.5, cmap='turbo')
            fig1.savefig(f"checkpoints/temp/temp_gt_{epoch}.png",
                            bbox_inches='tight', pad_inches=0)
            fig1.clear()
            plt.close('all')
            # ============ et ==============
            fig2 = plt.figure()
            plt.axis('off')
            plt.imshow(img_np)
            et_dmap_np = et_dmap[0].permute(1, 2, 0).cpu().numpy()
            et_dmap_np = cv2.resize(et_dmap_np, dsize=(w, h))
            plt.imshow(et_dmap_np, alpha=0.5, cmap='turbo')
            fig2.savefig(f"checkpoints/temp/temp_et_{epoch}.png",
                            bbox_inches='tight', pad_inches=0)
            fig2.clear()
            plt.close('all')
            # =========== upload ============
            if use_wandb:
                images.append(wandb.Image(
                    img_np, caption=f"image {epoch}"))
                images.append(wandb.Image(plt.imread(
                    f"checkpoints/temp/temp_gt_{epoch}.png"), caption=f"gt_density {epoch}"))
                images.append(wandb.Image(plt.imread(
                    f"checkpoints/temp/temp_et_{epoch}.png"), caption=f"et_density {epoch}"))
                wandb.log({"examples": [wandb.Image(im) for im in images]})
                print(f"[epoch {epoch}] wandb upload img done!")

        del img, gt_dmap, et_dmap

    return mae.item(), mse.item()
