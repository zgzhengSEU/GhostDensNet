import time
import torch
import os
from tqdm import tqdm as tqdm
import time

from model.GhostDensNet import GDNet
from model.ghostnetv2_torch import GhostNetV2DensNew
from model.efficientv2 import effnetv2_s
from model.effnetv2 import EfficientNet
from model.CrowdDataset import CrowdDataset
from utils.train_eval_utils import train_one_epoch_single_gpu, evaluate_single_gpu

import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_warmup as warmup
import wandb
from collections import OrderedDict
import math

def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict   

def load_state_dict(checkpoint_path, map_location, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint(model, checkpoint_path, map_location='cpu', use_ema=True, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, map_location, use_ema)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

"""
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 --use_env train.py   
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py   
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --use_env train.py
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default="./data/Shanghai_part_A/")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--show_images', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_id', type=str, default='5tpdfo8k')
    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='resume checkpoint path')
    parser.add_argument('--init_checkpoint', type=str, default='./checkpoints/ghostnetv2_torch/ck_ghostnetv2_10.pth',
                        help='initial weights path')
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    return args

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # configuration
    init_checkpoint = args.init_checkpoint
    temp_init_checkpoint_path = "checkpoints"
    resume_checkpoint = args.resume_checkpoint

    use_wandb = args.wandb
    show_images = args.show_images
    resume = args.resume
    resume_id = args.resume_id
    
    lr = args.lr
    gpu_or_cpu = args.device  # use cuda or cpu
    batch_size = args.batch_size

    start_epoch = 0
    epochs = args.epochs
    num_workers = 2
    seed = time.time()

    curtime = time.strftime(
        '%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
    print(f"[{curtime}] GPU train start!")
    print(args)

    if os.path.exists(temp_init_checkpoint_path) is False:
        os.makedirs(temp_init_checkpoint_path)
    if os.path.exists('checkpoints/temp/') is False:
        os.makedirs('checkpoints/temp/')
    
    if use_wandb:
        if resume:
            wandb.init(
                project="Density",
                group="ShanghaiTech",
                mode="online",
                resume='allow',
                id = resume_id,
                name='GhostDensNet')
        else:
            wandb.init(
                project="Density",
                group="ShanghaiTech",
                mode="online",
                name='GhostDensNetFPNnew')

    # ======================== cuda ====================================
    device = torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    # ==================================== data ==================
    # DataPath Shanghai_part_A
    # train_image_root = args.data_root + 'train_data/images'
    # train_dmap_root = args.data_root + 'train_data/ground_truth'
    # test_image_root = args.data_root + '.test_data/images'
    # test_dmap_root = args.data_root + '.test_data/ground_truth'
    train_image_root = 'data/shanghaitech/ShanghaiTech/part_A/train_data/images'
    train_dmap_root = 'data/shanghaitech/ShanghaiTech/part_A/train_data/ground-truth'
    test_image_root = 'data/shanghaitech/ShanghaiTech/part_A/test_data/images'
    test_dmap_root = 'data/shanghaitech/ShanghaiTech/part_A/test_data/ground-truth'

    train_dataset = CrowdDataset(
        train_image_root, train_dmap_root, gt_downsample=8, phase='train')
    test_dataset = CrowdDataset(
        test_image_root, test_dmap_root, gt_downsample=8, phase='test')


    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)

    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    # ========================================= model ===========================
    model = GhostNetV2DensNew().to(device)
    
    if resume:
        resume_load_checkpoint = torch.load(resume_checkpoint, map_location=device)
        start_epoch = resume_load_checkpoint['epoch']
        model.load_state_dict(resume_load_checkpoint['model_state_dict'])
        # ========================= optimizer ===================================
        pg = [p for p in model.parameters() if p.requires_grad]
        num_steps = len(train_loader) * epochs
        optimizer = optim.AdamW(pg, lr=lr, betas=(
            0.9, 0.999), weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
        
        optimizer.load_state_dict(resume_load_checkpoint['optim_state_dict'])
        scheduler.load_state_dict(resume_load_checkpoint['scheduler'])
        warmup_scheduler.load_state_dict(
            resume_load_checkpoint['warmup_scheduler'])

    elif os.path.exists(init_checkpoint):
        # weights_dict = torch.load(init_checkpoint, map_location=device)
        # model.load_state_dict(weights_dict, strict=False)
        load_checkpoint(model, init_checkpoint, strict=False, map_location=device)
        
        # load_checkpoint = torch.load(init_checkpoint)
        # model.load_state_dict(load_checkpoint['model'].state_dict(), strict=False)
        print(f'[load checkpoint from {init_checkpoint}]')

    # ===================================== optimizer ===========================================
    if not resume:
        pg = [p for p in model.parameters() if p.requires_grad]
        num_steps = len(train_loader) * epochs
        optimizer = optim.AdamW(pg, lr=lr, betas=(
            0.9, 0.999), weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    # ========================================= train and eval ============================================
    min_mae = 1e10
    min_mse = 1e10
    min_epoch = 0
    for epoch in range(start_epoch, epochs):
        # training phase
        mean_loss = train_one_epoch_single_gpu(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=scheduler,
            warmup_scheduler=warmup_scheduler
        )

        # testing phase
        mae_sum, mse_sum = evaluate_single_gpu(
            model=model,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            show_images=show_images,
            use_wandb=use_wandb
        )


        # eval and log
        mean_mae = mae_sum / len(test_loader)
        mean_mse = math.sqrt(mse_sum / len(test_loader))
        # checkpoints
        if os.path.exists(f'./checkpoints/epoch_{epoch - 1}.pth.tar') is True:
            os.remove(f'./checkpoints/epoch_{epoch - 1}.pth.tar')

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict()}
        torch.save(checkpoint_dict, f'./checkpoints/epoch_{epoch}.pth.tar')

        if mean_mae < min_mae:
            min_mae = mean_mae
            min_epoch = epoch
            torch.save(checkpoint_dict,
                        f'./checkpoints/best_epoch_{epoch}.pth.tar')
        
        if mean_mse < min_mse:
            min_mse = mean_mse
        
        print(
            f"[epoch {epoch}] mae: {mean_mae}, min_mae: {min_mae}, min_mse: {min_mse}, best_epoch: {min_epoch}")

        if use_wandb:
            wandb.log({'MSELoss': mean_loss})
            wandb.log({'MAE': mean_mae})
            wandb.log({'MSE': mean_mse})
            wandb.log({'lr': optimizer.param_groups[0]["lr"]})
            print(f"[epoch {epoch}] wandb log done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
