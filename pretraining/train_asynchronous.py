import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split  # 引入原生 DataLoader 和 random_split
from tqdm import tqdm

# 导入网络
from model_asynchronous import STUNet_MAE
# 导入第一段代码中的 Dataset (假设写在 dataloader.py 中)
from dataloader import NPZDataset


def visualize_val_samples(model, val_loader, device, epoch, save_dir, num_samples=2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    model.train()
    collected = 0

    with torch.no_grad():
        # 注意这里的拆包，兼容 NPZDataset 的返回格式
        for batch_data in val_loader:
            if collected >= num_samples:
                break

            # 兼容：如果返回的是 (data, label)，取第一个；如果只有 data，直接用
            data = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            if data is None: continue

            data = data.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                reconstruction, visible_mask = model(data)

            B = data.size(0)
            for b in range(B):
                if collected >= num_samples:
                    break

                D = data.shape[2]
                mid = D // 2

                orig_ct = data[b, 0, mid].cpu().float().numpy()
                orig_pet = data[b, 1, mid].cpu().float().numpy()

                vis_mask_ct = visible_mask[b, 0, mid].cpu().numpy()
                vis_mask_pet = visible_mask[b, 1, mid].cpu().numpy()

                recon_ct = reconstruction[b, 0, mid].cpu().float().numpy()
                recon_pet = reconstruction[b, 1, mid].cpu().float().numpy()

                masked_ct = np.where(vis_mask_ct, orig_ct, orig_ct.min())
                masked_pet = np.where(vis_mask_pet, orig_pet, orig_pet.min())

                comp_ct = np.where(vis_mask_ct, orig_ct, recon_ct)
                comp_pet = np.where(vis_mask_pet, orig_pet, recon_pet)

                fig, axes = plt.subplots(2, 4, figsize=(16, 8))

                m_ct = int((~vis_mask_ct).mean() * 100)
                m_pet = int((~vis_mask_pet).mean() * 100)
                avg_m = (m_ct + m_pet) // 2

                fig.suptitle(
                    f'Epoch {epoch + 1} | Sample {collected + 1} | Slice {mid}\n'
                    f'Masking: CT {m_ct}% / PET {m_pet}%', fontsize=12
                )

                titles = ['Original', f'Masked (~{avg_m}%)', 'Pure Recon', 'Composite']
                ct_imgs = [orig_ct, masked_ct, recon_ct, comp_ct]
                pet_imgs = [orig_pet, masked_pet, recon_pet, comp_pet]

                for col, (title, ci, pi) in enumerate(zip(titles, ct_imgs, pet_imgs)):
                    axes[0, col].imshow(ci, cmap='gray', aspect='auto')
                    axes[0, col].set_title(f'CT - {title}', fontsize=9)
                    axes[0, col].axis('off')
                    axes[1, col].imshow(pi, cmap='hot', aspect='auto')
                    axes[1, col].set_title(f'PET - {title}', fontsize=9)
                    axes[1, col].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'epoch_{epoch + 1:04d}_{collected + 1}.png'))
                plt.close(fig)
                collected += 1

    model.eval()


def train_mae(args):
    use_ddp = 'RANK' in os.environ

    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
        world_size = dist.get_world_size()
        if is_main:
            print(f"[DDP 模式] 使用 {world_size} 个 GPU")
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True
        print(f"[单机模式] 使用设备: {device}")
        if torch.cuda.device_count() > 1:
            print(f"  检测到 {torch.cuda.device_count()} 个 GPU，将使用 DataParallel")

    # --- 修改点 1：使用第一段代码的 NPZDataset ----------------------
    full_dataset = NPZDataset(args.data_dir)  # 注意参数名换成了 data_dir

    # 划分训练集和验证集 (保留 5% 作为验证集，保持老代码的功能)
    val_size = max(1, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - val_size

    # 固定随机种子确保 DDP 各个进程划分一致
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 如果是 DDP，需要 DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if use_ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True
    )
    # -----------------------------------------------------------

    model = STUNet_MAE(
        mask_ratio=args.mask_ratio,
        mask_block_size=(32, 32, 32),
        input_channels=2,
        num_classes=2,
        depth=[1, 1, 1, 1, 1, 1],
        dims=[16, 32, 64, 128, 256, 256],
        pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    )
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print("\n--- 开始 MAE 预训练 ---")
    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=not is_main)

        # --- 修改点 2：循环解包适配新的 Loader 返回格式 ---------
        for batch_data in progress_bar:
            # 兼容 NPZDataset 返回格式
            data = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            if data is None: continue

            data = data.to(device, non_blocking=True)
            target = data.clone()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                reconstruction, visible_mask = model(data)

                mask_region = ~visible_mask
                mask_region = mask_region.expand_as(target)

                if mask_region.sum() > 0:
                    loss = F.mse_loss(reconstruction[mask_region], target[mask_region])
                else:
                    loss = F.mse_loss(reconstruction, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * data.size(0)
            progress_bar.set_postfix({'Loss (Masked MSE)': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_dataset)
        if is_main:
            print(f"[Train] Epoch {epoch + 1} Avg Loss: {epoch_loss:.6f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_data in val_loader:
                data = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                if data is None: continue
                data = data.to(device, non_blocking=True)
                target = data.clone()

                with torch.cuda.amp.autocast():
                    reconstruction, _ = model(data)
                    loss = F.mse_loss(reconstruction, target)

                val_loss += loss.item() * data.size(0)
                val_samples += data.size(0)

        # 汇总所有 GPU 上的 val_loss
        if use_ddp:
            val_loss_tensor = torch.tensor([val_loss, val_samples], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor[0].item()
            val_samples = val_loss_tensor[1].item()

        avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
        if is_main:
            print(f"[ Val ] Epoch {epoch + 1} Global MSE Loss: {avg_val_loss:.6f}")

        if is_main and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            raw = model.module if hasattr(model, 'module') else model
            torch.save(raw.state_dict(), os.path.join(args.save_dir, 'mae_best.pth'))
            print(f"☆ Saved Best Model")

        if is_main:
            raw = model.module if hasattr(model, 'module') else model
            torch.save(raw.state_dict(), os.path.join(args.save_dir, 'mae_last_Spark3D_asy9.pth'))

            visualize_val_samples(
                model=raw,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                save_dir=args.save_dir,
                num_samples=2
            )

    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 注意：这里改为了 data_dir 以对应你的 NPZDataset 习惯 # /processing/c.lu/nnUNet_preprocessed/Dataset001_CropHNC/nnUNetPatchExtractorTrainer__nnUNetPlans__3d_fullres/fold_all
    #/processing/c.lu/nnUNet_preprocessed/Dataset245_AutoPET_psma/nnUNetPatchExtractorTrainer__nnUNetPlans__3d_fullres/fold_all
    #/processing/c.lu/pretrain_headnector/nnUNetPatchExtractorTrainer__nnUNetPlans__3d_fullres/fold_all
    #/processing/c.lu/nnUNet_preprocessed/Dataset100_autopet/nnUNetPatchExtractorTrainer__nnUNetPlans__3d_fullres/fold_all
    parser.add_argument('--data_dir', type=str,
                        default='/processing/c.lu/nnUNet_preprocessed/Dataset001_CropHNC/nnUNetPatchExtractorTrainer__nnUNetPlans__3d_fullres/fold_all')
    parser.add_argument('--save_dir', type=str,
                        default='/projects/lcy_data/pet_ct_challenge/xinglong/MAE_base_jiaju_dataloader/headnector')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.9)# psma 0.5 hN 0.8 FDG 0.5

    args = parser.parse_args()
    train_mae(args)