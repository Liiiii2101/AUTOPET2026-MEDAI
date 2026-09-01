import torch
import numpy as np
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from typing import Union

# 导入 nnU-Net 的基础组件
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.STUNetTrainer import STUNetTrainer_small_pretrain_location,STUNetTrainer_small_pretrain_STUNet_DualEncoder_gatefuse
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset


class MyCustomCurriculumTrainer(STUNetTrainer_small_pretrain_location):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # --- 遵循 CVPR 2025 论文的阶段定义 ---
        self.stage_1_end = 50   # 阶段 1：仅解码器预热 (0-50 epoch)
        self.stage_2_end = 100  # 阶段 2：全网联合预热 (50-100 epoch)
        self.initial_lr = 1e-4  # 论文建议微调使用较低的学习率 (如 1e-3 或 1e-4) [cite: 243, 252]

        # STU-Net 的编码器关键字
        self.encoder_key = "conv_blocks_context"

    def set_freeze_status(self, mode: str):
        """
        控制参数更新：
        - 'decoder_only': 冻结预训练编码器，仅允许随机初始化的解码器更新
        - 'all_parts': 解冻全网进行联合微调
        """
        mod = self.network.module if isinstance(self.network, (DDP, OptimizedModule)) else self.network
        if hasattr(mod, '_orig_mod'): mod = mod._orig_mod

        for name, param in mod.named_parameters():
            if mode == 'decoder_only':
                # 如果参数名包含编码器关键字，则冻结 (requires_grad = False)
                # 其余部分（解码器和分割头）保持活跃
                param.requires_grad = self.encoder_key not in name
            else:
                # 全网络解冻
                param.requires_grad = True

        msg = ">>> 仅解码器活跃 (编码器已冻结保护)" if mode == 'decoder_only' else ">>> 全网络活跃 (开始联合训练)"
        self.print_to_log_file(f"❄️ 状态切换: {msg}")

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        """核心：在不同阶段物理重置优化器"""
        self.set_freeze_status('decoder_only' if stage == 'warmup_decoder' else 'all_parts')

        # 重新获取可训练参数
        params = [p for p in self.network.parameters() if p.requires_grad]

        # 创建优化器
        optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        # ====================================================
        # 【新添加：修复 KeyError 的关键代码】
        # 为每个参数组手动设置 initial_lr 键，满足调度器的恢复检查
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        # ====================================================

        if stage == "warmup_decoder":
            # 阶段 1: 0->50 线性预热
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.stage_1_end)

        elif stage == "warmup_all":
            # 阶段 2: 再次线性预热。
            # 如果你想按照论文实现“再次从 0 爬升 50 轮”，建议将这里的总时长设为 50，
            # 并且 current_step 设为 -1 (或者不传)，让它从头起跳。
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, 50)

        else:
            # 阶段 3: 100->1000 正式 Poly 训练
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs,
                                                  self.stage_2_end)

        self.training_stage = stage
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        """
        手动重写以支持 CVPR 2025 论文的双重预热策略。
        实现：0-50 预热解码器 -> 50-100 联合预热 -> 100+ 正式训练
        """
        # 1. 阶段切换逻辑
        if self.current_epoch == 0:
            self.print_to_log_file(f"--- [Stage 1] 仅预热解码器 (0 -> {self.stage_1_end}) ---")
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_decoder")
        elif self.current_epoch == self.stage_1_end:
            self.print_to_log_file(f"--- [Stage 2] 解冻编码器，开始第二次 0->Max 预热 (50 -> {self.stage_2_end}) ---")
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.stage_2_end:
            self.print_to_log_file("--- [Stage 3] 联合预热结束，进入正式 Poly 衰减训练 ---")
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        # 2. 计算相对轮次 (核心修复)
        # 在 Stage 2 时，我们希望传给调度器的是 0, 1, 2... 而不是 50, 51, 52...
        if self.training_stage == "warmup_all":
            relative_epoch = self.current_epoch - self.stage_1_end
        else:
            relative_epoch = self.current_epoch

        # 3. 【手动执行基类核心任务】(替代 super().on_train_epoch_start)
        self.network.train() # 确保 BatchNorm/Dropout 正常运行
        self.lr_scheduler.step(relative_epoch) # 使用相对轮次，确保从 0 重新起爬

        # 4. 手动记录日志 (保持 nnU-Net 习惯)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch} (Stage: {self.training_stage}, Relative: {relative_epoch})')
        self.print_to_log_file(f"Current learning rate: {np.round(current_lr, decimals=8)}")
        self.logger.log('lrs', current_lr, self.current_epoch)
class MyCustomCurriculumTrainer_DualEncoder_gatefuse(STUNetTrainer_small_pretrain_STUNet_DualEncoder_gatefuse):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # --- 遵循 CVPR 2025 论文的阶段定义 ---
        self.stage_1_end = 50   # 阶段 1：仅解码器预热 (0-50 epoch)
        self.stage_2_end = 100  # 阶段 2：全网联合预热 (50-100 epoch)
        self.initial_lr = 1e-3  # 论文建议微调使用较低的学习率 (如 1e-3 或 1e-4) [cite: 243, 252]

        # STU-Net 的编码器关键字
        self.encoder_key = "conv_blocks_context"

    def set_freeze_status(self, mode: str):
        mod = self.network.module if isinstance(self.network, (DDP, OptimizedModule)) else self.network
        if hasattr(mod, '_orig_mod'): mod = mod._orig_mod

        # 定义需要保护（冻结）的预训练关键字
        # 包含两个编码器和门控融合层
        pretrained_keys = ["encoder_ct", "encoder_pet", "simple_fusion"]

        for name, param in mod.named_parameters():
            if mode == 'decoder_only':
                # 检查当前参数名是否包含任何预训练关键字
                is_pretrained = any(k in name for k in pretrained_keys)
                # 如果是预训练部分，则冻结（requires_grad = False）
                param.requires_grad = not is_pretrained
            else:
                # 全网络解冻
                param.requires_grad = True

        msg = ">>> 仅解码器活跃 (双编码器与融合层已冻结)" if mode == 'decoder_only' else ">>> 全网络活跃 (开始全量微调)"
        self.print_to_log_file(f"❄️ 状态切换: {msg}")

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        """核心：在不同阶段物理重置优化器"""
        self.set_freeze_status('decoder_only' if stage == 'warmup_decoder' else 'all_parts')

        # 重新获取可训练参数
        params = [p for p in self.network.parameters() if p.requires_grad]

        # 创建优化器
        optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        # ====================================================
        # 【新添加：修复 KeyError 的关键代码】
        # 为每个参数组手动设置 initial_lr 键，满足调度器的恢复检查
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        # ====================================================

        if stage == "warmup_decoder":
            # 阶段 1: 0->50 线性预热
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.stage_1_end)

        elif stage == "warmup_all":
            # 阶段 2: 再次线性预热。
            # 如果你想按照论文实现“再次从 0 爬升 50 轮”，建议将这里的总时长设为 50，
            # 并且 current_step 设为 -1 (或者不传)，让它从头起跳。
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, 50)

        else:
            # 阶段 3: 100->1000 正式 Poly 训练
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs,
                                                  self.stage_2_end)

        self.training_stage = stage
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        """
        手动重写以支持 CVPR 2025 论文的双重预热策略。
        实现：0-50 预热解码器 -> 50-100 联合预热 -> 100+ 正式训练
        """
        # 1. 阶段切换逻辑
        if self.current_epoch == 0:
            self.print_to_log_file(f"--- [Stage 1] 仅预热解码器 (0 -> {self.stage_1_end}) ---")
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_decoder")
        elif self.current_epoch == self.stage_1_end:
            self.print_to_log_file(f"--- [Stage 2] 解冻编码器，开始第二次 0->Max 预热 (50 -> {self.stage_2_end}) ---")
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.stage_2_end:
            self.print_to_log_file("--- [Stage 3] 联合预热结束，进入正式 Poly 衰减训练 ---")
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        # 2. 计算相对轮次 (核心修复)
        # 在 Stage 2 时，我们希望传给调度器的是 0, 1, 2... 而不是 50, 51, 52...
        if self.training_stage == "warmup_all":
            relative_epoch = self.current_epoch - self.stage_1_end
        else:
            relative_epoch = self.current_epoch

        # 3. 【手动执行基类核心任务】(替代 super().on_train_epoch_start)
        self.network.train() # 确保 BatchNorm/Dropout 正常运行
        self.lr_scheduler.step(relative_epoch) # 使用相对轮次，确保从 0 重新起爬

        # 4. 手动记录日志 (保持 nnU-Net 习惯)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch} (Stage: {self.training_stage}, Relative: {relative_epoch})')
        self.print_to_log_file(f"Current learning rate: {np.round(current_lr, decimals=8)}")
        self.logger.log('lrs', current_lr, self.current_epoch)

