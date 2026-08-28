# src/models/custom_swinunetr.py

import torch
from torch import nn
from monai.networks.nets import SwinUNETR
from .unet import PlainConvUNet  

class CustomSwinUNETRModel(nn.Module):
    def __init__(self, 
                 num_modalities=2, 
                 num_classes=4, 
                 img_size=(128, 128, 128),  # 根据需要调整
                 feature_size=48,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 dropout_path_rate=0.1,
                 spatial_dims=3,
                 ):
        super(CustomSwinUNETRModel, self).__init__()
        self.net = SwinUNETR(
            img_size=img_size,
            in_channels=num_modalities, 
            out_channels=num_classes,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            spatial_dims=spatial_dims,
            # deep_supervision=False,  # 根据需要调整
            )
        
        # 可选：加载预训练权重
        # pretrained_weights_path = "path_to_pretrained_weights.pth"
        # if pretrained_weights_path:
        #     self.net.load_from(pretrained_weights_path)

    def forward(self, x):
        # 假设输入 x 的形状为 [B, 4, D, H, W]
        # 将输入拆分为不同模态
        # pet = x[:, 0:1, :, :, :]     # [B,1,D,H,W]
        # ct = x[:, 1:2, :, :, :]      # [B,1,D,H,W]
        # suv = x[:, 2:3, :, :, :]    # [B,1,D,H,W]  # 忽略
        # organ = x[:, 3:4, :, :, :]   # [B,1,D,H,W]
        
        # 调试：打印形状
        # print(f"PET shape: {pet.shape}")          # 期望: [B,1,D,H,W]
        # print(f"CT shape: {ct.shape}")            # 期望: [B,1,D,H,W]
        # print(f"Organ label shape: {organ.shape}") # 期望: [B,1,D,H,W]
        
        # 将 PET 和 CT 拼接作为模型输入
        # input_x = torch.cat([pet, ct], dim=1)  # [B,2,D,H,W]
        # print(f"Input x shape (PET + CT): {input_x.shape}")  # 期望: [B,2,D,H,W]
        input_x = x
        # 通过 SwinUNETR 前向传播
        output = self.net(input_x)  # [B, n_classes, D, H, W]
        # print(f"Output shape: {output.shape}")  # 期望: [B, n_classes, D, H, W]
        
        # 添加更多调试信息
        # print(f"Output type: {type(output)}")
        # if isinstance(output, list):
        #     for i, o in enumerate(output):
        #         print(f"Output[{i}] type: {type(o)}, shape: {o.shape}")
        # else:
        #     print(f"Output shape: {output.shape}")
        
        return output


class MySwinUNETR(PlainConvUNet):
    def __init__(self, *args, **kwargs):
        super(MySwinUNETR, self).__init__(*args, **kwargs)
        # 初始化 CustomSwinUNETRModel
        self.net = CustomSwinUNETRModel(
            num_modalities=2,              # PET、CT 和其他模态（如 organ label）
            num_classes=2,                 # 包含背景 + 1 类
            img_size=(128, 128, 128),      # 根据需要调整
            feature_size=24,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.1,         # 根据需要调整
            spatial_dims=3,
        )
        print("**********************CustomSwinUNETR Initialized with Pretrained Weights and Frozen Encoder****************************")
    
    def forward(self, x):
        y = self.net(x)
        return y
