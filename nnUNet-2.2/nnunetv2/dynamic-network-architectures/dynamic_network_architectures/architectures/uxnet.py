# src/models/custom_uxnet.py

import torch
from torch import nn
from .uxnet3d.networks.UXNet_3D.network_backbone import UXNET  # 导入 UXNET
from .unet import PlainConvUNet

class CustomUXNETModel(nn.Module):
    def __init__(self, 
                 num_modalities=2, 
                 num_classes=4, 
                 depths=(2, 2, 2, 2),
                 feat_size=[48, 96, 192, 384],  # 对应 UXNET 的特征尺寸
                 drop_path_rate=0.0,
                 spatial_dims=3,
                 ) -> None:
        super(CustomUXNETModel, self).__init__()
        # 初始化 UXNET
        self.net = UXNET(
            in_chans=num_modalities,  # 输入通道数
            out_chans=num_classes,    # 输出通道数 (类别数)
            depths=depths,            # 每个阶段的深度
            feat_size=feat_size,      # 特征图大小
            drop_path_rate=drop_path_rate,  # Dropout 路径率
            spatial_dims=spatial_dims,  # 空间维度
        )

    def forward(self, x):
        input_x = x
        output = self.net(input_x)  # UXNET 前向传播
        return output

class MyUXNET(PlainConvUNet):
    def __init__(self, *args, **kwargs):
        super(MyUXNET, self).__init__(*args, **kwargs)
        # 初始化 CustomUXNETModel
        self.net = CustomUXNETModel(
            num_modalities=2,              # PET、CT 和其他模态（如 organ label）
            num_classes=3,                 # 包含背景 + 1 类
            depths=(2, 2, 2, 2),           # 每个阶段的深度
            feat_size=[48, 96, 192, 384],  # 特征图的尺寸
            drop_path_rate=0.1,            # 根据需要调整
            spatial_dims=3,                # 空间维度
        )
        print("**********************UXNET Initialized with Pretrained Weights****************************")

        print("**********************UXNET Initialized with Pretrained Weights****************************")
    def forward(self, x):
        y = self.net(x)
        return y
