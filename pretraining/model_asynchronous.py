# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


# --- 新增: MAE 包装器 ---
class STUNet_MAE(nn.Module):
    def __init__(self, mask_ratio=0.6, mask_block_size=(16, 16, 16), **stunet_kwargs):
        """
        mask_ratio: 遮挡比例 (通常 MAE 用 0.60 ~ 0.75)
        mask_block_size: 遮挡块的大小 (默认 16x16x16)
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

        # 强制关闭深度监督，因为 MAE 重建通常只需要最后一层输出计算 Loss
        stunet_kwargs['enable_deep_supervision'] = False
        self.backbone = STUNet(**stunet_kwargs)

        # 可学习的 Mask Token (每个通道一个独立的可学习标量/向量)
        in_channels = stunet_kwargs.get('input_channels', 2)
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)

    def generate_mask(self, x):
        """ 生成 3D Block-wise 掩码 """
        B, C, D, H, W = x.shape
        bd, bh, bw = self.mask_block_size

        # 计算 grid 数量
        grid_d, grid_h, grid_w = D // bd, H // bh, W // bw
        num_blocks = grid_d * grid_h * grid_w
        num_mask = int(num_blocks * self.mask_ratio) #default MAE

        # --- 修改这里：将固定比例改为 0.6 - 0.9 之间的随机值 - -- Spark3D
        # current_ratio = 0.6 + torch.rand(1).item() * 0.3
        # num_mask = int(num_blocks * current_ratio)
        # 并行生成随机噪声
        noise = torch.rand(B, num_blocks, device=x.device)

        # 获取要遮挡的索引
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_mask = ids_shuffle[:, :num_mask]

        # 生成平铺的 mask
        mask_flat = torch.ones(B, num_blocks, device=x.device)
        mask_flat.scatter_(1, ids_mask, 0.0)  # 0 表示被遮挡，1 表示保留可见
        mask_flat = mask_flat.bool()

        # 变回 3D 空间结构
        mask_grid = mask_flat.view(B, 1, grid_d, grid_h, grid_w)

        # 上采样到原图尺寸 (最近邻插值)
        visible_mask = F.interpolate(mask_grid.float(), size=(D, H, W), mode='nearest').bool()
        return visible_mask

    def forward(self, x):
        if self.training:
            # --- 最少修改核心点：分别调用两次生成掩码，并在通道维度拼接 ---
            mask_ct = self.generate_mask(x)   # [B, 1, Z, Y, X]
            mask_pet = self.generate_mask(x)  # [B, 1, Z, Y, X]
            visible_mask = torch.cat([mask_ct, mask_pet], dim=1)  # 拼接为 [B, 2, Z, Y, X]

            # 2. 扩展 mask token 到当前 batch size 和图像大小
            mask_token_expanded = self.mask_token.expand_as(x)

            # 3. 替换遮挡区域：如果 visible 为 True 保留原图，否则替换为 mask_token
            masked_x = torch.where(visible_mask, x, mask_token_expanded)
        else:
            # 验证/推理时，不进行遮挡
            masked_x = x
            # 修改：验证时的 mask 形状直接对齐 x 的形状 [B, 2, Z, Y, X]
            visible_mask = torch.ones_like(x, dtype=torch.bool)

        # 4. 送入 U-Net 进行重建
        reconstruction = self.backbone(masked_x)

        return reconstruction, visible_mask