from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn
import torch.nn.functional as F
import copy
class STUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-4

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes=label_manager.num_segmentation_heads
        kernel_sizes = [[3,3,3]] * 6
        strides=configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
        return STUNet(num_input_channels, num_classes, depth=[1]*6, dims= [32 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision)

class STUNetTrainer_small(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes=label_manager.num_segmentation_heads
        kernel_sizes = [[3,3,3]] * 6
        strides=configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
        return STUNet(num_input_channels, num_classes, depth=[1]*6, dims= [16 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision)

class STUNetTrainer_small_ft(STUNetTrainer_small):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000


class STUNetTrainer_base(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes=label_manager.num_segmentation_heads
        kernel_sizes = [[3,3,3]] * 6
        strides=configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
        return STUNet(num_input_channels, num_classes, depth=[1]*6, dims= [32 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision)

class STUNetTrainer_base_KI(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet_KI(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)
class STUNetTrainer_small_KI(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        model=STUNet_DualEncoder(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)
        model.load_pretrained_encoders(
            ct_ckpt_path='/projects/pet_ct_challenge/xinglong/pre_train_model/0.7_0.3/best_encoder_ct_epoch_97.pth',
            pet_ckpt_path='/projects/pet_ct_challenge/xinglong/pre_train_model/0.7_0.3/best_encoder_pet_epoch_97.pth',
            map_location='cuda'  # 或 'cpu'
        )
        return model

class STUNetTrainer_small_pretrain(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        model=STUNet_pretrain(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                        pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                        enable_deep_supervision=enable_deep_supervision)
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_mocov3/best_encoder_q.pth' mocov3
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/best_encoder.pth' MAE3D
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/final_encoder_swin.pth' swin
        model.load_pretrained_encoder(
            encoder_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_voco/latest_student_encoder.pth',
            map_location='cuda'  # 或 'cpu'
        )
        return model
#没有提示
class STUNetTrainer_small_pretrain_STUNet_DualEncoder(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        model=STUNet_DualEncoder(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                        pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                        enable_deep_supervision=enable_deep_supervision)
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_mocov3/best_encoder_q.pth' mocov3
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/best_encoder.pth' MAE3D
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/final_encoder_swin.pth' swin
        model.load_pretrained_encoders(
            ct_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000/best_encoder_ct_epoch_94.pth',
            pet_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000/best_encoder_pet_epoch_94.pth',
            map_location='cuda'  # 或 'cpu'
        )
        return model


#没有提示
class STUNetTrainer_small_pretrain_STUNet_DualEncoder_fuse(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        model=STUNet_DualEncoder_fuse(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                        pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                        enable_deep_supervision=enable_deep_supervision)
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_mocov3/best_encoder_q.pth' mocov3
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/best_encoder.pth' MAE3D
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/final_encoder_swin.pth' swin
        model.load_pretrained_weights(# 是因为训练头颈分割挑战赛注释 不然不注释
            pretrain_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000/best_checkpoint_epoch_94.pth',
            map_location='cuda'  # 或 'cpu'
        )
        return model

class STUNetTrainer_small_pretrain_STUNet_DualEncoder_1000_0_5(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        model=STUNet_DualEncoder(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                        pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                        enable_deep_supervision=enable_deep_supervision)
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_mocov3/best_encoder_q.pth' mocov3
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/best_encoder.pth' MAE3D
        #'/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_MAE3D/final_encoder_swin.pth' swin
        model.load_pretrained_encoders(
            ct_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000_0.9/best_encoder_ct_epoch_99.pth',
            pet_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000_0.9/best_encoder_pet_epoch_99.pth',
            map_location='cuda'  # 或 'cpu'
        )
        return model
class STUNetTrainer_base_ft(STUNetTrainer_base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000


class STUNetTrainer_large(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes=label_manager.num_segmentation_heads
        kernel_sizes = [[3,3,3]] * 6
        strides=configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
        return STUNet(num_input_channels, num_classes, depth=[2]*6, dims= [64 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision)
    
class STUNetTrainer_large_ft(STUNetTrainer_large):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000


class STUNetTrainer_huge(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes=label_manager.num_segmentation_heads
        kernel_sizes = [[3,3,3]] * 6
        strides=configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
        return STUNet(num_input_channels, num_classes, depth=[3]*6, dims= [96 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision)

class STUNetTrainer_huge_ft(STUNetTrainer_huge):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.final_nonlin = lambda x:x 
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        
        num_pool  = len(pool_op_kernel_sizes)
        
        assert num_pool == len(dims) - 1
        
        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True), 
                              *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool+1):
            stage = nn.Sequential(BasicResBlock(dims[d-1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1-u], dims[-2-u], pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2-u] * 2, dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u], use_1x1conv=True),
                *[BasicResBlock(dims[-2-u], dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u]) for _ in range(depth[-2-u]-1)])
            self.conv_blocks_localization.append(stage)
            
        # outputs    
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes, kernel_size=1))

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



class STUNet_pretrain(STUNet):
    def __init__(self, *args, **kwargs):
        """
        Inherits all init args from STUNet:
          - input_channels, num_classes, depth, dims, pool_op_kernel_sizes, conv_kernel_sizes, enable_deep_supervision
        """
        super().__init__(*args, **kwargs)
        # any additional setup can go here

    def load_pretrained_encoder(self,
                                encoder_ckpt_path: str,
                                map_location: str = 'cpu',
                                strict: bool = True):
        """
        Load pretrained STUNetEncoder weights into self.conv_blocks_context.

        Args:
            encoder_ckpt_path: path to a checkpoint saved from STUNetEncoder (state_dict).
            map_location: device mapping for torch.load.
            strict: whether to strictly enforce that the keys in state_dict match.
        """
        # 1) load the saved state dict
        enc_state = torch.load(encoder_ckpt_path, map_location=map_location)

        # 2) for each block in our conv_blocks_context, grab only the keys
        #    corresponding to that block (saved under 'blocks.{i}.<...>')
        for i, block in enumerate(self.conv_blocks_context):
            prefix = f'blocks.{i}.'
            # strip off the prefix to match the block's own state_dict keys
            subdict = {
                k[len(prefix):]: v
                for k, v in enc_state.items()
                if k.startswith(prefix)
            }
            # load into the block
            missing, unexpected = block.load_state_dict(subdict, strict=strict)
            if missing:
                print(f"[STUNet_pretrain] block {i}: missing keys: {missing}")
            if unexpected:
                print(f"[STUNet_pretrain] block {i}: unexpected keys: {unexpected}")

        print(f"✅ Loaded encoder weights from {encoder_ckpt_path}")

class STUNet_DualEncoder(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        # Ensure BasicResBlock and Upsample_Layer_nearest are correctly defined/imported

        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Assume CT is first, PET is second channel/group
        if input_channels == 2:
            self.split_sizes = [1, 1]
            print("Input channels = 2. Assuming channel 0 is CT, channel 1 is PET.")
        elif input_channels % 2 == 0:
            self.split_sizes = [input_channels // 2, input_channels // 2]
            print(f"Input channels = {input_channels} (even). Splitting equally: "
                  f"{self.split_sizes[0]} for CT, {self.split_sizes[1]} for PET.")
        else:
            # Fallback for odd channels, assign more to CT
            self.split_sizes = [input_channels // 2 + 1, input_channels // 2]
            print(f"Warning: input_channels ({input_channels}) is odd. "
                  f"Assigning first {self.split_sizes[0]} channels to CT encoder, "
                  f"and remaining {self.split_sizes[1]} to PET encoder.")


        self.final_nonlin = lambda x:x
        self.decoder = Decoder() # Using the dummy Decoder for attribute access
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False # Keep original setting

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])


        num_pool  = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1, f"num_pool ({num_pool}) != len(dims)-1 ({len(dims)-1})"
        assert len(depth) == len(dims), f"len(depth) ({len(depth)}) != len(dims) ({len(dims)})"
        assert len(conv_kernel_sizes) == len(dims), f"len(conv_kernel_sizes) ({len(conv_kernel_sizes)}) != len(dims) ({len(dims)})"


        # ==================== ENCODER MODIFICATION ====================
        # Create two separate encoder branches: CT and PET
        self.conv_blocks_context_ct = nn.ModuleList() # CT Encoder Branch
        self.conv_blocks_context_pet = nn.ModuleList() # PET Encoder Branch

        # --- Stage 0 ---
        # CT Encoder, Stage 0
        stage0_ct = nn.Sequential(
            BasicResBlock(self.split_sizes[0], dims[0], # Input channels for CT
                          self.conv_kernel_sizes[0], self.conv_pad_sizes[0],
                          use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0],
                            self.conv_kernel_sizes[0], self.conv_pad_sizes[0])
              for _ in range(depth[0]-1)]
        )
        self.conv_blocks_context_ct.append(stage0_ct)

        # PET Encoder, Stage 0
        stage0_pet = nn.Sequential(
            BasicResBlock(self.split_sizes[1], dims[0], # Input channels for PET
                          self.conv_kernel_sizes[0], self.conv_pad_sizes[0],
                          use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0],
                            self.conv_kernel_sizes[0], self.conv_pad_sizes[0])
              for _ in range(depth[0]-1)]
        )
        self.conv_blocks_context_pet.append(stage0_pet)


        # --- Stages 1 to num_pool ---
        for d in range(1, num_pool + 1):
            # CT Encoder Stage
            stage_d_ct = nn.Sequential(
                BasicResBlock(dims[d-1], dims[d],
                              self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                              stride=self.pool_op_kernel_sizes[d-1],
                              use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d],
                                self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                  for _ in range(depth[d]-1)]
            )
            self.conv_blocks_context_ct.append(stage_d_ct)

            # PET Encoder Stage (identical structure)
            # Use deepcopy to ensure weights are independent unless loading pretrained later
            stage_d_pet = copy.deepcopy(stage_d_ct)
            self.conv_blocks_context_pet.append(stage_d_pet)
        # ================= END OF ENCODER MODIFICATION ================


        # ================= DECODER DIMENSION ADJUSTMENTS =============
        # --- Upsample Layers ---
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            # The FIRST upsample layer (u=0) takes the *concatenated* bottleneck features
            if u == 0:
                # Bottleneck concatenation: dims[-1] from CT + dims[-1] from PET
                upsample_in_ch = dims[-1] * 2
            else:
                # Subsequent upsamplers take output from previous localization block
                upsample_in_ch = dims[-1-u] # Output dim of conv_blocks_localization[u-1]

            upsample_out_ch = dims[-2-u] # Target dimension for the current decoder level
            upsample_layer = Upsample_Layer_nearest(upsample_in_ch, upsample_out_ch, pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)


        # --- Localization Blocks ---
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            # Input features:
            # 1. From the upsample layer: dims[-2-u] channels
            # 2. From the skip connection: Concatenated features from CT+PET encoders
            #    CT skip: dims[-2-u] channels
            #    PET skip: dims[-2-u] channels
            # Total skip channels = dims[-2-u] * 2
            # Total input channels = dims[-2-u] (upsampled) + dims[-2-u] * 2 (skip) = dims[-2-u] * 3
            localization_input_ch = dims[-2-u] * 3
            localization_output_ch = dims[-2-u] # Output dim for this stage

            stage = nn.Sequential(
                # First block reduces channels from 3*dim to 1*dim
                BasicResBlock(localization_input_ch, localization_output_ch,
                              self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u],
                              use_1x1conv=True), # Use 1x1 conv in shortcut if needed for channel reduction
                # Subsequent blocks maintain the channel dimension
                *[BasicResBlock(localization_output_ch, localization_output_ch,
                                self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u])
                  for _ in range(depth[-2-u]-1)] # Use depth corresponding to the skip connection level
            )
            self.conv_blocks_localization.append(stage)
        # =============== END OF DECODER DIMENSION ADJUSTMENTS =============

        # --- Outputs ---
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)): # Should match num_pool
            output_in_ch = dims[-2-ds] # Input dim is the output of the corresponding localization block
            self.seg_outputs.append(nn.Conv3d(output_in_ch, num_classes, kernel_size=1))

        # Deep supervision upsampling - no change needed in definition
        self.upscale_logits_ops = []
        # Keeping the original lambda structure. Replace with actual upsampling if needed.
        cumulative_scale_factor = 1.0 # Track cumulative factor if needed for actual upsampling
        for stage in range(num_pool - 1):
            # Example: if pool sizes are (2,2,2), each step needs factor 2 upsampling
            # scale_factor = self.pool_op_kernel_sizes[stage + 1] # Check indexing relative to how pool sizes map to deep supervision levels
            # self.upscale_logits_ops.append(nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)) # Example
            self.upscale_logits_ops.append(lambda x: x) # Placeholder like original


    def forward(self, x):
        # ==================== FORWARD - ENCODER ====================
        # Split input into CT and PET paths
        x_ct = x[:, 0:1]
        x_pet = x[:, 1:2]
        skips = []
        # Process through encoder branches in parallel
        # Iterate through stages (0 to num_pool)
        for d in range(len(self.conv_blocks_context_ct)):
            x_ct = self.conv_blocks_context_ct[d](x_ct)
            x_pet = self.conv_blocks_context_pet[d](x_pet)
            # Store concatenated skips *before* the bottleneck stage
            if d < len(self.conv_blocks_context_ct) - 1:
                 skips.append(torch.cat((x_ct, x_pet), dim=1))

        # Bottleneck features are the concatenated outputs of the last encoder stage
        x = torch.cat((x_ct, x_pet), dim=1) # Shape: [B, dims[-1]*2, H_bottleneck, W_bottleneck, D_bottleneck]
        # ================== END FORWARD - ENCODER ==================


        # ==================== FORWARD - DECODER ====================
        seg_outputs_list = [] # Use a temporary list to collect outputs

        # Iterate through decoder stages (upsampling + localization)
        for u in range(len(self.conv_blocks_localization)): # This is num_pool
            x = self.upsample_layers[u](x)
            # Concatenate with the corresponding skip connection from the skips list
            skip = skips[-(u + 1)] # Access skips in reverse order
            x = torch.cat((x, skip), dim=1) # Shape: [B, dims[-2-u]*3, H, W, D]
            # Pass through localization block
            x = self.conv_blocks_localization[u](x) # Shape: [B, dims[-2-u], H, W, D]
            # Generate segmentation output for this level
            seg_output_level = self.seg_outputs[u](x)
            seg_outputs_list.append(self.final_nonlin(seg_output_level))
        # ================== END FORWARD - DECODER ==================


        # Handle deep supervision output formatting
        if self.decoder.deep_supervision:
            final_output = seg_outputs_list[-1]
            # Apply upsampling ops to the intermediate outputs
            deep_supervision_outputs = [op(out) for op, out in zip(list(self.upscale_logits_ops)[::-1], seg_outputs_list[:-1][::-1])]
            return tuple([final_output] + deep_supervision_outputs)
        else:
            # Return only the final, highest-resolution segmentation map
            return seg_outputs_list[-1]


    def load_pretrained_encoders(self, ct_ckpt_path, pet_ckpt_path, map_location='cpu', strict=True):
        """
        Loads pretrained weights into the CT and PET encoder branches.
        Assumes checkpoint keys match the structure within each encoder branch,
        using a prefix like 'blocks.i.' as in the example.
        Modify the expected prefix if your checkpoint keys are different.
        """
        # --- Load CT Encoder ---
        ct_state = torch.load(ct_ckpt_path, map_location=map_location)
        # Optional: Handle nested state dicts, e.g., ct_state = ct_state['state_dict']
        print(f"Loading weights for CT Encoder from {ct_ckpt_path}")

        # Determine the prefix used in the checkpoint keys (e.g., 'blocks.')
        # Using 'blocks.' as the default based on the example loading function provided
        key_prefix = 'blocks.' # Modify if your checkpoints use e.g. 'conv_blocks_context.'
        if not any(k.startswith(key_prefix) for k in ct_state.keys()):
             print(f"Warning: Keys in CT checkpoint do not seem to start with '{key_prefix}'. Trying without prefix.")
             # Attempt loading without assuming a prefix if 'blocks.' not found
             # This part needs careful adjustment based on actual key structure
             # For simplicity, we'll stick to requiring 'blocks.' or needing modification here.
             # key_prefix = '' # Alternative: try loading without prefix

        for i, block in enumerate(self.conv_blocks_context_ct): # Use the CT encoder attribute
            # Construct the expected prefix for this block's weights in the checkpoint
            current_prefix = f"{key_prefix}{i}."

            # Filter state_dict for the current block
            subdict = {
                k[len(current_prefix):]: v
                for k, v in ct_state.items()
                if k.startswith(current_prefix)
            }

            if not subdict:
                print(f"Warning: No weights found for CT Encoder, block {i} with prefix '{current_prefix}' in {ct_ckpt_path}")
                continue

            try:
                missing_keys, unexpected_keys = block.load_state_dict(subdict, strict=strict)
                if not strict:
                     if missing_keys: print(f"  Missing keys in CT block {i}: {missing_keys}")
                     if unexpected_keys: print(f"  Unexpected keys in CT block {i}: {unexpected_keys}")
                # print(f"  Loaded weights for CT block {i}")
            except Exception as e:
                print(f"Error loading weights for CT Encoder, block {i} with prefix '{current_prefix}': {e}")
                if strict: raise e


        # --- Load PET Encoder ---
        pet_state = torch.load(pet_ckpt_path, map_location=map_location)
        # Optional: Handle nested state dicts, e.g., pet_state = pet_state['state_dict']
        print(f"Loading weights for PET Encoder from {pet_ckpt_path}")

        # Assuming the same prefix structure for PET checkpoint
        if not any(k.startswith(key_prefix) for k in pet_state.keys()):
            print(f"Warning: Keys in PET checkpoint do not seem to start with '{key_prefix}'.")
            # Add handling if prefixes differ or are absent

        for i, block in enumerate(self.conv_blocks_context_pet): # Use the PET encoder attribute
            current_prefix = f"{key_prefix}{i}."

            subdict = {
                k[len(current_prefix):]: v
                for k, v in pet_state.items()
                if k.startswith(current_prefix)
            }
            if not subdict:
                print(f"Warning: No weights found for PET Encoder, block {i} with prefix '{current_prefix}' in {pet_ckpt_path}")
                continue

            try:
                missing_keys, unexpected_keys = block.load_state_dict(subdict, strict=strict)
                if not strict:
                     if missing_keys: print(f"  Missing keys in PET block {i}: {missing_keys}")
                     if unexpected_keys: print(f"  Unexpected keys in PET block {i}: {unexpected_keys}")
                # print(f"  Loaded weights for PET block {i}")
            except Exception as e:
                print(f"Error loading weights for PET Encoder, block {i} with prefix '{current_prefix}': {e}")
                if strict: raise e

        print(f"Finished loading pretrained CT and PET encoder weights.")

# Example Usage remains the same, but the model class is now STUNet_DualEncoder
# and the loading function correctly refers to self.conv_blocks_context_ct/pet.
# 增加了通道融合的步骤 之前是直接cat的
class STUNet_DualEncoder_fuse(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        # ... (input_channels, num_classes, final_nonlin, decoder, upscale_logits, pool_op_kernel_sizes, conv_kernel_sizes, conv_pad_sizes, num_pool, assertions - 保持不变) ...
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        if input_channels == 2:
            self.split_sizes = [1, 1]
            # print("Input channels = 2. Assuming channel 0 is CT, channel 1 is PET.")
        elif input_channels % 2 == 0:
            self.split_sizes = [input_channels // 2, input_channels // 2]
            # print(f"Input channels = {input_channels} (even). Splitting equally: "
                #   f"{self.split_sizes[0]} for CT, {self.split_sizes[1]} for PET.")
        else:
            self.split_sizes = [input_channels // 2 + 1, input_channels // 2]
            # print(f"Warning: input_channels ({input_channels}) is odd. "
                #   f"Assigning first {self.split_sizes[0]} channels to CT encoder, "
                #   f"and remaining {self.split_sizes[1]} to PET encoder.")


        self.final_nonlin = lambda x:x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])


        num_pool  = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1, f"num_pool ({num_pool}) != len(dims)-1 ({len(dims)-1})"
        assert len(depth) == len(dims), f"len(depth) ({len(depth)}) != len(dims) ({len(dims)})"
        assert len(conv_kernel_sizes) == len(dims), f"len(conv_kernel_sizes) ({len(conv_kernel_sizes)}) != len(dims) ({len(dims)})"


        # ==================== ENCODER DEFINITION (与预训练的STUNetEncoder类似) ====================
        # 我们需要能够独立获取每个编码器分支的各层特征
        # 预训练代码中 STUNetEncoder 的 __init__ 和 forward 返回特征列表的方式是正确的
        # 为了简单和与你的预训练代码对齐，我们直接使用 STUNetEncoder 类（假设它已定义）
        # 或者，你可以将预训练的 STUNetEncoder 类定义复制到这个文件中

        # 假设 STUNetEncoder 定义如下（与预训练脚本中的 STUNetEncoder 一致）：
        class STUNetEncoder(nn.Module):
            def __init__(self, input_channels_enc, depth_enc, dims_enc, pool_kernels_enc, conv_kernels_enc):
                super().__init__()
                assert len(dims_enc) == len(depth_enc)
                assert len(pool_kernels_enc) == len(dims_enc) - 1
                assert len(conv_kernels_enc) == len(dims_enc)
                self.conv_pads = [[k//2 for k in ks] for ks in conv_kernels_enc]
                self.blocks = nn.ModuleList()
                block0 = nn.Sequential(
                    BasicResBlock(input_channels_enc, dims_enc[0], conv_kernels_enc[0], self.conv_pads[0], use_1x1conv=True),
                    *[BasicResBlock(dims_enc[0], dims_enc[0], conv_kernels_enc[0], self.conv_pads[0]) for _ in range(depth_enc[0] - 1)]
                )
                self.blocks.append(block0)
                for i in range(1, len(dims_enc)):
                    blk = nn.Sequential(
                        BasicResBlock(dims_enc[i-1], dims_enc[i], conv_kernels_enc[i], self.conv_pads[i], stride=pool_kernels_enc[i-1], use_1x1conv=True),
                        *[BasicResBlock(dims_enc[i], dims_enc[i], conv_kernels_enc[i], self.conv_pads[i]) for _ in range(depth_enc[i] - 1)]
                    )
                    self.blocks.append(blk)
            def forward(self, x):
                feats = []
                for blk in self.blocks:
                    x = blk(x)
                    feats.append(x)
                return feats

        # 使用与预训练时 STUNetEncoder 相同的参数 (除了 input_channels)
        # 预训练时 STUNetEncoder 的 dims 是 [16, 32, 64, 128, 256, 256]
        # 这里的 dims 是 [32, 64, 128, 256, 512, 512]
        # 为了能加载预训练权重，这里的 dims 应该与预训练 Encoder 的 dims 对应阶段一致
        # 如果下游任务的 encoder dims 不同，加载权重会出问题。
        # 假设这里的 depth, pool_op_kernel_sizes, conv_kernel_sizes 与预训练的encoder部分匹配
        # **重要：这里的 dims 参数需要与预训练时 encoder_config['features_per_stage'] 对应**
        # 如果预训练 encoder_config 是 [16, 32, 64, 128, 256, 256]，那么这里也应该是
        # 为了加载，我们暂时假设 dims_enc 与预训练 encoder 一致
        dims_encoder_pretrain = [16, 32, 64, 128, 256, 256] # 从预训练配置中获取
        # depth_encoder_pretrain = [1, 1, 1, 1, 1, 1] # 从预训练配置中获取
        # pool_kernels_encoder_pretrain = [[2,2,2], [2,2,2], [2,2,2], [2,2,2], [1,2,2]] # 从预训练配置中获取
        # conv_kernels_encoder_pretrain = [[3,3,3]] * 6 # 从预训练配置中获取

        # 注意：这里的 depth, pool_op_kernel_sizes, conv_kernel_sizes 是传给 STUNet_DualEncoder 的
        # 我们需要确保 STUNetEncoder 内部使用的参数能正确构建与预训练编码器相同的结构
        # depth[0]到depth[len(dims_encoder_pretrain)-1] 应与预训练的depth匹配
        # pool_op_kernel_sizes[0]到pool_op_kernel_sizes[len(dims_encoder_pretrain)-2] 应与预训练的strides匹配
        # conv_kernel_sizes[0]到conv_kernel_sizes[len(dims_encoder_pretrain)-1] 应与预训练的kernel_sizes匹配

        self.encoder_ct = STUNetEncoder(
            input_channels_enc=self.split_sizes[0],
            depth_enc=depth, # 使用 STUNet_DualEncoder 的 depth
            dims_enc=dims_encoder_pretrain, # **使用预训练的 encoder dims**
            pool_kernels_enc=pool_op_kernel_sizes, # 使用 STUNet_DualEncoder 的 pool_op_kernel_sizes
            conv_kernels_enc=conv_kernel_sizes[:len(dims_encoder_pretrain)] # 使用 STUNet_DualEncoder 的 conv_kernel_sizes
        )
        self.encoder_pet = STUNetEncoder(
            input_channels_enc=self.split_sizes[1],
            depth_enc=depth,
            dims_enc=dims_encoder_pretrain, # **使用预训练的 encoder dims**
            pool_kernels_enc=pool_op_kernel_sizes,
            conv_kernels_enc=conv_kernel_sizes[:len(dims_encoder_pretrain)]
        )
        # ================= END OF ENCODER DEFINITION ================

        # ==================== SIMPLE FUSION LAYER (与预训练一致) ====================
        self.simple_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feat*2, feat, kernel_size=1),
                nn.GroupNorm(2, feat) # 或者 InstanceNorm3d(feat, affine=True)
            )
            # **使用预训练 encoder 的输出特征维度**
            for feat in dims_encoder_pretrain
        ])
        # ================= END OF SIMPLE FUSION LAYER ================


        # ================= DECODER DIMENSION ADJUSTMENTS (基于融合后的特征维度) =============
        # --- Upsample Layers ---
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            # 上采样层的输入维度是上一层解码器块的输出维度，或者是瓶颈层融合后的维度
            # 瓶颈层的融合特征维度是 dims_encoder_pretrain[-1]
            # 其他层的融合特征维度是 dims_encoder_pretrain[decoder_level]
            # 解码器的 dims 是传进来的 dims (e.g., [32, 64, ..., 512])
            # 融合后的特征维度是 dims_encoder_pretrain 的对应值
            if u == 0: # 从瓶颈层上采样
                upsample_in_ch = dims_encoder_pretrain[-1] # simple_fusion的输出
            else:
                # 输入来自上一个 conv_blocks_localization 的输出，其输出维度是下游任务的 dims
                # 这里的 dims 是 STUNet_DualEncoder 的 dims 参数
                upsample_in_ch = dims[-1-u]

            upsample_out_ch = dims[-2-u] # 目标是下游任务解码器对应层的维度
            upsample_layer = Upsample_Layer_nearest(upsample_in_ch, upsample_out_ch, pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)


        # --- Localization Blocks ---
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            # 输入特征:
            # 1. 来自上采样层: dims[-2-u] channels (下游任务的dims)
            # 2. 来自skip connection (simple_fusion的输出): dims_encoder_pretrain[-2-u] channels
            # 总输入 channels = dims[-2-u] + dims_encoder_pretrain[-2-u]
            localization_input_ch = dims[-2-u] + dims_encoder_pretrain[-2-u]
            localization_output_ch = dims[-2-u] # 输出是下游任务的dims

            stage = nn.Sequential(
                BasicResBlock(localization_input_ch, localization_output_ch,
                              self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u],
                              use_1x1conv=True),
                *[BasicResBlock(localization_output_ch, localization_output_ch,
                                self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u])
                  for _ in range(depth[-2-u]-1)] # 使用下游任务的 depth
            )
            self.conv_blocks_localization.append(stage)
        # =============== END OF DECODER DIMENSION ADJUSTMENTS =============

        # --- Outputs (与原来一致，基于下游任务的dims) ---
        self.seg_outputs = nn.ModuleList()
        for ds in range(num_pool): # 应该是 num_pool
            output_in_ch = dims[-2-ds]
            self.seg_outputs.append(nn.Conv3d(output_in_ch, num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for _ in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)


    def forward(self, x):
        # Split input
        x_ct_input = x[:, 0:self.split_sizes[0]] # 调整以适应 split_sizes
        x_pet_input = x[:, self.split_sizes[0]:self.split_sizes[0]+self.split_sizes[1]]

        # ==================== FORWARD - ENCODER & FUSION (与预训练一致) ====================
        ct_encoder_features = self.encoder_ct(x_ct_input) # list of features from CT encoder
        pet_encoder_features = self.encoder_pet(x_pet_input) # list of features from PET encoder

        fused_features = []
        for i, (ct_feat, pet_feat) in enumerate(zip(ct_encoder_features, pet_encoder_features)):
            concatenated = torch.cat([ct_feat, pet_feat], dim=1)
            fused = self.simple_fusion[i](concatenated)
            fused_features.append(fused)
        # fused_features 是一个列表，包含了每个编码器阶段融合后的特征
        # 其维度由 dims_encoder_pretrain 和 simple_fusion 决定
        # ================== END FORWARD - ENCODER & FUSION ==================

        # ==================== FORWARD - DECODER (使用融合后的特征) ====================
        # x_decode 初始为最高层 (瓶颈层) 的融合特征
        x_decode = fused_features[-1]
        # skips_decode 为其余各层融合后的特征，用作 skip connections (需要反转顺序)
        skips_decode = fused_features[:-1][::-1]

        seg_outputs_list = []

        for u in range(len(self.conv_blocks_localization)): # num_pool
            x_decode = self.upsample_layers[u](x_decode)
            skip_connection = skips_decode[u]
            x_decode = torch.cat((x_decode, skip_connection), dim=1)
            x_decode = self.conv_blocks_localization[u](x_decode)
            seg_output_level = self.seg_outputs[u](x_decode)
            seg_outputs_list.append(self.final_nonlin(seg_output_level))
        # ================== END FORWARD - DECODER ==================

        if self.decoder.deep_supervision:
            final_output = seg_outputs_list[-1]
            deep_supervision_outputs = [op(out) for op, out in zip(list(self.upscale_logits_ops)[::-1], seg_outputs_list[:-1][::-1])]
            return tuple([final_output] + deep_supervision_outputs)
        else:
            return seg_outputs_list[-1]


    def load_pretrained_weights(self, pretrain_ckpt_path, map_location='cpu', strict_encoder=True, strict_fusion=True):
        """
        Loads pretrained weights for encoders and simple_fusion layers.
        """
        print(f"Loading pretrained weights from {pretrain_ckpt_path}")
        pretrain_state = torch.load(pretrain_ckpt_path, map_location=map_location)
        model_state_dict = pretrain_state.get('model_state_dict', pretrain_state) # Handle nested dicts

        # --- Load CT Encoder ---
        ct_encoder_dict = {
            k[len('encoder_ct.'):]: v
            for k, v in model_state_dict.items()
            if k.startswith('encoder_ct.')
        }
        if ct_encoder_dict:
            try:
                missing, unexpected = self.encoder_ct.load_state_dict(ct_encoder_dict, strict=strict_encoder)
                if not strict_encoder:
                    if missing: print(f"  Missing keys in CT Encoder: {missing}")
                    if unexpected: print(f"  Unexpected keys in CT Encoder: {unexpected}")
                print("  Loaded CT Encoder weights.")
            except Exception as e:
                print(f"  Error loading CT Encoder weights: {e}")
        else:
            print("  Warning: No 'encoder_ct.' weights found in checkpoint.")


        # --- Load PET Encoder ---
        pet_encoder_dict = {
            k[len('encoder_pet.'):]: v
            for k, v in model_state_dict.items()
            if k.startswith('encoder_pet.')
        }
        if pet_encoder_dict:
            try:
                missing, unexpected = self.encoder_pet.load_state_dict(pet_encoder_dict, strict=strict_encoder)
                if not strict_encoder:
                    if missing: print(f"  Missing keys in PET Encoder: {missing}")
                    if unexpected: print(f"  Unexpected keys in PET Encoder: {unexpected}")
                print("  Loaded PET Encoder weights.")
            except Exception as e:
                print(f"  Error loading PET Encoder weights: {e}")
        else:
            print("  Warning: No 'encoder_pet.' weights found in checkpoint.")


        # --- Load Simple Fusion Layers ---
        simple_fusion_dict = {
            k[len('simple_fusion.'):]: v
            for k, v in model_state_dict.items()
            if k.startswith('simple_fusion.')
        }
        if simple_fusion_dict:
            try:
                missing, unexpected = self.simple_fusion.load_state_dict(simple_fusion_dict, strict=strict_fusion)
                if not strict_fusion:
                    if missing: print(f"  Missing keys in Simple Fusion: {missing}")
                    if unexpected: print(f"  Unexpected keys in Simple Fusion: {unexpected}")
                print("  Loaded Simple Fusion weights.")
            except Exception as e:
                print(f"  Error loading Simple Fusion weights: {e}")
        else:
            print("  Warning: No 'simple_fusion.' weights found in checkpoint.")

        print("Finished attempting to load pretrained weights.")


class STUNet_KI(nn.Module):
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

        self.leaky = nn.LeakyReLU(inplace=True)
        self.kencoder1 = nn.Conv3d(2, 32, 3, stride=1, padding=1)
        self.kdecoder1 = nn.Conv3d(32, 3, 3, stride=1, padding=1)# 肿瘤和淋巴结两个类别  所以输出是3
    def forward(self, x):
        skips = []
        seg_outputs = []
        k1 = self.leaky(F.interpolate(self.kencoder1(x), scale_factor=(1, 2, 2), mode='trilinear'))
        k2 = self.leaky(F.interpolate(self.kdecoder1(k1), scale_factor=(1, 0.5, 0.5), mode='trilinear'))
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
        seg_outputs[-1] = seg_outputs[-1] + k2
        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
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