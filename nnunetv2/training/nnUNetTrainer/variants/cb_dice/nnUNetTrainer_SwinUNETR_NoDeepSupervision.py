import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.variants.cb_dice.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


def _build_swinunetr(in_channels: int, out_channels: int, spatial_dims: int) -> nn.Module:
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as e:
        raise ImportError("SwinUNETR requires MONAI and a compatible PyTorch installation.") from e

    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        use_v2=False,
    )


class nnUNetTrainer_SwinUNETR_NoDeepSupervision(nnUNetTrainerNoDeepSupervision):
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            label_manager = self.plans_manager.get_label_manager(self.dataset_json)

            patch_size = self.configuration_manager.patch_size
            self.num_input_channels, label_manager.num_segmentation_heads

            self.network = _build_swinunetr(
                self.num_input_channels, label_manager.num_segmentation_heads, len(patch_size)
            ).to(self.device)

            # compile network for free speedup
            if ("nnUNet_compile" in os.environ.keys()) and (os.environ["nnUNet_compile"].lower() in ("true", "1", "t")):
                self.print_to_log_file("Compiling network...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

        label_manager = plans_manager.get_label_manager(dataset_json)

        return _build_swinunetr(
            num_input_channels, label_manager.num_segmentation_heads, len(configuration_manager.patch_size)
        )
