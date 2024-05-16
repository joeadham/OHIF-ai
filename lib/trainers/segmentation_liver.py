# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from monai.apps.deepedit.transforms import NormalizeLabelsInDatasetd
from monai.inferers import SlidingWindowInferer, SliceInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    SelectItemsd,
    Spacingd,
    ForegroundMaskD,
    ForegroundMaskd,
    ToTensord,
    NormalizeIntensityD,
    KeepLargestConnectedComponentD,
    FillHolesD,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)

class SegmentationLiver(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train Segmentation model for liver",
        **kwargs,
    ):
        self._network = network
        # self.target_spacing = target_spacing
        super().__init__(model_dir, description, **kwargs)


    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=0.01)

    # def scheduler(self, context: Context):
    #     return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5, verbose=False)

    def loss_function(self, context: Context):
        # return DiceLoss(include_background = 1, sigmoid = 1, batch = 1)
        return DiceLoss(batch=True, sigmoid=True, jaccard=1)


    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),

            EnsureChannelFirstd(keys=("image", "label")),
            EnsureTyped(keys=("image", "label"), device=context.device),
            Orientationd(keys=("image", "label"), axcodes="LAS"),

            ScaleIntensityRanged(keys="image", a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),

            ForegroundMaskd(
                        keys="label",
                        threshold = 0.5,
                        invert = True,
                        ),
            # RandCropByPosNegLabeld(
            #         keys=("image", "label"),
            #         image_key="image",
            #         label_key="label",
            #         spatial_size=(128,128,32),
            #         num_samples=12,
            #         pos=5,neg=8,
            #         allow_smaller=True,
            #         image_threshold=0.3),

            # ToTensord(
            #     keys=("image", "label"),
            #     # allow_mising_keys = True
            #     ),
        ]

    def train_post_transforms(self, context: Context):
        return [
            ToTensord(keys=("pred", "label")),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(
                keys=("pred", "label"),
                threshold = 0.5
            ),
            # FillHolesD(keys="pred"),
            # KeepLargestConnectedComponentD(keys="pred"),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),

            EnsureChannelFirstd(keys=("image", "label")),

            Orientationd(keys=("image", "label"), axcodes="LAS"),

            ScaleIntensityRanged(keys="image", a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),

            # ForegroundMaskD(
            #             keys="label",
            #             threshold = 0.5,
            #             invert = True,
            #             # allow_missing_keys = True
            #             ),

            # ToTensord(
            #     keys=("image", "label"),
            #     # allow_mising_keys = True
            #     ),
        ]

    # def val_inferer(self, context: Context):
    #     return SlidingWindowInferer(roi_size=[32, 32, 32],sw_batch_size = 2,overlap = 0.25)

    def val_inferer(self, context: Context):
        return SliceInferer(roi_size=(256,256), spatial_dim = 0)
