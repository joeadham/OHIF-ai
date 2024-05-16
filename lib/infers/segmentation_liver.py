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

from typing import Callable, Sequence
import torch

from monai.inferers import Inferer, SlidingWindowInferer, SliceInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstD,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImageD,
    OrientationD,
    ScaleIntensityD,
    ScaleIntensityRanged,
    SpacingD,
    FillHolesD,
    NormalizeIntensityD,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class SegmentationLiver(BasicInferTask):
    """
    This provides Inference Engine for pre-trained liver segmentation (UNet) model over MSD Dataset.
    """
    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained model for volumetric (2D) segmentation of the liver from CT image",
        **kwargs,
    ):
       super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=True,
            **kwargs,
        )
    #    self.target_spacing = target_spacing
    

   
    
    def pre_transforms(self, data=None) -> Sequence[Callable]:
         return [
                LoadImageD(keys="image"),
                EnsureChannelFirstD(keys="image"),
                OrientationD(keys="image", axcodes="LAS"),  # preferred by radiologists
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-135,
                    a_max=215,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                EnsureTyped(keys="image", device=data.get("device") if data else None),
                # NormalizeIntensityD(keys="image", channel_wise = True)
        ]


    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=[128, 128, 32],sw_batch_size = 2, overlap = 0.5
        )

    # def inferer(self, data=None) -> Inferer:
    #     return SliceInferer(roi_size=(256,256), spatial_dim = 0)


    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
            return [
                EnsureTyped(keys="pred", device=data.get("device") if data else None),
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold = 0.5),
                FillHolesD(keys="pred"),
                KeepLargestConnectedComponentd(keys="pred"),
                Restored(keys="pred", ref_image="image"),
            ]
