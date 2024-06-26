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

import torch
import logging
import copy
import os
import nibabel as nib
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Tuple, Union


from monai.inferers import Inferer, SlidingWindowInferer, SliceInferer
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.utils.transform import dump_data
from monailabel.utils.others.generic import name_to_device


# should be changed to the actual folders paths:
os.environ['nnUNet_preprocessed']= 'nnunet/Preprocessed'
os.environ['nnUNet_results']= 'nnunet/Results'
os.environ['nnUNet_raw']= 'nnunet/Raw'

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results


logger = logging.getLogger(__name__)

class CallBackTypes(str, Enum):
    PRE_TRANSFORMS = "PRE_TRANSFORMS"
    INFERER = "INFERER"
    INVERT_TRANSFORMS = "INVERT_TRANSFORMS"
    POST_TRANSFORMS = "POST_TRANSFORMS"
    WRITER = "WRITER"

class SegmentationNeuroblastoma(BasicInferTask):
    """
    This provides Inference Engine for pre-trained Neuroblastoma segmentation (NnUNet) model over AIN SHAMS Dataset.
    """
    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained NnUnet model for volumetric segmentation of the Neuroblastoma from CT image",
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
    #    self.temp_path='d:/Liver Segmentation Meena 2024/monailabel/apps/radiology2/temp'
       self.temp_path='/temp'


    
    


    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key 
        """
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            device=torch.device('cuda'),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        
        # Initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'Dataset200_blastoma/nnUNetTrainer__nnUNetPlans__3d_fullres'),
            use_folds=(0,),
            checkpoint_name='checkpoint_best.pth',
        )

        seg=predictor.predict_from_files([[data[self.input_key]]],
                                      self.temp_path,    
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        file_ending='.nii.gz' 
        basename = os.path.basename(data[self.input_key])[:-(len(file_ending) + 5)] + file_ending
        output_path=join(self.temp_path, basename)
        
        if os.path.exists(output_path):
            outputs = nib.load(output_path).get_fdata()
            outputs = torch.from_numpy(outputs)
            os.remove(output_path)
 
        data[self.output_label_key] = outputs

        return data
    
    

    def pre_transforms(self, data=None) -> Sequence[Callable]:
         return []

    
    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=[128, 128, 32],sw_batch_size = 2, overlap = 0.5
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
            return []
