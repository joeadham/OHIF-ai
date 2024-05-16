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
os.environ['nnUNet_preprocessed']= 'D:/nnunet/Preprocessed'
os.environ['nnUNet_results']= 'D:/nnunet/Results'
os.environ['nnUNet_raw']= 'D:/nnunet/Raw'

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
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


    def __call__(
        self, request, callbacks: Union[Dict[CallBackTypes, Any], None] = None
    ) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        """
        It provides basic implementation to run the following in order
            - Run Pre Transforms
            - Run Inferer
            - Run Invert Transforms
            - Run Post Transforms
            - Run Writer to save the label mask and result params

        You can provide callbacks which can be useful while writing pipelines to consume intermediate outputs
        Callback function should consume data and return data (modified/updated) e.g. `def my_cb(data): return data`

        Returns: Label (File Path) and Result Params (JSON)
        """
        req = copy.deepcopy(self._config)
        req.update(request)

        # device
        device = name_to_device(req.get("device", "cuda"))
        req["device"] = device

        logger.setLevel(req.get("logging", "INFO").upper())
        if req.get("image") is not None and isinstance(req.get("image"), str):
            logger.info(f"Infer Request (final): {req}")
            data = copy.deepcopy(req)
            data.update({"image_path": req.get("image")})
        else:
            dump_data(req, logger.level)
            data = req

        # callbacks useful in case of pipeliens to consume intermediate output from each of the following stages
        # callback function should consume data and returns data (modified/updated)
        callbacks = callbacks if callbacks else {}
        callback_run_inferer = callbacks.get(CallBackTypes.INFERER)
        callback_writer = callbacks.get(CallBackTypes.WRITER)
       
        data = self.run_inferer(data, device=device)

        if callback_run_inferer:
            data = callback_run_inferer(data)

        if self.skip_writer:
            return dict(data)

        result_file_name, result_json = self.writer(data)  

        if callback_writer:
            data = callback_writer(data)
      

        result_json["label_names"] = self.labels
     

        # Add Centroids to the result json to consume in OHIF v3
        centroids = data.get("centroids", None)
        if centroids is not None:
            centroids_dict = dict()
            for c in centroids:
                all_items = list(c.items())
                centroids_dict[all_items[0][0]] = [str(i) for i in all_items[0][1]]  # making it json compatible
            result_json["centroids"] = centroids_dict
        else:
            result_json["centroids"] = dict()

        if result_file_name is not None and isinstance(result_file_name, str):
            logger.info(f"Result File: {result_file_name}")
        logger.info(f"Result Json Keys: {list(result_json.keys())}")
        return result_file_name, result_json
    


    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key 
        """
        predict_from_raw_data(list_of_lists_or_source_folder= [[data[self.input_key]]],
                          output_folder= self.temp_path,
                          model_training_output_dir=join(nnUNet_results, 'Dataset200_blastoma/nnUNetTrainer__nnUNetPlans__3d_fullres'),
                          use_folds=(0,),
                          tile_step_size = 0.5,
                          use_gaussian = True,
                          use_mirroring= True,
                          perform_everything_on_gpu=True,
                          verbose = True,
                          overwrite = True,
                          checkpoint_name= 'checkpoint_best.pth',
                          num_processes_preprocessing = 1,
                          num_processes_segmentation_export = 1,
                          folder_with_segs_from_prev_stage = None,
                          num_parts = 1,
                          part_id = 0,
                          device= torch.device('cuda'))
        
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
