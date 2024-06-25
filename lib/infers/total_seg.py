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
import numpy as np
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from batchgenerators.utilities.file_and_folder_operations import join

from monai.inferers import Inferer, SlidingWindowInferer, SliceInferer
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.utils.transform import dump_data
from monailabel.utils.others.generic import name_to_device

import sys
import os
import argparse
from pkg_resources import require
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
logger = logging.getLogger(__name__)

class CallBackTypes(str, Enum):
    PRE_TRANSFORMS = "PRE_TRANSFORMS"
    INFERER = "INFERER"
    INVERT_TRANSFORMS = "INVERT_TRANSFORMS"
    POST_TRANSFORMS = "POST_TRANSFORMS"
    WRITER = "WRITER"


class TotalSegmentator(BasicInferTask):
    """
    This provides Inference Engine for total segmentator.
    """
    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="total segmentator",
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
       
       self.temp_path='C:/Users/youse/OneDrive/Desktop/GP/GP Codes/OHIF/monailabel/try'

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

    def run_inferer(self, data, device="cuda"):
        import subprocess

        # Run the external TotalSegmentator process
        subprocess.run(["TotalSegmentator", "-i", data[self.input_key], "-o", self.temp_path])

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Initialize an empty numpy array for the final label map
        final_label_data = None

        # Loop through the files in the temporary path
        for filename in os.listdir(self.temp_path):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                filepath = os.path.join(self.temp_path, filename)
                label_img = nib.load(filepath)
                label_data = label_img.get_fdata()
                # Determine the label from the filename (assuming the filename matches the labels dictionary values)
                label_name = filename.replace(".nii", "").replace(".gz", "")
                label_value = None
                for key, value in self.labels.items():
                    if value == label_name:
                        label_value = key
                        break

                if label_value is not None:
                    # Initialize final_label_data if it's the first file
                    if final_label_data is None:
                        final_label_data = np.zeros_like(label_data, dtype=int)

                    # Assign the label_value to the corresponding regions in the final_label_data
                    final_label_data[label_data != 0] = label_value

                # Print information (optional)
                unique_labels = np.unique(label_data)
                num_labels = len(unique_labels)
                print(f"File: {filename} - Number of labels: {num_labels} - Labels: {unique_labels} - Label: {label_name} - Value: {label_value}")

                # Clean up: remove the file after processing
                os.remove(filepath)

        # Convert to PyTorch tensor
        final_label_data = torch.from_numpy(final_label_data)

        # Assign the processed label data back to the output label key
        data[self.output_label_key] = final_label_data

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

