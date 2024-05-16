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
import os
from typing import Any, Dict, Optional, Union
import torch

import lib.infers
import lib.trainers
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)

class TotalSegmentator(TaskConfig):
    
            
    def __init__(self):
        super().__init__()

        self.epistemic_enabled = None
        self.epistemic_samples = None

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "aorta",
        8: "inferior_vena_cava",
        9: "portal_vein_and_splenic_vein",
        10: "pancreas",
        11: "adrenal_gland_right",
        12: "adrenal_gland_left",
        13: "lung_upper_lobe_left",
        14: "lung_lower_lobe_left",
        15: "lung_upper_lobe_right",
        16: "lung_middle_lobe_right",
        17: "lung_lower_lobe_right",
        18: "vertebrae_L5",
        19: "vertebrae_L4",
        20: "vertebrae_L3",
        21: "vertebrae_L2",
        22: "vertebrae_L1",
        23: "vertebrae_T12",
        24: "vertebrae_T11",
        25: "vertebrae_T10",
        26: "vertebrae_T9",
        27: "vertebrae_T8",
        28: "vertebrae_T7",
        29: "vertebrae_T6",
        30: "vertebrae_T5",
        31: "vertebrae_T4",
        32: "vertebrae_T3",
        33: "vertebrae_T2",
        34: "vertebrae_T1",
        35: "vertebrae_C7",
        36: "vertebrae_C6",
        37: "vertebrae_C5",
        38: "vertebrae_C4",
        39: "vertebrae_C3",
        40: "vertebrae_C2",
        41: "vertebrae_C1",
        42: "esophagus",
        43: "trachea",
        44: "heart_myocardium",
        45: "heart_atrium_left",
        46: "heart_ventricle_left",
        47: "heart_atrium_right",
        48: "heart_ventricle_right",
        49: "pulmonary_artery",
        50: "brain",
        51: "iliac_artery_left",
        52: "iliac_artery_right",
        53: "iliac_vena_left",
        54: "iliac_vena_right",
        55: "small_bowel",
        56: "duodenum",
        57: "colon",
        58: "rib_left_1",
        59: "rib_left_2",
        60: "rib_left_3",
        61: "rib_left_4",
        62: "rib_left_5",
        63: "rib_left_6",
        64: "rib_left_7",
        65: "rib_left_8",
        66: "rib_left_9",
        67: "rib_left_10",
        68: "rib_left_11",
        69: "rib_left_12",
        70: "rib_right_1",
        71: "rib_right_2",
        72: "rib_right_3",
        73: "rib_right_4",
        74: "rib_right_5",
        75: "rib_right_6",
        76: "rib_right_7",
        77: "rib_right_8",
        78: "rib_right_9",
        79: "rib_right_10",
        80: "rib_right_11",
        81: "rib_right_12",
        82: "humerus_left",
        83: "humerus_right",
        84: "scapula_left",
        85: "scapula_right",
        86: "clavicula_left",
        87: "clavicula_right",
        88: "femur_left",
        89: "femur_right",
        90: "hip_left",
        91: "hip_right",
        92: "sacrum",
        93: "face",
        94: "gluteus_maximus_left",
        95: "gluteus_maximus_right",
        96: "gluteus_medius_left",
        97: "gluteus_medius_right",
        98: "gluteus_minimus_left",
        99: "gluteus_minimus_right",
        100: "autochthon_left",
        101: "autochthon_right",
        102: "iliopsoas_left",
        103: "iliopsoas_right",
        104: "urinary_bladder"
        }
        
        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]
        
        
        # Network
        self.network = UNet(
                    spatial_dims = 3,
                    in_channels = 1,
                    out_channels = 1,
                    channels = [64, 128, 256, 512],
                    strides = [2, 2, 2],
                    num_res_units = 8,
                    norm = "BATCH",
                    bias = False
                    )        
        
        
        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")
        
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.TotalSegmentator(
        path=self.path,
        network=self.network,
        labels=self.labels,
        preload=strtobool(self.conf.get("preload", "false")),
    )
        return task
    
    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.TotalSegmentator(
            model_dir=output_dir,
            network=self.network,
            # target_spacing=self.target_spacing,
            description="Train total Segmentator Model",
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
            disable_meta_tracking=False,
        )
        return task
    
    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies
    
    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }
        
        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=UNet(
                        spatial_dims = 3,
                        in_channels = 1,
                        out_channels = 1,
                        channels = [64, 128, 256, 512],
                        strides = [2, 2, 2],
                        num_res_units = 4,
                        norm = "Batch",
                        bias = False,
                        dropout = 0.5
                        ),
                    transforms=lib.infers.TotalSegmentator(None).pre_transforms(),
                    num_samples=self.epistemic_samples,
                )
        return methods
  
