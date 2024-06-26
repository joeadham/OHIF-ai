# Using MONAILabel's pretrained spleen model as a Reference

## 1. Adding a New Configs File


### Step 1: Create a New File

Create a new file in the `configs` folder, for example, `segmentation_liver.py`, following the structure of `segmentation_spleen.py`.

### Step 2: Modify the  `init` Method

The `SegmentationLiver` class inherits from `TaskConfig` and defines a liver segmentation task. It includes methods for initializing the task, performing inference, training the model, selecting a strategy, and scoring the results. You will primarily modify the `init` method:

Set up Labels:
  
  Example:
  ```python
  self.labels = {
      "liver": 1,
  }
```

Set up your Model Checkpoints Paths:

  Example:
  ```python
  # self.path[0] is the path for our pre-trained model.
  # If using the default path, name your model as follows:  pretrained_{file name} in the directory ('Radiology/model')

  self.path = [
      os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
      os.path.join(self.model_dir, f"{name}.pt"),  # published
  ]
  ```

Set up your Network Architecture:

  Example:
  ```python
  self.network = UNet(
      spatial_dims=3,
      in_channels=1,
      out_channels=1,
      channels=[64, 128, 256, 512],
      strides=[2, 2, 2],
      num_res_units=8,
      norm="BATCH",
      bias=False
  )
  ```



---
---



## 2 Adding a New Inferer File

### Step 1: Create a New File
1. Create a new file in the `infers` folder, for example, `segmentation_liver.py`, following the structure of `segmentation_spleen.py`.
2. Add the import line in `__init__.py` to import the class that will be built in step 2:
   ```python
   from .segmentation_liver import SegmentationLiver
   ```

### Step 2: Modify the  `__init__` Method

Create a new class that inherits from `BasicInferTask`. This class will define the inference task for your new model. In the `__init__` method, You will need to modify the  `dimensions` and `description` according to your model:
- `dimension` (int, optional): Dimension of the input images.
- `description` (str, optional): Description of the model. 


#### Example:
```python

class SegmentationLiver(BasicInferTask):
   
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
```

### Step 3: Modify the `pre_transforms` Method

Define the preprocessing transforms to be applied to the input data before predicting.

#### Example:
```python
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
    ]
```

### Step 4: Modify the `inferer` Method

Define the inferer used in prediction.

#### Example for a 3D Model:
```python
def inferer(self, data=None) -> Inferer:
    return SlidingWindowInferer(
        roi_size=[128, 128, 32], sw_batch_size=2, overlap=0.5
    )
```

#### Example for a 2D Model:
```python
def inferer(self, data=None) -> Inferer:
    return SliceInferer(roi_size=(256,256), spatial_dim=0)
```

### Step 5: Modify the `post_transforms` Method

Define the postprocessing transforms to be applied to the output data after prediction.

#### Example:
```python
def post_transforms(self, data=None) -> Sequence[Callable]:
    return [
        EnsureTyped(keys="pred", device=data.get("device") if data else None),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
        FillHolesD(keys="pred"),
        KeepLargestConnectedComponentd(keys="pred"),
        Restored(keys="pred", ref_image="image"),
    ]
```

---
---



## 3 Adding a New Trainer File

### Step 1: Create a New File
1. Create a new file in the `trainers` folder, for example, `segmentation_liver.py`, following the structure of `segmentation_spleen.py`.
2. Add the import line in `__init__.py` to import the class that will be built in step 2:
   ```python
   from .segmentation_liver import SegmentationLiver
   ```
### Step 2: Modify the  `__init__` Method
Create a new class that inherits from `BasicTrainerTask`.

### Step 3: Modify the  `network` Method
Define the network used in training.

### Step 4: Modify the `optimizer` Method
Define the optimizer used in training.

### Step 5: Modify the `loss_function` Method

Define the scheduler used in training.
### Step 6: Modify the `train_pre_transforms` Method

Define the preprocessing transforms to be applied to the input data during training.
### Step 7: Modify the `train_post_transforms` Method
Define the postprocessing transforms to be applied to the output data during training.


### Step 8: Modify the `val_inferer` Method 
Define the inferer used in validation.


### Step 9: Modify `val_pre_transforms` & `val_post_transforms` Methods
(Optional if different from `train_pre_transforms` & `train_post_transforms`)
Define transformations for preprocessing and postprocessing input and output data during validation.




