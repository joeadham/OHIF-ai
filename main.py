from fastapi import FastAPI, UploadFile, File
import os
import shutil
from liver_imaging_analysis.models.liver_segmentation import segment_liver
from liver_imaging_analysis.engine.utils import Overlay
from typing import List
import nibabel as nib

app = FastAPI()

UPLOADS_DIR = "C:\\Studies\\Graduation-Project\\temp_codes\\img46\\"  # Update this with the path to your uploads directory

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

@app.post("/predict/")
async def predict_liver_segmentation(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        filename = file.filename
        file_path = os.path.join(UPLOADS_DIR, filename)  # Constructing absolute file path
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform inference using the loaded model
        prediction = segment_liver(file_path, cp_path="C:\\Studies\\Graduation-Project\\Codes\\liver-imaging-analysis\\liver_cp")
        # predictions.append(prediction)

        # Overlay the prediction on the original image
        animation = Overlay(file_path, prediction[0,0].cpu().numpy(), output_name= UPLOADS_DIR)

        original_header = nib.load(file_path).header
        original_affine = nib.load(file_path).affine
        prediction_nifti = nib.Nifti1Image(
                                prediction[0,0].cpu(),
                                affine = original_affine,
                                header = original_header
                                )
        nib.save(prediction_nifti, os.path.join(UPLOADS_DIR, "prediction.nii.gz"))

        # Remove the temporary file
        # os.remove(file_path)

    # Return the prediction results
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
