from fastapi import FastAPI, File, UploadFile
import os
from liver_imaging_analysis.models.liver_segmentation import segment_liver
import shutil
from typing import List

app = FastAPI()

@app.post("/predict/")
async def predict_liver_segmentation(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        filename = file.filename
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform inference using the loaded model
        prediction = segment_liver(filename, cp_path="C:\Studies\Graduation-Project\Codes\liver-imaging-analysis\liver_cp")
        predictions.append(prediction)

        # Remove the temporary file
        os.remove(filename)

    # Return the prediction results
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
