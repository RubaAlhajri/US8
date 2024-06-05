from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Load your models
model_dir = os.path.dirname(__file__)
kmeans_model = joblib.load(os.path.join(model_dir, 'Models/kmeans.pkl'))
dbscan_model = joblib.load(os.path.join(model_dir, 'Models/dbscan.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'Models/scaler.pkl'))

class PredictionRequest(BaseModel):
    Rating: float
    Rating_count: int
    Fiction: int
    Fantasy: int
    Young_Adult: int
    Classics: int
    Historical: int
    Romance: int
    Science_Fiction: int
    Adventure: int
    Nonfiction: int
    Contemporary: int
    Mystery: int
    Thriller: int
    Memoir: int
    Biography: int
    Horror: int
    Self_Help: int
    Graphic_Novels: int
    Short_Stories: int
    Science: int

@app.get("/")
async def read_root():
    return {"message": "Welcome to the book genre clustering API"}

@app.post("/predict/kmeans")
async def predict_kmeans(data: PredictionRequest):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    scaled_data = scaler.transform(df)
    prediction = kmeans_model.predict(scaled_data)
    return {"cluster": int(prediction[0])}

@app.post("/predict/dbscan")
async def predict_dbscan(data: PredictionRequest):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    scaled_data = scaler.transform(df)
    prediction = dbscan_model.fit_predict(scaled_data)
    cluster_label = int(prediction[0]) if prediction.size > 0 else -1  
    return {"cluster": cluster_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
