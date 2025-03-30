import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load trained model
model = joblib.load("final_model.pkl")

# Get the exact feature names used in training
EXPECTED_FEATURES = model.feature_names_in_.tolist()

# Define API
app = FastAPI()

# Define Model Input Schema
class ModelInput(BaseModel):
    Location: int 
    Lighting: int
    Traffic_Density: float
    Crime_Rate: float
    Police_Presence: int
    Public_Transport_Availability: int
    CCTV_Presence: int
    Population_Density: float
    Hour: int
    Is_Late_Night: int
    Is_Evening: int

@app.post("/predict/")
def predict(data: ModelInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Ensure required engineered features are present
        input_df["Crime_Population"] = input_df["Crime_Rate"] * input_df["Population_Density"]
        input_df["Lighting_Police"] = input_df["Lighting"] * input_df["Police_Presence"]

        # Ensure exact feature order
        input_df = input_df[EXPECTED_FEATURES]

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()

        return {"prediction": int(prediction), "probability": probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
