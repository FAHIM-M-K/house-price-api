import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI()

# Enable CORS for frontend access (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://your-frontend-domain.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("final_regression_pipeline.joblib")

# Define expected columns (order matters)
feature_columns = [
    "ID", "Property_ID", "Building_Class", "Zoning_Class", "Lot_Frontage", "Lot_Area", "Street_Type",
    "Alley_Access", "Lot_Shape", "Land_Contour", "Utility_Type", "Lot_Configuration", "Land_Slope",
    "Neighborhood", "Condition_1", "Condition_2", "Building_Type", "House_Style", "Overall_Quality",
    "Overall_Condition", "Year_Built", "Year_Remodeled", "Roof_Style", "Roof_Material",
    "Exterior_Material_1", "Exterior_Material_2", "Masonry_Veneer_Type", "Masonry_Veneer_Area",
    "Exterior_Quality", "Exterior_Condition", "Foundation_Type", "Basement_Quality",
    "Basement_Condition", "Basement_Exposure", "Basement_Finish_Type_1", "Basement_Finish_SF_1",
    "Basement_Finish_Type_2", "Basement_Finish_SF_2", "Basement_Unfinished_SF", "Total_Basement_SF",
    "Heating_Type", "Heating_Quality", "Central_Air", "Electrical_System", "First_Floor_SF",
    "Second_Floor_SF", "Low_Quality_Finished_SF", "Above_Ground_Living_Area", "Basement_Full_Bathrooms",
    "Basement_Half_Bathrooms", "Full_Bathrooms", "Half_Bathrooms", "Bedrooms_Above_Ground",
    "Kitchens_Above_Ground", "Kitchen_Quality", "Total_Rooms_Above_Ground", "Functionality",
    "Fireplaces", "Fireplace_Quality", "Garage_Type", "Garage_Year_Built", "Garage_Finish",
    "Garage_Capacity", "Garage_Area", "Garage_Quality", "Garage_Condition", "Paved_Driveway",
    "Wood_Deck_Area", "Open_Porch_Area", "Enclosed_Porch_Area", "Three_Season_Porch",
    "Screen_Porch_Area", "Pool_Area", "Pool_Quality", "Fence_Quality", "Miscellaneous_Feature",
    "Miscellaneous_Value", "Month_Sold", "Year_Sold", "Sale_Type", "Sale_Condition"
]

# Define input data model
class InputData(BaseModel):
    features: dict

@app.post("/predict")
def predict_price(data: InputData, x_api_key: str = Header(...)):
    # Check for valid API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    try:
        input_dict = data.features

        # Validate that all required features are present
        if set(feature_columns) != set(input_dict.keys()):
            missing = set(feature_columns) - set(input_dict.keys())
            return {"error": f"Missing fields: {missing}"}

        # Create ordered input list
        input_values = [input_dict[col] for col in feature_columns]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_values], columns=feature_columns)

        # Make prediction
        prediction = model.predict(input_df)[0]

        return {"predicted_price": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
