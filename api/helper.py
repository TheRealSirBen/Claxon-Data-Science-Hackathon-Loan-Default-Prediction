from pandas import DataFrame
from pandas import to_datetime


def check_date_columns(df: DataFrame, columns: list) -> list:
    """
    Check if the specified columns in the DataFrame can be converted to datetime.
    """
    not_convertible_columns: list = list()
    for column in columns:
        if column in df.columns:
            try:
                to_datetime(df[column])
            except (ValueError, TypeError):
                not_convertible_columns.append(column)
        else:
            not_convertible_columns.append(column)
    return not_convertible_columns


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load the saved model
model = joblib.load("path_to_your_model.joblib")

# Define the input data model
class LoanPredictionInput(BaseModel):
    # Define your input features here
    loan_id: str
    gender: str
    disbursemet_date: str
    currency: str
    country: str
    sex: str
    is_employed: bool
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    remaining_term: int
    salary: float
    marital_status: str

# Define the output data model
class LoanPredictionOutput(BaseModel):
    prediction: float
    prediction_probability: float

@app.post("/predict", response_model=LoanPredictionOutput)
async def predict_loan(input_data: LoanPredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Preprocess the input data if necessary
        # For example, you might need to encode categorical variables or scale numerical features
        # This step depends on how your model expects the input

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_probability = model.predict_proba(input_df)[0][1]  # Assuming binary classification

        return LoanPredictionOutput(
            prediction=float(prediction),
            prediction_probability=float(prediction_probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Loan Prediction API"}

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)