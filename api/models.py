from pydantic import BaseModel


# Define the model input schema
class ModelInput(BaseModel):
    loan_id: int
    gender: str
    disbursemet_date: str
    currency: str
    country: str
    sex: str
    is_employed: str
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    number_of_defaults_1: int
    remaining_term: int
    salary: float
    marital_status: str
    age_1: int
    loan_status: str
