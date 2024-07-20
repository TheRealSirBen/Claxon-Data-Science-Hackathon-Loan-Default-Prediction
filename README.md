# Loan Default Prediction Model

## Data Science Competition 2024 Submission

This project is a submission for the Data Science Competition 2024, focusing on predicting the probability of default (
PD) on loans using historical loan data.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Approach](#approach)
7. [Model Performance](#model-performance)
8. [API Endpoints](#api-endpoints)
9. [Data Drift Detection](#data-drift-detection)
10. [Model Analysis](#model-analysis)
11. [Future Improvements](#future-improvements)
12. [Business Implications](#business-implications)

## Project Overview

This project aims to develop a predictive model that estimates the probability of default on loans using historical loan
data. The model is crucial for risk management and strategic planning in financial institutions.

## Dataset Description

The dataset contains historical information about borrowers, including various features that may impact the probability
of default.
Here's an explanation of the variables in point form, using the column name and description:

- `Unnamed: 0`: Index of the dataset, usually not used in analysis.
- `loan_id`: Unique identifier for each loan.
- `gender`: Gender of the applicant (e.g., male, female).
- `disbursement_date`: Date when the loan was granted.
- `currency`: Currency in which the loan amount is stated.
- `country`: Country where the loan is issued.
- `sex`: May be a duplicate of gender; indicates applicant's sex.
- `is_employed`: Employment status of the applicant (e.g., employed, unemployed).
- `job`: Type of employment or job title of the applicant.
- `location`: Geographical location of the applicant.
- `loan_amount`: Total amount of the loan requested.
- `number_of_defaults`: Number of times the applicant has defaulted on previous loans.
- `outstanding_balance`: Remaining balance on the loan at the time of assessment.
- `interest_rate`: Interest rate applicable to the loan.
- `age`: Age of the applicant at the time of loan application.
- `number_of_defaults.1`: May be a duplicate of number of defaults; indicates defaults on loans.
- `remaining term`: Remaining term of the loan in months.
- `salary`: Monthly or annual salary of the applicant.
- `marital_status`: Marital status of the applicant (e.g., single, married).
- `age.1`: Could be a duplicate of age; indicates applicant's age.
- `Loan Status`: Current status of the loan (e.g., approved, rejected, defaulted).

## Project Structure

```

.
├── api/ # Directory for the FastAPI deployment project
   ├── logs/ # API app logs and data drift logs
   ├── prediction/ # Prediction results
   ├── .env/ # Environment file
   ├── _init.py # API app initialization file
   ├── main.py # FastAPI application
   ├── requirements.txt # Project dependencies
├── model/ # Directory for machine learning model development
   ├── cv_results/ # Cross-validation results
   ├── data/ # Directory for dataset storage
   ├── deployed_models/ # Saved models for deployment
   ├── eda/ # EDA reports and visualizations
   ├── model_fitting/ # Model fitting results
   ├── transformations/ # Data transformation scripts
      ├── train_model.ipynb # Jupyter notebook for model development
└── README.md # Project documentation

```

## Installation

1. Clone the repository:
   ```
    git clone https://github.com/TheRealSirBen/Claxon-Data-Science-Hackathon-Loan-Default-Prediction.git
   ```

## Model building approach

1. **Data Cleaning**: (Describe the data cleaning techniques used and justify decisions)

2. **Exploratory Data Analysis (EDA)**: (Summarize key findings from EDA, including visualizations and interesting observations)

3. **Feature Engineering**: (Explain the feature selection methods and justify choices)

4. **Data preprocessing**

5. **Hyperparameter Tuning and Model building**: (Discuss scaling and transformation techniques applied)

6. **Model evaluation**: (Discuss the model evaluation and model selection techniques)

### Model training and building

1. Open anaconda prompt

2. Open jupyter notebook. Run:
   ```
    jupyter notebook
   ```
3. Navigate to `Claxon-Data-Science-Hackathon-Loan-Default-Prediction` cloned dir
4. Navigate to `model` dir
5. Open `train_model.ipynb` file
6. Select the `Restart kernel and run all cell` button, then click to execute all commands
7. Wait for the program to request your input, thrice!
   1. For data expectations request 
   ![Alt text](data expectaction request.png)

   2. For duplicate drop columns list
   ![My Image](duplicate drop list.png)

   3. For cell value corrections
   ![My Image](value corrections list.png)

8. Wait until the whole program concludes

### Model Performance

(Provide details on model evaluation, including performance metrics on the validation set)

## API endpoints

1. Navigate to api directory
   ```
    cd api
   ```

2. Create a virtual environment:
   1. On windows
   ```
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. Navigate to api directory
   ```
    cd api
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

2. Access the API home page at `http://localhost:8000`
3. Access the API endpoints at page at `http://localhost:8000/docs`




## API Endpoints

The FastAPI application in `main.py` serves the trained model. Key endpoints include:

- `/predict`: Endpoint for model prediction
- `/train`: Endpoint for model training

(Provide more detailed documentation on how to use these endpoints)

## Data Drift Detection

(Explain the implemented mechanism for detecting data drift and its importance for model maintenance)

## Model Analysis

- Feature Importance: (Interpret model coefficients or feature importances)
- Error Analysis: (Investigate instances where the model performs poorly)
- Bias Analysis: (Analyze the model for biases in predictions)
- Interpretability: (Explain how the model is making predictions)

## Model Limitations

(Clearly communicate the limitations of the model, acknowledging situations where it might not perform well)

## Future Improvements

(Propose potential enhancements or future directions for model improvement)

## Business Implications

(Discuss potential business implications of your findings)

---

This project was developed as part of the Data Science Competition 2024. For any questions or clarifications, please contact competitions@claxonactuaries.com.