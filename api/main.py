import time
from os import environ

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Response
from fastapi import UploadFile
from fastapi import status
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pandas import DataFrame
from pandas import concat
from pandas import read_csv

from _init_ import app_logger
from helper import check_date_columns
from helper import predict_loan
from models import ModelInput
from transformations import transform_data

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Operations related to users providing data to get predictions.",
    },
    {
        "name": "Data management",
        "description": "Operations related to data management.",
    },
]

app = FastAPI(openapi_tags=tags_metadata)
# print(environ.get('TRANSFORMATION_DIR'))
# Initialize FastAPI app

# Define expected columns
EXPECTED_COLUMNS = [
    'Unnamed: 0', 'loan_id', 'gender', 'disbursemet_date', 'currency', 'country',
    'sex', 'is_employed', 'job', 'location', 'loan_amount', 'number_of_defaults',
    'outstanding_balance', 'interest_rate', 'age', 'number_of_defaults.1',
    'remaining term', 'salary', 'marital_status', 'age.1'
]

EXPECTED_DATE_LIST = ['disbursemet_date']

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    return HTMLResponse(content=open("static/index.html").read())


# Endpoint 1: Single Prediction
@app.post("/api/predict/entry", status_code=status.HTTP_200_OK, tags=["Predictions"])
async def prediction_for_single_entry(model_input: ModelInput, resp: Response):
    # Convert input to dataframe
    data = model_input.dict()
    data_df = DataFrame([data], columns=EXPECTED_COLUMNS)

    # Column fitness check
    date_fitness_check = check_date_columns(data_df, EXPECTED_DATE_LIST)

    if date_fitness_check:
        resp.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'The {} entry value(s) can not be converted to date'.format(date_fitness_check)}

    # Transform data
    data_df = transform_data(data_df)

    # Perform prediction
    prediction = predict_loan(data_df)

    prediction_value = prediction['probability'].values[0]
    data = {'prediction': prediction_value}
    return {'message': 'Prediction successful', 'data': data}


# Endpoint 2: File Upload Prediction
@app.post("/api/predict/file", tags=["Predictions"])
async def prediction_for_data_file(resp: Response, file: UploadFile = File(...)):
    # File extension format
    if file.filename.endswith('.csv'):

        file_size_limit = float(environ.get("FILE_SIZE_LIMIT_IN_MB"))
        # Validate file size
        if file.size > file_size_limit * 1024 * 1024:
            # Handle the case where the file is too large
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit.")

        app_logger.info('File size valid')

        # Read file
        input_df = read_csv(file.file)
        data_df = input_df.copy()

        # Validate columns
        if list(data_df.columns) != EXPECTED_COLUMNS:
            template = DataFrame(columns=EXPECTED_COLUMNS).iloc[:5]
            template.to_csv("template.csv", index=False)
            return FileResponse("template.csv", filename="template.csv")

        app_logger.info('File columns valid')

        # Column fitness check
        date_fitness_check = check_date_columns(data_df, EXPECTED_DATE_LIST)

        if date_fitness_check:
            resp.status_code = status.HTTP_400_BAD_REQUEST
            return {'message': 'The {} entry value(s) can not be converted to date'.format(date_fitness_check)}

        # Transform data
        data_df = transform_data(data_df)

        # Perform prediction
        prediction = predict_loan(data_df)

        # Save the file with predictions
        output_df = concat([input_df, prediction], axis=1)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = "predictions/{}_predicted_{}.csv".format(file.filename.split('.')[0], timestamp)
        output_df.to_csv(output_filename, index=False)

        return FileResponse(output_filename, filename=output_filename)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are supported.")


# Endpoint 4: Template CSV
@app.get("/api/download/csv-template", tags=["Data management"])
async def download_csv_template():
    template = DataFrame(columns=EXPECTED_COLUMNS).iloc[:5]
    template.to_csv("template.csv", index=False)
    return FileResponse("template.csv", filename="template.csv")


# Endpoint 4: Template CSV
@app.get("/api/data-drift", tags=["Data management"])
async def get_data_drift_logs():
    template = DataFrame(columns=EXPECTED_COLUMNS).iloc[:5]
    template.to_csv("template.csv", index=False)
    return FileResponse("template.csv", filename="template.csv")
