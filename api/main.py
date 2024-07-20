# This is my main fastapi code

import sqlite3
import time
from logging import info
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

from helper import check_date_columns
from helper import predict
from models import ModelInput
from transformations import transform_data

tags_metadata = [
    {
        "name": "Data management",
        "description": "Operations related to data management.",
    },
    {
        "name": "Predictions",
        "description": "Operations related to users providing data to get predictions.",
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


@app.get("/", response_class=HTMLResponse)
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
    prediction = predict(data_df)

    prediction_value = prediction['prediction'].values[0]
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

        info('File size valid')

        # Read file
        input_df = read_csv(file.file)
        data_df = input_df.copy()

        # Validate columns
        if list(data_df.columns) != EXPECTED_COLUMNS:
            template = DataFrame(columns=EXPECTED_COLUMNS).iloc[:5]
            template.to_csv("template.csv", index=False)
            return FileResponse("template.csv", filename="template.csv")
        info('File columns valid')

        # Column fitness check
        date_fitness_check = check_date_columns(data_df, EXPECTED_DATE_LIST)

        if date_fitness_check:
            resp.status_code = status.HTTP_400_BAD_REQUEST
            return {'message': 'The {} entry value(s) can not be converted to date'.format(date_fitness_check)}

        # Transform data
        data_df = transform_data(data_df)

        # Perform prediction
        prediction = predict(data_df)

        # Save the file with predictions
        output_df = concat([input_df, prediction], axis=1)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = "{}_predicted_{}.csv".format(file.filename.split('.')[0], timestamp)
        output_df.to_csv(output_filename, index=False)

        return FileResponse(output_filename, filename=output_filename)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are supported.")


# Endpoint 3: Database Connection
@app.post("/api/predict/database-table", tags=["Predictions"])
async def prediction_for_data_in_db_table():
    conn = sqlite3.connect('test.db')
    query = "SELECT * FROM loans"  # Assuming a table named 'loans'
    df = pd.read_sql_query(query, conn)

    # Validate columns
    if list(df.columns) != EXPECTED_COLUMNS:
        template = DataFrame(columns=EXPECTED_COLUMNS).iloc[:5]
        template.to_csv("template.csv", index=False)
        return FileResponse("template.csv", filename="template.csv")

    # Run predictions
    df['prediction'] = df.apply(predict, axis=1)

    # Save the file with predictions
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"database_predictions_{timestamp}.csv"
    df.to_csv(output_filename, index=False)

    return FileResponse(output_filename, filename=output_filename)


# Endpoint 4: Template CSV
@app.get("/api/download/csv-template", tags=["Data management"])
async def download_csv_template():
    template = DataFrame(columns=EXPECTED_COLUMNS).iloc[:5]
    template.to_csv("template.csv", index=False)
    return FileResponse("template.csv", filename="template.csv")

# I need to create 4 webpages.
# 1. home or welcome to demo page which let a user select either a single prediction, a prediction from a file, and predictions done on sql db table
#
# 2. A single prediction page. It should have a form that requests for all the expected coulmns. It should be able to navigate back to home page
#
# 3. A file prediction page which allows a user to upload a file. It should be able to navigate back to home page
#
# 4. a db tabkle page. Which should request for daabase connection details. It should be able to navigate back to home page
#
# I need the fucntionality of call the apis from the pages. No javascript
