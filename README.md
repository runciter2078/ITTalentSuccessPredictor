# ITTalentSuccessPredictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ITTalentSuccessPredictor is a complete end-to-end machine learning solution designed to predict the success probability of IT talent acquisition requests. The project is intended for HR and recruitment teams looking to optimize their talent acquisition process by using advanced predictive analytics. The solution includes a robust ETL pipeline, model training and evaluation routines, and a real-time prediction service deployed on AWS Lambda.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Schema](#data-schema)
  - [Requests Table](#requests-table)
  - [Salary Table](#salary-table)
- [ETL Pipeline](#etl-pipeline)
- [Model Training & Prediction Pipeline](#model-training--prediction-pipeline)
- [Real-Time Prediction with AWS Lambda](#real-time-prediction-with-aws-lambda)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Overview

This repository provides a generalized solution for predicting the success probability of IT talent acquisition requests. It contains three main components:

1. **ETL Pipeline (`etl_requests.py`):**  
   Transforms raw request and salary data into a clean, feature-engineered dataset ready for model ingestion.

2. **Production Pipeline (`production_request_pipeline.py`):**  
   Processes the transformed data, trains a RandomForestClassifier model using hyperparameter tuning and outlier removal, and saves the trained model.

3. **Streaming Pipeline (`stream_requests.py`):**  
   Implements an on-demand prediction function (designed for AWS Lambda) that processes a new request in real time and returns a prediction along with its probability.

---

## Features

- **Data Preprocessing and Feature Engineering:**  
  The ETL script processes raw data to create numerous engineered features such as:
  - **Coverage Rate:** Ratio between positions filled and required positions.
  - **Target Variable:** Encoded as -1 for open requests, 1 for successful closed requests, and 0 for unsuccessful closed requests. Success criteria depend on the number of positions and coverage thresholds.
  - **Additional Derived Variables:** Indicators such as `is_FullRemote`, `is_NegotiatedRate`, quartiles/deciles of salary and tenure, and others.

- **Model Training:**  
  Uses RandomForestClassifier with GridSearchCV for hyperparameter tuning and K-Fold cross-validation. Outlier removal is performed using Local Outlier Factor.

- **Real-Time Predictions via AWS Lambda:**  
  The streaming pipeline script (`stream_requests.py`) supports on-demand predictions using AWS Lambda. It loads the latest snapshots, preprocessed data, and the trained model, and returns predictions for new incoming requests.

- **Modular and Extensible Design:**  
  All code components are fully generalized and documented so that any organization can adapt them to its own database schema and business logic.

---

## Data Schema

For the solution to work correctly, the input data should follow a generic schema. Below are the expected variables for each dataset:

### Requests Table

The raw requests data (e.g., from a preload database) is expected to include the following columns:

- **request_code:** Unique identifier for the request.
- **status:** Request status (e.g., "Open" or "Closed").
- **request_creation_date:** Date when the request was created.
- **closure_date:** Date when the request was closed.
- **positions_filled:** Number of positions that have been filled.
- **required_positions:** Number of positions required (if not provided, it is adjusted to the maximum of this and positions_filled).
- **coverage_rate:** Calculated as positions_filled / required_positions.
- **desired_salary:** Salary expected by the candidate (expressed in thousands or full amount; the ETL converts values accordingly).
- **sales_agent:** Identifier of the sales/recruitment agent.
- **hr_manager:** Identifier of the HR manager responsible.
- **minimum_requirements:** Free text field describing the minimum qualifications required.
- **service_type:** Type of service (e.g., "T&M" for Time and Materials).
- **region:** Geographical region or location.
- **job_role:** The role or position being requested.
- **need_date:** Date when the resource is needed.
- **request_type:** Type of request (e.g., "Real" to indicate actual requests).
- **work_area:** Work area description.
- **remote_work:** Indicator of remote work (e.g., "100% Remote", "Hybrid", "On-site").
- **rate_type:** Type of rate (e.g., "Negotiated Rate").
- **tech_hours:** Number of hours for technicians.
- **community_tech_hours:** Community-level technician hours.
- **applicant_tech_hours_12:** Technician hours for applicants in the last 12 months.
- **applicant_tenure:** Number of months the applicant has been with their company.
- **tech_tenure:** Number of months the technician has been with their company.
- **average_salary:** Average salary used for comparison.
- **average_salary_12:** Average salary for new employees (less than 1 year) used for comparison.
- **consolidation_probability:** Probability value provided for consolidating a request (from the applicant side).
- **coverage_probability:** Probability value provided for covering a request (from the technician side).
- **languages:** Languages spoken or required.
- **request_difficulty:** A metric indicating the difficulty of the request.
- **job_level:** Job level (numerical value indicating seniority or complexity).
- **priority:** Request priority.
- **Other engineered variables:**  
  The ETL process creates many additional fields, including:
  - `is_FullRemote`
  - `is_NegotiatedRate`
  - `SalaryQuartile`, `SalaryDecile`
  - `ProfileAvgSalary`, `ProfileAvgSalary12`
  - `ConsolProbQuartile`, `ConsolProbCategory`
  - `CoverageProbQuartile`, `CoverageProbCategory`
  - `MixedProbCategory`
  - `TechHoursQuartile`, `CommunityTechHoursQuartile/Decile`, etc.
  - `EarlyRequest`
  - `DaysDiffCategory`
  - `month_created`
  - `GroupedRegion`
  - `GroupedPosition`
  - and others as derived in the ETL process.

### Salary Table

The salary data should contain the following columns:

- **salary:** The salary amount.
- **insertion_date:** The date when the salary record was inserted.
- **remote_percentage:** Percentage indicating the level of remote work.
- **profile:** The job profile (to be processed and adjusted by the ETL).
- **tenure:** The employee’s tenure in months.
- **region:** The geographical region.
- **Other derived columns:**  
  During processing, new columns are created (e.g., `RemoteStatus`, `NewProfile`).

---

## ETL Pipeline

The ETL process is implemented in `etl_requests.py` and performs the following steps:

1. **Data Loading:**  
   Loads raw request and salary data either from a preload source or CSV snapshots.

2. **Data Cleaning:**  
   - Filters requests based on age.
   - Adjusts the required positions and calculates the coverage rate.
   - Defines request success criteria (for instance, a closed request with 1–2 positions must have 100% coverage; 3 positions require at least 66% coverage, etc.).

3. **Target Variable Creation:**  
   - For open requests, the target is set to -1.
   - For closed requests, the target is 1 if the success criteria are met, otherwise 0.

4. **Feature Engineering:**  
   - Processes text fields (e.g., "minimum_requirements") using tokenization with English stop words.
   - Creates multiple derived variables (such as indicators for remote work, negotiated rates, and several quartile/decile features for salary and tenure).

5. **Encoding and Scaling:**  
   - One-hot encodes categorical variables.
   - Scales features using MinMaxScaler.
   - Performs feature selection based on correlation with the target and inter-feature correlations.

The output of the ETL is used both for model training and for real-time predictions.

---

## Model Training & Prediction Pipeline

The production pipeline, contained in `production_request_pipeline.py`, performs the following tasks:

1. **Database Connection and Data Retrieval:**  
   - Connects to a MySQL database to load previous predictions and check if the current date has been processed.
   - Loads raw request and salary data from preload tables.

2. **ETL Execution:**  
   - Calls the ETL pipeline (`etl_requests.py`) to process and transform the raw data.

3. **Data Splitting:**  
   - Splits the transformed data into training (closed requests) and prediction (open requests) sets.

4. **Outlier Removal:**  
   - Uses LocalOutlierFactor to remove outliers from the training set.

5. **Model Training:**  
   - Uses RandomForestClassifier with GridSearchCV to perform hyperparameter tuning.
   - Evaluates the model using cross-validation (roc_auc score) and outputs a confusion matrix.
   - Trains the final model on the entire training dataset and saves it to file.

6. **Output Generation:**  
   - Generates an output table that includes predictions, probabilities, and additional metadata (e.g., model performance metrics and relevant features).

7. **Database Write:**  
   - Writes the output table to a MySQL database table for further consumption.

---

## Real-Time Prediction with AWS Lambda

The real-time prediction functionality is implemented in `stream_requests.py`. Key points include:

- **Input Processing:**  
  - The Lambda function receives a JSON payload representing a new request.
  - The payload is converted into a DataFrame and compared with the latest CSV snapshots.

- **Record Update:**  
  - If the request already exists in the snapshot, it is updated using the most complete record.
  - If it is new, default values are assigned to missing fields (using pivot tables from the snapshot data).

- **ETL Processing:**  
  - The updated snapshot is processed by calling the ETL pipeline (with `use_preload=0` to process CSV data).

- **Prediction:**  
  - The processed data is reduced to the set of features expected by the trained model.
  - The latest trained model is loaded, and a prediction along with its probability is returned.

- **AWS Lambda Integration:**  
  - The `lambda_handler` function is defined as the entry point for AWS Lambda.
  - When deployed, this function can be triggered via API Gateway or other event sources to perform real-time predictions.

---

## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/runciter2078/ITTalentSuccessPredictor.git
   cd ITTalentSuccessPredictor
   ```

2. **Install Dependencies:**

   It is recommended to use a virtual environment. Then install dependencies from the provided `requirements.txt` file:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Configure Your Database:**

   - Ensure that you have a MySQL database with the necessary tables.
   - For the requests data, set up a table with the columns described in the [Data Schema](#data-schema).
   - For the salary data, ensure your salary table has the expected fields.
   - Update connection parameters in the production pipeline script as needed.

4. **AWS Lambda Setup (for Streaming Predictions):**

   - Package the `stream_requests.py` (along with the `etl_requests.py` module) and all required dependencies.
   - Create an AWS Lambda function and configure it to use your package.
   - Set the handler to `stream_requests.lambda_handler`.
   - Configure the Lambda function’s environment variables (if required) to point to your CSV and ETL directories.
   - You can trigger the Lambda function via API Gateway for real-time predictions.

---

## Usage

### Training the Model

Run the production pipeline script to train the model and save it:

```bash
python production_request_pipeline.py <host> <db> <user> <pwd> <directory>
```

This script will:

- Connect to your MySQL database.
- Load the raw request and salary data.
- Execute the ETL process.
- Train the RandomForestClassifier model with hyperparameter tuning.
- Save the trained model and output predictions to the database.

### Real-Time Prediction

For real-time prediction, the AWS Lambda function (implemented in `stream_requests.py`) will:

- Receive a JSON payload representing a new request.
- Update or add the request to the latest snapshot.
- Re-run the ETL process on the updated snapshot.
- Load the latest trained model and make a prediction.

The Lambda handler function can be invoked as follows:

```python
# Example JSON payload for a new request:
new_request = {
    "request_code": "REQ12345",
    "status": "Open",
    "request_creation_date": "2025-03-01",
    "positions_filled": 0,
    "required_positions": 3,
    "desired_salary": 60,
    "sales_agent": "Agent007",
    "hr_manager": "HR123",
    "minimum_requirements": "Experience in Python, SQL, and machine learning.",
    "service_type": "T&M",
    "region": "North America",
    "job_role": "Data Scientist",
    "need_date": "2025-04-01",
    "request_type": "Real",
    "work_area": "Remote",
    "remote_work": "100% Remote",
    "rate_type": "Negotiated Rate",
    "tech_hours": 40,
    "community_tech_hours": 50,
    "applicant_tech_hours_12": 35,
    "applicant_tenure": 24,
    "tech_tenure": 36,
    "average_salary": 70000,
    "average_salary_12": 50000,
    "consolidation_probability": 80,
    "coverage_probability": 70,
    "languages": "English|Spanish",
    "request_difficulty": 3,
    "job_level": 7,
    "priority": "High"
}

# In AWS Lambda, the event parameter will contain this JSON.
```

When the Lambda function is triggered, it returns an output similar to:

```json
{
  "pred": 1,
  "proba": 0.87
}
```

Where `pred` is the predicted class and `proba` is the associated probability.

---

## Folder Structure

```
ITTalentSuccessPredictor/
├── etl_requests.py              # ETL pipeline module for data transformation.
├── production_request_pipeline.py   # Script for training and evaluating the predictive model.
├── stream_requests.py           # AWS Lambda function for real-time predictions.
├── requirements.txt             # List of Python dependencies.
├── LICENSE                      # MIT License.
└── README.md                    # This file.
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests for improvements or bug fixes. Please follow standard guidelines and include tests for any new functionality.

---

## Contact

For questions or feedback, please open an issue in this repository or contact the repository maintainer.

---

This README provides all the details needed for a generic user to understand, adapt, and deploy the ITTalentSuccessPredictor solution.
