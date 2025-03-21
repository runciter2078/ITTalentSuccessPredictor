"""
stream_requests.py

On-demand prediction for a request using AWS Lambda.
"""

def stream_request(request_json, csv_dir, etl_dir):
    import sys, os, glob, gc, joblib
    import pandas as pd
    import numpy as np
    from datetime import date
    from unidecode import unidecode  # if needed in ETL
    # Append the ETL directory so that the ETL module can be imported
    sys.path.append(etl_dir)
    import etl_requests  # The generalized ETL module

    # Utility functions to temporarily block and enable printing
    def block_print():
        sys.stdout = open(os.devnull, 'w')
        
    def enable_print():
        sys.stdout = sys.__stdout__

    # Load the most recent snapshots and model files from the csv_dir.
    # Latest requests snapshot
    last_requests_file = max(glob.glob(os.path.join(csv_dir, 'data_*.csv')))
    snapshot_requests = pd.read_csv(last_requests_file)
    # Latest salary snapshot
    last_salary_file = max(glob.glob(os.path.join(csv_dir, 'salary_*.csv')))
    snapshot_salary = pd.read_csv(last_salary_file)
    # Latest model columns file
    last_modelcols_file = max(glob.glob(os.path.join(csv_dir, 'modelcols_*.csv')))
    model_cols_df = pd.read_csv(last_modelcols_file)
    # Latest trained model
    last_model_file = max(glob.glob(os.path.join(csv_dir, 'model_*.pkl')))
    rf_model = joblib.load(last_model_file)
    
    # Clean up temporary variables and run garbage collection
    del last_requests_file, last_salary_file, last_modelcols_file, last_model_file
    gc.collect()
    
    # Convert input JSON to a DataFrame (assumes JSON object with keys corresponding to column names)
    stream_df = pd.DataFrame([request_json])
    
    # Extract the request code from the new record.
    # It is assumed that the field is named "request_code".
    request_code = stream_df["request_code"].values[0]
    
    # If the request code is already in the snapshot, update the record.
    if request_code in snapshot_requests["request_code"].values:
        print("Request already exists in snapshot.")
        old_data = snapshot_requests[snapshot_requests["request_code"] == request_code]
        # Combine the existing record with the new one to update missing values
        combined = pd.concat([old_data, stream_df])
        # Use the second row to fill missing values from the first row
        row1 = combined.iloc[0]
        row2 = combined.iloc[1]
        updated_row = row2.combine_first(row1).fillna(row2)
        # Append the updated record as a DataFrame
        updated_df = pd.DataFrame([updated_row], columns=old_data.columns)
        # Remove the old record from the snapshot and add the updated one
        snapshot_requests = snapshot_requests[snapshot_requests["request_code"] != request_code]
        snapshot_requests = pd.concat([snapshot_requests, updated_df])
        # Ensure that the text field is of type string
        snapshot_requests["minimum_requirements"] = snapshot_requests["minimum_requirements"].astype(str)
    else:
        print("New request. Not found in snapshot.")
        # For new requests, assign default values to missing fields
        stream_df["status"] = "Open"
        stream_df["change_date"] = date.today()
        stream_df["positions_filled"] = 0
        stream_df["tech_hours"] = 0
        stream_df["average_salary"] = 0
        stream_df["average_salary_12"] = 0

        # Compute pivot tables on the snapshot to assign default values if possible
        community_hours = snapshot_requests.pivot_table('community_tech_hours', index=['region'], aggfunc=np.median) 
        applicant_hours = snapshot_requests.pivot_table('applicant_tech_hours_12', index=['hr_manager'], aggfunc=np.median)
        applicant_tenure = snapshot_requests.pivot_table('applicant_tenure', index=['sales_agent'], aggfunc=np.median)
        tech_tenure = snapshot_requests.pivot_table('tech_tenure', index=['hr_manager'], aggfunc=np.median)
        
        try:
            default_comm_hours = community_hours.loc[stream_df['region'][0]][0]
            stream_df["community_tech_hours"] = default_comm_hours
        except:
            stream_df["community_tech_hours"] = 0
        
        try:
            default_app_hours = applicant_hours.loc[stream_df['hr_manager'][0]][0]
            stream_df["applicant_tech_hours_12"] = default_app_hours
        except:
            stream_df["applicant_tech_hours_12"] = 0
        
        try:
            default_app_tenure = applicant_tenure.loc[stream_df['sales_agent'][0]][0]
            stream_df["applicant_tenure"] = default_app_tenure
        except:
            stream_df["applicant_tenure"] = 0
        
        try:
            default_tech_tenure = tech_tenure.loc[stream_df['hr_manager'][0]][0]
            stream_df["tech_tenure"] = default_tech_tenure
        except:
            stream_df["tech_tenure"] = 0
        
        gc.collect()
        # Append the new request to the snapshot
        snapshot_requests = pd.concat([snapshot_requests, stream_df])
        # Ensure the text field is string type
        snapshot_requests["minimum_requirements"] = snapshot_requests["minimum_requirements"].astype(str)
    
    # Convert date columns in the snapshot to proper date type
    date_cols = ["request_creation_date", "change_date", "closure_request_date", "closure_date", "need_date"]
    snapshot_requests[date_cols] = snapshot_requests[date_cols].apply(pd.to_datetime)
    for col in date_cols:
        snapshot_requests[col] = snapshot_requests[col].dt.date
    
    # Preprocess the updated snapshot via the ETL module.
    # Block print to avoid extraneous output during ETL processing.
    block_print()
    # Here, use_preload=0 to indicate processing of CSV snapshot data.
    raw_output, encoded_output, model_input = etl_requests.etl_requests(
        use_preload=0, data=snapshot_requests, salary=snapshot_salary, directory_path=csv_dir
    )
    enable_print()
    
    # Select only the model columns from the processed output.
    model_columns = model_cols_df.iloc[:, 0].tolist()
    model_input = model_input[model_columns]
    
    # Extract the row corresponding to the streamed request.
    stream_features = model_input.loc[[request_code]].values
    
    # Perform prediction using the loaded model.
    prediction = int(rf_model.predict(stream_features))
    probability = float(rf_model.predict_proba(stream_features)[:, 1])
    
    output = {
        "pred": prediction,
        "proba": probability
    }
    
    return output

# AWS Lambda handler function
def lambda_handler(event, context):
    # Define the directories for CSV snapshots and the ETL module.
    csv_directory = "/mnt/efs0/RequestsData/"
    etl_directory = "/mnt/efs0/MachineLearning/"
    
    result = stream_request(event, csv_directory, etl_directory)
    return result
