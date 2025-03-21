"""
production_request_pipeline.py

This script implements the production pipeline for processing IT talent acquisition requests.
It performs the following steps:
  1. Connects to a MySQL database and checks whether the predictions for the current date have already been processed.
     If so, it deletes the existing records for that date.
  2. Loads raw request and salary data from generic preload tables.
  3. Calls the ETL module (etl_requests.py) to process the raw data.
  4. Splits the resulting dataset into training (closed requests) and prediction (open requests) sets.
  5. Removes outliers using Local Outlier Factor.
  6. Trains a RandomForestClassifier using GridSearchCV and cross-validation to obtain optimal parameters.
  7. Saves the final model and makes predictions on open requests.
  8. Generates an output DataFrame with predictions, probabilities, and additional metadata.
  9. Writes the output table to a MySQL database.

Usage (command line):
  python production_request_pipeline.py <host> <db> <user> <pwd> <directory>
  
  where:
    <host>      : Database host IP.
    <db>        : Database name.
    <user>      : Database user.
    <pwd>       : Database password.
    <directory> : Directory path to save intermediate files and the trained model.
"""

import os
import pandas as pd
import numpy as np
from datetime import date
import time

from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import gc

import mysql.connector as connection
import argparse
from sqlalchemy import create_engine
import joblib

import etl_requests  # Import the generalized ETL module

import warnings
warnings.filterwarnings("ignore")

import random
np.random.seed(1)
random.seed(1)

start_time = time.time()

# ---------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Configuration parameters for the IT Talent Acquisition model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("host", type=str, help="Database host IP")
parser.add_argument("db", type=str, help="Database name")
parser.add_argument("user", type=str, help="Database user")
parser.add_argument("pwd", type=str, help="Database password")
parser.add_argument("directory", type=str, help="Directory path for saving files")
args = parser.parse_args()

# ---------------------- PARAMETERS ----------------------
n_neighbors = 7       # Number of neighbors for outlier removal.
train_size = 0.85     # Training fraction.
verbose = 1           # Verbosity for grid search.

# ---------------------------------------------------------------------------
# Load prediction table to check if current date's predictions already exist.
try:
    db_conn = connection.connect(host=args.host, database=args.db,
                                 user=args.user, passwd=args.pwd, use_pure=True)
    query = "SELECT * FROM request_success_probability;"
    predictions_table = pd.read_sql(query, db_conn)
    db_conn.close()
except Exception as e:
    db_conn.close()
    print(str(e))

# Convert insertion_date column to string and check current date.
predictions_table['insertion_date'] = predictions_table['insertion_date'].astype(str)
current_date_str = date.today().strftime("%Y-%m-%d")

if current_date_str in predictions_table['insertion_date'].values:
    print("Current date already processed. Deleting existing records for date %s..." % current_date_str)
    try:
        db_conn = connection.connect(host=args.host, database=args.db, user=args.user, passwd=args.pwd, use_pure=True)
        cursor = db_conn.cursor()
        data = (current_date_str,)
        delete_stmt = "DELETE FROM request_success_probability WHERE insertion_date = %s"
        cursor.execute(delete_stmt, data)
        db_conn.commit()
        print("Deleted %d records for date %s." % (cursor.rowcount, current_date_str))
        db_conn.close()
    except Exception as e:
        db_conn.close()
        print(str(e))
else:
    print("No existing records for date %s. Proceeding without deletion." % current_date_str)

del query, db_conn, current_date_str

# ---------------------------------------------------------------------------
# Load raw request data from the preload table.
print("Loading raw request data...\n")
try:
    db_conn = connection.connect(host=args.host, database=args.db, user=args.user, passwd=args.pwd, use_pure=True)
    query = "SELECT * FROM preload_requests_data;"
    raw_requests = pd.read_sql(query, db_conn)
    db_conn.close()
except Exception as e:
    db_conn.close()
    print(str(e))
del query, db_conn

# Load raw salary data from the preload salary table.
print("Loading salary data...\n")
try:
    db_conn = connection.connect(host=args.host, database=args.db, user=args.user, passwd=args.pwd, use_pure=True)
    query = "SELECT * FROM preload_salary_profile;"
    raw_salary = pd.read_sql(query, db_conn)
    db_conn.close()
except Exception as e:
    db_conn.close()
    print(str(e))
del query, db_conn

# ---------------------------------------------------------------------------
# Run the ETL process on the raw data.
print("Running ETL process on requests data...\n")
# Call the ETL function (use_preload=1 indicates using the raw preload data)
initial_requests_df, encoded_df, final_df_for_model = etl_requests.etl_requests(
    use_preload=1, 
    preload_requests_df=raw_requests, 
    preload_salary_df=raw_salary, 
    directory_path=args.directory
)

# ---------------------------------------------------------------------------
# Split data into training (closed requests) and prediction (open requests) sets.
print("Splitting data into training and prediction sets...")
df_copy = final_df_for_model.copy(deep=True)
open_requests_df = df_copy[df_copy["target"] == -1]
open_requests_index = open_requests_df.index
# Remove target column from open requests set
open_requests_df.drop(columns=["target"], inplace=True)
closed_requests_df = df_copy[df_copy["target"] != -1]
target_series = closed_requests_df["target"]
print("Number of closed requests for training:", closed_requests_df.shape[0])
print("Number of open requests for prediction:", open_requests_df.shape[0])
del df_copy
gc.collect()
print("\n")

# ---------------------------------------------------------------------------
# Outlier removal using LocalOutlierFactor.
print("Removing outliers from training data...")
# Temporarily remove target column for outlier detection.
closed_requests_no_target = closed_requests_df.drop(columns=["target"])
initial_shape = closed_requests_no_target.shape
lof = LocalOutlierFactor(n_neighbors=n_neighbors)
closed_requests_no_target["outlier"] = lof.fit_predict(closed_requests_no_target)
closed_requests_no_target = closed_requests_no_target[closed_requests_no_target["outlier"] == 1]
closed_requests_no_target.drop(columns=["outlier"], inplace=True)
print("Initial shape:", initial_shape, " Final shape:", closed_requests_no_target.shape)
gc.collect()
# Merge the target back using the request code (assumed to be the index)
closed_requests_df = closed_requests_no_target.merge(target_series, left_index=True, right_index=True, how="left")
del closed_requests_no_target, lof
print("\n")

# ---------------------------------------------------------------------------
# Model training using RandomForestClassifier and GridSearchCV.
print("Training RandomForestClassifier model...")
X = closed_requests_df.drop("target", axis=1)
Y = closed_requests_df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=1)

# Grid search parameters
rf_params = {
    "bootstrap": [True, False],
    "criterion": ["entropy", "gini"],
    "max_depth": [2, 5, 10, 30, None],
    "max_features": ["sqrt", "log2"],
    "n_jobs": [-1],
    "random_state": [1],
    "min_samples_leaf": [2, 5, 10, 20],
    "min_samples_split": [2, 5, 10, 30, 50],
    "n_estimators": [128],
    "class_weight": ["balanced"]
}

grid_rf = RandomForestClassifier()
cv_rf = GridSearchCV(estimator=grid_rf, param_grid=rf_params, cv=5,
                     verbose=verbose, scoring="roc_auc")
cv_rf.fit(X_train, Y_train)

# Train with optimal parameters
rf_model_train = RandomForestClassifier(**cv_rf.best_params_)
print("Optimal parameters:", cv_rf.best_params_)

rf_model_train.fit(X_train, Y_train)
predictions_test = rf_model_train.predict(X_test)

# Confusion matrix
cm = confusion_matrix(Y_test, predictions_test, labels=rf_model_train.classes_)
print("Confusion Matrix:\n", cm)

# Cross-validation roc_auc score
kf = KFold(n_splits=5)
roc_auc_scores = cross_val_score(rf_model_train, X_train, Y_train, cv=kf, scoring="roc_auc")
cv_roc_auc = round(roc_auc_scores.mean(), 4)
print("Average cross-validation roc_auc:", cv_roc_auc)

# Retrain on all data
rf_model = RandomForestClassifier(**cv_rf.best_params_)
rf_model.fit(X, Y)

# Save the trained model to file
model_filename = "model_" + str(date.today()) + ".pkl"
model_filepath = os.path.join(args.directory, model_filename)
joblib.dump(rf_model, model_filepath)

# Make predictions on open requests
pred_open = rf_model.predict(open_requests_df)
prob_open = rf_model.predict_proba(open_requests_df)[:, 1]

del kf, rf_params, grid_rf, roc_auc_scores, X, Y, X_test, Y_test, Y_train
print("\n")

# Calculate feature importances and select top features.
print("Calculating feature importances...")
features = X_train.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": rf_model.feature_importances_})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top_threshold = importance_df["Importance"].quantile(0.75)
top_features = importance_df[importance_df["Importance"] >= top_threshold]["Feature"].tolist()

# ---------------------------------------------------------------------------
# Prepare output DataFrame with predictions and metadata.
print("Generating output DataFrame...")
# Merge output with a subset of initial request fields for context.
# (Select a few generic columns; adjust as needed.)
output_df = pd.DataFrame(open_requests_index)
# For example, merge with "priority", "region", "required_positions", "desired_salary"
output_df = output_df.merge(initial_requests_df[["priority", "region", "required_positions", "desired_salary"]],
                            left_index=True, right_index=True, how="left")
output_df["prediction"] = pred_open
output_df["probability"] = prob_open
output_df["probability_80"] = None
output_df["salary_80"] = None
output_df["roc_auc"] = round(cv_roc_auc, 2)
output_df["insertion_date"] = str(date.today())
output_df["relevant_features"] = 0  # This field will hold a dictionary of top features per request.
output_df["model_version"] = "1.0"    # Generic model version.
output_df["recalculated_probability"] = prob_open  # In this version, recalculated probability equals probability.
output_df["pending_update"] = False

# Populate the 'relevant_features' field with the top features for each open request.
# (Here we simply store a dictionary of the top features and their values.)
relevant_features_dict = {}
for idx in output_df.index:
    for feat in top_features:
        relevant_features_dict[feat] = output_df.loc[idx, feat] if feat in output_df.columns else None
    output_df.at[idx, "relevant_features"] = str(relevant_features_dict)
    
# ---------------------------------------------------------------------------
# Write output table to the database.
print("Writing output table to database...")
# Build connection string (use a generic database name, e.g., "generic_management_db")
conn_string = "mysql+mysqlconnector://" + args.user + ":" + args.pwd + "@" + args.host + ":3306/generic_management_db"
engine = create_engine(conn_string, echo=False)

try:
    output_df.to_sql(name="request_success_probability", con=engine, if_exists="append", index=False)
except Exception as e:
    print("Error writing output table:", e)
    
# End of process
end_time = time.time()
elapsed_minutes = round((end_time - start_time) / 60, 2)
print("Process completed in %.2f minutes." % elapsed_minutes)
