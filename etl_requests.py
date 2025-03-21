"""
etl_requests.py

This module defines the function etl_requests() which performs all the pre-transformations
required for a machine learning model that predicts the success probability of IT talent acquisition requests.
It accepts input DataFrames (either loaded from a preload database or from CSV files) containing
request and salary data, and returns three DataFrames:
    - initial_requests_df: the initial filtered DataFrame with the computed target variable.
    - encoded_df: the processed and one-hot encoded DataFrame.
    - final_df_for_model: the final scaled and feature-selected DataFrame ready for model ingestion.

Input parameters:
    - use_preload (int): If 1, data is read from the preload database; if 0, data is read from CSV files.
    - preload_requests_df (DataFrame): Raw requests DataFrame from the preload source.
    - preload_salary_df (DataFrame): Raw salary DataFrame from the preload source.
    - csv_requests_df (DataFrame): Requests DataFrame read from CSV.
    - csv_salary_df (DataFrame): Salary DataFrame read from CSV.
    - directory_path (str): Directory path for saving/reading CSV snapshots.
"""

def etl_requests(use_preload, preload_requests_df=None, preload_salary_df=None, 
                 csv_requests_df=None, csv_salary_df=None, directory_path=None):
    """
    use_preload = 1 indicates data is read from the preload database.
    use_preload = 0 indicates data is read from saved CSV files.
    """
    # ---------------------- IMPORT LIBRARIES ----------------------
    import os
    import pandas as pd
    import numpy as np
    import re
    import gc
    import joblib
    import glob
    from datetime import date, timedelta
    import time
    from unidecode import unidecode
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import MinMaxScaler
    import warnings
    warnings.filterwarnings("ignore")
    import random
    np.random.seed(1)
    random.seed(1)
    
    start_time = time.time()
    
    # ---------------------- PARAMETERS ----------------------
    target_days = 90         # Maximum days until request closure for a successful outcome.
    model_days = 540         # Maximum age (in days) of data relative to today.
    closing_days = 180       # Maximum days an open request is allowed before forced closure.
    min_word_freq = 0.05     # Minimum word frequency threshold for text tokenization.
    pearson_threshold = 0.85 # Threshold to drop features that are highly correlated.
    target_corr_threshold = 0.25 # Minimum correlation with target for feature selection.
    current_date = date.today()
    
    # ---------------------- DATA LOADING ----------------------
    if use_preload == 1:
        print("Loading data from preload source...\n")
        raw_requests_df = preload_requests_df.copy()
        del preload_requests_df
        salary_df = preload_salary_df.copy()
        del preload_salary_df
    elif use_preload == 0:
        print("Loading data from CSV files...\n")
        raw_requests_df = csv_requests_df.copy()
        salary_df = csv_salary_df.copy()
    else:
        print("Error: Invalid parameter for data source.")
        return None

    # It is assumed that all column names are already in English.
    # ---------------------- FILTER DATA BY DATE ----------------------
    min_date_threshold = date.today() - timedelta(days=model_days)
    raw_requests_df = raw_requests_df[raw_requests_df["request_creation_date"] >= min_date_threshold]
    
    if use_preload == 1:
        data_filename = "data_" + str(current_date) + ".csv"
        data_filepath = os.path.join(directory_path, data_filename)
        raw_requests_df.to_csv(data_filepath, index=False)
        
        # For salary data, keep only the most recent records (using insertion_date)
        salary_df.sort_values("insertion_date", ascending=False, inplace=True)
        latest_date = salary_df["insertion_date"].iloc[0]
        salary_df = salary_df[salary_df["insertion_date"] == latest_date]
        del latest_date
        salary_df = salary_df[salary_df["salary"] > 0]
        
        salary_filename = "salary_" + str(current_date) + ".csv"
        salary_filepath = os.path.join(directory_path, salary_filename)
        salary_df.to_csv(salary_filepath, index=False)
    
    gc.collect()
    
    print("------------ STARTING REQUEST PROCESSING -----------------")
    raw_requests_df = raw_requests_df.sample(frac=1)  # Shuffle data
    raw_requests_df = raw_requests_df[raw_requests_df["request_creation_date"] >= min_date_threshold]
    
    # ---------------------- INITIAL DATA CLEANING ----------------------
    print("------------ INITIAL DATA CLEANING -----------------")
    # Filter out records with unwanted statuses (assumes statuses are already in English)
    df_filtered = raw_requests_df[~raw_requests_df["status"].isin(["Discarded", "Hibernated", "Active"])]
    
    # Adjust required positions: use the maximum between required_positions and positions_filled
    df_filtered["required_positions"] = np.where(
        df_filtered["required_positions"] > df_filtered["positions_filled"],
        df_filtered["required_positions"],
        df_filtered["positions_filled"]
    )
    
    # Calculate coverage rate
    df_filtered["coverage_rate"] = df_filtered["positions_filled"] / df_filtered["required_positions"]
    
    # --- Define request success criteria based on number of positions ---
    # For 1 or 2 positions: 100% coverage is needed.
    # For 3 to 5 positions: at least 66% coverage is required.
    # For 6 or more positions: at least 25% coverage is needed.
    conditions_status = [
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 1) & (df_filtered["coverage_rate"] >= 1.0),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 2) & (df_filtered["coverage_rate"] >= 1.0),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 3) & (df_filtered["coverage_rate"] >= 0.66),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 4) & (df_filtered["coverage_rate"] >= 0.75),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 5) & (df_filtered["coverage_rate"] >= 0.70),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] >= 6) & (df_filtered["coverage_rate"] >= 0.25),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 1) & (df_filtered["coverage_rate"] < 1.0),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 2) & (df_filtered["coverage_rate"] < 1.0),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 3) & (df_filtered["coverage_rate"] < 0.80),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 4) & (df_filtered["coverage_rate"] < 0.75),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 5) & (df_filtered["coverage_rate"] < 0.70),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] == 6) & (df_filtered["coverage_rate"] < 0.25),
        (df_filtered["status"] == "Open") & (df_filtered["positions_filled"] < 1),
        (df_filtered["status"] == "Closed")
    ]
    # For the first six conditions, set status to "Closed"; for the next seven, "Open"; last one "Closed"
    status_values = ["Closed"] * 6 + ["Open"] * 7 + ["Closed"]
    df_filtered["status"] = np.select(conditions_status, status_values, default=-1)
    
    # Force closure for open requests older than closing_days
    days_diff = (current_date - df_filtered["request_creation_date"]).dt.days
    closure_conditions = [
        (df_filtered["status"] == "Open") & (days_diff > closing_days),
        (df_filtered["status"] == "Open") & (days_diff <= closing_days),
        (df_filtered["status"] == "Closed")
    ]
    closure_choices = ["Closed", "Open", "Closed"]
    df_filtered["status"] = np.where(
        df_filtered["status"].isin(["Open", "Closed"]),
        np.select(closure_conditions, closure_choices, default=-1),
        df_filtered["status"]
    )
    del days_diff
    
    # Drop closed requests with missing closure_date
    df_filtered.drop(df_filtered[(df_filtered["closure_date"].isna()) & (df_filtered["status"] == "Closed")].index, inplace=True)
    
    # Drop rows with a region value that should be excluded (e.g., Administrative Management)
    df_filtered = df_filtered.drop(df_filtered[df_filtered["region"] == "Administrative Management"].index)
    
    # Remove rows with anomalous desired salary values
    df_filtered.drop(df_filtered[df_filtered["desired_salary"] <= 5].index, inplace=True)
    
    # Set negative technician hours to zero
    df_filtered["tech_hours"] = np.where(df_filtered["tech_hours"] < 0, 0, df_filtered["tech_hours"])
    
    # Keep only requests with request_type equal to "Real"
    df_filtered = df_filtered.drop(df_filtered[df_filtered["request_type"] != "Real"].index)
    
    # Set the request code as index
    initial_requests_df = df_filtered.set_index("request_code")
    
    print("Dropping unnecessary columns...\n")
    cols_to_drop = ["id", "direct_client", "category", "name", "technology", "observations",
                    "associates", "change_date", "closure_request_date", "alias",
                    "response_time", "rate", "final_client_address", "tech_hours", 
                    "closed_salary", "current_salary", "min_salary", "max_salary", "average_salary"]
    initial_requests_df.drop(columns=cols_to_drop, inplace=True)
    
    # ---------------------- TARGET CREATION ----------------------
    print("------------ CALCULATING TARGET -----------------")
    open_reqs = initial_requests_df[initial_requests_df["status"] != "Closed"]
    closed_reqs = initial_requests_df[initial_requests_df["status"] != "Open"]
    closed_reqs["duration"] = (closed_reqs["closure_date"] - closed_reqs["request_creation_date"]).dt.days
    initial_requests_df = pd.concat([closed_reqs, open_reqs])
    del open_reqs, closed_reqs
    
    # Define the target variable:
    # For open requests: target = -1
    # For closed requests, success is defined as:
    #   - 1 or 2 positions with 100% coverage,
    #   - 3 positions with at least 66% coverage,
    #   - 4 positions with at least 70% coverage,
    #   - 5 positions with at least 60% coverage,
    #   - 6 or more positions with at least 25% coverage.
    target_conditions = [
        initial_requests_df["status"] == "Open",
        (initial_requests_df["status"] == "Closed") & (initial_requests_df["positions_filled"] == 1) &
            (initial_requests_df["duration"] <= target_days) & (initial_requests_df["coverage_rate"] >= 1.0),
        (initial_requests_df["status"] == "Closed") & (initial_requests_df["positions_filled"] == 2) &
            (initial_requests_df["duration"] <= target_days) & (initial_requests_df["coverage_rate"] >= 1.0),
        (initial_requests_df["status"] == "Closed") & (initial_requests_df["positions_filled"] == 3) &
            (initial_requests_df["duration"] <= target_days) & (initial_requests_df["coverage_rate"] >= 0.66),
        (initial_requests_df["status"] == "Closed") & (initial_requests_df["positions_filled"] == 4) &
            (initial_requests_df["duration"] <= target_days) & (initial_requests_df["coverage_rate"] >= 0.70),
        (initial_requests_df["status"] == "Closed") & (initial_requests_df["positions_filled"] == 5) &
            (initial_requests_df["duration"] <= target_days) & (initial_requests_df["coverage_rate"] >= 0.60),
        (initial_requests_df["status"] == "Closed") & (initial_requests_df["positions_filled"] >= 6) &
            (initial_requests_df["duration"] <= target_days) & (initial_requests_df["coverage_rate"] >= 0.25)
    ]
    target_values = [-1, 1, 1, 1, 1, 1, 0]
    initial_requests_df["target"] = np.select(target_conditions, target_values, default=0)
    print("Target distribution: \n", initial_requests_df["target"].value_counts(normalize=True))
    
    # Remove auxiliary columns used in target calculation
    initial_requests_df.drop(columns=["duration", "sub_status", "closure_date", 
                                      "positions_filled", "status", "coverage_rate"],
                              inplace=True)
    
    # Shuffle the final initial DataFrame
    initial_requests_df = initial_requests_df.sample(frac=1)
    print("\n")
    
    # ---------------------- VARIABLE SELECTION ----------------------
    selected_columns = ["final_client", "client_code", "sales_agent", "hr_manager", 
                        "minimum_requirements", "desired_salary", "required_positions", 
                        "priority", "service_type", "request_creation_date", "region", 
                        "job_role", "need_date", "request_type", "work_area", "remote_work", 
                        "rate_type", "tech_hours", "community_tech_hours", "applicant_tech_hours_12", 
                        "applicant_tenure", "tech_tenure", "average_salary_12", 
                        "consolidation_probability", "coverage_probability", "languages", 
                        "request_difficulty", "job_level", "target"]
    requests_df = initial_requests_df[selected_columns].copy()
    del selected_columns
    
    # ---------------------- DATA TRANSFORMATION ----------------------
    print("-------------- TRANSFORMING DATA -------------------")
    # NOTE: The original code contained a section to create an indicator column based on a specific request code ("ATSI").
    # Since that code is company-specific, it has been removed in this generalized version.
    
    # Process text field "minimum_requirements"
    print("Processing text field 'minimum_requirements'...")
    requests_df["minimum_requirements"] = requests_df["minimum_requirements"].str.lower().apply(unidecode)
    req_text = requests_df["minimum_requirements"]
    
    try:
        vectorizer = CountVectorizer(min_df=min_word_freq, stop_words="english")
        cv_matrix = vectorizer.fit_transform(req_text)
        req_features = pd.DataFrame(cv_matrix.toarray(), index=requests_df.index,
                                    columns=vectorizer.get_feature_names_out())
        # Optional: group similar terms if applicable
        try:
            req_features["api_group"] = req_features["api"] + req_features["apis"]
            del req_features["api"], req_features["apis"]
        except Exception:
            print("Warning: Could not group terms 'api/apis'.")
        try:
            req_features["english_group"] = req_features["ingles"] + req_features["english"]
            del req_features["ingles"], req_features["english"]
        except Exception:
            print("Warning: Could not group terms 'ingles/english'.")
        try:
            req_features["management_group"] = req_features["gestion"] + req_features["management"]
            del req_features["gestion"], req_features["management"]
        except Exception:
            print("Warning: Could not group terms 'gestion/management'.")
        print("Adding %u new keyword columns from 'minimum_requirements'..." % (req_features.shape[1]))
        requests_df = requests_df.merge(req_features, left_index=True, right_index=True, how="left")
        del req_features, req_text, vectorizer, cv_matrix
    except Exception as e:
        print("WARNING: Skipping processing of 'minimum_requirements' due to error...", e)
        del req_text, requests_df["minimum_requirements"]
        
    # Process client success indicator
    print("Creating column: FavorableClient...")
    requests_df = requests_df.dropna(subset=["client_code"])
    client_data = pd.DataFrame(columns=["Client", "SuccessRate"])
    i = 0
    for client in requests_df["client_code"].unique():
        sub_df = requests_df[requests_df["client_code"] == client]
        if sub_df.shape[0] >= 5:
            ratio = sub_df[sub_df["target"] == 1].shape[0] / sub_df.shape[0]
            if ratio > 0:
                client_data.loc[i] = (client, ratio * 100)
            i += 1
    client_summary = client_data.groupby("Client").sum().sort_values(by=["SuccessRate"], ascending=False)
    favorable_clients = client_summary[client_summary["SuccessRate"] >= client_summary["SuccessRate"].quantile(0.75)].index.to_list()
    requests_df["FavorableClient"] = requests_df.apply(lambda row: 1 if row["client_code"] in favorable_clients else 0, axis=1)
    requests_df.drop(columns=["client_code", "final_client"], inplace=True)
    del client_summary, client_data, favorable_clients
    
    # Process sales agent indicators
    print("Creating columns: CommercialFavorable and CommercialExpert...")
    sales_data = pd.DataFrame(columns=["SalesAgent", "SuccessRate"])
    i = 0
    for agent in requests_df["sales_agent"].unique():
        sub_df = requests_df[requests_df["sales_agent"] == agent]
        if sub_df.shape[0] >= 5:
            ratio = sub_df[sub_df["target"] == 1].shape[0] / sub_df.shape[0]
            if ratio > 0:
                sales_data.loc[i] = (agent, ratio * 100)
            i += 1
    sales_summary = sales_data.groupby("SalesAgent").sum().sort_values(by=["SuccessRate"], ascending=False)
    favorable_sales = sales_summary[sales_summary["SuccessRate"] >= sales_summary["SuccessRate"].quantile(0.75)].index.to_list()
    requests_df["CommercialFavorable"] = requests_df.apply(lambda row: 1 if row["sales_agent"] in favorable_sales else 0, axis=1)
    # For CommercialExpert, one could add additional criteria; here it is omitted for simplicity.
    requests_df.drop(columns=["sales_agent"], inplace=True)
    del sales_summary, sales_data, favorable_sales
    
    # Process HR manager indicators
    print("Creating columns: MultiResponsible and ResponsibleFavorable...")
    requests_df["NumResponsible"] = requests_df["hr_manager"].apply(lambda x: x.count("|") + 1)
    requests_df["MultiResponsible"] = requests_df["NumResponsible"].apply(lambda x: 1 if x > 1 else 0)
    hr_data = pd.DataFrame(columns=["HR", "SuccessRate"])
    i = 0
    for hr in requests_df["hr_manager"].unique():
        sub_df = requests_df[requests_df["hr_manager"] == hr]
        if sub_df.shape[0] >= 5:
            ratio = sub_df[sub_df["target"] == 1].shape[0] / sub_df.shape[0]
            if ratio > 0:
                hr_data.loc[i] = (hr, ratio * 100)
            i += 1
    hr_summary = hr_data.groupby("HR").sum().sort_values(by=["SuccessRate"], ascending=False)
    favorable_hr = hr_summary[hr_summary["SuccessRate"] >= hr_summary["SuccessRate"].quantile(0.75)].index.to_list()
    requests_df["ResponsibleFavorable"] = requests_df.apply(lambda row: 1 if row["hr_manager"] in favorable_hr else 0, axis=1)
    requests_df.drop(columns=["hr_manager"], inplace=True)
    del hr_summary, hr_data, favorable_hr
    
    # Process languages
    print("Encoding languages and creating column: NumLanguages...")
    requests_df["languages"] = requests_df["languages"].fillna("||||||||||||||||||||||||||||")
    requests_df["NumLanguages"] = requests_df["languages"].apply(lambda x: x.count("|") + 1)
    requests_df["NumLanguages"] = np.where(requests_df["NumLanguages"] >= 20, 0, requests_df["NumLanguages"])
    requests_df["languages"] = np.where(requests_df["languages"] == "||||||||||||||||||||||||||||", "No", requests_df["languages"])
    requests_df["language"] = requests_df["languages"].str.split("|")
    df_lang = requests_df.explode("language")
    df_lang["language"] = df_lang["language"].str.strip()
    lang_tab = df_lang.reset_index()
    lang_dummies = pd.crosstab(index=lang_tab["request_code"], columns=lang_tab["language"])
    orig_col_count = requests_df.shape[1]
    requests_df = requests_df.merge(lang_dummies, left_on="request_code", right_index=True, how="left")
    requests_df.drop(columns=["No"], inplace=True)
    print("Created %u new language encoding columns." % (requests_df.shape[1] - orig_col_count))
    del requests_df["languages"], lang_tab, lang_dummies, requests_df["language"], orig_col_count, df_lang
    
    # Process job level
    print("Encoding job level and creating column: JobLevelCategory...")
    requests_df = requests_df.dropna(subset=["job_level"])
    level_conditions = [
        requests_df["job_level"] <= 5,
        (requests_df["job_level"] >= 6) & (requests_df["job_level"] <= 9),
        requests_df["job_level"] >= 10
    ]
    level_labels = ["high_level", "senior", "junior"]
    requests_df["JobLevelCategory"] = np.select(level_conditions, level_labels)
    
    # Process desired salary
    print("Processing desired salary: creating RequestSalary, SalaryQuartile and SalaryDecile...")
    requests_df["RequestSalary"] = np.where(
        (requests_df["desired_salary"] >= 5) & (requests_df["desired_salary"] < 100),
        requests_df["desired_salary"] * 1000,
        requests_df["desired_salary"]
    )
    requests_df = requests_df.drop(requests_df[(requests_df["RequestSalary"] >= 100) & (requests_df["RequestSalary"] < 8000)].index)
    requests_df["SalaryQuartile"] = pd.qcut(requests_df["RequestSalary"], q=4, labels=False)
    requests_df["SalaryDecile"] = pd.qcut(requests_df["RequestSalary"], q=10, labels=False, duplicates="drop")
    requests_df.drop(columns=["desired_salary"], inplace=True)
    
    # Adjust salary table
    print("Adjusting salary table...")
    salary_df = salary_df.dropna().reset_index(drop=True)
    remote_conditions = [
        salary_df["remote_percentage"] == 100,
        (salary_df["remote_percentage"] > 0) & (salary_df["remote_percentage"] < 100)
    ]
    remote_values = ["Fully Remote", "Hybrid"]
    salary_df["RemoteStatus"] = np.select(remote_conditions, remote_values, default="On-site")
    salary_df["NewProfile"] = salary_df["profile"].str.replace("Area 3", "")
    salary_df["NewProfile"] = salary_df["NewProfile"].apply(lambda x: re.sub(r"\s+$", "", x[:-4]))
    salary_48 = salary_df[(salary_df["tenure"] <= 48) & (salary_df["tenure"] > 12)]
    salary_12 = salary_df[salary_df["tenure"] <= 12]
    
    # Process average salary for employees with 1-4 years
    print("Calculating average salary per profile: creating ProfileAvgSalary, SalaryMedQuartile and SalaryMedDecile...")
    salary_table_pp = salary_48.pivot_table("salary", index=["NewProfile", "job_level", "RemoteStatus", "region"], aggfunc=np.median)
    salary_table_np = salary_48.pivot_table("salary", index=["job_level", "RemoteStatus", "region"], aggfunc=np.median)
    salary_table_tt = salary_48.pivot_table("salary", index=["RemoteStatus", "region"], aggfunc=np.median)
    salary_table_com = salary_48.pivot_table("salary", index=["region"], aggfunc=np.median)
    requests_df["ProfileAvgSalary"] = requests_df.apply(
        lambda x: fill_median_salary(x["average_salary"], x["job_role"], x["job_level"], x["remote_work"], x["region"]),
        axis=1
    )
    requests_df["SalaryMedQuartile"] = pd.qcut(requests_df["ProfileAvgSalary"], q=4, labels=False, duplicates="drop")
    requests_df["SalaryMedDecile"] = pd.qcut(requests_df["ProfileAvgSalary"], q=10, labels=False, duplicates="drop")
    del requests_df["average_salary"], salary_table_pp, salary_table_tt, salary_table_com
    
    # Process average salary for new employees (less than 1 year)
    print("Calculating average salary for new employees: creating ProfileAvgSalary12, SalaryMed12Quartile and SalaryMed12Decile...")
    salary_table_pp = salary_12.pivot_table("salary", index=["NewProfile", "job_level", "RemoteStatus", "region"], aggfunc=np.median)
    salary_table_np = salary_12.pivot_table("salary", index=["job_level", "RemoteStatus", "region"], aggfunc=np.median)
    salary_table_tt = salary_12.pivot_table("salary", index=["RemoteStatus", "region"], aggfunc=np.median)
    salary_table_com = salary_12.pivot_table("salary", index=["region"], aggfunc=np.median)
    requests_df["ProfileAvgSalary12"] = requests_df.apply(
        lambda x: fill_median_salary(x["average_salary_12"], x["job_role"], x["job_level"], x["remote_work"], x["region"]),
        axis=1
    )
    requests_df["SalaryMed12Quartile"] = pd.qcut(requests_df["ProfileAvgSalary12"], q=4, labels=False, duplicates="drop")
    requests_df["SalaryMed12Decile"] = pd.qcut(requests_df["ProfileAvgSalary12"], q=10, labels=False, duplicates="drop")
    del requests_df["average_salary_12"], salary_table_pp, salary_table_tt, salary_table_com, salary_df, salary_12, salary_48
    
    # Compute salary ratios and growth indicator
    print("Creating salary ratio variables and growth flag...")
    requests_df["DesiredSalaryRatio"] = requests_df["RequestSalary"] / requests_df["ProfileAvgSalary"]
    requests_df["DesiredSalaryRatio12"] = requests_df["RequestSalary"] / requests_df["ProfileAvgSalary12"]
    requests_df["SalaryGrowthFlag"] = np.where(requests_df["ProfileAvgSalary12"] / requests_df["ProfileAvgSalary"] > 1.0, 1, 0)
    requests_df["SalaryRatioQuartile"] = pd.qcut(requests_df["DesiredSalaryRatio"], q=4, labels=False, duplicates="drop")
    requests_df["SalaryRatioDecile"] = pd.qcut(requests_df["DesiredSalaryRatio"], q=10, labels=False, duplicates="drop")
    requests_df["SalaryRatio12Quartile"] = pd.qcut(requests_df["DesiredSalaryRatio12"], q=4, labels=False, duplicates="drop")
    requests_df["SalaryRatio12Decile"] = pd.qcut(requests_df["DesiredSalaryRatio12"], q=10, labels=False, duplicates="drop")
    
    # Process consolidation probability (applicant side)
    print("Processing consolidation probability: creating ConsolProbQuartile and ConsolProbCategory...")
    requests_df["consolidation_probability"] = requests_df["consolidation_probability"].fillna(-100)
    prob_clean = requests_df[requests_df["consolidation_probability"] >= 0]
    prob_table_pc = prob_clean.pivot_table("consolidation_probability", index=["job_role", "region", "work_area", "remote_work"], aggfunc=np.median)
    prob_table_com = prob_clean.pivot_table("consolidation_probability", index=["region", "work_area", "remote_work"], aggfunc=np.median)
    prob_table_at = prob_clean.pivot_table("consolidation_probability", index=["work_area", "remote_work"], aggfunc=np.median)
    prob_table_zon = prob_clean.pivot_table("consolidation_probability", index=["work_area"], aggfunc=np.median)
    requests_df["ApplicantProbability"] = requests_df.apply(
        lambda x: fill_probability(x["consolidation_probability"], x["work_area"], x["remote_work"], x["region"], x["job_role"]),
        axis=1
    )
    requests_df["ConsolProbQuartile"] = pd.qcut(requests_df["ApplicantProbability"], q=4, labels=False)
    consolidation_conditions = [
        requests_df["ApplicantProbability"] >= 75,
        (requests_df["ApplicantProbability"] >= 40) & (requests_df["ApplicantProbability"] < 75)
    ]
    consolidation_values = [1, 0]
    requests_df["ConsolProbCategory"] = np.select(consolidation_conditions, consolidation_values, default=-1)
    del prob_table_pc, prob_table_com, prob_table_at, prob_table_zon, prob_clean, requests_df["consolidation_probability"]
    
    # Process coverage probability (technician side)
    print("Processing coverage probability: creating CoverageProbQuartile and CoverageProbCategory...")
    requests_df["coverage_probability"] = requests_df["coverage_probability"].fillna(-100)
    prob_clean = requests_df[requests_df["coverage_probability"] >= 0]
    prob_table_pc = prob_clean.pivot_table("coverage_probability", index=["job_role", "region", "work_area", "remote_work"], aggfunc=np.median)
    prob_table_com = prob_clean.pivot_table("coverage_probability", index=["region", "work_area", "remote_work"], aggfunc=np.median)
    prob_table_at = prob_clean.pivot_table("coverage_probability", index=["work_area", "remote_work"], aggfunc=np.median)
    prob_table_zon = prob_clean.pivot_table("coverage_probability", index=["remote_work"], aggfunc=np.median)
    requests_df["TechProbability"] = requests_df.apply(
        lambda x: fill_probability(x["coverage_probability"], x["work_area"], x["remote_work"], x["region"], x["job_role"]),
        axis=1
    )
    requests_df["CoverageProbQuartile"] = pd.qcut(requests_df["TechProbability"], q=4, labels=False)
    coverage_conditions = [
        requests_df["TechProbability"] >= 75,
        (requests_df["TechProbability"] >= 40) & (requests_df["TechProbability"] < 75)
    ]
    coverage_values = [1, 0]
    requests_df["CoverageProbCategory"] = np.select(coverage_conditions, coverage_values, default=-1)
    del prob_table_pc, prob_table_com, prob_table_at, prob_table_zon, prob_clean, requests_df["coverage_probability"]
    
    # Create combined probability category
    print("Creating combined probability category: MixedProbCategory...")
    mixed_conditions = [
        (requests_df["TechProbability"] >= 75) & (requests_df["ApplicantProbability"] >= 75),
        (requests_df["TechProbability"] >= 75) | (requests_df["ApplicantProbability"] >= 75),
        (requests_df["TechProbability"] <= 30) & (requests_df["ApplicantProbability"] <= 30)
    ]
    mixed_values = [2, 1, -1]
    requests_df["MixedProbCategory"] = np.select(mixed_conditions, mixed_values, default=0)
    
    # Process required positions indicator
    print("Creating multiple resources indicator...")
    resource_conditions = [
        requests_df["required_positions"] > 1,
        requests_df["required_positions"] == 1
    ]
    resource_values = [1, 0]
    requests_df["MultipleResources"] = np.select(resource_conditions, resource_values, default=-1)
    
    # Process service type indicator
    print("Creating negotiated rate indicator...")
    requests_df["is_NegotiatedRate"] = np.where(requests_df["rate_type"] == "Negotiated Rate", 1, 0)
    requests_df.drop(columns=["rate_type"], inplace=True)
    
    # Process creation month extraction
    print("Extracting creation month...")
    requests_df["month_created"] = pd.DatetimeIndex(requests_df["request_creation_date"]).month
    month_conditions = [
        requests_df["month_created"] == 1,
        requests_df["month_created"] == 2,
        requests_df["month_created"] == 3,
        requests_df["month_created"] == 4,
        requests_df["month_created"] == 5,
        requests_df["month_created"] == 6,
        requests_df["month_created"] == 7,
        requests_df["month_created"] == 8,
        requests_df["month_created"] == 9,
        requests_df["month_created"] == 10,
        requests_df["month_created"] == 11,
        requests_df["month_created"] == 12
    ]
    month_labels = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    requests_df["month_created"] = np.select(month_conditions, month_labels, default=None)
    
    # Group small regions into "Other"
    print("Grouping small regions...")
    region_map = {}
    norm_counts = requests_df["region"].value_counts(normalize=True)
    for reg in norm_counts.index:
        region_map[reg] = (norm_counts[reg] * 100) >= 2
    requests_df["GroupedRegion"] = requests_df.apply(
        lambda row: row["region"] if region_map.get(row["region"]) else "Other",
        axis=1
    )
    del requests_df["region"], region_map
    
    # Drop request_type column as it is not needed
    requests_df.drop(columns=["request_type"], inplace=True)
    
    # Group small job roles into "Other"
    print("Grouping small job roles...")
    role_map = {}
    norm_roles = requests_df["job_role"].value_counts(normalize=True)
    for role in norm_roles.index:
        role_map[role] = (norm_roles[role] * 100) >= 1
    requests_df["GroupedPosition"] = requests_df.apply(
        lambda row: row["job_role"] if role_map.get(row["job_role"]) else "Other",
        axis=1
    )
    del requests_df["job_role"], role_map
    
    # Process request creation and need dates
    print("Creating early request flag and days difference categories...")
    requests_df["EarlyRequest"] = np.where(
        pd.to_datetime(requests_df["request_creation_date"]) < pd.to_datetime(requests_df["need_date"]),
        1, 0
    )
    requests_df[["need_date", "request_creation_date"]] = requests_df[["need_date", "request_creation_date"]].apply(pd.to_datetime)
    requests_df["days_between"] = (requests_df["need_date"] - requests_df["request_creation_date"]).dt.days
    days_conditions = [
        requests_df["days_between"] == 0,
        requests_df["days_between"] > 0,
        requests_df["days_between"] < 0
    ]
    days_values = [0, 1, -1]
    requests_df["DaysDiffCategory"] = np.select(days_conditions, days_values, default=0)
    del requests_df["request_creation_date"], requests_df["need_date"]
    gc.collect()
    
    # Map work area to geographic zones
    print("Mapping work areas to geographic zones...")
    requests_df["work_area"] = requests_df["work_area"].fillna("Any location")
    work_area_mapping = {
        "madrid": "Madrid",
        "barcelona": "Barcelona",
        "cualquier": "Any location",
        "cádiz": "Andalusia",
        "coruña": "Galicia",
        "lisboa": "Lisbon",
        "baleares": "Balearic Islands",
        "sevilla": "Andalusia",
        "zaragoza": "Zaragoza",
        "huelva": "Andalusia",
        "milán": "Milan",
        "compostela": "Galicia",
        "valencia": "Valencia"
    }
    requests_df["work_zone"] = requests_df["work_area"].str.lower().str.split().map(
        lambda words: next((work_area_mapping[word] for word in words if word in work_area_mapping), "Other")
    )
    zone_map = {}
    zone_counts = requests_df["work_zone"].value_counts(normalize=True)
    for zone in zone_counts.index:
        zone_map[zone] = (zone_counts[zone] * 100) >= 2
    requests_df["GroupedWorkZone"] = requests_df.apply(
        lambda row: row["work_zone"] if zone_map.get(row["work_zone"]) else "Other",
        axis=1
    )
    del requests_df["work_area"], requests_df["work_zone"], zone_map
    requests_df["GroupedWorkZone"] = np.where(requests_df["GroupedWorkZone"] != "Any location", "Localized", requests_df["GroupedWorkZone"])
    
    # Process remote work indicator
    print("Creating full remote indicator...")
    requests_df["is_FullRemote"] = np.where(requests_df["remote_work"] == "100% Remote", 1, 0)
    
    # Process rate type indicator
    print("Creating negotiated rate indicator...")
    requests_df["is_NegotiatedRate"] = np.where(requests_df["rate_type"] == "Negotiated Rate", 1, 0)
    requests_df.drop(columns=["rate_type"], inplace=True)
    
    # Process technician hours
    print("Creating technician hours quartile...")
    requests_df["TechHoursQuartile"] = pd.qcut(requests_df["tech_hours"], q=4, labels=False)
    
    # Process community technician hours
    print("Creating community technician hours quartile and decile...")
    requests_df["CommunityTechHoursQuartile"] = pd.qcut(requests_df["community_tech_hours"], q=4, labels=False)
    requests_df["CommunityTechHoursDecile"] = pd.qcut(requests_df["community_tech_hours"], q=10, labels=False, duplicates="drop")
    requests_df.drop(columns=["community_tech_hours"], inplace=True)
    
    # Process applicant technician hours (last 12 months)
    print("Creating applicant technician hours (12 months) quartile and decile...")
    requests_df["ApplicantTechHours12Quartile"] = pd.qcut(requests_df["applicant_tech_hours_12"], q=4, labels=False)
    requests_df["ApplicantTechHours12Decile"] = pd.qcut(requests_df["applicant_tech_hours_12"], q=10, labels=False)
    requests_df.drop(columns=["applicant_tech_hours_12"], inplace=True)
    
    # Process applicant tenure
    print("Creating applicant tenure quartile and decile...")
    requests_df["ApplicantTenureQuartile"] = pd.qcut(requests_df["applicant_tenure"], q=4, labels=False)
    requests_df["ApplicantTenureDecile"] = pd.qcut(requests_df["applicant_tenure"], q=10, labels=False)
    requests_df.drop(columns=["applicant_tenure"], inplace=True)
    
    # Process technician tenure
    print("Creating technician tenure quartile and decile...")
    requests_df["TechTenureQuartile"] = pd.qcut(requests_df["tech_tenure"], q=4, labels=False)
    requests_df["TechTenureDecile"] = pd.qcut(requests_df["tech_tenure"], q=10, labels=False, duplicates="drop")
    requests_df.drop(columns=["tech_tenure"], inplace=True)
    
    # Process negative correlation flag based on RequestSalary vs. target
    print("Creating negative correlation flag (neg_corr)...")
    closed_req = requests_df[requests_df["target"] != -1]
    pivot_tbl = closed_req.pivot_table("RequestSalary", index=["GroupedPosition"], columns="target")
    m = pivot_tbl.mean(axis=1)
    for i, col in enumerate(pivot_tbl):
        pivot_tbl.iloc[:, i] = pivot_tbl.iloc[:, i].fillna(m)
    pivot_tbl["neg_corr"] = np.where(pivot_tbl[1] < pivot_tbl[0], 1, 0)
    pos_idx = pivot_tbl[pivot_tbl["neg_corr"] == 0].index
    neg_idx = pivot_tbl[pivot_tbl["neg_corr"] == 1].index
    requests_df["neg_corr"] = np.where(requests_df["GroupedPosition"].isin(neg_idx), 1, 0)
    del closed_req, pivot_tbl, m, pos_idx, neg_idx
    gc.collect()
    
    # Create an intermediate raw copy excluding salary-dependent variables
    df_raw = requests_df.copy(deep=True)
    remove_cols = ["RequestSalary", "SalaryQuartile", "SalaryDecile", "DesiredSalaryRatio", "DesiredSalaryRatio12", "neg_corr"]
    df_raw.drop(columns=remove_cols, inplace=True)
    
    # ---------------------- ENCODING CATEGORICAL VARIABLES ----------------------
    print("--------------- ENCODING --------------------")
    target_series = df_raw.pop("target")
    categorical_columns = df_raw.select_dtypes(include="object").columns
    encoded_df = pd.get_dummies(df_raw, columns=categorical_columns)
    print("Encoded categorical variables:", list(categorical_columns))
    print("Number of columns after encoding:", encoded_df.shape[1])
    print("\n")
    
    final_df = encoded_df.copy(deep=True)
    print("Clean DataFrame contains %d rows and %d columns" % (final_df.shape[0], final_df.shape[1]))
    print("Total missing values =", final_df.isnull().sum().sum())
    
    del df_raw, raw_requests_df, requests_df, encoded_df
    gc.collect()
    print("\n")
    
    # ---------------------- SCALING AND FEATURE SELECTION ----------------------
    if use_preload == 1:
        print("--------------- SCALING ---------------------")
        scaled_df = final_df.copy(deep=True)
        features = scaled_df.columns
        scaler = MinMaxScaler()
        scaled_df[features] = scaler.fit_transform(scaled_df[features])
        scaler_file = "scaler_" + str(current_date) + ".pkl"
        scaler_path = os.path.join(directory_path, scaler_file)
        joblib.dump(scaler, scaler_path)
        del categorical_columns, features
        gc.collect()
    
        print("--------------- TARGET CORRELATION -----------------")
        corr_df = scaled_df.merge(target_series, left_index=True, right_index=True, how="left")
        target_corr = abs(corr_df[corr_df["target"] != -1].corr()["target"]).sort_values(ascending=False)
        top_thresh = target_corr.quantile(target_corr_threshold)
        corr_feats = target_corr[target_corr >= top_thresh].sort_values(ascending=False)
        reduced_df = corr_df[corr_feats.index].copy()
        del reduced_df["target"]
        gc.collect()
        print("Removed %u variables with low target correlation." % (corr_df.shape[1] - reduced_df.shape[1]))
    
        print("--------------- FEATURE CORRELATION -----------------")
        corr_matrix = reduced_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] >= pearson_threshold)]
        eliminated = reduced_df.loc[:, high_corr_cols]
        final_df_for_model = reduced_df.copy(deep=True)
        final_df_for_model.drop(columns=high_corr_cols, inplace=True)
        print("Removed %u highly correlated features." % (reduced_df.shape[1] - final_df_for_model.shape[1]))
        print("Final number of features:", final_df_for_model.shape[1])
        del corr_matrix, upper_tri, high_corr_cols, eliminated, pearson_threshold
        gc.collect()
        print("\n")
        
        model_columns = pd.DataFrame(final_df_for_model.columns)
        cols_file = "modelcols_" + str(current_date) + ".csv"
        cols_path = os.path.join(directory_path, cols_file)
        model_columns.to_csv(cols_path, index=False)
        
        final_df_for_model = final_df_for_model.merge(target_series, left_index=True, right_index=True, how="left")
    
    elif use_preload == 0:
        print("Loading scaler from file...\n")
        last_scaler = max(glob.glob(os.path.join(directory_path, "scaler_*.pkl")))
        scaler = joblib.load(last_scaler)
        scaled_df = final_df.copy(deep=True)
        features = scaled_df.columns
        scaled_df[features] = scaler.fit_transform(scaled_df[features])
        final_df_for_model = scaled_df.copy(deep=True)
        final_df_for_model = final_df_for_model.merge(target_series, left_index=True, right_index=True, how="left")
        del scaled_df, features
        gc.collect()
    else:
        print("Error: Invalid parameter for preprocessing.")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("ETL process completed in %.2f seconds." % round(elapsed_time, 2))
    
    return initial_requests_df, encoded_df, final_df_for_model
