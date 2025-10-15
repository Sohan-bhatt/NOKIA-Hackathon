import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# --- 1. DATA PREPROCESSING + FEATURE ENGINEERING ---
def preprocess_alarm_data(df, cutoff=100):
    # Handle columns possibly missing from user-uploaded data
    if 'Alarmed Object Source System' in df.columns:
        df = df.drop('Alarmed Object Source System', axis=1)
    df = df.dropna(subset=['Alarm Name','Site Name'])
    df['Additional Text'] = df.get('Additional Text', 'Unknown')
    df['Additional Text'].fillna('Unknown', inplace=True)
    df['Is Service Affecting'] = df.get('Is Service Affecting', 1)
    df['Is Service Affecting'].fillna(1, inplace=True)

    # is_active logic: if "Last Time Cleared" exists, otherwise fill with zeros
    if 'Last Time Cleared' in df.columns:
        df['is_active'] = df['Last Time Cleared'].isnull().astype(int)
    elif 'is_active' not in df.columns:
        df['is_active'] = 0

    # Merge rare causes for "Probable Cause"
    cause_counts = df['Probable Cause'].value_counts()
    common_causes = set(cause_counts[cause_counts >= cutoff].index)
    df['Probable Cause Merged'] = df['Probable Cause'].apply(lambda cause: cause if cause in common_causes else 'Other')

    # Timestamp features
    df['First Time Detected Clean'] = pd.to_datetime(df['First Time Detected'].str.slice(0, 19), format="%Y/%m/%d %H:%M:%S")
    df['hour'] = df['First Time Detected Clean'].dt.hour
    df['dayofweek'] = df['First Time Detected Clean'].dt.dayofweek

    # Label encoding for categorical columns
    categorical_cols = [
        'Severity', 'Site Name', 'Source System', 'Alarm Name',
        'Alarmed Object Name', 'Alarmed Object Type', 'Alarm Type',
        'Probable Cause Merged', 'Specific Problem', 'Previous Severity'
    ]
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = 0  # if missing, fill with 0 and encode a dummy
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Numeric clean - always ensure columns exist!
    for col in ['Life Span (minutes)', 'Number Of Occurrences', 'Is Service Affecting', 'is_active']:
        if col not in df.columns:
            df[col] = 0 if col != 'Number Of Occurrences' else 1
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0 if col != 'Number Of Occurrences' else 1)

    # Drop unused columns
    drop_cols = [
        'Unnamed: 0', 'Alarm ID', 'First Time Detected', 'Last Time Cleared',
        'Last Time Detected', 'Additional Text', 'Probable Cause'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Next alarm timestamp (for time regression)
    df = df.sort_values('First Time Detected Clean').reset_index(drop=True)
    df['Next Alarm Timestamp'] = df['First Time Detected Clean'].shift(-1)
    df['Next Alarm Unix'] = df['Next Alarm Timestamp'].astype(np.int64) // 10**9
    df['This Alarm Unix'] = df['First Time Detected Clean'].astype(np.int64) // 10**9
    df_time = df[df['Next Alarm Unix'].notnull()]
    df_time = df_time[df_time['Next Alarm Unix'] > df_time['This Alarm Unix']]

    return df, df_time, encoders

# --- 2. MODEL TRAINING ---
def train_all_models(df, df_time):
    # Feature lists
    input_features_alarm_type = [
        'Source System', 'Alarm Name', 'Alarmed Object Name', 'Alarmed Object Type',
        'Previous Severity', 'Is Service Affecting', 'Number Of Occurrences', 'is_active', 'hour', 'dayofweek'
    ]
    input_features_cause = input_features_alarm_type + ['Alarm Type']
    input_features_duration = input_features_cause + ['Probable Cause Merged']
    input_features_severity = input_features_duration + ['Life Span (minutes)']
    input_features_timestamp = [
        'Severity', 'Site Name', 'Source System', 'Probable Cause Merged',
        'Alarmed Object Name', 'Alarmed Object Type', 'Previous Severity',
        'Is Service Affecting', 'Number Of Occurrences', 'is_active',
        'Alarm Type', 'Life Span (minutes)', 'hour', 'dayofweek'
    ]
    target_alarm_type = 'Alarm Type'
    target_cause = 'Probable Cause Merged'
    target_duration = 'Life Span (minutes)'
    target_severity = 'Previous Severity'
    target_timestamp = 'Next Alarm Unix'

    # Alarm Type
    X = df[input_features_alarm_type]
    y_alarm_type = df[target_alarm_type]
    X_train, X_test, y_train_alarm_type, y_test_alarm_type = train_test_split(
        X, y_alarm_type, test_size=0.2, random_state=42, stratify=y_alarm_type
    )

    # Timestamp regression split
    X_time = df_time[input_features_timestamp]
    y_time = df_time[target_timestamp]
    X_time_train, X_time_test, y_time_train, y_time_test = train_test_split(
        X_time, y_time, test_size=0.2, random_state=42
    )

    # 1. Alarm Type
    alarm_type_clf = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric='mlogloss')
    alarm_type_clf.fit(X_train, y_train_alarm_type)

    # 2. Probable Cause
    X_cause = df[input_features_cause]
    y_cause = df[target_cause]
    cause_clf = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric='mlogloss')
    cause_clf.fit(X_cause, y_cause)

    # 3. Duration
    X_duration = df[input_features_duration]
    y_duration = df[target_duration]
    duration_reg = XGBRegressor(tree_method="hist")
    duration_reg.fit(X_duration, y_duration)

    # 4. Severity
    X_severity = df[input_features_severity]
    y_severity = df[target_severity]
    severity_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    severity_clf.fit(X_severity, y_severity)

    # 5. Next Alarm Timestamp (independent regression)
    reg_time = XGBRegressor(tree_method="hist")
    reg_time.fit(X_time_train, y_time_train)

    # Return all models and relevant feature lists
    return {
        'alarm_type_clf': alarm_type_clf,
        'cause_clf': cause_clf,
        'duration_reg': duration_reg,
        'severity_clf': severity_clf,
        'timestamp_reg': reg_time,
        'input_features': {
            'alarm_type': input_features_alarm_type,
            'cause': input_features_cause,
            'duration': input_features_duration,
            'severity': input_features_severity,
            'timestamp': input_features_timestamp
        }
    }

# --- 3. SAVE/LOAD MODEL BUNDLE + ENCODERS ---
def save_all_models(model_dict, encoders, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    for k, model in model_dict.items():
        if "input_features" not in k:
            joblib.dump(model, os.path.join(out_dir, f"{k}.joblib"))
    # Save encoders and feature lists
    joblib.dump(encoders, os.path.join(out_dir, "encoders.joblib"))
    joblib.dump(model_dict['input_features'], os.path.join(out_dir, "input_features.joblib"))

def load_all_models(out_dir="models"):
    model_dict = {}
    for k in ['alarm_type_clf', 'cause_clf', 'duration_reg', 'severity_clf', 'timestamp_reg']:
        model_dict[k] = joblib.load(os.path.join(out_dir, f"{k}.joblib"))
    model_dict['input_features'] = joblib.load(os.path.join(out_dir, "input_features.joblib"))
    encoders = joblib.load(os.path.join(out_dir, "encoders.joblib"))
    return model_dict, encoders

# --- 4. CHAINED PREDICTION FUNCTION (for a single sample) ---
def chained_predict(row, model_dict, encoders):
    features = model_dict['input_features']
    alarm_type_clf = model_dict['alarm_type_clf']
    cause_clf = model_dict['cause_clf']
    duration_reg = model_dict['duration_reg']
    severity_clf = model_dict['severity_clf']

    # Predict Alarm Type
    alarm_type_proba = alarm_type_clf.predict_proba(row[features['alarm_type']])
    pred_alarm_type = alarm_type_clf.predict(row[features['alarm_type']])[0]
    alarm_type_conf = alarm_type_proba[0][pred_alarm_type]

    # Probable Cause
    row_cause = row.copy()
    row_cause['Alarm Type'] = pred_alarm_type
    cause_proba = cause_clf.predict_proba(row_cause[features['cause']])
    pred_cause = cause_clf.predict(row_cause[features['cause']])[0]
    cause_conf = cause_proba[0][pred_cause]

    # Duration
    row_duration = row_cause.copy()
    row_duration['Probable Cause Merged'] = pred_cause
    pred_duration = duration_reg.predict(row_duration[features['duration']])[0]

    # Severity
    row_sev = row_duration.copy()
    row_sev['Life Span (minutes)'] = pred_duration
    sev_proba = severity_clf.predict_proba(row_sev[features['severity']])
    pred_severity = severity_clf.predict(row_sev[features['severity']])[0]
    sev_conf = sev_proba[0][pred_severity]

    # Decode outputs
    decoded_alarm_type = encoders['Alarm Type'].inverse_transform([pred_alarm_type])[0]
    decoded_cause = encoders['Probable Cause Merged'].inverse_transform([pred_cause])[0]
    decoded_severity = encoders['Previous Severity'].inverse_transform([pred_severity])[0]

    return {
        'Predicted Alarm Type': decoded_alarm_type,
        'Alarm Type Confidence': alarm_type_conf,
        'Predicted Probable Cause': decoded_cause,
        'Cause Confidence': cause_conf,
        'Predicted Duration (minutes)': float(pred_duration),
        'Predicted Severity': decoded_severity,
        'Severity Confidence': sev_conf
    }

# --- 5. BATCH PREDICT (for top 10 or many) ---
def batch_chained_predict(X, model_dict, encoders):
    results = []
    for idx, row in X.iterrows():
        # row.to_frame().T is a single‐row DataFrame
        single = row.to_frame().T
        result = chained_predict(single, model_dict, encoders)
        result['row'] = idx
        results.append(result)
    return pd.DataFrame(results)
