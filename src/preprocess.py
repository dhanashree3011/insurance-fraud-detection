import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

def load_and_clean(path):
    df = pd.read_csv(path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df

def engineer_features(df):
    # Example feature engineering (adjust to your dataset columns)
    if 'incident_date' in df.columns and 'policy_bind_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
        df['days_since_policy'] = (df['incident_date'] - df['policy_bind_date']).dt.days

    # Claim to premium ratio (if columns exist)
    if 'total_claim_amount' in df.columns and 'policy_annual_premium' in df.columns:
        df['claim_premium_ratio'] = df['total_claim_amount'] / (df['policy_annual_premium'] + 1)

    return df

def encode_and_scale(df, target_col='fraud_reported'):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != target_col:
            df[col] = le.fit_transform(df[col].astype(str))

    # Encode target
    df[target_col] = df[target_col].map({'Y': 1, 'N': 0})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_scaled, y, X.columns.tolist()

if __name__ == "__main__":
    df = load_and_clean('data/raw/insurance_claims.csv')
    df = engineer_features(df)
    X, y, features = encode_and_scale(df)
    pd.DataFrame(X, columns=features).assign(fraud_reported=y.values)\
      .to_csv('data/processed/cleaned_data.csv', index=False)
    print("✅ Preprocessing done!")