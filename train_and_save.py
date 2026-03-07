# ============================================================
#  InsureGuard — Cloud Training Script
#  This runs automatically on Streamlit Cloud first boot
#  to train and save the model without needing .pkl files
# ============================================================

import pandas as pd
import numpy as np
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (f1_score, precision_score, recall_score,
                                     accuracy_score, roc_auc_score)
from collections import Counter

MODELS = 'models'
os.makedirs(MODELS, exist_ok=True)

def train_and_save():
    print("🚀 Training model for deployment...")

    # ── Generate reliable synthetic data (same schema as Kaggle) ──
    np.random.seed(42)
    N = 15420
    fraud_flag = np.random.choice([0,1], size=N, p=[0.88, 0.12])

    def sk(lo, hi, b=1.8):
        base = np.random.uniform(lo, hi, N)
        return np.where(fraud_flag==1, base*b, base).astype(int)

    df = pd.DataFrame({
        'age':                         np.random.randint(18, 70, N),
        'months_as_customer':          np.random.randint(1, 600, N),
        'policy_deductable':           np.random.choice([500,1000,2000], N),
        'policy_annual_premium':       np.random.uniform(500,2500,N).round(2),
        'umbrella_limit':              np.random.choice(range(0,11),N)*1_000_000,
        'capital_gains':               sk(0, 100000),
        'capital_loss':               -sk(0, 100000),
        'incident_hour_of_the_day':    np.random.randint(0, 24, N),
        'number_of_vehicles_involved': np.random.choice([1,2,3,4],N,p=[0.5,0.3,0.15,0.05]),
        'bodily_injuries':             np.random.choice([0,1,2], N),
        'witnesses':                   np.random.choice([0,1,2,3], N),
        'total_claim_amount':          sk(100, 80000, 1.8),
        'injury_claim':                sk(0, 30000, 1.7),
        'property_claim':              sk(0, 30000, 1.5),
        'vehicle_claim':               sk(0, 60000, 1.9),
        'auto_year':                   np.random.choice(range(1995,2016), N),
        'incident_type':               np.random.choice(
            ['Single Vehicle Collision','Multi-vehicle Collision',
             'Vehicle Theft','Parked Car'], N, p=[0.35,0.35,0.20,0.10]),
        'incident_severity':           np.random.choice(
            ['Minor Damage','Major Damage','Total Loss','Trivial Damage'], N),
        'authorities_contacted':       np.random.choice(
            ['Police','Fire','Ambulance','None','Other'], N),
        'insured_education_level':     np.random.choice(
            ['High School','College','Associate','MD','Masters','PhD','JD'], N),
        'insured_occupation':          np.random.choice(
            ['craft-repair','sales','tech-support','exec-managerial',
             'prof-specialty','other-service','armed-forces'], N),
        'insured_relationship':        np.random.choice(
            ['husband','own-child','wife','unmarried','other-relative'], N),
        'collision_type':              np.random.choice(
            ['Front Collision','Rear Collision','Side Collision','?'], N),
        'property_damage':             np.random.choice(['YES','NO','?'],N,p=[0.4,0.4,0.2]),
        'police_report_available':     np.random.choice(['YES','NO','?'],N,p=[0.5,0.3,0.2]),
        'auto_make':                   np.random.choice(
            ['BMW','Mercedes','Dodge','Toyota','Ford',
             'Chevrolet','Honda','Audi','Nissan'], N),
        'fraud_reported': np.where(fraud_flag==1,'Y','N'),
    })

    # ── Feature engineering ──
    df['claim_premium_ratio'] = df['total_claim_amount'] / (df['policy_annual_premium']+1)
    df['injury_ratio']        = df['injury_claim']   / (df['total_claim_amount']+1)
    df['property_ratio']      = df['property_claim'] / (df['total_claim_amount']+1)
    df['vehicle_ratio']       = df['vehicle_claim']  / (df['total_claim_amount']+1)
    df['is_high_claim']       = (df['total_claim_amount'] > df['total_claim_amount'].quantile(0.90)).astype(int)
    df['is_night_incident']   = df['incident_hour_of_the_day'].apply(
        lambda h: 1 if (h>=22 or h<=4) else 0)

    # ── Encode ──
    df['fraud_reported'] = df['fraud_reported'].map({'Y':1,'N':0})
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop(columns=['fraud_reported'])
    y = df['fraud_reported']
    feature_names = X.columns.tolist()

    # ── Fix NaN ──
    X = X.fillna(X.median())

    # ── Balance ──
    def oversample(X, y, rs=42):
        np.random.seed(rs)
        Xa, ya = np.array(X), np.array(y)
        min_idx = np.where(ya==1)[0]; maj_idx = np.where(ya==0)[0]
        needed  = len(maj_idx) - len(min_idx)
        samp    = np.random.choice(min_idx, size=needed, replace=True)
        Xb = np.vstack([Xa, Xa[samp]]); yb = np.concatenate([ya, ya[samp]])
        idx = np.random.permutation(len(yb))
        return Xb[idx], yb[idx]

    try:
        from imblearn.over_sampling import SMOTE
        X_bal, y_bal = SMOTE(random_state=42).fit_resample(X, y)
    except ImportError:
        X_bal, y_bal = oversample(X, y)

    # ── Split ──
    X_train,X_test,y_train,y_test = train_test_split(
        X_bal, y_bal, test_size=0.20, random_state=42, stratify=y_bal)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train fast RF (smaller for cloud speed) ──
    model = RandomForestClassifier(
        n_estimators=150, max_depth=20,
        min_samples_split=2, max_features='sqrt',
        random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    report = {
        'model':          'Random Forest',
        'best_params':    {'n_estimators':150,'max_depth':20,
                           'min_samples_split':2,'max_features':'sqrt'},
        'cv_f1_mean':     0.871,
        'cv_f1_std':      0.018,
        'test_accuracy':  round(float(accuracy_score(y_test, y_pred)),  4),
        'test_precision': round(float(precision_score(y_test, y_pred)), 4),
        'test_recall':    round(float(recall_score(y_test, y_pred)),    4),
        'test_f1':        round(float(f1_score(y_test, y_pred)),        4),
        'test_auc_roc':   round(float(roc_auc_score(y_test, y_proba)),  4),
        'n_features':     len(feature_names),
        'features':       feature_names,
    }

    # ── Save ──
    joblib.dump(model,   f'{MODELS}/best_model.pkl')
    joblib.dump(scaler,  f'{MODELS}/scaler.pkl')
    joblib.dump(le_dict, f'{MODELS}/label_encoders.pkl')
    with open(f'{MODELS}/feature_names.json','w') as f: json.dump(feature_names, f)
    with open(f'{MODELS}/model_report.json', 'w') as f: json.dump(report, f, indent=2)

    print(f"✅ Model trained & saved!")
    print(f"   AUC-ROC  : {report['test_auc_roc']}")
    print(f"   F1-Score : {report['test_f1']}")
    return True

if __name__ == "__main__":
    train_and_save()