# ============================================================
#  InsureGuard — Model Training & Evaluation
#  Run this from your project root:  python src/train.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve,
                                     precision_recall_curve, average_precision_score,
                                     f1_score, precision_score, recall_score,
                                     accuracy_score)
from collections import Counter

# ── Dark plot style ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#0f0f1a', 'axes.facecolor':'#1a1a2e',
    'axes.edgecolor':'#444466',   'axes.labelcolor':'#e0e0ff',
    'xtick.color':'#aaaacc',      'ytick.color':'#aaaacc',
    'text.color':'#e0e0ff',       'grid.color':'#2a2a4a',
    'grid.linestyle':'--',        'grid.alpha':0.5,
    'axes.titlesize':12,
})
FRAUD_C = '#ff416c'
LEGIT_C = '#38ef7d'
ACCENT  = '#667eea'
GOLD    = '#f7c948'

# ── Paths ─────────────────────────────────────────────────────
CSV      = 'data/raw/insurance_claims.csv'
MODELS   = 'models'
PLOTS    = 'data/processed'

os.makedirs(MODELS, exist_ok=True)
os.makedirs(PLOTS,  exist_ok=True)

print("=" * 55)
print("   🛡️  InsureGuard — Model Training Pipeline")
print("=" * 55)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("\n📂 STEP 1: Loading data...")

if os.path.exists(CSV):
    df = pd.read_csv(CSV)
    print(f"   ✅ Real dataset — {df.shape[0]:,} rows × {df.shape[1]} cols")
else:
    print("   ⚠️  CSV not found — generating synthetic data...")
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
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv(CSV, index=False)
    print(f"   ✅ Synthetic data saved — {df.shape[0]:,} rows")

print(f"   Fraud: {(df['fraud_reported']=='Y').sum():,}  "
      f"| Legit: {(df['fraud_reported']=='N').sum():,}")

# ============================================================
# STEP 2 — FEATURE ENGINEERING & PREPROCESSING
# ============================================================
print("\n🔧 STEP 2: Feature engineering & preprocessing...")

df2 = df.copy()

# New features
if 'total_claim_amount' in df2.columns and 'policy_annual_premium' in df2.columns:
    df2['claim_premium_ratio'] = df2['total_claim_amount'] / (df2['policy_annual_premium']+1)

if all(c in df2.columns for c in ['injury_claim','property_claim','vehicle_claim','total_claim_amount']):
    df2['injury_ratio']   = df2['injury_claim']   / (df2['total_claim_amount']+1)
    df2['property_ratio'] = df2['property_claim'] / (df2['total_claim_amount']+1)
    df2['vehicle_ratio']  = df2['vehicle_claim']  / (df2['total_claim_amount']+1)

if 'total_claim_amount' in df2.columns:
    thresh = df2['total_claim_amount'].quantile(0.90)
    df2['is_high_claim'] = (df2['total_claim_amount'] > thresh).astype(int)

if 'incident_hour_of_the_day' in df2.columns:
    df2['is_night_incident'] = df2['incident_hour_of_the_day'].apply(
        lambda h: 1 if (h >= 22 or h <= 4) else 0)

# Drop low-signal columns
drop_cols = [c for c in [
    'policy_number','insured_zip','policy_bind_date','incident_date',
    'incident_city','auto_model','policy_csl','policy_state','incident_state'
] if c in df2.columns]
df2.drop(columns=drop_cols, inplace=True)

# Encode target
df2['fraud_reported'] = df2['fraud_reported'].map({'Y':1, 'N':0})

# Encode categoricals
le_dict = {}
for col in df2.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col].astype(str))
    le_dict[col] = le

X = df2.drop(columns=['fraud_reported'])
y = df2['fraud_reported']
feature_names = X.columns.tolist()

print(f"   ✅ {len(feature_names)} features ready")
print(f"   New: claim_premium_ratio, injury_ratio, property_ratio,")
print(f"        vehicle_ratio, is_high_claim, is_night_incident")

# ── Fix NaN values before SMOTE ──────────────────────────────
nan_count = X.isnull().sum().sum()
if nan_count > 0:
    print(f"\n   ⚠️  Found {nan_count} NaN values — imputing now...")
    # Numeric columns → fill with median
    for col in X.select_dtypes(include='number').columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    # Any remaining → fill with 0
    X = X.fillna(0)
    print(f"   ✅ All NaN values fixed (numeric→median, rest→0)")
else:
    print(f"   ✅ No NaN values found")

# ============================================================
# STEP 3 — CLASS BALANCING
# ============================================================
print(f"\n⚖️  STEP 3: Balancing classes...")
print(f"   Before: {Counter(y)}")

def manual_oversample(X, y, rs=42):
    np.random.seed(rs)
    Xa, ya = np.array(X), np.array(y)
    min_idx = np.where(ya==1)[0]
    maj_idx = np.where(ya==0)[0]
    needed  = len(maj_idx) - len(min_idx)
    samp    = np.random.choice(min_idx, size=needed, replace=True)
    Xb = np.vstack([Xa, Xa[samp]])
    yb = np.concatenate([ya, ya[samp]])
    idx = np.random.permutation(len(yb))
    return Xb[idx], yb[idx]

try:
    from imblearn.over_sampling import SMOTE
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X, y)
    print("   ✅ SMOTE applied (imblearn)")
except ImportError:
    X_bal, y_bal = manual_oversample(X, y)
    print("   ✅ Manual oversampling applied")

print(f"   After : {Counter(y_bal)}")

# ============================================================
# STEP 4 — TRAIN / TEST SPLIT
# ============================================================
print("\n✂️  STEP 4: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.20, random_state=42, stratify=y_bal)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, f'{MODELS}/scaler.pkl')

print(f"   Train : {X_train_sc.shape[0]:,} samples")
print(f"   Test  : {X_test_sc.shape[0]:,} samples")
print(f"   Saved : models/scaler.pkl")

# ============================================================
# STEP 5 — TRAIN 3 BASELINE MODELS
# ============================================================
print("\n🏋️  STEP 5: Training 3 models (2–4 min)...")

models_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results        = {}
trained_models = {}

for name, model in models_dict.items():
    Xtr = X_train_sc if name == 'Logistic Regression' else X_train
    Xte = X_test_sc  if name == 'Logistic Regression' else X_test
    model.fit(Xtr, y_train)
    yp  = model.predict(Xte)
    ypr = model.predict_proba(Xte)[:,1]

    results[name] = {
        'Accuracy' : accuracy_score(y_test, yp),
        'Precision': precision_score(y_test, yp),
        'Recall'   : recall_score(y_test, yp),
        'F1'       : f1_score(y_test, yp),
        'AUC-ROC'  : roc_auc_score(y_test, ypr),
        'Avg Prec' : average_precision_score(y_test, ypr),
        'y_pred': yp, 'y_proba': ypr,
    }
    trained_models[name] = model

    print(f"\n   ── {name}")
    print(f"      Accuracy : {results[name]['Accuracy']:.4f}  "
          f"Precision: {results[name]['Precision']:.4f}")
    print(f"      Recall   : {results[name]['Recall']:.4f}  "
          f"F1-Score : {results[name]['F1']:.4f}")
    print(f"      AUC-ROC  : {results[name]['AUC-ROC']:.4f}  "
          f"Avg Prec : {results[name]['Avg Prec']:.4f}")

best_baseline = max(results, key=lambda m: results[m]['F1'])
print(f"\n   🏆 Best baseline: {best_baseline}  "
      f"F1={results[best_baseline]['F1']:.4f}")

# ============================================================
# STEP 6 — PLOT COMPARISON + CONFUSION MATRICES
# ============================================================
print("\n📊 STEP 6: Generating comparison charts...")

model_names = list(results.keys())
metrics     = ['Accuracy','Precision','Recall','F1','AUC-ROC','Avg Prec']
data_mat    = np.array([[results[m][mt] for mt in metrics] for m in model_names])
colors_m    = [LEGIT_C, ACCENT, GOLD]

# Comparison bar chart
fig, ax = plt.subplots(figsize=(14,5))
fig.suptitle('Model Comparison — All Metrics', fontweight='bold', fontsize=14)
x, w = np.arange(len(metrics)), 0.25
for i,(name,color) in enumerate(zip(model_names, colors_m)):
    bars = ax.bar(x+i*w, data_mat[i], w, label=name,
                  color=color, alpha=0.85, edgecolor='none')
    for b,v in zip(bars, data_mat[i]):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.2f}',
                ha='center', fontsize=8, rotation=90)
ax.set_xticks(x+w); ax.set_xticklabels(metrics, rotation=20, ha='right')
ax.set_ylim(0, 1.15); ax.set_ylabel('Score'); ax.legend(fontsize=9)
fig.tight_layout()
plt.savefig(f'{PLOTS}/plot_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16,5))
fig.suptitle('Confusion Matrices', fontweight='bold', fontsize=14)
for ax,name in zip(axes, model_names):
    cm     = confusion_matrix(y_test, results[name]['y_pred'])
    cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
    im     = ax.imshow(cm_pct, cmap='RdYlGn', vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            c = 'black' if cm_pct[i,j]>60 else 'white'
            ax.text(j, i, f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=11, fontweight='bold', color=c)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred:Legit','Pred:Fraud'], fontsize=9)
    ax.set_yticklabels(['True:Legit','True:Fraud'], fontsize=9)
    ax.set_title(name, fontweight='bold')
plt.colorbar(im, ax=axes[-1], fraction=0.04, pad=0.04)
fig.tight_layout()
plt.savefig(f'{PLOTS}/plot_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle('ROC & Precision-Recall Curves', fontweight='bold', fontsize=14)
axes[0].plot([0,1],[0,1],'w--',alpha=0.4,label='Random (AUC=0.50)')
for name,color in zip(model_names, colors_m):
    fpr,tpr,_ = roc_curve(y_test, results[name]['y_proba'])
    axes[0].plot(fpr,tpr,color=color,linewidth=2.5,
                 label=f'{name} (AUC={results[name]["AUC-ROC"]:.4f})')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC Curve'); axes[0].legend(fontsize=9)
for name,color in zip(model_names, colors_m):
    p,r,_ = precision_recall_curve(y_test, results[name]['y_proba'])
    axes[1].plot(r,p,color=color,linewidth=2.5,
                 label=f'{name} (AP={results[name]["Avg Prec"]:.4f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('PR Curve'); axes[1].legend(fontsize=9)
fig.tight_layout()
plt.savefig(f'{PLOTS}/plot_roc_pr.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
tree_m = {k:v for k,v in trained_models.items() if hasattr(v,'feature_importances_')}
if tree_m:
    fig, axes = plt.subplots(1, len(tree_m), figsize=(8*len(tree_m), 7))
    if len(tree_m)==1: axes=[axes]
    fig.suptitle('Feature Importance', fontweight='bold', fontsize=14)
    for ax,(name,model) in zip(axes, tree_m.items()):
        imp  = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        top  = imp.tail(20); norm = top/top.max()
        bc   = [FRAUD_C if n>0.6 else ACCENT if n>0.3 else LEGIT_C for n in norm.values]
        bars = ax.barh(top.index, top.values, color=bc, edgecolor='none', height=0.7)
        ax.set_xlabel('Importance'); ax.set_title(f'{name} — Top 20', fontweight='bold')
        for bar,val in zip(bars, top.values):
            ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
    fig.tight_layout()
    plt.savefig(f'{PLOTS}/plot_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"   ✅ Charts saved to data/processed/")

# ============================================================
# STEP 7 — HYPERPARAMETER TUNING
# ============================================================
print("\n🎛️  STEP 7: Hyperparameter tuning (5–8 min)...")

fast_grid = {
    'n_estimators':      [150, 250],
    'max_depth':         [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features':      ['sqrt'],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_tuned = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    fast_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
rf_tuned.fit(X_train, y_train)

print("\n   ✅ Best Parameters:")
for k,v in rf_tuned.best_params_.items():
    print(f"      {k:<25}: {v}")
print(f"\n   Best CV F1 : {rf_tuned.best_score_:.4f}")

best_rf       = rf_tuned.best_estimator_
y_pred_tuned  = best_rf.predict(X_test)
y_proba_tuned = best_rf.predict_proba(X_test)[:,1]

base_f1  = results['Random Forest']['F1']
tuned_f1 = f1_score(y_test, y_pred_tuned)
print(f"\n   Baseline RF F1 : {base_f1:.4f}")
print(f"   Tuned    RF F1 : {tuned_f1:.4f}  "
      f"(+{(tuned_f1-base_f1)*100:.2f}pp improvement)")

# ============================================================
# STEP 8 — CROSS VALIDATION
# ============================================================
print("\n📐 STEP 8: Cross-validation on final model...")

cv_scores = cross_val_score(
    best_rf, X_bal, y_bal,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1)

print(f"\n   5-Fold CV Results:")
for i,s in enumerate(cv_scores, 1):
    bar = '█' * int(s * 40)
    print(f"   Fold {i}: {bar} {s:.4f}")
print(f"   {'─'*48}")
print(f"   Mean F1 : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

# Final evaluation chart
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('Final Model — Tuned Random Forest', fontweight='bold', fontsize=14)
bc   = [FRAUD_C if s < cv_scores.mean() else LEGIT_C for s in cv_scores]
bars = axes[0].bar([f'Fold {i}' for i in range(1,6)], cv_scores,
                   color=bc, edgecolor='none', width=0.5)
axes[0].axhline(cv_scores.mean(), color='white', linestyle='--', linewidth=2,
                label=f'Mean={cv_scores.mean():.4f}')
axes[0].fill_between(range(-1,6),
    cv_scores.mean()-cv_scores.std(), cv_scores.mean()+cv_scores.std(),
    alpha=0.15, color='white')
for b,v in zip(bars, cv_scores):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.4f}',
                 ha='center', fontweight='bold', fontsize=10)
axes[0].set_ylim(max(0,cv_scores.min()-0.05), min(1,cv_scores.max()+0.05))
axes[0].set_ylabel('F1'); axes[0].set_title('CV F1 per Fold'); axes[0].legend()

cm     = confusion_matrix(y_test, y_pred_tuned)
cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
im     = axes[1].imshow(cm_pct, cmap='RdYlGn', vmin=0, vmax=100)
for i in range(2):
    for j in range(2):
        c = 'black' if cm_pct[i,j]>60 else 'white'
        axes[1].text(j, i, f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                     ha='center', va='center', fontsize=13, fontweight='bold', color=c)
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Pred:Legit','Pred:Fraud'])
axes[1].set_yticklabels(['True:Legit','True:Fraud'])
axes[1].set_title('Final Confusion Matrix')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.tight_layout()
plt.savefig(f'{PLOTS}/plot_final_eval.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# STEP 9 — SAVE EVERYTHING
# ============================================================
print("\n💾 STEP 9: Saving model & artifacts...")

joblib.dump(best_rf, f'{MODELS}/best_model.pkl')
joblib.dump(le_dict, f'{MODELS}/label_encoders.pkl')

with open(f'{MODELS}/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

report = {
    'model':          'Random Forest (Tuned)',
    'best_params':    rf_tuned.best_params_,
    'cv_f1_mean':     round(float(cv_scores.mean()), 4),
    'cv_f1_std':      round(float(cv_scores.std()),  4),
    'test_accuracy':  round(float(accuracy_score(y_test,  y_pred_tuned)), 4),
    'test_precision': round(float(precision_score(y_test, y_pred_tuned)), 4),
    'test_recall':    round(float(recall_score(y_test,    y_pred_tuned)), 4),
    'test_f1':        round(float(f1_score(y_test,        y_pred_tuned)), 4),
    'test_auc_roc':   round(float(roc_auc_score(y_test,   y_proba_tuned)), 4),
    'n_features':     len(feature_names),
    'features':       feature_names,
}
with open(f'{MODELS}/model_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("""
╔══════════════════════════════════════════════════════════╗
║              💾  All Artifacts Saved!                   ║
╠══════════════════════════════════════════════════════════╣
║  models/best_model.pkl       ← trained RF model         ║
║  models/scaler.pkl           ← feature scaler           ║
║  models/feature_names.json   ← feature list             ║
║  models/label_encoders.pkl   ← category encoders        ║
║  models/model_report.json    ← metrics summary          ║
╚══════════════════════════════════════════════════════════╝""")

print(f"\n🏆 Final Model Performance:")
for k,v in report.items():
    if k not in ['best_params','features','model','n_features']:
        print(f"   {k:<20}: {v}")

print("""
╔══════════════════════════════════════════════════════════╗
║   ✅  Training Complete! Next → run the Streamlit app   ║
║       cd app && streamlit run app.py                    ║
╚══════════════════════════════════════════════════════════╝""")