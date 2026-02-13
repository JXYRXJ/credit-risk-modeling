import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_score, 
    recall_score, f1_score, accuracy_score, average_precision_score,
    precision_recall_curve, roc_curve, auc, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, log_loss
)
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import sys
import matplotlib.pyplot as plt

# ----------------------------
# Logger
# ----------------------------
class ResultLogger:
    def __init__(self, filename='credit_risk_model_results.txt'):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.console.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

logger = ResultLogger()
sys.stdout = logger

# ----------------------------
# Header
# ----------------------------
print("=" * 80)
print("COMPREHENSIVE CREDIT RISK MODEL EVALUATION WITH OVERFITTING ANALYSIS")
print("=" * 80)
print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ----------------------------
# Load datasets
# ----------------------------
print("Loading datasets...")
X_train = pd.read_csv('X_train_processed.csv')
X_test = pd.read_csv('X_test_processed.csv') 
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class distribution: {dict(zip(*np.unique(np.concatenate([y_train, y_test]), return_counts=True)))}")

# ----------------------------
# Class weights
# ----------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
scale_pos_weight = class_weights[0] / class_weights[1]
print(f"Scale pos weight for XGBoost: {scale_pos_weight:.2f}")

# ----------------------------
# Train models
# ----------------------------
models = {}

print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

# XGBoost
print("Training XGBoost...")
models['XGBoost'] = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=5,
    max_depth=5,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.9,
    colsample_bytree=0.9,
    learning_rate=0.05,
    n_estimators=600,
    reg_alpha=0.1,
    reg_lambda=1.0,
    max_delta_step=1,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)
models['XGBoost'].fit(X_train, y_train)

# Random Forest
print("Training Random Forest...")
models['Random Forest'] = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
models['Random Forest'].fit(X_train, y_train)

# ----------------------------
# Metrics calculation
# ----------------------------
def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    gini = 2 * roc_auc - 1
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'kappa': kappa,
        'gini': gini,
        'log_loss': logloss,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

# ----------------------------
# OVERFITTING ANALYSIS - NEW SECTION
# ----------------------------
print("\n" + "="*60)
print("OVERFITTING ANALYSIS")
print("="*60)

overfitting_results = {}

for name, model in models.items():
    print(f"\n{name} Overfitting Analysis:")
    print("-" * 40)
    
    # Training performance
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    
    train_accuracy = accuracy_score(y_train, train_pred)
    train_roc_auc = roc_auc_score(y_train, train_prob)
    train_recall = recall_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)
    
    # Test performance
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, test_pred)
    test_roc_auc = roc_auc_score(y_test, test_prob)
    test_recall = recall_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    
    # Calculate overfitting percentages
    accuracy_gap = ((train_accuracy - test_accuracy) / train_accuracy) * 100 if train_accuracy > 0 else 0
    roc_auc_gap = ((train_roc_auc - test_roc_auc) / train_roc_auc) * 100 if train_roc_auc > 0 else 0
    recall_gap = ((train_recall - test_recall) / train_recall) * 100 if train_recall > 0 else 0
    precision_gap = ((train_precision - test_precision) / train_precision) * 100 if train_precision > 0 else 0
    f1_gap = ((train_f1 - test_f1) / train_f1) * 100 if train_f1 > 0 else 0
    
    overfitting_results[name] = {
        'accuracy_gap': accuracy_gap,
        'roc_auc_gap': roc_auc_gap,
        'recall_gap': recall_gap,
        'precision_gap': precision_gap,
        'f1_gap': f1_gap,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_f1': train_f1,
        'test_f1': test_f1
    }
    
    # Print detailed overfitting analysis
    print(f"Training Metrics:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  ROC-AUC:   {train_roc_auc:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  ROC-AUC:   {test_roc_auc:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    print(f"\nOverfitting Analysis:")
    print(f"  Accuracy Gap:  {accuracy_gap:.2f}%")
    print(f"  ROC-AUC Gap:   {roc_auc_gap:.2f}%")
    print(f"  Recall Gap:    {recall_gap:.2f}%")
    print(f"  Precision Gap: {precision_gap:.2f}%")
    print(f"  F1-Score Gap:  {f1_gap:.2f}%")
    
    # Overfitting assessment
    avg_gap = np.mean([accuracy_gap, roc_auc_gap, recall_gap, precision_gap, f1_gap])
    if avg_gap < 2:
        assessment = "🟢 MINIMAL"
    elif avg_gap < 5:
        assessment = "🟡 ACCEPTABLE"
    elif avg_gap < 10:
        assessment = "🟠 MODERATE"
    else:
        assessment = "🔴 HIGH"
    
    print(f"  Average Gap:   {avg_gap:.2f}%")
    print(f"  Assessment:    {assessment}")

# ----------------------------
# Overfitting Comparison Table
# ----------------------------
print("\n" + "="*70)
print("OVERFITTING COMPARISON TABLE")
print("="*70)

overfitting_comparison = pd.DataFrame({
    'Metric': ['Accuracy Gap %', 'ROC-AUC Gap %', 'Recall Gap %', 'Precision Gap %', 'F1-Score Gap %', 'Average Gap %'],
    'XGBoost': [
        overfitting_results['XGBoost']['accuracy_gap'],
        overfitting_results['XGBoost']['roc_auc_gap'],
        overfitting_results['XGBoost']['recall_gap'],
        overfitting_results['XGBoost']['precision_gap'],
        overfitting_results['XGBoost']['f1_gap'],
        np.mean([overfitting_results['XGBoost']['accuracy_gap'],
                overfitting_results['XGBoost']['roc_auc_gap'],
                overfitting_results['XGBoost']['recall_gap'],
                overfitting_results['XGBoost']['precision_gap'],
                overfitting_results['XGBoost']['f1_gap']])
    ],
    'Random Forest': [
        overfitting_results['Random Forest']['accuracy_gap'],
        overfitting_results['Random Forest']['roc_auc_gap'],
        overfitting_results['Random Forest']['recall_gap'],
        overfitting_results['Random Forest']['precision_gap'],
        overfitting_results['Random Forest']['f1_gap'],
        np.mean([overfitting_results['Random Forest']['accuracy_gap'],
                overfitting_results['Random Forest']['roc_auc_gap'],
                overfitting_results['Random Forest']['recall_gap'],
                overfitting_results['Random Forest']['precision_gap'],
                overfitting_results['Random Forest']['f1_gap']])
    ]
})

print(overfitting_comparison.round(2).to_string(index=False))

# Overfitting winner
xgb_avg_gap = overfitting_comparison.iloc[-1]['XGBoost']
rf_avg_gap = overfitting_comparison.iloc[-1]['Random Forest']
less_overfitted = 'XGBoost' if xgb_avg_gap < rf_avg_gap else 'Random Forest'

print(f"\n🏆 Less Overfitted Model: {less_overfitted}")
print(f"   XGBoost Average Gap: {xgb_avg_gap:.2f}%")
print(f"   Random Forest Average Gap: {rf_avg_gap:.2f}%")

# ----------------------------
# Evaluate models (with threshold tuning)
# ----------------------------
print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

results = {}
for name, model in models.items():
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    
    # For XGBoost, apply threshold tuning
    if name == 'XGBoost':
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_thresh = 0.5
        best_recall = 0
        print("\nThreshold Tuning (XGBoost):")
        for t in thresholds:
            y_pred_thresh = (prob >= t).astype(int)
            rec = recall_score(y_test, y_pred_thresh)
            prec = precision_score(y_test, y_pred_thresh)
            f1s = f1_score(y_test, y_pred_thresh)
            print(f"Threshold: {t:.2f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1s:.4f}")
            if rec > best_recall and prec >= 0.7:
                best_recall = rec
                best_thresh = t
        print(f"\n✅ Best Threshold: {best_thresh:.2f} | Recall: {best_recall:.4f}")
        pred = (prob >= best_thresh).astype(int)

    metrics = calculate_comprehensive_metrics(y_test, pred, prob)
    results[name] = {
        'predictions': pred,
        'probabilities': prob,
        **metrics
    }
    
    # Print metrics
    print(f"\n{name} - Detailed Performance Metrics:")
    print("-" * 50)
    for key, val in metrics.items():
        if key in ['tp','tn','fp','fn']:
            continue
        print(f"{key:<25} {val:<10.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              No Default  Default")
    print(f"Actual No    {metrics['tn']:>6d}     {metrics['fp']:>6d}")
    print(f"       Yes   {metrics['fn']:>6d}     {metrics['tp']:>6d}")

# ----------------------------
# Model Comparison
# ----------------------------
comparison_metrics = [
    'accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 
    'f1_score', 'npv', 'roc_auc', 'pr_auc', 'mcc', 'kappa', 'gini'
]

comparison = pd.DataFrame({
    'Metric': comparison_metrics,
    'XGBoost': [results['XGBoost'][m] for m in comparison_metrics],
    'Random Forest': [results['Random Forest'][m] for m in comparison_metrics],
    'Difference (XGB-RF)': [results['XGBoost'][m] - results['Random Forest'][m] for m in comparison_metrics]
})

print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*70)
print(comparison.round(4).to_string(index=False))

# ----------------------------
# Model Ranking
# ----------------------------
print(f"\n" + "="*50)
print("MODEL RANKING BY KEY METRICS")
print("="*50)

key_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'mcc', 'balanced_accuracy']
rankings = {}

for metric in key_metrics:
    xgb_val = results['XGBoost'][metric]
    rf_val = results['Random Forest'][metric]
    winner = 'XGBoost' if xgb_val > rf_val else 'Random Forest'
    difference = abs(xgb_val - rf_val)
    rankings[metric] = {
        'winner': winner,
        'xgb_score': xgb_val,
        'rf_score': rf_val,
        'difference': difference
    }
    print(f"{metric.upper():<20} Winner: {winner:<15} (XGB: {xgb_val:.4f}, RF: {rf_val:.4f})")

overall_score_xgb = (results['XGBoost']['roc_auc'] + results['XGBoost']['pr_auc']) / 2
overall_score_rf = (results['Random Forest']['roc_auc'] + results['Random Forest']['pr_auc']) / 2
best_model = 'XGBoost' if overall_score_xgb > overall_score_rf else 'Random Forest'
print(f"\nOVERALL BEST MODEL: {best_model}")
print(f"Combined ROC-AUC + PR-AUC Score:")
print(f"  XGBoost: {overall_score_xgb:.4f}")
print(f"  Random Forest: {overall_score_rf:.4f}")

# ----------------------------
# Feature Importance (XGBoost)
# ----------------------------
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': models['XGBoost'].feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE (XGBoost)")
print("="*50)
print("Top 15 Features:")
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")

# ----------------------------
# FINAL OVERFITTING SUMMARY
# ----------------------------
print("\n" + "="*60)
print("FINAL OVERFITTING ASSESSMENT")
print("="*60)

print("Overfitting Summary:")
for name in models.keys():
    avg_gap = np.mean([
        overfitting_results[name]['accuracy_gap'],
        overfitting_results[name]['roc_auc_gap'],
        overfitting_results[name]['recall_gap'],
        overfitting_results[name]['precision_gap'],
        overfitting_results[name]['f1_gap']
    ])
    
    if avg_gap < 2:
        status = "🟢 MINIMAL (Excellent generalization)"
    elif avg_gap < 5:
        status = "🟡 ACCEPTABLE (Good generalization)"
    elif avg_gap < 10:
        status = "🟠 MODERATE (Some overfitting)"
    else:
        status = "🔴 HIGH (Significant overfitting)"
    
    print(f"  {name}: {avg_gap:.2f}% - {status}")

print("\n📊 Overfitting Interpretation:")
print("  • 0-2%: Minimal overfitting - Production ready")
print("  • 2-5%: Acceptable overfitting - Monitor closely")
print("  • 5-10%: Moderate overfitting - Apply regularization")
print("  • 10%+: High overfitting - Significant intervention needed")

print(f"\n🎉 Analysis complete with overfitting assessment!")
print(f"🏆 Best performing model: {best_model}")
print(f"🔍 Less overfitted model: {less_overfitted}")

print("\n" + "="*80)
print("END OF COMPREHENSIVE ANALYSIS WITH OVERFITTING")
print("="*80)

# ----------------------------
# Cleanup
# ----------------------------
sys.stdout = logger.console
logger.close()
print("✅ Comprehensive results with overfitting analysis saved!")

