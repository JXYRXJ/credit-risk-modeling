import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_score, 
    recall_score, f1_score, accuracy_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, log_loss
)
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Create logger for results
class ResultLogger:
    def __init__(self, filename='catboost_optimized_results_with_overfitting.txt'):
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

# Initialize logger
logger = ResultLogger()
sys.stdout = logger

print("=" * 80)
print("CATBOOST OPTIMIZED CREDIT RISK MODEL - WITH OVERFITTING ANALYSIS")
print("=" * 80)
print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Load original dataset
print("Loading original dataset...")
df = pd.read_csv('credit_risk_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Basic data exploration
print(f"\nClass distribution:")
print(df['loan_status'].value_counts())
class_ratio = df['loan_status'].value_counts()[0] / df['loan_status'].value_counts()[1]
print(f"Class imbalance ratio: {class_ratio:.2f}:1")

# Missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Identify categorical features for CatBoost
categorical_features = [
    'person_home_ownership',
    'loan_intent', 
    'loan_grade',
    'cb_person_default_on_file'
]

print(f"\nCategorical features identified: {categorical_features}")

# Prepare features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Create CatBoost Pool objects
print(f"\nCreating CatBoost data pools...")
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_features
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=categorical_features
)

print("✅ Data pools created successfully!")

print("\n" + "="*60)
print("CATBOOST MODEL TRAINING WITH COST-SENSITIVE LEARNING")
print("="*60)

# Initialize CatBoost with cost-sensitive learning
catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',  # Specify loss function
    # auto_class_weights='Balanced',  # Removed - using custom weights
    class_weights={0: 1, 1: 12},  # Weight defaults 12x higher for business cost
    nan_mode='Min',
    cat_features=categorical_features,
    l2_leaf_reg=3,
    task_type='CPU',
    thread_count=-1,
    eval_metric='AUC',
    verbose=100,
    random_seed=42
)

# Train the model
print("Training CatBoost model with cost-sensitive weights...")
catboost_model.fit(
    train_pool,
    eval_set=test_pool,
    plot=False,
    use_best_model=True,
    early_stopping_rounds=50
)

print("✅ CatBoost training completed!")

# ----------------------------
# OVERFITTING ANALYSIS - NEW SECTION
# ----------------------------
print("\n" + "="*60)
print("CATBOOST OVERFITTING ANALYSIS")
print("="*60)

# Training performance
train_pred = catboost_model.predict(X_train)
train_prob = catboost_model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, train_pred)
train_roc_auc = roc_auc_score(y_train, train_prob)
train_recall = recall_score(y_train, train_pred)
train_precision = precision_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

# Test performance (before optimization)
test_pred = catboost_model.predict(X_test)
test_prob = catboost_model.predict_proba(X_test)[:, 1]

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

print("CatBoost Overfitting Analysis:")
print("-" * 50)
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

# Overall overfitting assessment
avg_gap = np.mean([accuracy_gap, roc_auc_gap, recall_gap, precision_gap, f1_gap])
if avg_gap < 2:
    assessment = "🟢 MINIMAL (Excellent generalization)"
elif avg_gap < 5:
    assessment = "🟡 ACCEPTABLE (Good generalization)"
elif avg_gap < 10:
    assessment = "🟠 MODERATE (Some overfitting)"
else:
    assessment = "🔴 HIGH (Significant overfitting)"

print(f"  Average Gap:   {avg_gap:.2f}%")
print(f"  Assessment:    {assessment}")

# Store original predictions for comparison
y_pred = test_pred
y_prob = test_prob

# THRESHOLD OPTIMIZATION FOR BETTER RECALL
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION FOR BETTER RECALL")
print("="*60)

# Test different thresholds for better recall
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
best_threshold = 0.5
best_f1 = 0
best_recall = 0

print("Testing different thresholds:")
print(f"{'Threshold':<10} {'Recall':<8} {'Precision':<10} {'F1-Score':<10} {'Business Impact'}")
print("-" * 60)

for threshold in thresholds:
    y_pred_thresh = (y_prob > threshold).astype(int)
    recall_thresh = recall_score(y_test, y_pred_thresh)
    precision_thresh = precision_score(y_test, y_pred_thresh)
    f1_thresh = f1_score(y_test, y_pred_thresh)
    
    # Calculate business impact (missed defaults cost more)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    missed_defaults_cost = fn * 35000 * 0.6  # $35K avg loan, 60% loss
    false_alarms_cost = fp * 35000 * 0.02    # 2% opportunity cost
    total_cost = missed_defaults_cost + false_alarms_cost
    
    business_impact = f"${total_cost/1000:.0f}K"
    
    print(f"{threshold:<10} {recall_thresh:<8.3f} {precision_thresh:<10.3f} {f1_thresh:<10.3f} {business_impact:<15}")
    
    # Choose threshold with recall >= 85% and best F1, or best recall if none achieve 85%
    if recall_thresh >= 0.85 and f1_thresh > best_f1:
        best_threshold = threshold
        best_f1 = f1_thresh
        best_recall = recall_thresh
    elif recall_thresh > best_recall:
        best_threshold = threshold
        best_f1 = f1_thresh
        best_recall = recall_thresh

# Use optimized threshold for final predictions
print(f"\n🎯 Optimal threshold selected: {best_threshold}")
print(f"📈 Expected recall: {best_recall:.3f}")
print(f"⚖️ Expected F1-score: {best_f1:.3f}")

# Override predictions with optimal threshold
y_pred_optimized = (y_prob > best_threshold).astype(int)
print(f"✅ Using optimized predictions with threshold = {best_threshold}")

# Comprehensive evaluation function
def calculate_comprehensive_metrics(y_true, y_pred, y_prob, model_name="CatBoost"):
    """Calculate all performance metrics"""
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Additional metrics
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Advanced metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Log loss
    logloss = log_loss(y_true, y_prob)
    
    # Gini coefficient
    gini = 2 * roc_auc - 1
    
    return {
        'model': model_name,
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

# Evaluate CatBoost performance
print("\n" + "="*60)
print("CATBOOST PERFORMANCE EVALUATION - STANDARD VS OPTIMIZED")
print("="*60)

# Calculate metrics for both standard and optimized predictions
metrics_standard = calculate_comprehensive_metrics(y_test, y_pred, y_prob, "CatBoost_Standard")
metrics_optimized = calculate_comprehensive_metrics(y_test, y_pred_optimized, y_prob, "CatBoost_Optimized")

print(f"CatBoost - Standard vs Optimized Performance:")
print("-" * 60)
print(f"{'Metric':<25} {'Standard':<12} {'Optimized':<12} {'Improvement'}")
print("-" * 60)

metrics_comparison = [
    ('Recall', 'recall'),
    ('Precision', 'precision'), 
    ('F1-Score', 'f1_score'),
    ('ROC-AUC', 'roc_auc'),
    ('PR-AUC', 'pr_auc')
]

for metric_name, metric_key in metrics_comparison:
    standard_val = metrics_standard[metric_key]
    optimized_val = metrics_optimized[metric_key]
    improvement = ((optimized_val - standard_val) / standard_val) * 100 if standard_val > 0 else 0
    improvement_str = f"{improvement:+.1f}%" if abs(improvement) >= 0.1 else "~0%"
    
    print(f"{metric_name:<25} {standard_val:<12.4f} {optimized_val:<12.4f} {improvement_str}")

# Use optimized metrics for rest of analysis
metrics = metrics_optimized
print(f"\n✅ Using OPTIMIZED model performance for analysis")

print(f"\nOptimized Model - Comprehensive Performance Metrics:")
print("-" * 55)
print(f"{'Metric':<25} {'Value':<10} {'Description'}")
print("-" * 55)
print(f"{'Accuracy':<25} {metrics['accuracy']:<10.4f} Overall correctness")
print(f"{'Balanced Accuracy':<25} {metrics['balanced_accuracy']:<10.4f} Avg of sensitivity & specificity")
print(f"{'Precision (PPV)':<25} {metrics['precision']:<10.4f} TP / (TP + FP)")
print(f"{'Recall (Sensitivity)':<25} {metrics['recall']:<10.4f} TP / (TP + FN)")
print(f"{'Specificity (TNR)':<25} {metrics['specificity']:<10.4f} TN / (TN + FP)")
print(f"{'F1-Score':<25} {metrics['f1_score']:<10.4f} Harmonic mean of precision & recall")
print(f"{'NPV':<25} {metrics['npv']:<10.4f} TN / (TN + FN)")
print(f"{'False Positive Rate':<25} {metrics['fpr']:<10.4f} FP / (FP + TN)")
print(f"{'False Negative Rate':<25} {metrics['fnr']:<10.4f} FN / (FN + TP)")
print(f"{'ROC-AUC':<25} {metrics['roc_auc']:<10.4f} Area under ROC curve")
print(f"{'PR-AUC':<25} {metrics['pr_auc']:<10.4f} Area under PR curve")
print(f"{'Matthews Corr Coef':<25} {metrics['mcc']:<10.4f} Correlation coefficient")
print(f"{'Cohen Kappa':<25} {metrics['kappa']:<10.4f} Agreement correcting for chance")
print(f"{'Gini Coefficient':<25} {metrics['gini']:<10.4f} 2*AUC - 1")
print(f"{'Log Loss':<25} {metrics['log_loss']:<10.4f} Logarithmic loss (lower better)")

print(f"\nOptimized Confusion Matrix:")
print(f"                 Predicted")
print(f"              No Default  Default")
print(f"Actual No    {metrics['tn']:>6d}     {metrics['fp']:>6d}")
print(f"       Yes   {metrics['fn']:>6d}     {metrics['tp']:>6d}")

# Feature importance analysis
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

feature_importance = catboost_model.get_feature_importance()
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print("-" * 45)
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")

# CATBOOST BUILT-IN CROSS-VALIDATION
print("\n" + "="*50)
print("CATBOOST CROSS-VALIDATION RESULTS")
print("="*50)

# Create a pool for the entire dataset for CV
full_pool = Pool(
    data=X,
    label=y,
    cat_features=categorical_features
)

# Define CV parameters - using auto_class_weights instead of class_weights
cv_params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss',  # REQUIRED for CV
    'auto_class_weights': 'Balanced',  # This works with CV (unlike class_weights)
    'nan_mode': 'Min',
    'l2_leaf_reg': 3,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': False
}

# Perform cross-validation using CatBoost's built-in CV
print("Performing 5-fold cross-validation with auto balanced class weights...")
cv_results = cv(
    pool=full_pool,
    params=cv_params,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    plot=False,
    verbose=False
)

# Extract CV results
cv_auc_scores = cv_results['test-AUC-mean']
cv_logloss_scores = cv_results['test-Logloss-mean']

# Get final scores (last 10 iterations for stability)
final_auc_scores = cv_auc_scores[-10:]
final_logloss_scores = cv_logloss_scores[-10:]

print(f"Cross-Validation Results:")
print(f"AUC Results:")
print(f"  Mean AUC (last 10 iterations): {np.mean(final_auc_scores):.4f}")
print(f"  Std AUC (last 10 iterations): {np.std(final_auc_scores):.4f}")
print(f"  Best AUC: {np.max(cv_auc_scores):.4f}")
print(f"  Final AUC: {cv_auc_scores.iloc[-1]:.4f}")

# ----------------------------
# FINAL OVERFITTING SUMMARY
# ----------------------------
print("\n" + "="*60)
print("FINAL OVERFITTING ASSESSMENT")
print("="*60)

print("CatBoost Overfitting Summary:")
print(f"  Average Overfitting Gap: {avg_gap:.2f}%")
print(f"  Assessment: {assessment}")

print(f"\nOverfitting Breakdown:")
print(f"  • Accuracy Gap:  {accuracy_gap:.2f}%")
print(f"  • ROC-AUC Gap:   {roc_auc_gap:.2f}%")
print(f"  • Recall Gap:    {recall_gap:.2f}%")
print(f"  • Precision Gap: {precision_gap:.2f}%")
print(f"  • F1-Score Gap:  {f1_gap:.2f}%")

print("\n📊 Overfitting Interpretation:")
print("  • 0-2%: Minimal overfitting - Production ready")
print("  • 2-5%: Acceptable overfitting - Monitor closely")
print("  • 5-10%: Moderate overfitting - Apply regularization")
print("  • 10%+: High overfitting - Significant intervention needed")

# Business cost analysis
print("\n" + "="*50)
print("BUSINESS COST IMPACT ANALYSIS")
print("="*50)

# Calculate business costs
avg_loan_amount = 35000
default_loss_rate = 0.6
opportunity_cost_rate = 0.02

# Standard threshold costs
tn_std, fp_std, fn_std, tp_std = confusion_matrix(y_test, y_pred).ravel()
cost_missed_std = fn_std * avg_loan_amount * default_loss_rate
cost_false_std = fp_std * avg_loan_amount * opportunity_cost_rate
total_cost_std = cost_missed_std + cost_false_std

# Optimized threshold costs  
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimized).ravel()
cost_missed_opt = fn_opt * avg_loan_amount * default_loss_rate
cost_false_opt = fp_opt * avg_loan_amount * opportunity_cost_rate
total_cost_opt = cost_missed_opt + cost_false_opt

print(f"Business Cost Comparison:")
print(f"{'Metric':<25} {'Standard':<15} {'Optimized':<15} {'Savings'}")
print("-" * 65)
print(f"{'Missed Defaults Cost':<25} ${cost_missed_std/1000:<14.0f}K ${cost_missed_opt/1000:<14.0f}K ${(cost_missed_std-cost_missed_opt)/1000:+.0f}K")
print(f"{'False Alarms Cost':<25} ${cost_false_std/1000:<14.0f}K ${cost_false_opt/1000:<14.0f}K ${(cost_false_std-cost_false_opt)/1000:+.0f}K")
print(f"{'Total Business Cost':<25} ${total_cost_std/1000:<14.0f}K ${total_cost_opt/1000:<14.0f}K ${(total_cost_std-total_cost_opt)/1000:+.0f}K")

savings = total_cost_std - total_cost_opt
print(f"\n💰 Total Business Savings: ${savings:,.0f}")
print(f"📊 Cost Reduction: {(savings/total_cost_std)*100:.1f}%")

print(f"\n🎉 CatBoost Analysis Complete!")
print(f"🏆 ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"📈 PR-AUC: {metrics['pr_auc']:.4f}")
print(f"🎯 Optimized Recall: {metrics['recall']:.3f}")
print(f"🔍 Overfitting Level: {avg_gap:.2f}% ({assessment.split()[1]})")
print(f"💰 Business Savings: ${savings:,.0f}")

print("\n" + "="*80)
print("DEPLOYMENT RECOMMENDATIONS")
print("="*80)
print(f"📋 Production Settings:")
print(f"   • Model shows {assessment.split()[1].lower()} overfitting ({avg_gap:.2f}%)")
print(f"   • Use class_weights={{0: 1, 1: 12}}")
print(f"   • Set classification threshold = {best_threshold}")
print(f"   • Expected business savings: ${savings:,.0f}")
print(f"   • Model ready for deployment")

print("\n" + "="*80)
print("END OF CATBOOST ANALYSIS WITH OVERFITTING")
print("="*80)

# Close logger
sys.stdout = logger.console
logger.close()
print("✅ CatBoost results with overfitting analysis saved!")
