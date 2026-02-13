import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def preprocess_credit_data(df):
    """
    Comprehensive preprocessing pipeline for credit risk dataset
    """
    df_processed = df.copy()
    
    print("=== STARTING DATA PREPROCESSING ===")
    
    # 1. HANDLE OUTLIERS
    print("\n1. Handling Outliers...")
    
    # Age outliers (cap at 18-85)
    df_processed['person_age'] = df_processed['person_age'].clip(18, 85)
    print(f"   - Capped person_age to 18-85 range")
    
    # Income outliers (winsorize at 99th percentile)
    income_99th = df_processed['person_income'].quantile(0.99)
    df_processed['person_income'] = df_processed['person_income'].clip(upper=income_99th)
    print(f"   - Winsorized person_income at 99th percentile: {income_99th}")
    
    # Employment length outliers (cap at 50)
    df_processed.loc[df_processed['person_emp_length'] > 50, 'person_emp_length'] = 50
    print(f"   - Capped person_emp_length at 50 years")
    
    # 2. HANDLE MISSING VALUES
    print("\n2. Handling Missing Values...")
    
    # Create missing indicators (important for credit models)
    df_processed['emp_length_missing'] = df_processed['person_emp_length'].isnull().astype(int)
    df_processed['int_rate_missing'] = df_processed['loan_int_rate'].isnull().astype(int)
    
    # Impute employment length with median
    emp_length_median = df_processed['person_emp_length'].median()
    df_processed['person_emp_length'].fillna(emp_length_median, inplace=True)
    
    # Impute interest rate using loan_grade median (more sophisticated imputation)
    for grade in df_processed['loan_grade'].unique():
        grade_median = df_processed[df_processed['loan_grade'] == grade]['loan_int_rate'].median()
        mask = (df_processed['loan_grade'] == grade) & (df_processed['loan_int_rate'].isnull())
        df_processed.loc[mask, 'loan_int_rate'] = grade_median
    
    # 3. FEATURE ENGINEERING
    print("\n3. Feature Engineering...")
    
    # Age bins
    df_processed['age_group'] = pd.cut(df_processed['person_age'], 
                                     bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['young', 'young_adult', 'middle', 'mature', 'senior'])
    
    # Income transformations
    df_processed['log_income'] = np.log1p(df_processed['person_income'])
    df_processed['income_bracket'] = pd.cut(df_processed['person_income'],
                                          bins=[0, 30000, 50000, 75000, 100000, np.inf],
                                          labels=['low', 'lower_mid', 'mid', 'upper_mid', 'high'])
    
    # Employment length bins
    df_processed['emp_length_group'] = pd.cut(df_processed['person_emp_length'],
                                            bins=[-1, 0, 2, 5, 10, np.inf],
                                            labels=['zero', 'short', 'medium', 'long', 'very_long'])
    
    # Loan amount transformations
    df_processed['log_loan_amnt'] = np.log1p(df_processed['loan_amnt'])
    df_processed['loan_bracket'] = pd.cut(df_processed['loan_amnt'],
                                        bins=[0, 5000, 10000, 20000, np.inf],
                                        labels=['small', 'medium', 'large', 'very_large'])
    
    # Debt burden categories
    df_processed['debt_burden'] = pd.cut(df_processed['loan_percent_income'],
                                       bins=[0, 0.2, 0.4, 0.6, np.inf],
                                       labels=['low', 'moderate', 'high', 'extreme'])
    
    # Credit history bins
    df_processed['credit_hist_group'] = pd.cut(df_processed['cb_person_cred_hist_length'],
                                             bins=[0, 2, 5, 10, np.inf],
                                             labels=['very_short', 'short', 'medium', 'long'])
    
    # Interest rate bins
    df_processed['int_rate_group'] = pd.cut(df_processed['loan_int_rate'],
                                          bins=[0, 8, 12, 16, np.inf],
                                          labels=['low', 'medium', 'high', 'very_high'])
    
    # Advanced engineered features
    df_processed['affordability_ratio'] = 1 - df_processed['loan_percent_income']
    df_processed['income_per_year_employed'] = df_processed['person_income'] / (df_processed['person_emp_length'] + 1)
    df_processed['loan_to_credit_hist'] = df_processed['loan_amnt'] / (df_processed['cb_person_cred_hist_length'] + 1)
    df_processed['age_to_credit_hist_ratio'] = df_processed['person_age'] / (df_processed['cb_person_cred_hist_length'] + 1)
    
    # Risk indicators
    df_processed['thin_file'] = ((df_processed['cb_person_cred_hist_length'] <= 2) & 
                               (df_processed['cb_person_default_on_file'] == 'N')).astype(int)
    df_processed['high_debt_burden'] = (df_processed['loan_percent_income'] > 0.4).astype(int)
    df_processed['has_prior_default'] = (df_processed['cb_person_default_on_file'] == 'Y').astype(int)
    
    # Interaction features
    grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    df_processed['income_grade_interaction'] = df_processed['log_income'] * df_processed['loan_grade'].map(grade_mapping)
    
    return df_processed

def encode_categorical_features(df_processed):
    """
    Encode all categorical variables
    """
    df_encoded = df_processed.copy()
    
    print("4. Encoding Categorical Variables...")
    
    # Binary encoding
    df_encoded['cb_person_default_on_file'] = df_encoded['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
    
    # Ordinal encoding for loan grade
    grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    df_encoded['loan_grade_ordinal'] = df_encoded['loan_grade'].map(grade_mapping)
    
    # One-hot encoding for nominal variables
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade']
    categorical_engineered_cols = ['age_group', 'income_bracket', 'emp_length_group', 
                                 'loan_bracket', 'debt_burden', 'credit_hist_group', 'int_rate_group']
    
    all_categorical = categorical_cols + categorical_engineered_cols
    df_encoded = pd.get_dummies(df_encoded, columns=all_categorical, prefix=all_categorical, drop_first=False)
    
    return df_encoded

def prepare_for_modeling(df_encoded, target_col='loan_status', test_size=0.2, random_state=42):
    """
    Final preparation for modeling including scaling and train-test split
    """
    print("5. Preparing for Modeling...")
    
    # Separate features and target
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    # Train-test split with stratification (important for imbalanced classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numerical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    print(f"   - Training set: {X_train_scaled.shape}")
    print(f"   - Test set: {X_test_scaled.shape}")
    print(f"   - Class distribution in training: {y_train.value_counts().to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Usage example:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('credit_risk_dataset.csv')
    
    # Apply preprocessing pipeline
    df_processed = preprocess_credit_data(df)
    df_encoded = encode_categorical_features(df_processed)
    X_train, X_test, y_train, y_test, scaler = prepare_for_modeling(df_encoded)
    
    # Calculate class weights for handling imbalance
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"\n6. Class Imbalance Handling:")
    print(f"   - Class weights: {class_weight_dict}")
    print(f"   - Scale_pos_weight for XGBoost: {class_weights[0]/class_weights[1]:.2f}")
    
    # Save processed data
    X_train.to_csv('X_train_processed.csv', index=False)
    X_test.to_csv('X_test_processed.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print(f"\n✅ Preprocessing Complete!")
    print(f"   - Original features: 12")
    print(f"   - Final features: {X_train.shape[1]}")
    print(f"   - Files saved: X_train_processed.csv, X_test_processed.csv, y_train.csv, y_test.csv")
