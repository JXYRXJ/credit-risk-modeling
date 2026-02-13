# Credit Risk Modeling

Machine learning models for credit risk prediction and classification using CatBoost, XGBoost, and ensemble methods.

## Overview

This project implements and compares multiple machine learning algorithms for credit risk assessment, predicting the probability of credit default using various borrower features. The models help support data-driven lending decisions by identifying high-risk borrowers.

## Dataset

**Credit Risk Dataset**  
Source: [Kaggle - Credit Risk Dataset by Laotse](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

The dataset contains various features about borrowers including:
- Personal information (age, income, employment length)
- Loan details (amount, interest rate, purpose)
- Credit history
- Home ownership status
- Default status (target variable)

**Credit:** Special thanks to [Laotse](https://www.kaggle.com/laotse) for providing this comprehensive credit risk dataset on Kaggle.

## Project Structure

```
credit-risk-modeling/
├── catboost_example/          # CatBoost implementation
│   ├── model.py              # CatBoost model training and evaluation
│   ├── simple.ipynb          # Interactive notebook
│   ├── credit_risk_dataset.csv
│   ├── catboost_credit_risk_results.txt
│   ├── catboost_optimized_results.txt
│   └── shap_*.png            # SHAP visualization plots
│
├── xgboost_example/           # XGBoost implementation
│   ├── model.py              # XGBoost model training
│   ├── data_processing.py    # Data preprocessing pipeline
│   ├── simple.ipynb          # Interactive notebook
│   ├── credit_risk_dataset.csv
│   ├── credit_risk_model_results.txt
│   └── best_result.txt
│
├── ensemble_example/          # Ensemble methods
│   ├── simple.ipynb          # Ensemble model implementation
│   └── credit_risk_dataset.csv
│
├── .gitignore
└── README.md
```

## Models Implemented

### 1. CatBoost Classifier
- Optimized hyperparameters using cross-validation
- Feature importance analysis
- SHAP (SHapley Additive exPlanations) for model interpretability:
  - Feature importance plots
  - Summary plots
  - Individual prediction analysis
  - Partial dependence plots
  - Feature impact analysis

### 2. XGBoost Classifier
- Custom data preprocessing pipeline
- Hyperparameter tuning
- Performance comparison with baseline models
- Detailed evaluation metrics

### 3. Ensemble Methods
- Combining multiple algorithms
- Improved prediction accuracy through model aggregation

## Features

- **Comprehensive Model Comparison**: Side-by-side evaluation of different gradient boosting algorithms
- **Explainable AI**: SHAP visualizations for model interpretability
- **Cross-validation**: Robust model evaluation and hyperparameter optimization
- **Production-ready**: Modular code structure with separate data processing and modeling scripts
- **Interactive Notebooks**: Jupyter notebooks for exploratory analysis and experimentation

## Requirements

```
python>=3.8
catboost
xgboost
scikit-learn
pandas
numpy
matplotlib
seaborn
shap
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JXYRXJ/credit-risk-modeling.git
cd credit-risk-modeling
```

2. Install required packages:
```bash
pip install catboost xgboost scikit-learn pandas numpy matplotlib seaborn shap jupyter
```

## Usage

### CatBoost Model
```bash
cd catboost_example
python model.py
```

Or explore interactively:
```bash
jupyter notebook simple.ipynb
```

### XGBoost Model
```bash
cd xgboost_example
python model.py
```

### Ensemble Model
```bash
cd ensemble_example
jupyter notebook simple.ipynb
```

## Results

Detailed results for each model can be found in:
- `catboost_example/catboost_optimized_results.txt` - CatBoost performance metrics
- `xgboost_example/best_result.txt` - XGBoost performance metrics
- `compare.txt` - Comparative analysis across models

## Model Interpretability

The CatBoost implementation includes comprehensive SHAP analysis:
- **Feature Importance**: Identifies which features most influence credit risk predictions
- **Summary Plots**: Visual representation of feature impact across all predictions
- **Individual Analysis**: Detailed breakdown of predictions for specific cases
- **Partial Dependence**: Shows how individual features affect predictions

## License

This project is available for educational and research purposes.

## Acknowledgments

- **Dataset**: [Laotse](https://www.kaggle.com/laotse) - [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Libraries**: CatBoost, XGBoost, scikit-learn, SHAP

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
