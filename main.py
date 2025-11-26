import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_fscore_support, accuracy_score)
import joblib
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv" 
OUTPUT_DIR = "output"
RANDOM_STATE = 42
TEST_SIZE = 0.2
os.makedirs(OUTPUT_DIR, exist_ok=True)
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved figure: {path}")
def compute_hr_metrics(df):
    metrics = {}
    if 'Attrition' in df.columns:
        total = len(df)
        left = df['Attrition'].map({'Yes':1, 'No':0}).sum()
        metrics['attrition_rate'] = left / total
    if 'YearsAtCompany' in df.columns:
        metrics['average_tenure_years'] = df['YearsAtCompany'].mean()
    sat_cols = [c for c in ['JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','RelationshipSatisfaction','JobInvolvement'] if c in df.columns]
    if sat_cols:
        metrics['satisfaction_index'] = df[sat_cols].mean(axis=1).mean()
    if 'YearsSinceLastPromotion' in df.columns:
        metrics['avg_years_since_last_promotion'] = df['YearsSinceLastPromotion'].mean()
    if 'MonthlyIncome' in df.columns:
        metrics['average_monthly_income'] = df['MonthlyIncome'].mean()

    return metrics

def quick_eda(df):
    print('\n=== Head & Info ===')
    print(df.head())
    print('\n=== Dtypes ===')
    print(df.dtypes)
    print('\n=== Missing values ===')
    print(df.isna().sum()[lambda x: x>0])

    if 'Attrition' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x='Attrition', ax=ax)
        ax.set_title('Attrition Counts')
        save_fig(fig, 'attrition_counts.png')
        plt.close(fig)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0)
        ax.set_title('Numeric Feature Correlation')
        save_fig(fig, 'correlation_heatmap.png')
        plt.close(fig)


def make_preprocessor(X):
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    for t in ['Attrition']:
        if t in numeric_cols: numeric_cols.remove(t)
        if t in cat_cols: cat_cols.remove(t)

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')

    return preprocessor, numeric_cols, cat_cols

def train_and_evaluate(X, y):
    preprocessor, num_cols, cat_cols = make_preprocessor(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Logistic Regression pipeline
    pipe_lr = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    # Random Forest pipeline
    pipe_rf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # Simple hyperparameter grids
    param_grid_lr = {
        'clf__C': [0.01, 0.1, 1, 10]
    }
    param_grid_rf = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 6, 12],
        'clf__min_samples_split': [2, 5]
    }

    # Grid search (use CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv, scoring='roc_auc', n_jobs=-1)
    gs_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv, scoring='roc_auc', n_jobs=-1)

    print('\nTraining Logistic Regression grid...')
    gs_lr.fit(X_train, y_train)
    print('Best LR params:', gs_lr.best_params_)

    print('\nTraining Random Forest grid...')
    gs_rf.fit(X_train, y_train)
    print('Best RF params:', gs_rf.best_params_)

    # Evaluate function
    def evaluate_model(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"\n{name} — Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0,1],[0,1],'--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        save_fig(fig, f'roc_{name}.png')
        plt.close(fig)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        save_fig(fig, f'cm_{name}.png')
        plt.close(fig)

        return {'accuracy': acc, 'roc_auc': auc, 'confusion_matrix': cm}

    # Evaluate best estimators
    results = {}
    results['logistic'] = evaluate_model(gs_lr.best_estimator_, X_test, y_test, 'LogisticRegression')
    results['random_forest'] = evaluate_model(gs_rf.best_estimator_, X_test, y_test, 'RandomForest')

    # Feature importance for Random Forest (requires access to transformed feature names)
    # Build transformed feature names
    pre = gs_rf.best_estimator_.named_steps['pre']
    # Get numeric features
    num_features = num_cols
    # Get categorical feature names from OneHotEncoder
    cat_encoder = pre.named_transformers_['cat'].named_steps['onehot']
    try:
        cat_ohe_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    except Exception:
        # fallback
        cat_ohe_names = []
    feature_names = num_features + cat_ohe_names

    rf_model = gs_rf.best_estimator_.named_steps['clf']
    importances = rf_model.feature_importances_

    # Keep top 25 features
    idx_sorted = np.argsort(importances)[::-1]
    top_n = min(25, len(importances))
    top_idx = idx_sorted[:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = importances[top_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, max(4, top_n*0.3)))
    sns.barplot(x=top_importances, y=top_features, ax=ax)
    ax.set_title('Top feature importances (Random Forest)')
    save_fig(fig, 'rf_top_features.png')
    plt.close(fig)

    # Save models
    joblib.dump(gs_lr.best_estimator_, os.path.join(OUTPUT_DIR,'best_logistic.pkl'))
    joblib.dump(gs_rf.best_estimator_, os.path.join(OUTPUT_DIR,'best_rf.pkl'))
    print('Saved best models to output/')

    return results, (feature_names, importances)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'Data file not found: {DATA_PATH} — please update DATA_PATH')

    df = load_data(DATA_PATH)

    # Compute HR metrics
    metrics = compute_hr_metrics(df)
    print('\n=== HR Key Metrics ===')
    for k,v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Quick EDA (saves plots)
    quick_eda(df)

    # Prepare target and features
    if 'Attrition' not in df.columns:
        raise KeyError('Target column Attrition not found in dataset')

    y = df['Attrition'].map({'Yes':1, 'No':0})
    X = df.drop(columns=['Attrition', 'EmployeeNumber'] if 'EmployeeNumber' in df.columns else ['Attrition'])

    # Train and evaluate
    results, rf_info = train_and_evaluate(X, y)

    print('\n=== Done ===')
    print('Results summary:')
    print(results)

if __name__ == '__main__':
    main()
