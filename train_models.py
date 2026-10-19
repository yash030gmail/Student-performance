"""
Machine Learning Training Pipeline for Student Performance Prediction.
Implements:
  1. Regression Models to predict Average/Overall Score and Math Score.
  2. Classification Models to predict Academic Risk (Pass/Fail) and Performance Tiers.
  3. 5-Fold Cross Validation & Hyperparameter Tuning.
  4. Feature Importance & Model Explainability.
  5. Artifact & Model serialization.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Regression Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

DATA_PATH = os.path.join(os.path.dirname(__file__), "StudentsPerformance.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
VIS_DIR = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = df['total_score'] / 3.0
    
    # Binary Target: Passing all subjects (standard passing threshold: 50)
    df['passed_all'] = (
        (df['math score'] >= 50) & 
        (df['reading score'] >= 50) & 
        (df['writing score'] >= 50)
    ).astype(int)

    # Multi-class Target: Performance Tier
    def assign_tier(avg):
        if avg >= 80:
            return 'High Performer'
        elif avg >= 60:
            return 'Moderate Performer'
        else:
            return 'At-Risk'

    df['tier'] = df['average_score'].apply(assign_tier)

    # Features
    demographic_features = [
        'gender', 'race/ethnicity', 'parental level of education', 
        'lunch', 'test preparation course'
    ]

    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, demographic_features)
        ],
        remainder='passthrough'
    )

    X_demo = df[demographic_features]
    y_reg_avg = df['average_score']
    y_cls_pass = df['passed_all']

    # Train-test split (80/20) with random seed for reproducibility
    X_train_d, X_test_d, y_train_avg, y_test_avg = train_test_split(
        X_demo, y_reg_avg, test_size=0.20, random_state=42
    )

    _, _, y_train_pass, y_test_pass = train_test_split(
        X_demo, y_cls_pass, test_size=0.20, random_state=42, stratify=y_cls_pass
    )

    print("===============================================================")
    print(" 1. REGRESSION BENCHMARK: PREDICTING STUDENT OVERALL SCORE")
    print("===============================================================")
    
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        "Support Vector Regressor": SVR(C=10.0, epsilon=0.2),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=9)
    }

    reg_results = {}
    best_reg_model_name = None
    best_reg_r2 = -float("inf")
    best_reg_pipeline = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in regression_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # 5-fold Cross-validation
        cv_scores = cross_val_score(pipeline, X_train_d, y_train_avg, cv=kf, scoring='r2')
        pipeline.fit(X_train_d, y_train_avg)
        
        preds = pipeline.predict(X_test_d)
        mae = mean_absolute_error(y_test_avg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_avg, preds))
        r2 = r2_score(y_test_avg, preds)

        reg_results[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2_Test": round(float(r2), 4),
            "CV_R2_Mean": round(float(cv_scores.mean()), 4),
            "CV_R2_Std": round(float(cv_scores.std()), 4)
        }

        print(f"{name:25s} | R²: {r2:7.4f} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_reg_model_name = name
            best_reg_pipeline = pipeline

    print(f"\n>>> Best Regression Model: {best_reg_model_name} with R² = {best_reg_r2:.4f}")
    joblib.dump(best_reg_pipeline, os.path.join(MODEL_DIR, "best_overall_score_regressor.joblib"))

    # Also build multi-score regressor (predict Math, Reading, Writing all together)
    from sklearn.multioutput import MultiOutputRegressor
    multi_target = df[['math score', 'reading score', 'writing score']]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_demo, multi_target, test_size=0.20, random_state=42
    )
    multi_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('multi_reg', MultiOutputRegressor(Ridge(alpha=1.0)))
    ])
    multi_pipeline.fit(X_train_m, y_train_m)
    m_preds = multi_pipeline.predict(X_test_m)
    multi_r2 = r2_score(y_test_m, m_preds, multioutput='raw_values')
    print(f"Multi-Subject Prediction R² - Math: {multi_r2[0]:.4f}, Reading: {multi_r2[1]:.4f}, Writing: {multi_r2[2]:.4f}")
    joblib.dump(multi_pipeline, os.path.join(MODEL_DIR, "multi_subject_regressor.joblib"))

    # Also build Math score predictor using demographics + reading + writing scores
    X_math = df[demographic_features + ['reading score', 'writing score']]
    y_math = df['math score']
    X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(
        X_math, y_math, test_size=0.20, random_state=42
    )
    math_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), demographic_features),
            ('num', StandardScaler(), ['reading score', 'writing score'])
        ]
    )
    math_pipeline = Pipeline([
        ('prep', math_preprocessor),
        ('reg', Ridge(alpha=1.0))
    ])
    math_pipeline.fit(X_train_math, y_train_math)
    math_preds = math_pipeline.predict(X_test_math)
    math_r2 = r2_score(y_test_math, math_preds)
    math_mae = mean_absolute_error(y_test_math, math_preds)
    print(f"Math Score Prediction (using Demographics + Reading/Writing) | R²: {math_r2:.4f} | MAE: {math_mae:.3f}")
    joblib.dump(math_pipeline, os.path.join(MODEL_DIR, "math_score_predictor.joblib"))

    print("\n===============================================================")
    print(" 2. CLASSIFICATION BENCHMARK: PREDICTING ACADEMIC PASS / FAIL")
    print("===============================================================")

    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        "Support Vector Classifier": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9)
    }

    cls_results = {}
    best_cls_model_name = None
    best_cls_f1 = -float("inf")
    best_cls_pipeline = None

    for name, model in classification_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        cv_scores = cross_val_score(pipeline, X_train_d, y_train_pass, cv=5, scoring='f1')
        pipeline.fit(X_train_d, y_train_pass)
        
        preds = pipeline.predict(X_test_d)
        probs = pipeline.predict_proba(X_test_d)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test_pass, preds)
        prec = precision_score(y_test_pass, preds, zero_division=0)
        rec = recall_score(y_test_pass, preds)
        f1 = f1_score(y_test_pass, preds)
        roc_auc = roc_auc_score(y_test_pass, probs) if probs is not None else 0.0

        cls_results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(roc_auc), 4),
            "CV_F1_Mean": round(float(cv_scores.mean()), 4),
            "CV_F1_Std": round(float(cv_scores.std()), 4)
        }

        print(f"{name:30s} | Acc: {acc:6.4f} | Prec: {prec:6.4f} | Rec: {rec:6.4f} | F1: {f1:6.4f} | AUC: {roc_auc:6.4f} | CV F1: {cv_scores.mean():.4f}")

        if f1 > best_cls_f1:
            best_cls_f1 = f1
            best_cls_model_name = name
            best_cls_pipeline = pipeline

    print(f"\n>>> Best Classification Model: {best_cls_model_name} with F1 = {best_cls_f1:.4f}")
    joblib.dump(best_cls_pipeline, os.path.join(MODEL_DIR, "best_pass_fail_classifier.joblib"))

    # Also build Multi-class Tier Classifier
    y_tier = df['tier']
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_demo, y_tier, test_size=0.20, random_state=42, stratify=y_tier
    )
    tier_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
    ])
    tier_pipeline.fit(X_train_t, y_train_t)
    tier_preds = tier_pipeline.predict(X_test_t)
    tier_acc = accuracy_score(y_test_t, tier_preds)
    print(f"\nPerformance Tier Classifier (3 Classes) Accuracy: {tier_acc:.4f}")
    print(classification_report(y_test_t, tier_preds))
    joblib.dump(tier_pipeline, os.path.join(MODEL_DIR, "tier_classifier.joblib"))

    # 3. Save All Metrics
    full_metrics = {
        "regression_benchmark": reg_results,
        "best_regression_model": best_reg_model_name,
        "classification_benchmark": cls_results,
        "best_classification_model": best_cls_model_name,
        "multi_subject_r2": {
            "math": float(multi_r2[0]),
            "reading": float(multi_r2[1]),
            "writing": float(multi_r2[2])
        },
        "math_given_reading_writing": {
            "R2": float(math_r2),
            "MAE": float(math_mae)
        },
        "tier_classification_accuracy": float(tier_acc)
    }

    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(full_metrics, f, indent=4)
    print("\nSaved model_metrics.json")

    # ---------------- PLOT 5: Model Comparisons ----------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Regression Comparison (R2 and MAE)
    reg_df = pd.DataFrame(reg_results).T.reset_index().rename(columns={'index': 'Model'})
    sns.barplot(data=reg_df, x='R2_Test', y='Model', palette='Blues_r', ax=axes[0])
    axes[0].set_title("Regression Benchmark ($R^2$ Score)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("$R^2$ (Higher is Better)")
    for p in axes[0].patches:
        val = p.get_width()
        axes[0].annotate(f'{val:.3f}', (val + 0.005, p.get_y() + p.get_height() / 2),
                         va='center', fontsize=10, fontweight='bold')

    # Classification Comparison (F1 and Accuracy)
    cls_df = pd.DataFrame(cls_results).T.reset_index().rename(columns={'index': 'Model'})
    sns.barplot(data=cls_df, x='F1_Score', y='Model', palette='Greens_r', ax=axes[1])
    axes[1].set_title("Classification Benchmark (F1-Score)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("F1-Score (Higher is Better)")
    for p in axes[1].patches:
        val = p.get_width()
        axes[1].annotate(f'{val:.3f}', (val + 0.005, p.get_y() + p.get_height() / 2),
                         va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot5_path = os.path.join(VIS_DIR, "05_model_benchmark_comparison.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"Saved {plot5_path}")

    # ---------------- PLOT 6: Regression Residuals and Actual vs Predicted ----------------
    best_reg_preds = best_reg_pipeline.predict(X_test_d)
    residuals = y_test_avg - best_reg_preds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Actual vs Predicted
    axes[0].scatter(y_test_avg, best_reg_preds, alpha=0.6, color='#1f77b4', edgecolor='k')
    min_val = min(y_test_avg.min(), best_reg_preds.min())
    max_val = max(y_test_avg.max(), best_reg_preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    axes[0].set_title(f"Actual vs Predicted Score ({best_reg_model_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual Average Score")
    axes[0].set_ylabel("Predicted Average Score")
    axes[0].legend()

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#e76f51', bins=20, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_title("Residuals (Error) Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    plot6_path = os.path.join(VIS_DIR, "06_regression_evaluation.png")
    plt.savefig(plot6_path, dpi=300)
    plt.close()
    print(f"Saved {plot6_path}")

    # ---------------- PLOT 7: Confusion Matrix & ROC Curve for Classifier ----------------
    from sklearn.metrics import roc_curve, auc
    best_cls_preds = best_cls_pipeline.predict(X_test_d)
    best_cls_probs = best_cls_pipeline.predict_proba(X_test_d)[:, 1]
    cm = confusion_matrix(y_test_pass, best_cls_preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Fail (At-Risk)', 'Pass'], yticklabels=['Fail (At-Risk)', 'Pass'])
    axes[0].set_title(f"Confusion Matrix ({best_cls_model_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_pass, best_cls_probs)
    roc_val = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='#2a9d8f', lw=2.5, label=f'ROC curve (AUC = {roc_val:.3f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
    axes[1].set_title("ROC Curve (Pass/Fail Prediction)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plot7_path = os.path.join(VIS_DIR, "07_classification_evaluation.png")
    plt.savefig(plot7_path, dpi=300)
    plt.close()
    print(f"Saved {plot7_path}")

    # ---------------- PLOT 8: Feature Importance ----------------
    # Compute permutation importance on the best model to inspect key driving factors
    perm_importance = permutation_importance(best_reg_pipeline, X_test_d, y_test_avg, n_repeats=10, random_state=42)
    feat_names = demographic_features
    importances = perm_importance.importances_mean

    fig, ax = plt.subplots(figsize=(8, 5))
    perm_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)
    sns.barplot(data=perm_df, x='Importance', y='Feature', palette='mako', ax=ax)
    ax.set_title("Demographic Feature Importance (Permutation Importance)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Decrease in R² Score upon Permutation")
    plt.tight_layout()
    plot8_path = os.path.join(VIS_DIR, "08_feature_importance.png")
    plt.savefig(plot8_path, dpi=300)
    plt.close()
    print(f"Saved {plot8_path}")

    print("\nTraining and evaluation pipeline finished with excellence!")

if __name__ == "__main__":
    train_and_evaluate()
