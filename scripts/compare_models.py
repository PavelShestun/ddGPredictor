import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix
from imblearn.combine import SMOTEENN
from joblib import dump
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ddGPredictor.data_loader import load_skempi_data
from ddGPredictor.preprocessing import preprocess_data

# Use non-GUI backend to avoid Qt issues
import matplotlib
matplotlib.use('Agg')

def plot_roc_curve(y_test, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'models/roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'models/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

def find_optimal_threshold(y_test, y_scores, target_recall=0.5):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) == 0:
        optimal_idx = np.argmax(recalls)
    else:
        f1_scores_valid = f1_scores[valid_indices]
        optimal_idx = valid_indices[np.argmax(f1_scores_valid)]
    
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, f1_scores[optimal_idx]

def compare_models():
    data_path = "data/SKEMPI2.csv"
    try:
        df = load_skempi_data(data_path)
    except FileNotFoundError:
        print("Please place SKEMPI2.csv in the data/ folder or update the URL in data_loader.py")
        return

    if 'ddG' not in df.columns:
        R = 0.001987
        T = 298
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce').fillna(T)
        df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed'])
        df['ddG'] = -R * df['Temperature'] * np.log(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ddG'])

    X, y = preprocess_data(df, ddg_threshold=0.0)

    X = X.to_numpy().astype(float)
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    smoteenn = SMOTEENN(random_state=42, sampling_strategy=0.3)
    X_train, y_train = smoteenn.fit_resample(X_train, y_train)

    rf_for_selection = RandomForestClassifier(
        random_state=42, 
        class_weight="balanced",
        min_samples_leaf=10,
        max_depth=15
    )
    rf_for_selection.fit(X_train, y_train)

    selector = SelectFromModel(rf_for_selection, threshold=0.005, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    feature_names = pd.read_csv("models/feature_names.txt", header=None)[0].tolist()
    selected_features = np.array(feature_names)[selector.get_support()]
    print(f"Selected features: {selected_features}")

    base_estimators = [
        ('rf', RandomForestClassifier(random_state=42, class_weight="balanced", min_samples_leaf=10, max_depth=5)),
        ('catboost', CatBoostClassifier(random_state=42, auto_class_weights='Balanced', iterations=100, depth=4, l2_leaf_reg=7, verbose=0)),
        ('lr', LogisticRegression(random_state=42, class_weight="balanced")),
        ('lgb', lgb.LGBMClassifier(random_state=42, class_weight="balanced", reg_alpha=5.0, reg_lambda=5.0))
    ]
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=5
    )

    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, class_weight="balanced"))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(random_state=42, class_weight="balanced", max_iter=5000))
        ]),
        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                random_state=42, 
                class_weight="balanced",
                min_samples_leaf=10,
                max_depth=15
            ))
        ]),
        "CatBoost": Pipeline([
            ("model", CatBoostClassifier(random_state=42, auto_class_weights='Balanced', iterations=100, depth=4, l2_leaf_reg=7, verbose=0))
        ]),
        "Stacking Ensemble": Pipeline([
            ("model", stacking_clf)
        ])
    }

    results = []
    best_pipeline = None
    best_model_name = None
    best_f1 = 0
    y_scores_dict = {}
    optimal_thresholds = {}

    for name, pipeline in pipelines.items():
        calibrated_model = CalibratedClassifierCV(pipeline.named_steps['model'], cv=5, method='sigmoid')
        calibrated_pipeline = Pipeline([
            ("scaler", pipeline.named_steps.get('scaler', None)),
            ("model", calibrated_model)
        ])
        if 'scaler' not in pipeline.named_steps:
            calibrated_pipeline.steps.pop(0)

        start_time = time.perf_counter()
        calibrated_pipeline.fit(X_train_selected, y_train)
        train_time = time.perf_counter() - start_time
        train_time = max(train_time, 0)

        start_time = time.perf_counter()
        if hasattr(calibrated_pipeline.named_steps['model'], "predict_proba"):
            y_scores = calibrated_pipeline.named_steps['model'].predict_proba(X_test_selected)[:, 1]
        else:
            y_scores = calibrated_pipeline.named_steps['model'].decision_function(X_test_selected)
        inference_time = time.perf_counter() - start_time
        inference_time = max(inference_time, 0)

        target_recall = 0.65 if name == "Random Forest" else 0.5
        if hasattr(calibrated_pipeline.named_steps['model'], "predict_proba"):
            optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_scores, target_recall=target_recall)
            y_pred = (y_scores >= optimal_threshold).astype(int)
            optimal_thresholds[name] = optimal_threshold
        else:
            y_pred = calibrated_pipeline.predict(X_test_selected)
            optimal_thresholds[name] = None

        print(f"\nModel: {name}")
        print(f"y_test shape: {y_test.shape}, unique values: {np.unique(y_test)}")
        print(f"y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")

        try:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)

            roc_auc = roc_auc_score(y_test, y_scores)
            y_scores_dict[name] = y_scores

            precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_scores)
            pr_auc = auc(recall_pr, precision_pr)

            # Кросс-валидация временно отключена
            cv_f1_mean = 0.0
            cv_f1_std = 0.0

            results.append({
                "Model": name,
                "Accuracy": round(accuracy, 3),
                "F1-Score": round(f1, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "ROC-AUC": round(roc_auc, 3),
                "PR-AUC": round(pr_auc, 3),
                "CV F1 Mean": round(cv_f1_mean, 3),
                "CV F1 Std": round(cv_f1_std, 3),
                "Train Time (s)": round(train_time, 4),
                "Inference Time (s)": round(inference_time, 4)
            })
        except Exception as e:
            print(f"Error computing metrics for {name}: {e}")
            continue

    results_df = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(results_df.to_markdown(index=False))

    if not results:
        print("No models were successfully evaluated.")
        return

    best_idx = results_df["F1-Score"].idxmax()
    if results_df["F1-Score"].duplicated().any():
        best_idx = results_df.loc[results_df["F1-Score"] == results_df["F1-Score"].max()]["Inference Time (s)"].idxmin()
    best_model_name = results_df.loc[best_idx, "Model"]
    best_pipeline = pipelines[best_model_name]
    best_f1 = results_df.loc[best_idx, "F1-Score"]

    print(f"\nEvaluating {best_model_name} before hyperparameter tuning:")
    y_pred = (y_scores_dict[best_model_name] >= optimal_thresholds[best_model_name]).astype(int) if optimal_thresholds[best_model_name] is not None else best_pipeline.predict(X_test_selected)
    plot_roc_curve(y_test, y_scores_dict[best_model_name], best_model_name)
    plot_confusion_matrix(y_test, y_pred, best_model_name)

    print(f"\nTuning hyperparameters for {best_model_name}...")
    if best_model_name == "Logistic Regression":
        param_grid = {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__solver": ["lbfgs", "liblinear", "saga"],
            "model__max_iter": [5000]
        }
    elif best_model_name == "Linear SVM":
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__max_iter": [10000]
        }
    elif best_model_name == "Decision Tree":
        param_grid = {
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4]
        }
    elif best_model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [4, 8, 12],
            "model__max_features": ["sqrt", "log2"]
        }
    elif best_model_name == "CatBoost":
        param_grid = {
            "model__iterations": [100, 200, 500],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__l2_leaf_reg": [5, 7, 9]
        }
    elif best_model_name == "Stacking Ensemble":
        param_grid = {
            "model__rf__n_estimators": [50, 100],
            "model__rf__max_depth": [3, 5],
            "model__catboost__iterations": [50, 100],
            "model__catboost__depth": [3, 4],
            "model__lgb__n_estimators": [50, 100],
            "model__lgb__max_depth": [3, 5]
        }

    grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring="f1", n_jobs=2)
    grid_search.fit(X_train_selected, y_train)

    best_pipeline = grid_search.best_estimator_
    calibrated_best = CalibratedClassifierCV(best_pipeline.named_steps['model'], cv=5, method='sigmoid')
    final_pipeline = Pipeline([
        ("scaler", best_pipeline.named_steps.get('scaler', None)),
        ("model", calibrated_best)
    ])
    if 'scaler' not in best_pipeline.named_steps:
        final_pipeline.steps.pop(0)

    final_pipeline.fit(X_train_selected, y_train)

    if hasattr(final_pipeline.named_steps['model'], "predict_proba"):
        y_scores_tuned = final_pipeline.named_steps['model'].predict_proba(X_test_selected)[:, 1]
        optimal_threshold_tuned, _ = find_optimal_threshold(y_test, y_scores_tuned, target_recall=0.65 if best_model_name == "Random Forest" else 0.5)
        y_pred_tuned = (y_scores_tuned >= optimal_threshold_tuned).astype(int)
    else:
        y_pred_tuned = final_pipeline.predict(X_test_selected)
        y_scores_tuned = final_pipeline.named_steps['model'].decision_function(X_test_selected)

    f1_tuned = f1_score(y_pred_tuned, y_test)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"F1-Score after tuning: {f1_tuned:.3f} (before: {best_f1:.3f})")

    plot_roc_curve(y_test, y_scores_tuned, f"{best_model_name} (Tuned)")
    plot_confusion_matrix(y_test, y_pred_tuned, f"{best_model_name} (Tuned)")

    if best_model_name == "Logistic Regression":
        coef = grid_search.best_estimator_.named_steps['model'].coef_[0]
        coef_df = pd.DataFrame({"Feature": selected_features, "Coefficient": coef})
        coef_df = coef_df.sort_values(by="Coefficient", ascending=False, key=abs)
        print("\nLogistic Regression Coefficients:")
        print(coef_df.head(10).to_markdown(index=False))

    if best_model_name in ["Random Forest", "CatBoost", "Decision Tree", "Stacking Ensemble"]:
        if best_model_name == "Stacking Ensemble":
            model = final_pipeline.named_steps['model'].base_estimator.named_estimators_['rf']
        else:
            model = final_pipeline.named_steps['model'].base_estimator
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({"Feature": selected_features, "Importance": importances})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
        print("\nFeature Importance:")
        print(feature_importance_df.head(10).to_markdown(index=False))

    Path("models").mkdir(exist_ok=True)
    dump(final_pipeline, "models/best_ddg_predictor_pipeline.joblib")
    print(f"\nBest model saved: {best_model_name} (tuned)")

if __name__ == "__main__":
    compare_models()