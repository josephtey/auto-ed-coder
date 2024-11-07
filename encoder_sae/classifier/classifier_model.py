import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import pickle
from datetime import datetime
import os

class BinaryClassifierModel:
    def __init__(self, wandb=None):
        self.wandb = wandb
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.best_model = None
        self.best_model_params = None
        self.selected_feature_indices = None

        # Create output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = f"output/{timestamp}_binary_classifier"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def boruta_filter(self, X, y):
        boruta_selector = BorutaPy(
            RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5),
            n_estimators="auto",
            verbose=0,
            random_state=1,
        )
        boruta_selector.fit(X, y)

        self.selected_feature_indices = boruta_selector.support_
        X_filtered = X[:, self.selected_feature_indices]

        selected_feature_count = X_filtered.shape[1]
        print(f"Number of features selected by Boruta: {selected_feature_count}")

        if self.wandb:
            self.wandb.log({"features_selected": selected_feature_count})

        return X_filtered

    def train_model(self, X_train, y_train, X_val, y_val, use_feature_selection=True):
        if use_feature_selection:
            print("Applying Boruta feature selection...")
            X_train = self.boruta_filter(X_train, y_train)
            X_val = X_val[:, self.selected_feature_indices]

        print("Scaling features...")
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Define hyperparameter ranges
        max_depths = [10, 20, 30, 50, 100]
        n_estimators_list = [10, 50, 100]
        print(f"Will try {len(max_depths)} different max_depths and {len(n_estimators_list)} different n_estimators")
        print(f"Total combinations to try: {len(max_depths) * len(n_estimators_list)}")

        best_f1 = -1
        best_params = None
        best_clf = None
        best_metrics = None
        results = []

        from tqdm.notebook import tqdm

        total_iterations = len(max_depths) * len(n_estimators_list)
        pbar = tqdm(total=total_iterations, desc="Training Progress")

        for max_depth in max_depths:
            for n_estimators in n_estimators_list:
                clf = RandomForestClassifier(
                    max_depth=max_depth, n_estimators=n_estimators
                )
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_val)
                y_pred_proba = clf.predict_proba(X_val)[:, 1]
                
                # Calculate multiple metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                auc_roc = roc_auc_score(y_val, y_pred_proba)

                results.append({
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc_roc": auc_roc,
                })

                # Use F1-score as the primary metric for model selection
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        "max_depth": max_depth,
                        "n_estimators": n_estimators,
                    }
                    best_clf = clf
                    best_metrics = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "auc_roc": auc_roc
                    }

                pbar.update(1)
                pbar.set_postfix({"Best F1": best_f1})

                # Print current iteration results
                print(f"\nIteration Results (max_depth={max_depth}, n_estimators={n_estimators}):")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1: {f1:.4f}")
                print(f"AUC-ROC: {auc_roc:.4f}")

        pbar.close()

        print(f"\nBest Hyperparameters: {best_params}")
        print("Best Validation Metrics:")
        for metric, value in best_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{self.output_folder}/validation_results.csv', index=False)

        with open(f"{self.output_folder}/best_model.pkl", "wb") as model_file:
            pickle.dump(best_clf, model_file)

        with open(f"{self.output_folder}/scaler.pkl", "wb") as scaler_file:
            pickle.dump(self.scaler, scaler_file)

        self.best_model = best_clf
        self.best_model_params = best_params

        if self.wandb:
            self.wandb.log({
                "val/best_results": {
                    "accuracy": {
                        "score": best_metrics["accuracy"],
                        "params": best_params,
                    },
                },
                "val/results": results,
            })

    def evaluate_model(self, X_test, y_test):
        if not hasattr(self, "best_model") or self.best_model is None:
            raise ValueError("Best model has not been set. Please run the training process first.")

        if self.selected_feature_indices is not None:
            X_test = X_test[:, self.selected_feature_indices]

        X_test = self.scaler.transform(X_test)
        y_pred_test = self.best_model.predict(X_test)
        y_pred_proba_test = self.best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test),
            "Recall": recall_score(y_test, y_pred_test),
            "F1": f1_score(y_test, y_pred_test),
            "AUC-ROC": roc_auc_score(y_test, y_pred_proba_test)
        }

        print("Evaluation on Test Set:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

        evaluation_results_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        })
        evaluation_results_df.to_csv(f"{self.output_folder}/test_set_results.csv", index=False)

        if self.wandb:
            self.wandb.log({
                "test/results": {
                    **metrics,
                    "confusion_matrix": confusion_matrix(y_test, y_pred_test),
                    "params": self.best_model_params,
                },
            })