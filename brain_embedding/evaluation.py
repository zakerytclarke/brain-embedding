import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from typing import Dict, Any, Tuple, Optional


class DownstreamEvaluator:
    """Evaluates frozen embeddings on downstream classification tasks with balanced classes."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.oversampler = RandomOverSampler(random_state=random_seed)

    def _prepare_data(self, embeddings: np.ndarray, subject_ids: list, targets_df: pd.DataFrame, task_col: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Filter and align embeddings with targets."""
        X, y = [], []
        # targets_df is indexed by subject_id
        for i, sub_id in enumerate(subject_ids):
            if sub_id in targets_df.index:
                val = targets_df.loc[sub_id, task_col]
                if not pd.isna(val):
                    X.append(embeddings[i])
                    y.append(val)
        
        if not X:
            return None, None
        return np.array(X), np.array(y).astype(int)

    def evaluate_task(self, train_emb, train_ids, test_emb, test_ids, 
                      targets_df, task_col, task_name, num_classes) -> Dict[str, Any]:
        """Train a balanced MLP probe and return evaluation metrics."""
        
        X_train, y_train = self._prepare_data(train_emb, train_ids, targets_df, task_col)
        X_test, y_test = self._prepare_data(test_emb, test_ids, targets_df, task_col)
        
        if X_train is None or X_test is None or len(np.unique(y_train)) < 2:
            # Check for cases where test might only have 1 class
            if X_test is not None and len(np.unique(y_test)) < 2:
                 return [{"Task": task_name, "Status": f"Insufficient Test Classes ({len(np.unique(y_test))})"}]
            return [{"Task": task_name, "Status": "Insufficient Training Data"}]

        # 1. Apply Oversampling to Balance Training Classes
        try:
            X_train_bal, y_train_bal = self.oversampler.fit_resample(X_train, y_train)
        except ValueError:
            # Fallback if oversampling fails (e.g. only 1 sample in a class)
            X_train_bal, y_train_bal = X_train, y_train

        # 2. Train Sklearn MLP
        # 1-layer MLP is just a linear classifier if no hidden layers, 
        # but user asked for MLPClassifier. Default is (100,).
        # We'll use a single hidden layer of 128 for a bit more power than pure linear.
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            alpha=1e-4,
            solver='adam',
            random_state=self.random_seed,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        clf.fit(X_train_bal, y_train_bal)
        
        # 3. Evaluate
        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
        results = []
        
        # Calculate precision and recall first
        if num_classes == 2:
            precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            tn, fp, fn, tp = cm.ravel()
            cm_str = f"[{tn} {fp} / {fn} {tp}]"
            try:
                auc = roc_auc_score(y_test, y_probs[:, 1])
            except ValueError:
                auc = float('nan')
        else:
            precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            cm_str = "Multi-class matrix"
            auc = float('nan') # Will be overwritten if available classes > 1
            
        # Overall Task Dictionary
        main_result = {
            "Task": task_name,
            "Test Ex.": len(y_test),
            "Distribution": f"Bal: {np.bincount(y_test).tolist()}",
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "Conf. Matrix": cm_str,
            "AUC": auc
        }

        # Format Confusion Matrix and sub-tasks
        if num_classes == 2:
            try:
                auc = roc_auc_score(y_test, y_probs[:, 1])
            except ValueError:
                auc = float('nan')
            
            tn, fp, fn, tp = cm.ravel()
            results.append({
                "Task": task_name,
                "Test Ex.": len(y_test),
                "Distribution": f"Bal: {np.bincount(y_test).tolist()}",
                "Accuracy": acc,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "Conf. Matrix": f"[{tn} {fp} / {fn} {tp}]"
            })
            
        else:
            # Multi-class breakdown: Return only the individual OvR binary tasks
            available_classes = np.unique(y_test)
            if len(available_classes) > 1:
                for cls in available_classes:
                    y_test_binary = (y_test == cls).astype(int)
                    if len(np.unique(y_test_binary)) == 2:
                        if cls in clf.classes_:
                            cls_idx = np.where(clf.classes_ == cls)[0][0]
                            try:
                                score = roc_auc_score(y_test_binary, y_probs[:, cls_idx])
                                b_acc = accuracy_score(y_test_binary, (y_pred == cls).astype(int))
                                results.append({
                                    "Task": f"{task_name} (Class {cls} vs Rest)",
                                    "Test Ex.": len(y_test),
                                    "Distribution": f"Bal: {np.bincount(y_test_binary).tolist()}",
                                    "Accuracy": b_acc,
                                    "AUC": score,
                                    "Precision": float('nan'),
                                    "Recall": float('nan'),
                                    "Conf. Matrix": ""
                                })
                            except ValueError:
                                pass

        return results
