import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import yaml

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        self.best_model = None
        self.best_score = 0
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def compare_models(self, X_train, X_test, y_train, y_test):
        """Compare multiple algorithms"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            results[name] = self.evaluate_model(model, X_test, y_test, name)
            
            # Update best model
            if results[name]['accuracy'] > self.best_score:
                self.best_score = results[name]['accuracy']
                self.best_model = model
        
        return results
    
    def optimize_xgboost(self, X_train, X_test, y_train, y_test):
        """Optimize XGBoost using GridSearchCV"""
        print("\nOptimizing XGBoost with GridSearchCV...")
        
        param_grid = self.config['hyperparameters']['xgboost']
        
        grid_search = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'),
            param_grid,
            cv=self.config['model']['cv_folds'],
            scoring='precision',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate optimized model
        optimized_model = grid_search.best_estimator_
        y_pred_optimized = optimized_model.predict(X_test)
        
        original_precision = precision_score(y_test, self.models['xgboost'].predict(X_test))
        optimized_precision = precision_score(y_test, y_pred_optimized)
        
        improvement = ((optimized_precision - original_precision) / original_precision) * 100
        
        print(f"\nPrecision Improvement: {improvement:.1f}%")
        print(f"Original Precision: {original_precision:.4f}")
        print(f"Optimized Precision: {optimized_precision:.4f}")
        
        self.best_model = optimized_model
        return optimized_model, improvement
    
    def save_model(self, model, filename='models/best_churn_model.pkl'):
        """Save the trained model"""
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")