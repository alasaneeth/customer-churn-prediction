import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc

class Visualizer:
    def __init__(self):
        plt.style.use('default')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.barh(range(top_n), importances[indices[:top_n]][::-1])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self, model, X_test, feature_names):
        """Create SHAP summary plot for model interpretation"""
        print("Generating SHAP explanations...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Churn Prediction')
        plt.tight_layout()
        plt.show()
        
        # Force plot for a single observation
        plt.figure(figsize=(12, 6))
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:], 
                       feature_names=feature_names, matplotlib=True, show=False)
        plt.title('SHAP Force Plot for First Customer')
        plt.tight_layout()
        plt.show()
        
        return explainer, shap_values
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        return roc_auc