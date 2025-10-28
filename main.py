import warnings
warnings.filterwarnings('ignore')

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.visualization import Visualizer
import pandas as pd
import numpy as np

def main():
    print("=== Customer Churn Prediction Project ===\n")
    
    # Step 1: Data Processing
    print("1. Loading and cleaning data...")
    processor = DataProcessor()
    df = processor.load_data()
    df = processor.clean_data()
    
    # Step 2: Feature Engineering
    print("\n2. Performing feature engineering...")
    engineer = FeatureEngineer()
    df = engineer.prepare_features(df)
    
    # Split data before preprocessing
    X_train, X_test, y_train, y_test = processor.split_data()
    
    # Create and fit preprocessor
    preprocessor = engineer.create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    feature_names = (engineer.numerical_features + 
                    list(preprocessor.named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(engineer.categorical_features)))
    
    # Step 3: Model Training and Comparison
    print("\n3. Training and comparing models...")
    trainer = ModelTrainer()
    
    # Compare multiple algorithms
    results = trainer.compare_models(X_train_processed, X_test_processed, y_train, y_test)
    
    # Step 4: Model Optimization
    print("\n4. Optimizing the best model...")
    optimized_model, improvement = trainer.optimize_xgboost(
        X_train_processed, X_test_processed, y_train, y_test
    )
    
    # Step 5: Visualization and Interpretation
    print("\n5. Creating visualizations and interpretations...")
    viz = Visualizer()
    
    # Feature Importance
    viz.plot_feature_importance(optimized_model, feature_names)
    
    # SHAP Analysis
    explainer, shap_values = viz.plot_shap_summary(
        optimized_model, X_test_processed, feature_names
    )
    
    # Confusion Matrix
    y_pred = optimized_model.predict(X_test_processed)
    viz.plot_confusion_matrix(y_test, y_pred)
    
    # ROC Curve
    y_pred_proba = optimized_model.predict_proba(X_test_processed)[:, 1]
    auc_score = viz.plot_roc_curve(y_test, y_pred_proba)
    
    # Step 6: Save Model
    print("\n6. Saving the final model...")
    trainer.save_model(optimized_model)
    
    # Final Summary
    print("\n=== PROJECT SUMMARY ===")
    print(f"📊 Dataset: {len(df)} customers, {df['Churn'].sum()} churn cases")
    print(f"🎯 Best Model: Optimized XGBoost")
    print(f"📈 Accuracy: {trainer.best_score:.2%}")
    print(f"🎯 Precision Improvement: {improvement:.1f}%")
    print(f"📊 AUC Score: {auc_score:.2%}")
    print(f"💡 Key Insights: SHAP analysis revealed top factors driving churn")
    print("✅ Project completed successfully!")

if __name__ == "__main__":
    main()