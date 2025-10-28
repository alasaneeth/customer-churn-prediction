import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

class DataProcessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def load_data(self):
        """Load and initial data exploration"""
        # For demo purposes, creating sample data
        # In practice, you would load from CSV file
        np.random.seed(42)
        n_samples = 7043
        
        data = {
            'customerID': [f'C{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
            'TotalCharges': np.round(np.random.uniform(20, 8000, n_samples), 2)
        }
        
        self.df = pd.DataFrame(data)
        
        # Create target variable with some logic
        churn_proba = (
            (self.df['tenure'] < 12) * 0.3 +
            (self.df['Contract'] == 'Month-to-month') * 0.2 +
            (self.df['MonthlyCharges'] > 80) * 0.2 +
            (self.df['InternetService'] == 'Fiber optic') * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        self.df['Churn'] = (churn_proba > 0.5).astype(int)
        
        return self.df
    
    def clean_data(self):
        """Perform data cleaning operations"""
        print("Original data shape:", self.df.shape)
        
        # Handle missing values in TotalCharges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'].fillna(self.df['MonthlyCharges'], inplace=True)
        
        # Remove customerID as it's not a feature
        self.df.drop('customerID', axis=1, inplace=True)
        
        print("After cleaning data shape:", self.df.shape)
        return self.df
    
    def split_data(self):
        """Split data into train and test sets"""
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test