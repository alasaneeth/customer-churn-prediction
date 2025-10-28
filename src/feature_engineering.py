import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                   'MultipleLines', 'InternetService', 'OnlineSecurity',
                                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                   'StreamingTV', 'StreamingMovies', 'Contract',
                                   'PaperlessBilling', 'PaymentMethod']
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    def create_new_features(self, df):
        """Create new engineered features"""
        # Tenure Group (as mentioned in CV)
        df['TenureGroup'] = pd.cut(df['tenure'], 
                                 bins=[0, 12, 24, 48, 72], 
                                 labels=['0-1Year', '1-2Years', '2-4Years', '4+Years'])
        
        # Monthly Spend Category (as mentioned in CV)
        df['MonthlySpendCategory'] = pd.cut(df['MonthlyCharges'],
                                          bins=[0, 35, 70, 120],
                                          labels=['Low', 'Medium', 'High'])
        
        # Additional engineered features
        df['AvgChargePerTenure'] = df['TotalCharges'] / df['tenure']
        df['AvgChargePerTenure'].fillna(df['MonthlyCharges'], inplace=True)
        
        df['HasMultipleServices'] = ((df['OnlineSecurity'] == 'Yes') | 
                                   (df['OnlineBackup'] == 'Yes') | 
                                   (df['DeviceProtection'] == 'Yes') | 
                                   (df['TechSupport'] == 'Yes')).astype(int)
        
        return df
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        from sklearn.preprocessing import OneHotEncoder
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor
    
    def prepare_features(self, df):
        """Apply all feature engineering steps"""
        df = self.create_new_features(df)
        return df