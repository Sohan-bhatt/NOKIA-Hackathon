import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib

class NetworkAlarmPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.alarm_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.time_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
    def preprocess_features(self, df, training=True):
        # Create time-based features
        df['hourOfDay'] = pd.to_datetime(df['firstOccurrenceTime']).dt.hour
        df['dayOfWeek'] = pd.to_datetime(df['firstOccurrenceTime']).dt.dayofweek
        
        # Calculate time since last alarm per site
        df = df.sort_values('firstOccurrenceTime')
        df['timeSinceLastAlarm'] = df.groupby('siteId')['firstOccurrenceTime'].diff().dt.total_seconds() / 60
        
        # Calculate rolling counts
        for window in [60, 360, 1440]:  # 1h, 6h, 24h in minutes
            df[f'rollingAlarmCount{window//60}h'] = df.groupby('siteId').rolling(
                window='{}min'.format(window), 
                on='firstOccurrenceTime'
            ).count()['alarmCode']
        
        # Encode categorical variables
        categorical_cols = ['siteId', 'nodeId', 'region', 'equipmentType', 
                          'alarmCode', 'probableCause', 'alarmCategory']
        
        for col in categorical_cols:
            if training:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Prepare feature matrix
        feature_cols = ['hourOfDay', 'dayOfWeek', 'timeSinceLastAlarm',
                       'rollingAlarmCount1h', 'rollingAlarmCount6h', 'rollingAlarmCount24h'] + \
                      [col + '_encoded' for col in categorical_cols]
        
        X = df[feature_cols].fillna(0)
        
        if training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X
    
    def prepare_targets(self, df):
        # Prepare next alarm code target
        df['nextAlarmCode'] = df.groupby('siteId')['alarmCode'].shift(-1)
        df['timeToNextAlarm'] = df.groupby('siteId')['firstOccurrenceTime'].diff(-1).dt.total_seconds() / 60
        df['nextSeverity'] = df.groupby('siteId')['severity'].shift(-1)
        
        y_code = self.label_encoders['alarmCode'].transform(df['nextAlarmCode'].fillna(df['alarmCode']))
        y_time = df['timeToNextAlarm'].fillna(df['timeToNextAlarm'].mean())
        
        return y_code, y_time
    
    def train(self, df):
        X = self.preprocess_features(df, training=True)
        y_code, y_time = self.prepare_targets(df)
        
        # Train models
        self.alarm_classifier.fit(X, y_code)
        self.time_regressor.fit(X, y_time)
        
        # Save models
        joblib.dump(self.alarm_classifier, 'alarm_classifier.joblib')
        joblib.dump(self.time_regressor, 'time_regressor.joblib')
        joblib.dump(self.label_encoders, 'label_encoders.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
    
    def predict(self, df):
        X = self.preprocess_features(df, training=False)
        
        # Make predictions
        next_alarm_encoded = self.alarm_classifier.predict(X)
        next_alarm_proba = self.alarm_classifier.predict_proba(X)
        time_to_next = self.time_regressor.predict(X)
        
        # Decode predictions
        predictions = pd.DataFrame({
            'predictedNextAlarmCode': self.label_encoders['alarmCode'].inverse_transform(next_alarm_encoded),
            'probability': np.max(next_alarm_proba, axis=1),
            'predictedTimeToNextAlarm': time_to_next
        })
        
        return predictions

# Example usage:
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('alarm_data.csv')
    
    # Initialize and train the predictor
    predictor = NetworkAlarmPredictor()
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train the model
    predictor.train(train_df)
    
    # Make predictions
    predictions = predictor.predict(test_df)
    print("Predictions:", predictions.head())