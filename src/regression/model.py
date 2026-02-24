import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class RiskPredictor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.model_path = os.path.join(self.config['paths']['checkpoints'], "risk_model.joblib")

    def train(self, data_path="data/features/infrastructure_features.csv"):
        """
        Train XGBoost regression model on tabular features.
        """
        if not os.path.exists(data_path):
            logger.error(f"Data not found at {data_path}")
            return
            
        df = pd.read_csv(data_path)
        
        # Simulated target: risk_score (0-100)
        # In a real scenario, this would be provided in the dataset.
        if 'risk_score' not in df.columns:
            logger.info("Simulating risk_score for training.")
            df['risk_score'] = (df['crack_area_percent'] * 5 + df['growth_rate'] * 10 + df['change_score'] * 2).clip(0, 100)
            
        X = df.drop(columns=['location_id', 'date', 'risk_score'])
        y = df['risk_score']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['models']['regression']['test_size'], 
            random_state=self.config['models']['regression']['random_state']
        )
        
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        logger.info(f"Risk Model Trained. RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def predict(self, features_dict):
        """
        Predict risk for a single set of features.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                logger.warning("Model not trained yet. Returning default risk.")
                return {"risk_score": 0.0, "risk_level": "UNKNOWN", "estimated_failure_months": 0}

        # Convert dict to DF
        df = pd.DataFrame([features_dict]).drop(columns=['location_id', 'date'], errors='ignore')
        risk_score = float(self.model.predict(df)[0])
        
        risk_level = "LOW"
        if risk_score > 70:
            risk_level = "HIGH"
        elif risk_score > 30:
            risk_level = "MEDIUM"
            
        estimated_failure_months = max(1, int(100 - risk_score)) # Dummy logic
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "estimated_failure_months": estimated_failure_months
        }

if __name__ == "__main__":
    predictor = RiskPredictor()
    # Generate some dummy data to train
    dummy_data = {
        "location_id": ["L1"]*10,
        "date": ["D"]*10,
        "crack_area_percent": np.random.rand(10) * 10,
        "growth_rate": np.random.rand(10) * 2,
        "texture_variance": np.random.rand(10),
        "color_degradation": np.random.rand(10),
        "change_score": np.random.rand(10)
    }
    pd.DataFrame(dummy_data).to_csv("data/features/infrastructure_features.csv", index=False)
    
    predictor.train()
    res = predictor.predict({
        "crack_area_percent": 5.0,
        "growth_rate": 0.5,
        "texture_variance": 0.2,
        "color_degradation": 0.1,
        "change_score": 0.3
    })
    print(res)
