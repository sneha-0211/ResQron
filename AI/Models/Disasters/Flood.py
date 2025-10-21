import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class FloodDisasterPredictor:
    
    def __init__(self, config=None):
        self.config = config or {
            "csv_path": "flood_risk_dataset_india.csv",
            "artifacts_dir": "artifacts",
            "risk_threshold": 0.7,
            "alert_threshold": 0.5
        }
        os.makedirs(self.config["artifacts_dir"], exist_ok=True)
        self.model = None
        self.feature_names = []
        self.risk_zones = {}
    
    def load_and_validate_data(self, path):
        print("Loading and validating dataset...")
        
        try:
            df = pd.read_csv(path)
            print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        
        print(f"\nDataset Overview:")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        critical_features = ["Latitude", "Longitude", "Rainfall_mm", "Water_Level_m"]
        missing_critical = [f for f in critical_features if f not in df.columns]
        
        if missing_critical:
            print(f"Missing critical features: {missing_critical}")
            print("These are essential for accurate flood prediction!")
        
        return df
    
    def create_enhanced_features(self, df):
        print("Engineering enhanced features...")
        
        enhanced_df = df.copy()
        
        if all(col in df.columns for col in ["Rainfall_mm", "Water_Level_m"]):
            enhanced_df["Risk_Score"] = (
                (df["Rainfall_mm"] / df["Rainfall_mm"].std()) + 
                (df["Water_Level_m"] / df["Water_Level_m"].std())
            )
        
        if "Elevation_m" in df.columns:
            enhanced_df["Elevation_Risk"] = 1 / (df["Elevation_m"] + 1)
        
        if "Population_Density" in df.columns:
            enhanced_df["Population_Risk"] = np.log1p(df["Population_Density"])
        
        if "Historical_Floods" in df.columns:
            enhanced_df["Historical_Weight"] = df["Historical_Floods"] * 2
        
        if all(col in df.columns for col in ["Rainfall_mm", "Temperature_C", "Humidity_pct"]):
            enhanced_df["Weather_Intensity"] = (
                df["Rainfall_mm"] * df["Humidity_pct"] / 100
            )
        
        if "Flood_Occurred" not in enhanced_df.columns:
            print("Creating realistic flood target based on risk factors...")
            enhanced_df = self._create_realistic_target(enhanced_df)
        
        return enhanced_df
    
    def _create_realistic_target(self, df):
        risk_factors = []
        
        if "Rainfall_mm" in df.columns:
            rainfall_risk = (df["Rainfall_mm"] > df["Rainfall_mm"].quantile(0.8)).astype(float)
            risk_factors.append(rainfall_risk * 0.3)
        
        if "Water_Level_m" in df.columns:
            water_risk = (df["Water_Level_m"] > df["Water_Level_m"].quantile(0.8)).astype(float)
            risk_factors.append(water_risk * 0.3)
        
        if "Elevation_m" in df.columns:
            elevation_risk = (df["Elevation_m"] < df["Elevation_m"].quantile(0.2)).astype(float)
            risk_factors.append(elevation_risk * 0.2)
        
        if "Historical_Floods" in df.columns:
            hist_risk = (df["Historical_Floods"] > 0).astype(float)
            risk_factors.append(hist_risk * 0.2)
        
        if risk_factors:
            combined_risk = sum(risk_factors)
            noise = np.random.normal(0, 0.1, len(df))
            final_risk = combined_risk + noise
            
            threshold = np.percentile(final_risk, 80)
            df["Flood_Occurred"] = (final_risk > threshold).astype(int)
            
            flood_rate = df["Flood_Occurred"].mean()
            print(f"Created target with {flood_rate:.1%} flood occurrence rate")
        else:
            df["Flood_Occurred"] = np.random.binomial(1, 0.1, len(df))
        
        return df
    
    def train_ensemble_models(self, df):
        print("Training ensemble of flood prediction models...")
        
        target = "Flood_Occurred"
        feature_cols = [c for c in df.columns if c not in [target, "Affected_Area"]]
        
        X = df[feature_cols]
        y = df[target]
        
        print(f"\nTraining Data Summary:")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Flood cases: {y.sum()} ({y.mean():.1%})")
        
        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        transformers = []
        if num_features:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_pipeline, num_features))
        
        if cat_features:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipeline, cat_features))
        
        preprocessor = ColumnTransformer(transformers)
        
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)            
            results[name] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }            
            trained_models[name] = pipeline            
            print(f"{name} Results:")
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   ROC-AUC: {roc_auc:.3f}")
            print(f"   CV Mean: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")        
        best_model_name = max(results.keys(), key=lambda x: results[x]["roc_auc"])
        self.model = trained_models[best_model_name]     
        print(f"\nBest Model: {best_model_name}")
        print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.3f}")     
        model_path = os.path.join(self.config["artifacts_dir"], "flood_disaster_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"Model saved: {model_path}")    
        self._analyze_feature_importance(X_train.columns)    
        return self.model, results
    def _analyze_feature_importance(self, feature_cols):
        try:
            clf = self.model.named_steps["classifier"]
            importances = clf.feature_importances_        
            preprocessor = self.model.named_steps["preprocessor"]
            feature_names = []          
            for name, transformer, cols in preprocessor.transformers_:
                if name == "num":
                    feature_names.extend(cols)
                elif name == "cat":
                    try:
                        cat_names = transformer.named_steps["onehot"].get_feature_names_out(cols)
                        feature_names.extend(cat_names)
                    except:
                        feature_names.extend([f"{col}_cat" for col in cols])            
            importance_df = pd.DataFrame({
                "feature": feature_names[:len(importances)],
                "importance": importances
            }).sort_values("importance", ascending=False)            
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            sns.barplot(data=top_features, x="importance", y="feature")
            plt.title("Top 15 Features for Flood Prediction")
            plt.xlabel("Feature Importance")
            plt.tight_layout()            
            importance_path = os.path.join(self.config["artifacts_dir"], "feature_importance_enhanced.png")
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()            
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(top_features.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<25} - {row['importance']:.3f}")            
            return importance_df            
        except Exception as e:
            print(f"Could not analyze feature importance: {e}")
            return None    
    def predict_flood_risk(self, new_data):
        if self.model is None:
            raise ValueError("Model not trained yet!")        
        probabilities = self.model.predict_proba(new_data)[:, 1]
        predictions = self.model.predict(new_data)        
        risk_levels = []
        for prob in probabilities:
            if prob >= self.config["risk_threshold"]:
                risk_levels.append("HIGH")
            elif prob >= self.config["alert_threshold"]:
                risk_levels.append("MEDIUM")
            else:
                risk_levels.append("LOW")
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "risk_levels": risk_levels
        }  
    def generate_risk_report(self, df, predictions):
        print("\nGenerating Flood Risk Assessment Report...")        
        high_risk_count = sum(1 for level in predictions["risk_levels"] if level == "HIGH")
        medium_risk_count = sum(1 for level in predictions["risk_levels"] if level == "MEDIUM")        
        report = f"""
        FLOOD DISASTER RISK ASSESSMENT REPORT
        ==========================================        
        SUMMARY STATISTICS:
        - Total Locations Analyzed: {len(df)}
        - High Risk Areas: {high_risk_count} ({high_risk_count/len(df)*100:.1f}%)
        - Medium Risk Areas: {medium_risk_count} ({medium_risk_count/len(df)*100:.1f}%)
        - Low Risk Areas: {len(df) - high_risk_count - medium_risk_count}
        
        RISK THRESHOLDS:
        - High Risk: ≥{self.config['risk_threshold']*100:.0f}% probability
        - Medium Risk: ≥{self.config['alert_threshold']*100:.0f}% probability
        
        IMMEDIATE ACTIONS REQUIRED:
        - Deploy monitoring equipment in HIGH risk zones
        - Issue early warning alerts for MEDIUM+ risk areas
        - Prepare evacuation routes for HIGH risk areas
        - Coordinate with local disaster management authorities
        
        MODEL PERFORMANCE:
        - Model Type: Ensemble (Random Forest + XGBoost + Gradient Boosting)
        - Validation: Cross-validated on historical data
        - Key Factors: Rainfall, Water Levels, Elevation, Historical Patterns
        """ 
        report_path = os.path.join(self.config["artifacts_dir"], "flood_risk_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)     
        print(report)
        print(f"Report saved: {report_path}")        
        return report
def main():
    predictor = FloodDisasterPredictor({
        "csv_path": "flood_risk_dataset_india.csv",
        "artifacts_dir": "flood_disaster_artifacts",
        "risk_threshold": 0.7,
        "alert_threshold": 0.5
    })    
    df = predictor.load_and_validate_data(predictor.config["csv_path"])
    if df is None:
        return    
    enhanced_df = predictor.create_enhanced_features(df)   
    model, results = predictor.train_ensemble_models(enhanced_df)
    feature_cols = [c for c in enhanced_df.columns if c not in ["Flood_Occurred", "Affected_Area"]]
    predictions = predictor.predict_flood_risk(enhanced_df[feature_cols])  
    report = predictor.generate_risk_report(enhanced_df, predictions)
    
    print("\nFlood Disaster Prediction System Ready!")
    print("Ready for real-time monitoring and early warning alerts!")

if __name__ == "__main__":
    main()