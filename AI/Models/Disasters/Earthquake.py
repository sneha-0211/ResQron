import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class EarthquakePredictionPipeline:
    
    def __init__(self, config=None):
        self.config = config or {
            "n_samples": 2500,
            "artifacts_dir": "earthquake_artifacts",
            "alert_threshold": 0.5,
            "random_state": 42
        }
        
        os.makedirs(self.config["artifacts_dir"], exist_ok=True)
        
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.metrics = {}
        
    def generate_synthetic_dataset(self):
        print("Generating synthetic earthquake dataset...")
        
        np.random.seed(self.config["random_state"])
        n = self.config["n_samples"]
        
        # Time series data - 6 months of hourly data
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(n)]
        
        # Geographic features (India/Nepal seismic zone)
        latitudes = np.random.uniform(25.0, 32.0, n)  # Himalayan region
        longitudes = np.random.uniform(75.0, 88.0, n)
        
        # Seismic features
        magnitudes = np.random.exponential(1.5, n) + 4.0  # Exponential distribution
        magnitudes = np.clip(magnitudes, 4.0, 8.5)
        
        depths = np.random.lognormal(2.5, 0.8, n)  # Log-normal distribution
        depths = np.clip(depths, 5, 80)
        
        # Historical earthquake activity
        past_30d_quakes = np.random.poisson(2, n)  # Poisson distribution
        past_30d_quakes = np.clip(past_30d_quakes, 0, 20)
        
        # Distance to fault lines
        distance_to_fault = np.random.exponential(10, n)
        distance_to_fault = np.clip(distance_to_fault, 0, 50)
        
        # Soil characteristics
        vs30 = np.random.normal(400, 150, n)
        vs30 = np.clip(vs30, 150, 800)
        
        # Population and infrastructure
        population_density = np.random.lognormal(6, 1, n)
        population_density = np.clip(population_density, 100, 5000)
        
        building_quality = np.random.beta(2, 3, n)  # Skewed towards lower quality
        building_quality = 0.2 + building_quality * 0.8
        
        infrastructure = np.random.binomial(1, 0.6, n)  # 60% have infrastructure
        
        # Create realistic target variables
        event_24h, damage_area = self._generate_targets(
            magnitudes, depths, distance_to_fault, past_30d_quakes,
            population_density, building_quality, infrastructure
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Latitude': latitudes,
            'Longitude': longitudes,
            'Magnitude': magnitudes,
            'Depth_km': depths,
            'Past_30d_quakes': past_30d_quakes,
            'Distance_to_fault_km': distance_to_fault,
            'Vs30': vs30,
            'Population_Density': population_density,
            'Building_Quality': building_quality,
            'Infrastructure': infrastructure,
            'Event_24h': event_24h,
            'Damage_Area_km2': damage_area
        })
        
        # Save dataset
        dataset_path = os.path.join(self.config["artifacts_dir"], "earthquake_dataset.csv")
        df.to_csv(dataset_path, index=False)
        
        print(f"Dataset created with {len(df)} samples")
        print(f"Event rate: {df['Event_24h'].mean():.1%}")
        print(f"Average damage area: {df['Damage_Area_km2'].mean():.1f} km²")
        print(f"Dataset saved to: {dataset_path}")
        
        return df
    
    def _generate_targets(self, magnitudes, depths, distance_to_fault, 
                          past_30d_quakes, population_density, building_quality, infrastructure):
        
        n = len(magnitudes)
        
        # Event probability based on seismic factors
        event_prob = (
            0.3 * (magnitudes - 4.0) / 4.5 +  # Higher magnitude = higher probability
            0.2 * (1 / (depths / 10)) +        # Shallower = higher probability
            0.25 * (1 / (distance_to_fault + 1)) +  # Closer to fault = higher probability
            0.25 * (past_30d_quakes / 20)      # Recent activity = higher probability
        )
        
        # Add noise and convert to probability
        event_prob += np.random.normal(0, 0.1, n)
        event_prob = np.clip(event_prob, 0, 1)
        
        # Generate binary events
        event_24h = np.random.binomial(1, event_prob)
        
        # Damage area calculation (only for events)
        base_damage = np.power(10, magnitudes - 4) * 2  # Exponential with magnitude
        
        # Depth factor (shallower = more damage)
        depth_factor = np.maximum(0.1, 1 - depths / 100)
        
        # Population factor
        pop_factor = np.log1p(population_density) / 10
        
        # Building quality factor (poor quality = more damage)
        quality_factor = 2 - building_quality
        
        # Infrastructure factor
        infra_factor = 1.5 - 0.3 * infrastructure
        
        damage_area = (base_damage * depth_factor * pop_factor * quality_factor * infra_factor)
        
        # Add noise
        damage_area *= np.random.lognormal(0, 0.3, n)
        
        # Set damage to 0 for non-events, but keep some background damage
        damage_area = np.where(event_24h == 1, damage_area, damage_area * 0.1)
        damage_area = np.clip(damage_area, 0.1, 1000)
        
        return event_24h, damage_area
    
    def preprocess_data(self, df):
        print("Preprocessing data...")
        
        # Feature engineering
        df_processed = df.copy()
        
        # Time-based features
        df_processed['Hour'] = df_processed['Timestamp'].dt.hour
        df_processed['DayOfWeek'] = df_processed['Timestamp'].dt.dayofweek
        df_processed['Month'] = df_processed['Timestamp'].dt.month
        
        # Risk interaction features
        df_processed['Magnitude_Depth_Ratio'] = df_processed['Magnitude'] / df_processed['Depth_km']
        df_processed['Population_Building_Risk'] = (
            df_processed['Population_Density'] * (2 - df_processed['Building_Quality'])
        )
        df_processed['Seismic_Risk_Score'] = (
            df_processed['Magnitude'] * (1 / (df_processed['Distance_to_fault_km'] + 1)) *
            (1 / (df_processed['Depth_km'] / 10))
        )
        
        # Define feature columns
        feature_cols = [
            'Latitude', 'Longitude', 'Magnitude', 'Depth_km', 'Past_30d_quakes',
            'Distance_to_fault_km', 'Vs30', 'Population_Density', 'Building_Quality',
            'Infrastructure', 'Hour', 'DayOfWeek', 'Month', 'Magnitude_Depth_Ratio',
            'Population_Building_Risk', 'Seismic_Risk_Score'
        ]
        
        # Time-based split (80% train, 20% test)
        split_idx = int(0.8 * len(df_processed))
        
        train_df = df_processed.iloc[:split_idx]
        test_df = df_processed.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train_class = train_df['Event_24h']
        y_test_class = test_df['Event_24h']
        y_train_reg = train_df['Damage_Area_km2']
        y_test_reg = test_df['Damage_Area_km2']
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_cols)}")
        
        return (X_train_scaled, X_test_scaled, y_train_class, y_test_class, 
                y_train_reg, y_test_reg, feature_cols, test_df)
    
    def train_classifier(self, X_train, y_train, X_test, y_test):
        print("Training earthquake occurrence classifier...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Base XGBoost classifier
        base_clf = xgb.XGBClassifier(
            random_state=self.config["random_state"],
            eval_metric='logloss'
        )
        
        # Hyperparameter tuning
        random_search = RandomizedSearchCV(
            base_clf,
            param_grid,
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.config["random_state"]
        )
        
        random_search.fit(X_train, y_train)
        
        # Best model with calibration
        self.classifier = CalibratedClassifierCV(
            random_search.best_estimator_,
            method='sigmoid',
            cv=5
        )
        self.classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.config["alert_threshold"]).astype(int)
        
        # Evaluate classifier
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        self.metrics['classifier'] = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'best_params': random_search.best_params_
        }
        
        print(f"Classifier ROC-AUC: {roc_auc:.3f}")
        print(f"Classifier PR-AUC: {pr_auc:.3f}")
        
        # Save classifier
        classifier_path = os.path.join(self.config["artifacts_dir"], "earthquake_classifier.joblib")
        joblib.dump(self.classifier, classifier_path)
        
        return y_pred_proba, y_pred
    
    def train_regressor(self, X_train, y_train, X_test, y_test):
        print("Training damage area regressor...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Base XGBoost regressor
        base_reg = xgb.XGBRegressor(
            random_state=self.config["random_state"]
        )
        
        # Hyperparameter tuning
        random_search = RandomizedSearchCV(
            base_reg,
            param_grid,
            n_iter=20,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=self.config["random_state"]
        )
        
        random_search.fit(X_train, y_train)
        self.regressor = random_search.best_estimator_
        
        # Predictions
        y_pred = self.regressor.predict(X_test)
        
        # Evaluate regressor
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.metrics['regressor'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'best_params': random_search.best_params_
        }
        
        print(f"Regressor MAE: {mae:.2f} km²")
        print(f"Regressor RMSE: {rmse:.2f} km²")
        print(f"Regressor R²: {r2:.3f}")
        
        # Save regressor
        regressor_path = os.path.join(self.config["artifacts_dir"], "earthquake_regressor.joblib")
        joblib.dump(self.regressor, regressor_path)
        
        return y_pred
    
    def create_visualizations(self, y_test_class, y_pred_proba_class, y_pred_class,
                              y_test_reg, y_pred_reg, feature_cols):
        print("Creating visualizations...")
        
        # 1. Feature importance plots
        self._plot_feature_importance(feature_cols)
        
        # 2. Precision-Recall curve
        self._plot_precision_recall_curve(y_test_class, y_pred_proba_class)
        
        # 3. Calibration plot
        self._plot_calibration_curve(y_test_class, y_pred_proba_class)
        
        # 4. Regression performance
        self._plot_regression_performance(y_test_reg, y_pred_reg)
        
        # 5. Confusion matrix
        self._plot_confusion_matrix(y_test_class, y_pred_class)
    
    def _plot_feature_importance(self, feature_cols):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        clf_model = None
        if isinstance(self.classifier, CalibratedClassifierCV):
            try:
                clf_model = self.classifier.calibrated_classifiers_[0].estimator
            except Exception:
                clf_model = None
        else:
            clf_model = self.classifier
        
        if clf_model is not None and hasattr(clf_model, "feature_importances_"):
            importances_clf = clf_model.feature_importances_
            indices_clf = np.argsort(importances_clf)[::-1][:10]
            ax1.bar(range(len(indices_clf)), importances_clf[indices_clf])
            ax1.set_title("Top 10 Features - Earthquake Classifier")
            ax1.set_xlabel("Feature Importance")
            ax1.set_xticks(range(len(indices_clf)))
            ax1.set_xticklabels([feature_cols[i] for i in indices_clf], rotation=45)
        else:
            ax1.text(0.5, 0.5, "Feature importances not available",
                     ha="center", va="center", fontsize=12)
            ax1.set_title("Earthquake Classifier")
            
        if hasattr(self.regressor, "feature_importances_"):
            importances_reg = self.regressor.feature_importances_
            indices_reg = np.argsort(importances_reg)[::-1][:10]
            ax2.bar(range(len(indices_reg)), importances_reg[indices_reg])
            ax2.set_title("Top 10 Features - Damage Regressor")
            ax2.set_xlabel("Feature Importance")
            ax2.set_xticks(range(len(indices_reg)))
            ax2.set_xticklabels([feature_cols[i] for i in indices_reg], rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["artifacts_dir"], "feature_importance.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_precision_recall_curve(self, y_true, y_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Earthquake Classifier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config["artifacts_dir"], "precision_recall_curve.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, y_true, y_proba):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Classifier")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot - Earthquake Classifier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config["artifacts_dir"], "calibration_curve.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regression_performance(self, y_true, y_pred):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Predicted vs Actual
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Damage Area (km²)')
        ax1.set_ylabel('Predicted Damage Area (km²)')
        ax1.set_title('Predicted vs Actual Damage Area')
        
        # Residuals
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Damage Area (km²)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["artifacts_dir"], "regression_performance.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Earthquake Classifier')
        plt.savefig(os.path.join(self.config["artifacts_dir"], "confusion_matrix.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_alerts(self, test_df, X_test, y_pred_proba, y_pred_reg):
        print("Generating earthquake alerts...")
        
        # Create alerts dataframe
        alerts_df = pd.DataFrame({
            'Timestamp': test_df['Timestamp'].values,
            'Latitude': test_df['Latitude'].values,
            'Longitude': test_df['Longitude'].values,
            'Event_Prob': y_pred_proba,
            'Predicted_Event_24h': (y_pred_proba >= self.config["alert_threshold"]).astype(int),
            'Predicted_Damage_Area': y_pred_reg,
            'Actual_Event': test_df['Event_24h'].values,
            'Actual_Damage': test_df['Damage_Area_km2'].values
        })
        
        # Risk categories
        alerts_df['Risk_Level'] = pd.cut(
            alerts_df['Event_Prob'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH'],
            include_lowest=True
        )
        
        # Filter high-risk alerts
        high_risk_alerts = alerts_df[alerts_df['Event_Prob'] >= self.config["alert_threshold"]]
        
        # Save alerts
        alerts_path = os.path.join(self.config["artifacts_dir"], "earthquake_alerts.csv")
        alerts_df.to_csv(alerts_path, index=False)
        
        print(f"Total alerts generated: {len(alerts_df)}")
        print(f"High-risk alerts: {len(high_risk_alerts)}")
        
        return alerts_df, high_risk_alerts
    
    def save_metrics_and_report(self, alerts_df, high_risk_alerts):
        """
        Save metrics JSON and a UTF-8 encoded human-readable risk report.
        """
        print("Saving metrics and generating report...")

        # Save metrics to JSON (UTF-8)
        metrics_path = os.path.join(self.config["artifacts_dir"], "model_metrics.json")
        with open(metrics_path, 'w', encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)

        # Prepare summary stats
        total_alerts = len(alerts_df)
        high_risk_count = len(high_risk_alerts)
        medium_risk_count = len(alerts_df[alerts_df['Risk_Level'] == 'MEDIUM'])
        avg_damage_high_risk = high_risk_alerts['Predicted_Damage_Area'].mean() if len(high_risk_alerts) > 0 else 0.0

        # Build report (avoid special Unicode characters to be safe)
        report = f"""EARTHQUAKE 24H RISK ASSESSMENT REPORT
====================================

ALERT SUMMARY:
- Total Monitoring Points: {total_alerts}
- High Risk Areas (>=50% probability): {high_risk_count} ({(high_risk_count/total_alerts*100) if total_alerts>0 else 0:.1f}%)
- Medium Risk Areas (30-70% probability): {medium_risk_count} ({(medium_risk_count/total_alerts*100) if total_alerts>0 else 0:.1f}%)
- Average Predicted Damage (High Risk): {avg_damage_high_risk:.1f} km^2

MODEL PERFORMANCE:
Classifier:
- ROC-AUC: {self.metrics.get('classifier',{}).get('roc_auc', 0):.3f}
- PR-AUC: {self.metrics.get('classifier',{}).get('pr_auc', 0):.3f}

Regressor:
- MAE: {self.metrics.get('regressor',{}).get('mae', 0):.2f} km^2
- RMSE: {self.metrics.get('regressor',{}).get('rmse', 0):.2f} km^2
- R^2: {self.metrics.get('regressor',{}).get('r2', 0):.3f}

IMMEDIATE ACTIONS:
1. Deploy monitoring in HIGH risk zones
2. Issue early warnings for MEDIUM+ risk areas
3. Prepare emergency response for predicted damage areas
4. Coordinate with local disaster management

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Save report with UTF-8 encoding
        report_path = os.path.join(self.config["artifacts_dir"], "earthquake_risk_report.txt")
        with open(report_path, 'w', encoding="utf-8") as f:
            f.write(report)

        print(f"Metrics saved to: {metrics_path}")
        print(f"Risk report saved to: {report_path}")

        return report

    def run_pipeline(self):
        """
        Full pipeline orchestration: dataset -> preprocess -> train -> visualize -> alerts -> report
        """
        print("Starting Earthquake Prediction ML Pipeline...")
        print("=" * 50)

        # Step 1: Generate dataset
        df = self.generate_synthetic_dataset()

        # Step 2: Preprocess data
        (X_train, X_test, y_train_class, y_test_class,
         y_train_reg, y_test_reg, feature_cols, test_df) = self.preprocess_data(df)

        # Step 3: Train classifier
        y_pred_proba_class, y_pred_class = self.train_classifier(
            X_train, y_train_class, X_test, y_test_class
        )

        # Step 4: Train regressor
        y_pred_reg = self.train_regressor(X_train, y_train_reg, X_test, y_test_reg)

        # Step 5: Create visualizations
        self.create_visualizations(
            y_test_class, y_pred_proba_class, y_pred_class,
            y_test_reg, y_pred_reg, feature_cols
        )

        # Step 6: Generate alerts
        alerts_df, high_risk_alerts = self.generate_alerts(
            test_df, X_test, y_pred_proba_class, y_pred_reg
        )

        # Step 7: Save metrics and report
        report = self.save_metrics_and_report(alerts_df, high_risk_alerts)

        # Step 8: Save scaler
        scaler_path = os.path.join(self.config["artifacts_dir"], "feature_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        print("\nPipeline completed successfully!")
        print(f"All artifacts saved to: {self.config['artifacts_dir']}")

        return alerts_df, high_risk_alerts, report

def main():
    # Configuration
    config = {
        "n_samples": 2500,
        "artifacts_dir": "earthquake_artifacts",
        "alert_threshold": 0.5,
        "random_state": 42
    }
    
    # Initialize and run pipeline
    pipeline = EarthquakePredictionPipeline(config)
    alerts_df, high_risk_alerts, report = pipeline.run_pipeline()
    
    # Display sample alerts
    print("\n" + "="*60)
    print("SAMPLE HIGH-RISK EARTHQUAKE ALERTS")
    print("="*60)
    
    if len(high_risk_alerts) > 0:
        sample_alerts = high_risk_alerts.head(10)
        for idx, alert in sample_alerts.iterrows():
            print(f"\nAlert #{idx + 1}:")
            print(f"   Time: {alert['Timestamp']}")
            print(f"   Location: {alert['Latitude']:.3f}°N, {alert['Longitude']:.3f}°E")
            print(f"   Event Probability: {alert['Event_Prob']:.1%}")
            print(f"   Predicted Damage Area: {alert['Predicted_Damage_Area']:.1f} km²")
            print(f"   Risk Level: {alert['Risk_Level']}")
    else:
        print("No high-risk alerts in current test period.")
    
    print(f"\nTotal Alerts: {len(alerts_df)}")
    print(f"High-Risk Alerts: {len(high_risk_alerts)}")
    
    # Print summary metrics
    print(f"\nModel Performance Summary:")
    print(f"Classifier ROC-AUC: {pipeline.metrics['classifier']['roc_auc']:.3f}")
    print(f"Regressor RMSE: {pipeline.metrics['regressor']['rmse']:.2f} km²")
    
    print("\nEarthquake ML Pipeline completed successfully!")

if __name__ == "__main__":
    main()
