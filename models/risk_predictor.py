"""
TestScope AI - Risk Tahmin Modelleri
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RiskPredictor:
    """Çevresel test risk tahmin modeli"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['temperature', 'humidity', 'vibration', 'pressure']
        
        # Model seçimi
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Modeli eğitir"""
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Özellik ölçeklendirme
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model eğitimi
        self.model.fit(X_train_scaled, y_train)
        
        # Tahminler
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Performans metrikleri
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
        self.f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        self.cv_mean = cv_scores.mean()
        self.cv_std = cv_scores.std()
        
        self.is_trained = True
        
        # Sonuçları yazdır
        print(f"Model Eğitimi Tamamlandı - {self.model_type.upper()}")
        print(f"Test Doğruluğu: {self.accuracy:.3f}")
        print(f"Precision: {self.precision:.3f}")
        print(f"Recall: {self.recall:.3f}")
        print(f"F1-Score: {self.f1:.3f}")
        print(f"Cross-Validation: {self.cv_mean:.3f} (+/- {self.cv_std * 2:.3f})")
        
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std
        }
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Risk tahmini yapar"""
        
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Özellik ölçeklendirme
        X_scaled = self.scaler.transform(X)
        
        # Tahminler
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        # Risk skoru (FAIL olasılığı)
        risk_score = probability[1] if len(probability) > 1 else 0.0
        
        # Sonuç
        result = {
            'prediction': 'FAIL' if prediction == 1 else 'PASS',
            'risk_score': round(risk_score, 3),
            'confidence': round(max(probability), 3),
            'pass_probability': round(probability[0], 3),
            'fail_probability': round(probability[1], 3)
        }
        
        return result
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Toplu tahmin yapar"""
        
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Özellik ölçeklendirme
        X_scaled = self.scaler.transform(X)
        
        # Tahminler
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Sonuçları DataFrame'e ekle
        results = X.copy()
        results['prediction'] = ['FAIL' if p == 1 else 'PASS' for p in predictions]
        results['risk_score'] = [round(prob[1], 3) for prob in probabilities]
        results['confidence'] = [round(max(prob), 3) for prob in probabilities]
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Özellik önem derecelerini döndürür"""
        
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, filepath: str = 'models/risk_predictor.joblib'):
        """Modeli kaydeder"""
        
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Dizin oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Model ve scaler'ı kaydet
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1
            }
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath: str = 'models/risk_predictor.joblib'):
        """Modeli yükler"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        
        # Model verilerini yükle
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        # Metrikleri yükle
        if 'metrics' in model_data:
            self.accuracy = model_data['metrics']['accuracy']
            self.precision = model_data['metrics']['precision']
            self.recall = model_data['metrics']['recall']
            self.f1 = model_data['metrics']['f1_score']
        
        self.is_trained = True
        print(f"Model yüklendi: {filepath}")
    
    def get_model_info(self) -> dict:
        """Model bilgilerini döndürür"""
        
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        if self.is_trained:
            info.update({
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1
            })
        
        return info 