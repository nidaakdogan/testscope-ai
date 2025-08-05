"""
TestScope AI - Model Eğitim Araçları
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

from .risk_predictor import RiskPredictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import TestDataGenerator

class ModelTrainer:
    """Model eğitim ve değerlendirme araçları"""
    
    def __init__(self):
        self.data_generator = TestDataGenerator()
        self.models = {}
        self.best_model = None
        self.training_results = {}
    
    def generate_training_data(self, num_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """Eğitim verisi üretir"""
        
        print(f"{num_samples} adet eğitim verisi üretiliyor...")
        X, y = self.data_generator.generate_training_data(num_samples)
        
        print(f"Veri dağılımı:")
        print(f"PASS: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"FAIL: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        
        return X, y
    
    def train_multiple_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Birden fazla model eğitir ve karşılaştırır"""
        
        models_to_train = {
            'random_forest': RiskPredictor('random_forest'),
            'logistic_regression': RiskPredictor('logistic_regression')
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\n{name.upper()} modeli eğitiliyor...")
            
            # Model eğitimi
            metrics = model.train(X, y)
            
            # Sonuçları kaydet
            self.models[name] = model
            results[name] = metrics
            
            print(f"{name} eğitimi tamamlandı!")
        
        # En iyi modeli seç
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x]['f1_score'])
        self.best_model = self.models[best_model_name]
        
        print(f"\nEn iyi model: {best_model_name}")
        print(f"F1-Score: {results[best_model_name]['f1_score']:.3f}")
        
        self.training_results = results
        return results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'random_forest') -> RiskPredictor:
        """Hiperparametre optimizasyonu yapar"""
        
        print(f"{model_type} için hiperparametre optimizasyonu...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            
        elif model_type == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            base_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        # Veri ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Grid search eğitimi
        grid_search.fit(X_scaled, y)
        
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi F1-Score: {grid_search.best_score_:.3f}")
        
        # Optimize edilmiş modeli oluştur
        optimized_model = RiskPredictor(model_type)
        optimized_model.model = grid_search.best_estimator_
        optimized_model.scaler = scaler
        optimized_model.is_trained = True
        
        return optimized_model
    
    def evaluate_model(self, model: RiskPredictor, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Model performansını değerlendirir"""
        
        # Tahminler
        predictions = model.predict_batch(X)
        y_pred = (predictions['prediction'] == 'FAIL').astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Sınıflandırma raporu
        report = classification_report(y, y_pred, output_dict=True)
        
        # Risk skorları
        risk_scores = predictions['risk_score'].values
        
        evaluation = {
            'confusion_matrix': cm,
            'classification_report': report,
            'risk_scores': risk_scores,
            'predictions': predictions
        }
        
        return evaluation
    
    def create_evaluation_plots(self, evaluation: Dict, save_path: str = 'notebooks/'):
        """Değerlendirme grafikleri oluşturur"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = evaluation['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['PASS', 'FAIL'], 
                   yticklabels=['PASS', 'FAIL'])
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek Değerler')
        plt.xlabel('Tahmin Edilen Değerler')
        plt.tight_layout()
        plt.savefig(f'{save_path}confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Risk Skor Dağılımı
        plt.figure(figsize=(10, 6))
        risk_scores = evaluation['risk_scores']
        plt.hist(risk_scores, bins=30, alpha=0.7, color='green')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Risk Eşiği')
        plt.xlabel('Risk Skoru')
        plt.ylabel('Frekans')
        plt.title('Risk Skor Dağılımı')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}risk_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Özellik Önem Dereceleri
        if self.best_model:
            feature_importance = self.best_model.get_feature_importance()
            if not feature_importance.empty:
                plt.figure(figsize=(8, 6))
                sns.barplot(data=feature_importance, x='importance', y='feature')
                plt.title('Özellik Önem Dereceleri')
                plt.xlabel('Önem Derecesi')
                plt.tight_layout()
                plt.savefig(f'{save_path}feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Grafikler kaydedildi: {save_path}")
    
    def save_training_report(self, save_path: str = 'notebooks/training_report.txt'):
        """Eğitim raporunu kaydeder"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("TestScope AI - Model Eğitim Raporu\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Performans Karşılaştırması:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, metrics in self.training_results.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Doğruluk: {metrics['accuracy']:.3f}\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
                f.write(f"  CV Mean: {metrics['cv_mean']:.3f}\n")
                f.write(f"  CV Std: {metrics['cv_std']:.3f}\n")
            
            if self.best_model:
                f.write(f"\nEn İyi Model: {self.best_model.model_type}\n")
                f.write(f"Model Bilgileri: {self.best_model.get_model_info()}\n")
        
        print(f"Eğitim raporu kaydedildi: {save_path}")
    
    def full_training_pipeline(self, num_samples: int = 5000) -> RiskPredictor:
        """Tam eğitim pipeline'ı çalıştırır"""
        
        print("TestScope AI - Model Eğitim Pipeline'ı Başlatılıyor...")
        print("=" * 60)
        
        # 1. Veri üretimi
        print("\n1. Eğitim verisi üretiliyor...")
        X, y = self.generate_training_data(num_samples)
        
        # 2. Model eğitimi
        print("\n2. Modeller eğitiliyor...")
        results = self.train_multiple_models(X, y)
        
        # 3. Model değerlendirmesi
        print("\n3. Model değerlendirmesi yapılıyor...")
        evaluation = self.evaluate_model(self.best_model, X, y)
        
        # 4. Grafikler oluşturuluyor
        print("\n4. Değerlendirme grafikleri oluşturuluyor...")
        self.create_evaluation_plots(evaluation)
        
        # 5. Rapor kaydediliyor
        print("\n5. Eğitim raporu kaydediliyor...")
        self.save_training_report()
        
        # 6. En iyi modeli kaydet
        print("\n6. En iyi model kaydediliyor...")
        self.best_model.save_model()
        
        print("\nEğitim pipeline'ı tamamlandı!")
        print(f"En iyi model: {self.best_model.model_type}")
        print(f"F1-Score: {results[self.best_model.model_type]['f1_score']:.3f}")
        
        return self.best_model

if __name__ == "__main__":
    # Tam eğitim pipeline'ı çalıştır
    trainer = ModelTrainer()
    best_model = trainer.full_training_pipeline() 