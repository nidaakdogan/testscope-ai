"""
TestScope AI - Veri İşleme Araçları
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class DataProcessor:
    """Veri işleme ve analiz araçları"""
    
    def __init__(self):
        self.test_limits = {
            'temperature': {'min': -40, 'max': 70, 'unit': '°C'},
            'humidity': {'min': 10, 'max': 95, 'unit': '%'},
            'vibration': {'min': 0.1, 'max': 50.0, 'unit': 'g'},
            'pressure': {'min': 800, 'max': 1200, 'unit': 'hPa'}
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """CSV dosyasından veri yükler"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Veri yüklendi: {len(df)} kayıt")
        return df
    
    def validate_test_parameters(self, temperature: float, humidity: float, 
                               vibration: float, pressure: float) -> Dict:
        """Test parametrelerini doğrular"""
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Sıcaklık kontrolü
        if temperature < self.test_limits['temperature']['min']:
            validation['warnings'].append(
                f"Sıcaklık çok düşük: {temperature}°C "
                f"(minimum: {self.test_limits['temperature']['min']}°C)"
            )
        elif temperature > self.test_limits['temperature']['max']:
            validation['warnings'].append(
                f"Sıcaklık çok yüksek: {temperature}°C "
                f"(maksimum: {self.test_limits['temperature']['max']}°C)"
            )
        
        # Nem kontrolü
        if humidity < self.test_limits['humidity']['min']:
            validation['warnings'].append(
                f"Nem çok düşük: {humidity}% "
                f"(minimum: {self.test_limits['humidity']['min']}%)"
            )
        elif humidity > self.test_limits['humidity']['max']:
            validation['warnings'].append(
                f"Nem çok yüksek: {humidity}% "
                f"(maksimum: {self.test_limits['humidity']['max']}%)"
            )
        
        # Titreşim kontrolü
        if vibration < self.test_limits['vibration']['min']:
            validation['warnings'].append(
                f"Titreşim çok düşük: {vibration}g "
                f"(minimum: {self.test_limits['vibration']['min']}g)"
            )
        elif vibration > self.test_limits['vibration']['max']:
            validation['warnings'].append(
                f"Titreşim çok yüksek: {vibration}g "
                f"(maksimum: {self.test_limits['vibration']['max']}g)"
            )
        
        # Basınç kontrolü
        if pressure < self.test_limits['pressure']['min']:
            validation['warnings'].append(
                f"Basınç çok düşük: {pressure}hPa "
                f"(minimum: {self.test_limits['pressure']['min']}hPa)"
            )
        elif pressure > self.test_limits['pressure']['max']:
            validation['warnings'].append(
                f"Basınç çok yüksek: {pressure}hPa "
                f"(maksimum: {self.test_limits['pressure']['max']}hPa)"
            )
        
        # Kritik hatalar
        if len(validation['errors']) > 0:
            validation['is_valid'] = False
        
        return validation
    
    def calculate_risk_factors(self, temperature: float, humidity: float, 
                             vibration: float, pressure: float) -> Dict:
        """Risk faktörlerini hesaplar - Gerçek risk değerleri (0-1 aralığı)"""
        
        risk_factors = {}
        
        # Sıcaklık riski - 60°C üzeri yüksek risk
        if temperature >= 60:
            temp_risk = 0.95  # %95 risk (neredeyse %100)
        elif temperature >= 50:
            temp_risk = 0.8   # %80 risk
        elif temperature >= 40:
            temp_risk = 0.6   # %60 risk
        elif temperature >= 30:
            temp_risk = 0.4   # %40 risk
        elif temperature >= 20:
            temp_risk = 0.2   # %20 risk
        else:
            temp_risk = 0.1   # %10 risk (düşük)
        
        risk_factors['temperature_risk'] = temp_risk
        
        # Nem riski - %85 nem yüksek risk
        if humidity >= 90:
            humidity_risk = 0.95  # %95 risk
        elif humidity >= 85:
            humidity_risk = 0.85  # %85 risk (görseldeki değer)
        elif humidity >= 80:
            humidity_risk = 0.7   # %70 risk
        elif humidity >= 70:
            humidity_risk = 0.5   # %50 risk
        elif humidity >= 60:
            humidity_risk = 0.3   # %30 risk
        else:
            humidity_risk = 0.1   # %10 risk
        
        risk_factors['humidity_risk'] = humidity_risk
        
        # Titreşim riski - 25g orta risk
        if vibration >= 40:
            vibration_risk = 0.9   # %90 risk
        elif vibration >= 30:
            vibration_risk = 0.8   # %80 risk
        elif vibration >= 25:
            vibration_risk = 0.5   # %50 risk (görseldeki değer)
        elif vibration >= 20:
            vibration_risk = 0.4   # %40 risk
        elif vibration >= 10:
            vibration_risk = 0.2   # %20 risk
        else:
            vibration_risk = 0.1   # %10 risk
        
        risk_factors['vibration_risk'] = vibration_risk
        
        # Basınç riski - 1013hPa normal basınç, düşük risk
        if pressure <= 850 or pressure >= 1150:
            pressure_risk = 0.8   # %80 risk
        elif pressure <= 900 or pressure >= 1100:
            pressure_risk = 0.5   # %50 risk
        elif pressure <= 950 or pressure >= 1050:
            pressure_risk = 0.2   # %20 risk
        else:
            pressure_risk = 0.05  # %5 risk (1013hPa normal)
        
        risk_factors['pressure_risk'] = pressure_risk
        
        # Toplam risk - Ağırlıklı ortalama
        total_risk = (temp_risk + humidity_risk + vibration_risk + pressure_risk) / 4
        risk_factors['total_risk'] = total_risk
        
        return risk_factors
    
    def get_test_recommendations(self, risk_score: float) -> List[str]:
        """Risk skoruna göre test önerileri"""
        
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.extend([
                "Yüksek risk! Detaylı test planı hazırlanmalı.",
                "Test süresi kısaltılmalı.",
                "Sürekli izleme gerekli.",
                "Yedek ekipman hazır bulundurulmalı."
            ])
        elif risk_score > 0.6:
            recommendations.extend([
                "Orta-yüksek risk. Dikkatli test yapılmalı.",
                "Test parametreleri aşamalı olarak artırılmalı.",
                "Düzenli kontrol noktaları belirlenmeli."
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Orta risk. Standart test prosedürü uygulanabilir.",
                "Test süresi normal tutulabilir."
            ])
        else:
            recommendations.extend([
                "Düşük risk. Güvenli test koşulları.",
                "Standart test protokolü uygulanabilir."
            ])
        
        return recommendations
    
    def analyze_test_trends(self, df: pd.DataFrame) -> Dict:
        """Test verilerindeki trendleri analiz eder"""
        
        analysis = {}
        
        # Test kategorilerine göre dağılım
        if 'test_category' in df.columns:
            category_counts = df['test_category'].value_counts()
            analysis['category_distribution'] = category_counts.to_dict()
        
        # Pass/Fail oranları
        if 'pass_fail' in df.columns:
            pass_fail_counts = df['pass_fail'].value_counts()
            analysis['pass_fail_ratio'] = {
                'pass_count': pass_fail_counts.get('PASS', 0),
                'fail_count': pass_fail_counts.get('FAIL', 0),
                'pass_rate': pass_fail_counts.get('PASS', 0) / len(df) * 100
            }
        
        # Risk skoru analizi
        if 'risk_score' in df.columns:
            analysis['risk_analysis'] = {
                'mean_risk': df['risk_score'].mean(),
                'median_risk': df['risk_score'].median(),
                'std_risk': df['risk_score'].std(),
                'min_risk': df['risk_score'].min(),
                'max_risk': df['risk_score'].max()
            }
        
        # Test süresi analizi
        if 'test_duration' in df.columns:
            analysis['duration_analysis'] = {
                'mean_duration': df['test_duration'].mean(),
                'median_duration': df['test_duration'].median(),
                'total_hours': df['test_duration'].sum() / 60  # Saat cinsinden
            }
        
        return analysis
    
    def export_test_report(self, test_data: Dict, analysis_results: Dict, 
                          filename: str = 'test_report.txt') -> None:
        """Test raporunu dosyaya kaydeder"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("TestScope AI - Test Raporu\n")
            f.write("=" * 40 + "\n\n")
            
            # Test parametreleri
            f.write("Test Parametreleri:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Sıcaklık: {test_data.get('temperature', 'N/A')}°C\n")
            f.write(f"Nem: {test_data.get('humidity', 'N/A')}%\n")
            f.write(f"Titreşim: {test_data.get('vibration', 'N/A')}g\n")
            f.write(f"Basınç: {test_data.get('pressure', 'N/A')}hPa\n\n")
            
            # Tahmin sonuçları
            if 'prediction' in test_data:
                f.write("Tahmin Sonuçları:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Tahmin: {test_data['prediction']}\n")
                f.write(f"Risk Skoru: {test_data.get('risk_score', 'N/A')}\n")
                f.write(f"Güven: {test_data.get('confidence', 'N/A')}\n\n")
            
            # Analiz sonuçları
            if analysis_results:
                f.write("Analiz Sonuçları:\n")
                f.write("-" * 20 + "\n")
                for key, value in analysis_results.items():
                    f.write(f"{key}: {value}\n")
        
        print(f"Test raporu kaydedildi: {filename}")
    
    def get_test_standards_info(self) -> Dict:
        """Test standartları hakkında bilgi döndürür"""
        
        standards_info = {
            'MIL-STD-810': {
                'description': 'Askeri ekipmanlar için çevresel test standartları',
                'methods': {
                    '501.7': 'Yüksek sıcaklık testi',
                    '502.7': 'Düşük sıcaklık testi',
                    '507.7': 'Nem testi',
                    '514.8': 'Mekanik titreşim testi',
                    '515.8': 'Akustik titreşim testi'
                }
            },
            'ISO 16750': {
                'description': 'Otomotiv elektronik ekipmanları için test standartları',
                'methods': {
                    'ISO 16750-4': 'Nem ve su testleri',
                    'ISO 16750-5': 'Kimyasal testler'
                }
            },
            'IEC 60068': {
                'description': 'Çevresel test standartları',
                'methods': {
                    'IEC 60068-2-14': 'Termal şok testi',
                    'IEC 60068-2-27': 'Darbe testi',
                    'IEC 60529': 'Su sıçrama testi'
                }
            }
        }
        
        return standards_info 