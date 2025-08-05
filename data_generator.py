"""
TestScope AI - Sentetik Veri Üretici
MIL-STD-810, ISO 16750, IEC 60068 standartlarına uygun mock data üretir.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import random

class TestDataGenerator:
    """Çevresel test standartlarına uygun sentetik veri üretici"""
    
    def __init__(self):
        # Test standartları limit değerleri
        self.test_limits = {
            'temperature': {
                'min': -40,
                'max': 70,
                'unit': '°C'
            },
            'humidity': {
                'min': 10,
                'max': 95,
                'unit': '%'
            },
            'vibration': {
                'min': 0.1,
                'max': 50.0,
                'unit': 'g'
            },
            'pressure': {
                'min': 800,
                'max': 1200,
                'unit': 'hPa'
            }
        }
        
        # Test kategorileri
        self.test_categories = {
            'temperature': ['high_temp', 'low_temp', 'thermal_shock'],
            'humidity': ['humidity_resistance', 'condensation', 'water_splash'],
            'vibration': ['mechanical_vibration', 'acoustic_vibration', 'shock']
        }
    
    def generate_test_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Ana test verisi üretir"""
        
        data = []
        
        for i in range(num_samples):
            # Test parametreleri
            temp = np.random.uniform(self.test_limits['temperature']['min'], 
                                   self.test_limits['temperature']['max'])
            humidity = np.random.uniform(self.test_limits['humidity']['min'], 
                                       self.test_limits['humidity']['max'])
            vibration = np.random.uniform(self.test_limits['vibration']['min'], 
                                        self.test_limits['vibration']['max'])
            pressure = np.random.uniform(self.test_limits['pressure']['min'], 
                                       self.test_limits['pressure']['max'])
            
            # Test kategorisi seçimi
            test_category = random.choice(list(self.test_categories.keys()))
            test_type = random.choice(self.test_categories[test_category])
            
            # Risk hesaplama (basit kurallar)
            risk_score = self._calculate_risk_score(temp, humidity, vibration, pressure)
            
            # Pass/Fail belirleme
            pass_fail = 'PASS' if risk_score < 0.7 else 'FAIL'
            
            # Test süresi (dakika)
            test_duration = np.random.randint(30, 480)  # 30 dakika - 8 saat
            
            # Test tarihi
            test_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
            
            data.append({
                'test_id': f'TEST_{i+1:06d}',
                'test_category': test_category,
                'test_type': test_type,
                'temperature': round(temp, 2),
                'humidity': round(humidity, 2),
                'vibration': round(vibration, 2),
                'pressure': round(pressure, 2),
                'risk_score': round(risk_score, 3),
                'pass_fail': pass_fail,
                'test_duration': test_duration,
                'test_date': test_date,
                'standard': self._get_standard(test_category, test_type)
            })
        
        return pd.DataFrame(data)
    
    def _calculate_risk_score(self, temp: float, humidity: float, 
                             vibration: float, pressure: float) -> float:
        """Risk skoru hesaplar (0-1 arası) - Güncellenmiş mantık"""
        
        risk = 0.0
        
        # Sıcaklık riski - Daha katı kurallar
        if temp > 65 or temp < -35:
            risk += 0.5  # Çok yüksek risk
        elif temp > 60 or temp < -30:
            risk += 0.4  # Yüksek risk
        elif temp > 50 or temp < -20:
            risk += 0.3  # Orta-yüksek risk
        elif temp > 40 or temp < -10:
            risk += 0.2  # Orta risk
        elif temp > 30 or temp < 0:
            risk += 0.1  # Düşük risk
        
        # Nem riski - Daha katı kurallar
        if humidity > 95:
            risk += 0.4  # Çok yüksek risk
        elif humidity > 90:
            risk += 0.3  # Yüksek risk
        elif humidity > 80:
            risk += 0.2  # Orta risk
        elif humidity < 15:
            risk += 0.15  # Düşük risk
        
        # Titreşim riski - Daha katı kurallar
        if vibration > 40:
            risk += 0.5  # Çok yüksek risk
        elif vibration > 30:
            risk += 0.4  # Yüksek risk
        elif vibration > 20:
            risk += 0.3  # Orta risk
        elif vibration > 10:
            risk += 0.2  # Düşük risk
        
        # Basınç riski - Daha katı kurallar
        if pressure < 850 or pressure > 1150:
            risk += 0.3  # Yüksek risk
        elif pressure < 900 or pressure > 1100:
            risk += 0.2  # Orta risk
        elif pressure < 950 or pressure > 1050:
            risk += 0.1  # Düşük risk
        
        # Kombinasyon riski - Birden fazla yüksek parametre varsa ek risk
        high_risk_params = 0
        if temp > 60 or temp < -30:
            high_risk_params += 1
        if humidity > 90:
            high_risk_params += 1
        if vibration > 30:
            high_risk_params += 1
        if pressure < 900 or pressure > 1100:
            high_risk_params += 1
        
        if high_risk_params >= 2:
            risk += 0.2  # Kombinasyon riski
        
        # Rastgele faktör (daha az etkili)
        risk += np.random.normal(0, 0.03)
        
        return max(0.0, min(1.0, risk))
    
    def _get_standard(self, category: str, test_type: str) -> str:
        """Test standardını belirler"""
        
        standards = {
            'temperature': {
                'high_temp': 'MIL-STD-810 Method 501.7',
                'low_temp': 'MIL-STD-810 Method 502.7',
                'thermal_shock': 'IEC 60068-2-14'
            },
            'humidity': {
                'humidity_resistance': 'MIL-STD-810 Method 507.7',
                'condensation': 'ISO 16750-4',
                'water_splash': 'IEC 60529'
            },
            'vibration': {
                'mechanical_vibration': 'MIL-STD-810 Method 514.8',
                'acoustic_vibration': 'MIL-STD-810 Method 515.8',
                'shock': 'IEC 60068-2-27'
            }
        }
        
        return standards.get(category, {}).get(test_type, 'ISO 16750')
    
    def generate_training_data(self, num_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """Eğitim için veri üretir"""
        
        df = self.generate_test_data(num_samples)
        
        # Özellikler
        X = df[['temperature', 'humidity', 'vibration', 'pressure']]
        
        # Hedef değişken (0: PASS, 1: FAIL)
        y = (df['pass_fail'] == 'FAIL').astype(int)
        
        return X, y
    
    def save_mock_data(self, filename: str = 'data/mock_data.csv'):
        """Mock veriyi CSV dosyasına kaydeder"""
        
        df = self.generate_test_data(2000)
        df.to_csv(filename, index=False)
        print(f"Mock data kaydedildi: {filename}")
        print(f"Toplam kayıt sayısı: {len(df)}")
        print(f"PASS oranı: {(df['pass_fail'] == 'PASS').mean():.2%}")
        print(f"FAIL oranı: {(df['pass_fail'] == 'FAIL').mean():.2%}")

if __name__ == "__main__":
    # Test verisi üretimi
    generator = TestDataGenerator()
    generator.save_mock_data() 