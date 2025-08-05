"""
TestScope AI - Ana Web Uygulaması
Çevresel test risk tahmin sistemi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import TestDataGenerator
from models.risk_predictor import RiskPredictor
from models.model_trainer import ModelTrainer
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="TestScope AI - Risk Tahmin Sistemi",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

    # CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .risk-high {
        color: #DC143C;
        font-weight: bold;
    }
    .risk-medium {
        color: #FFD700;
        font-weight: bold;
    }
    .risk-low {
        color: #32CD32;
        font-weight: bold;
    }
    
    /* Sidebar font boyutları düzenlemesi */
    .sidebar-metric-label {
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    .sidebar-metric-value {
        font-size: 16px !important;
        font-weight: 400 !important;
        margin-top: 2px !important;
    }
    .sidebar-subheader {
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    .sidebar-header {
        font-size: 18px !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;
    }
    
    /* Tooltip iyileştirmeleri - Düzeltilmiş konumlandırma */
    .stTooltip, [data-testid="tooltip"], .tooltip, div[role="tooltip"] {
        background-color: #2c3e50 !important;
        color: white !important;
        border: 2px solid #34495e !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        max-width: 320px !important;
        z-index: 99999 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        position: fixed !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        left: auto !important;
        right: auto !important;
        top: auto !important;
        bottom: auto !important;
    }
    
    /* Tooltip hover efekti */
    .stTooltip:hover, [data-testid="tooltip"]:hover, .tooltip:hover, div[role="tooltip"]:hover {
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        transform: scale(1.02) !important;
        transition: all 0.2s ease !important;
    }
    
    /* Tooltip konumlandırma düzeltmeleri */
    [data-testid="stTooltip"] {
        position: absolute !important;
        left: auto !important;
        right: auto !important;
        top: auto !important;
        bottom: auto !important;
    }
    
    /* Sidebar tooltip'leri için özel konumlandırma */
    .sidebar [data-testid="stTooltip"] {
        position: fixed !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 99999 !important;
        max-width: 300px !important;
        background-color: #2c3e50 !important;
        color: white !important;
        border: 2px solid #34495e !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Buton hover durumunda tooltip'i göster */
    .sidebar button:hover + [data-testid="stTooltip"],
    .sidebar button:focus + [data-testid="stTooltip"] {
        display: block !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)

class TestScopeApp:
    """TestScope AI ana uygulama sınıfı"""
    
    def __init__(self):
        self.data_generator = TestDataGenerator()
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.model = None
        self.load_or_train_model()
        
        # Session state'i initialize et
        if 'temp_slider' not in st.session_state:
            st.session_state.temp_slider = 25
        if 'hum_slider' not in st.session_state:
            st.session_state.hum_slider = 50
        if 'vib_slider' not in st.session_state:
            st.session_state.vib_slider = 5.0
        if 'pres_slider' not in st.session_state:
            st.session_state.pres_slider = 1013
        
        # Profesyonel renk paleti
        self.colors = {
            'success': '#28A745',       # Bootstrap Green (düşük risk)
            'warning': '#FFC107',       # Bootstrap Yellow (orta risk)
            'danger': '#DC3545',        # Bootstrap Red (yüksek risk)
            'light_success': '#D4EDDA', # Açık yeşil (arka plan)
            'light_warning': '#FFF3CD', # Açık sarı (arka plan)
            'light_danger': '#F8D7DA',  # Açık kırmızı (arka plan)
            'text_success': '#155724',  # Koyu yeşil (metin)
            'text_warning': '#856404',  # Koyu sarı (metin)
            'text_danger': '#721C24'    # Koyu kırmızı (metin)
        }
    
    def get_risk_colors(self, risk_value: float):
        """Risk değerine göre renk ve emoji döndürür - Normalize edilmiş değerler (0-1 aralığı)"""
        if risk_value < 0.3:  # Düşük risk (0-30%)
            return {
                'emoji': '🟢',
                'bg_color': self.colors['light_success'],
                'text_color': self.colors['text_success']
            }
        elif risk_value < 0.6:  # Orta risk (30-60%)
            return {
                'emoji': '🟡',
                'bg_color': self.colors['light_warning'],
                'text_color': self.colors['text_warning']
            }
        else:  # Yüksek risk (60-100%)
            return {
                'emoji': '🔴',
                'bg_color': self.colors['light_danger'],
                'text_color': self.colors['text_danger']
            }
    
    def get_test_scenarios(self, standard):
        """Seçilen standarda göre test senaryolarını döndürür"""
        scenarios = {
            "MIL-STD-810": {
                "Yüksek Sıcaklık": {
                    "temp": 65, "humidity": 50, "vibration": 5.0, "pressure": 1013,
                    "method": "501.7", "duration": "6 saat", "color": "#FF6B35"
                },
                "Yüksek Nem": {
                    "temp": 25, "humidity": 90, "vibration": 5.0, "pressure": 1013,
                    "method": "507.6", "duration": "24 saat", "color": "#4ECDC4"
                },
                "Yüksek Titreşim": {
                    "temp": 25, "humidity": 50, "vibration": 35.0, "pressure": 1013,
                    "method": "514.7", "duration": "2 saat", "color": "#9B59B6"
                },
                "Kombine Test": {
                    "temp": 60, "humidity": 85, "vibration": 25.0, "pressure": 1013,
                    "method": "520.3", "duration": "4 saat", "color": "#F39C12"
                }
            },
            "ISO 16750": {
                "Yüksek Sıcaklık": {
                    "temp": 70, "humidity": 45, "vibration": 3.0, "pressure": 1013,
                    "method": "5.1.1", "duration": "8 saat", "color": "#E74C3C"
                },
                "Yüksek Nem": {
                    "temp": 30, "humidity": 95, "vibration": 3.0, "pressure": 1013,
                    "method": "5.2.1", "duration": "48 saat", "color": "#3498DB"
                },
                "Yüksek Titreşim": {
                    "temp": 30, "humidity": 45, "vibration": 40.0, "pressure": 1013,
                    "method": "5.3.1", "duration": "1 saat", "color": "#8E44AD"
                },
                "Kombine Test": {
                    "temp": 65, "humidity": 80, "vibration": 20.0, "pressure": 1013,
                    "method": "5.4.1", "duration": "6 saat", "color": "#F1C40F"
                }
            },
            "IEC 60068": {
                "Yüksek Sıcaklık": {
                    "temp": 60, "humidity": 40, "vibration": 4.0, "pressure": 1013,
                    "method": "2-14", "duration": "5 saat", "color": "#D35400"
                },
                "Yüksek Nem": {
                    "temp": 25, "humidity": 85, "vibration": 4.0, "pressure": 1013,
                    "method": "2-30", "duration": "12 saat", "color": "#2980B9"
                },
                "Yüksek Titreşim": {
                    "temp": 25, "humidity": 40, "vibration": 30.0, "pressure": 1013,
                    "method": "2-6", "duration": "3 saat", "color": "#7D3C98"
                },
                "Kombine Test": {
                    "temp": 55, "humidity": 75, "vibration": 15.0, "pressure": 1013,
                    "method": "2-1", "duration": "4 saat", "color": "#E67E22"
                }
            }
        }
        return scenarios.get(standard, scenarios["MIL-STD-810"])
    
    def create_visual_tooltip(self, temp, humidity, vibration, pressure, standard="MIL-STD-810", method="514.7", duration="2 saat"):
        """Gelişmiş hazır senaryo tooltip formatı - renkli ikon ve metin"""
        risk_factors = self.data_processor.calculate_risk_factors(temp, humidity, vibration, pressure)
        total_risk = (risk_factors['temperature_risk'] + risk_factors['humidity_risk'] +
                       risk_factors['vibration_risk'] + risk_factors['pressure_risk']) / 4

        # Risk seviyesi ve renk kodu
        if total_risk <= 0.3:
            risk_icon = "&#x1F7E2;"  # yeşil daire
            risk_text = "Düşük Risk"
            risk_color = "#43A047"
        elif total_risk <= 0.6:
            risk_icon = "&#x1F7E1;"  # sarı daire
            risk_text = "Orta Risk"
            risk_color = "#FFC107"
        else:
            risk_icon = "&#x1F534;"  # kırmızı daire
            risk_text = "Yüksek Risk"
            risk_color = "#E53935"

        # Risk renk kodlaması için emoji
        if total_risk <= 0.3:
            risk_emoji = "🟢"  # yeşil daire
        elif total_risk <= 0.6:
            risk_emoji = "🟡"  # sarı daire
        else:
            risk_emoji = "🔴"  # kırmızı daire
            
        tooltip = f"""HAZIR SENARYO - {standard}
Method {method} | {duration}
🌡 {temp}°C | 💧 {humidity}% | 📈 {vibration}g | 🌬 {pressure}hPa | Risk: {risk_emoji} {risk_text} | Sabit Değerler"""
        return tooltip
    
    def load_or_train_model(self):
        """Modeli yükle veya eğit"""
        
        model_path = 'models/risk_predictor.joblib'
        
        if os.path.exists(model_path):
            try:
                self.model = RiskPredictor()
                self.model.load_model(model_path)
                st.sidebar.success("✅ Model başarıyla yüklendi!")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Model yüklenemedi: {e}")
                self.train_new_model()
        else:
            st.sidebar.info("🔄 Model bulunamadı, yeni model eğitiliyor...")
            self.train_new_model()
    
    def train_new_model(self):
        """Yeni model eğit"""
        
        with st.spinner("Model eğitiliyor..."):
            trainer = ModelTrainer()
            self.model = trainer.full_training_pipeline(3000)  # Daha az veri ile hızlı eğitim
            st.sidebar.success("✅ Model eğitimi tamamlandı!")
    
    def main(self):
        """Ana uygulama"""
        
        # Sidebar
        self.sidebar()
        
        # 1️⃣ Üst Header / Başlık Alanı
        self.create_header()
        
        # 2️⃣ Test Seçim ve Parametre Girişi Paneli
        self.create_test_selection_panel()
        
        # 3️⃣ Tahmin Sonucu ve Risk Göstergesi - Sadece analiz yapıldığında göster
        if hasattr(st.session_state, 'analysis_performed') and st.session_state.analysis_performed:
            self.create_prediction_panel()
        
        # 4️⃣ Bilgi ve Standart Referans Paneli
        self.create_info_panel()
    
    def create_header(self):
        """Üst header alanını oluşturur"""
        
        # Ana başlık
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2E8B57 0%, #4682B4 100%); 
                    border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">TestScope AI</h1>
            <h2 style="color: white; margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Çevresel Test Risk Tahmin Sistemi
            </h2>
            <p style="color: white; margin: 10px 0 0 0; opacity: 0.8;">
                MIL‑STD‑810, ISO 16750 ve IEC 60068 standartlarına uygun sentetik verilerle çalışan PoC sistemi
            </p>
            <div style="position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.2); 
                        padding: 5px 10px; border-radius: 15px;">
                <span style="color: white; font-size: 0.8rem;">v1.0 PoC</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_test_selection_panel(self):
        """Test seçim ve parametre girişi panelini oluşturur"""
        
        st.markdown("## Test Konfigürasyonu")
        
        # Sol ve sağ kolonlar
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### Test Ayarları")
            
            # Test standardı seçimi - Sidebar ile senkronize
            if 'selected_standard' not in st.session_state:
                st.session_state.selected_standard = "MIL-STD-810"
            
            test_standard = st.selectbox(
                "Test Standardı",
                ["MIL-STD-810", "ISO 16750", "IEC 60068"],
                index=["MIL-STD-810", "ISO 16750", "IEC 60068"].index(st.session_state.selected_standard),
                help="Kullanılacak test standardını seçin"
            )
            
            # Seçimi session state'e kaydet
            st.session_state.selected_standard = test_standard
            
            # Test tipi seçimi
            test_type = st.selectbox(
                "Test Tipi",
                ["Yüksek Sıcaklık", "Düşük Sıcaklık", "Nem", "Titreşim", "Termal Şok"],
                help="Gerçekleştirilecek test tipini seçin"
            )
            
            # Model seçimi
            model_type = st.selectbox(
                "Model",
                ["Random Forest", "Logistic Regression"],
                help="Risk tahmini için kullanılacak model"
            )
            
            # Kabul eşiği
            confidence_threshold = st.slider(
                "Kabul Eşiği (%)",
                min_value=50,
                max_value=99,
                value=70,
                help="Tahminin kabul edilmesi için gereken minimum güven seviyesi"
            )
            
            # Risk analizi butonu - Gradient ve ikon ile
            st.markdown("""
            <style>
            .gradient-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 15px 30px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 18px;
                font-weight: bold;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .gradient-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("Risk Analizi Yap", type="primary", use_container_width=True, 
                        help="Seçilen parametrelerle risk analizi gerçekleştir"):
                self.perform_analysis()
        
        with col_right:
            st.markdown("### Test Parametreleri")
            
            # Parametre kartları - 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Sıcaklık kartı
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; 
                            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">Sıcaklık</h4>
                    <p style="margin: 0 0 10px 0; font-size: 0.9rem; color: #495057;">
                        Test sıcaklığı (-40°C ile +70°C arası)
                    </p>
                    <p style="margin: 0; font-size: 0.8rem; color: #dc3545; font-style: italic;">
                        Yüksek sıcaklık malzemeye zarar verebilir
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sıcaklık slider'ı - Sadece key kullan, value parametresi kullanma
                temperature = st.slider(
                    "Sıcaklık (°C)",
                    min_value=-40,
                    max_value=70,
                    key="temp_slider"
                )
                
                # Risk renk kodu ve büyük gösterim
                temp_risk = self.data_processor.calculate_risk_factors(temperature, 50, 5, 1013)['temperature_risk']
                temp_colors = self.get_risk_colors(temp_risk)
                
                st.markdown(f"""
                <div style="background: {temp_colors['bg_color']}; border: 2px solid {temp_colors['text_color']}; 
                            border-radius: 8px; padding: 10px; margin: 10px 0; text-align: center;">
                    <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {temp_colors['text_color']};">
                        {temp_colors['emoji']} <strong>{temperature}°C</strong>
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: {temp_colors['text_color']};">
                        {temp_risk:.1%} risk seviyesi
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Nem kartı
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; 
                            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">Nem</h4>
                    <p style="margin: 0 0 10px 0; font-size: 0.9rem; color: #495057;">
                        Test nem oranı (%10 ile %95 arası)
                    </p>
                    <p style="margin: 0; font-size: 0.8rem; color: #dc3545; font-style: italic;">
                        Yüksek nem korozyona neden olabilir
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Nem slider'ı - Sadece key kullan, value parametresi kullanma
                humidity = st.slider(
                    "Nem (%)",
                    min_value=10,
                    max_value=95,
                    key="hum_slider"
                )
                
                # Risk renk kodu ve büyük gösterim
                hum_risk = self.data_processor.calculate_risk_factors(25, humidity, 5, 1013)['humidity_risk']
                hum_colors = self.get_risk_colors(hum_risk)
                
                st.markdown(f"""
                <div style="background: {hum_colors['bg_color']}; border: 2px solid {hum_colors['text_color']}; 
                            border-radius: 8px; padding: 10px; margin: 10px 0; text-align: center;">
                    <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {hum_colors['text_color']};">
                        {hum_colors['emoji']} <strong>{humidity}%</strong>
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: {hum_colors['text_color']};">
                        {hum_risk:.1%} risk seviyesi
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Titreşim kartı
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; 
                            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">Titreşim</h4>
                    <p style="margin: 0 0 10px 0; font-size: 0.9rem; color: #495057;">
                        Test titreşim değeri (0.1g ile 50g arası)
                    </p>
                    <p style="margin: 0; font-size: 0.8rem; color: #dc3545; font-style: italic;">
                        Yüksek titreşim bağlantıları gevşetebilir
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Titreşim slider'ı - Sadece key kullan, value parametresi kullanma
                vibration = st.slider(
                    "Titreşim (g)",
                    min_value=0.1,
                    max_value=50.0,
                    step=0.1,
                    key="vib_slider"
                )
                
                # Risk renk kodu ve büyük gösterim
                vib_risk = self.data_processor.calculate_risk_factors(25, 50, vibration, 1013)['vibration_risk']
                vib_colors = self.get_risk_colors(vib_risk)
                
                st.markdown(f"""
                <div style="background: {vib_colors['bg_color']}; border: 2px solid {vib_colors['text_color']}; 
                            border-radius: 8px; padding: 10px; margin: 10px 0; text-align: center;">
                    <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {vib_colors['text_color']};">
                        {vib_colors['emoji']} <strong>{vibration}g</strong>
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: {vib_colors['text_color']};">
                        {vib_risk:.1%} risk seviyesi
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Basınç kartı
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; 
                            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">Basınç</h4>
                    <p style="margin: 0 0 10px 0; font-size: 0.9rem; color: #495057;">
                        Test basınç değeri (800hPa ile 1200hPa arası)
                    </p>
                    <p style="margin: 0; font-size: 0.8rem; color: #dc3545; font-style: italic;">
                        Aşırı basınç sızdırmazlığı etkileyebilir
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Basınç slider'ı - Sadece key kullan, value parametresi kullanma
                pressure = st.slider(
                    "Basınç (hPa)",
                    min_value=800,
                    max_value=1200,
                    key="pres_slider"
                )
                
                # Risk renk kodu ve büyük gösterim
                pres_risk = self.data_processor.calculate_risk_factors(25, 50, 5, pressure)['pressure_risk']
                pres_colors = self.get_risk_colors(pres_risk)
                
                st.markdown(f"""
                <div style="background: {pres_colors['bg_color']}; border: 2px solid {pres_colors['text_color']}; 
                            border-radius: 8px; padding: 10px; margin: 10px 0; text-align: center;">
                    <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {pres_colors['text_color']};">
                        {pres_colors['emoji']} <strong>{pressure}hPa</strong>
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: {pres_colors['text_color']};">
                        {pres_risk:.1%} risk seviyesi
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Parametreleri session state'e kaydet
        st.session_state.test_standard = test_standard
        st.session_state.test_type = test_type
        st.session_state.model_type = model_type
        st.session_state.confidence_threshold = confidence_threshold
    
    def create_prediction_panel(self):
        """Tahmin sonucu ve risk göstergesi panelini oluşturur"""
        
        st.markdown("## Risk Analiz Sonuçları")
        
        # Eğer analiz yapılmışsa sonuçları göster
        if hasattr(st.session_state, 'prediction_result'):
            prediction = st.session_state.prediction_result
            risk_factors = st.session_state.risk_factors
            
            # Tahmin Bazlı Metrikler
            st.markdown("### Tahmin Bazlı Metrikler")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Tahmin sonucu - yumuşak kırmızı ton
                result_color = "🟢" if prediction['prediction'] == 'PASS' else "🔴"
                result_bg = "#d4edda" if prediction['prediction'] == 'PASS' else "#ffe6e6"  # Yumuşak kırmızı
                result_border = "#c3e6cb" if prediction['prediction'] == 'PASS' else "#ffcccc"  # Yumuşak kırmızı border
                result_text = "#155724" if prediction['prediction'] == 'PASS' else "#8b0000"  # Koyu kırmızı
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: {result_bg}; 
                            border-radius: 10px; border: 2px solid {result_border}; margin: 5px;">
                    <h3 style="margin: 0; color: {result_text}; font-size: 1.3rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                        <span style="margin-right: 8px;">{result_color}</span> {prediction['prediction']}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Risk skoru - tutarlı yeşil ton
                risk_score = prediction['risk_score']
                risk_level = "Düşük" if risk_score < 0.3 else "Orta" if risk_score < 0.6 else "Yüksek"
                risk_color = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.6 else "🔴"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #dee2e6; margin: 5px;">
                    <h3 style="margin: 0; color: #495057; font-size: 1.1rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                        <span style="margin-right: 8px;">{risk_color}</span> Risk Seviyesi
                    </h3>
                    <p style="margin: 8px 0 0 0; font-size: 1.2rem; font-weight: bold; color: #495057;">
                        {risk_level}
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #28a745;">
                        {risk_score:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Tahmin güveni - mavi ton
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #dee2e6; margin: 5px;">
                    <h3 style="margin: 0; color: #495057; font-size: 1.1rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                        <span style="margin-right: 8px;">🎯</span> Tahmin Güveni
                        <span style="margin-left: 5px; font-size: 0.8rem; color: #6c757d; cursor: help;" title="Bu tahmin özelinde modelin kendine güven seviyesi">ℹ️</span>
                    </h3>
                    <p style="margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #4682B4;">
                        {prediction['confidence']:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # PASS olasılığı - tutarlı yeşil ton
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #dee2e6; margin: 5px;">
                    <h3 style="margin: 0; color: #495057; font-size: 1.1rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                        <span style="margin-right: 8px;">✅</span> PASS Olasılığı
                    </h3>
                    <p style="margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #28a745;">
                        {prediction['pass_probability']:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            

            
            # Risk gauge grafiği
            col_gauge, col_factors = st.columns(2)
            
            with col_gauge:
                fig_gauge = self.visualizer.create_risk_gauge(prediction['risk_score'])
                st.plotly_chart(fig_gauge, use_container_width=True, key="prediction_gauge")
            
            with col_factors:
                fig_risk = self.visualizer.create_risk_breakdown(risk_factors, prediction['risk_score'])
                st.plotly_chart(fig_risk, use_container_width=True, key="prediction_risk")
            
            # Test parametreleri radar grafiği
            test_params = {
                'temperature': st.session_state.temp_slider,
                'humidity': st.session_state.hum_slider,
                'vibration': st.session_state.vib_slider,
                'pressure': st.session_state.pres_slider
            }
            # Radar grafiği
            radar_fig = self.visualizer.create_parameter_radar(
                risk_factors['temperature_risk'],
                risk_factors['humidity_risk'], 
                risk_factors['vibration_risk'],
                risk_factors['pressure_risk']
            )
            st.plotly_chart(radar_fig, use_container_width=True, key="prediction_radar")
            
            # Öneriler
            recommendations = self.data_processor.get_test_recommendations(prediction['risk_score'])
            
            st.markdown("### Test Önerileri")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
    
    def create_info_panel(self):
        """Bilgi ve standart referans panelini oluşturur"""
        
        st.markdown("## Test Standartları ve Bilgiler")
        st.markdown("*Test standartları ve metodları hakkında detaylı bilgi için başlıklara tıklayın*")
        
        # Standart bilgileri
        standards_info = self.data_processor.get_test_standards_info()
        
        # Accordion ile standartları göster (hepsi kapalı olsun)
        standard_keys = list(standards_info.keys())
        
        for i, (standard, info) in enumerate(standards_info.items()):
            # Tüm standartlar kapalı olsun
            expanded = False
            
            with st.expander(f"📖 {standard}", expanded=expanded):
                st.markdown(f"**Açıklama:** {info['description']}")
                st.markdown("**Test Metodları:**")
                
                for method, description in info['methods'].items():
                    st.markdown(f"• **{method}:** {description}")
        
        # Test limitleri tablosu
        st.markdown("### Test Limitleri")
        
        limits_df = pd.DataFrame([
            {"Parametre": "Sıcaklık", "Min": "-40°C", "Max": "+70°C", "Birim": "°C", "Risk": "Yüksek sıcaklık > 60°C"},
            {"Parametre": "Nem", "Min": "10%", "Max": "95%", "Birim": "%", "Risk": "Yüksek nem > 90%"},
            {"Parametre": "Titreşim", "Min": "0.1g", "Max": "50.0g", "Birim": "g", "Risk": "Yüksek titreşim > 30g"},
            {"Parametre": "Basınç", "Min": "800hPa", "Max": "1200hPa", "Birim": "hPa", "Risk": "Düşük/yüksek basınç"}
        ])
        
        st.dataframe(limits_df, use_container_width=True)
        
        # PDF bağlantıları
        st.markdown("### Standart Dokümanları")
        st.markdown("""
        - [MIL-STD-810H](https://www.everyspec.com/MIL-STD/MIL-STD-0800-0899/MIL-STD-810H_55998/) - Askeri ekipmanlar için çevresel test standartları
        - [ISO 16750](https://www.iso.org/standard/55998.html) - Otomotiv elektronik ekipmanları için test standartları
        - [IEC 60068](https://webstore.iec.ch/publication/61611) - Çevresel test standartları
        """)
    
    def perform_analysis(self):
        """Risk analizi yapar"""
        
        if not self.model or not self.model.is_trained:
            st.error("❌ Model henüz eğitilmemiş!")
            return
        
        # Session state'den parametreleri al
        temperature = st.session_state.temp_slider
        humidity = st.session_state.hum_slider
        vibration = st.session_state.vib_slider
        pressure = st.session_state.pres_slider
        
        # Test parametrelerini doğrula
        validation = self.data_processor.validate_test_parameters(
            temperature, humidity, vibration, pressure
        )
        
        # Uyarıları göster
        if validation['warnings']:
            for warning in validation['warnings']:
                st.warning(f"⚠️ {warning}")
        
        # Risk faktörlerini hesapla
        risk_factors = self.data_processor.calculate_risk_factors(
            temperature, humidity, vibration, pressure
        )
        
        # Model tahmini
        test_data = pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity,
            'vibration': vibration,
            'pressure': pressure
        }])
        
        prediction = self.model.predict(test_data)
        
        # Sonuçları session state'e kaydet
        st.session_state.prediction_result = prediction
        st.session_state.risk_factors = risk_factors
        st.session_state.analysis_performed = True
        
        # Sayfayı yenile
        st.rerun()
    
    def display_analysis_results(self, prediction, risk_factors, test_params):
        """Analiz sonuçlarını gösterir"""
        
        st.markdown("## 🎯 Risk Analiz Sonuçları")
        
        # Risk skoru gauge - Daha güvenli işleme
        if isinstance(prediction, (list, np.ndarray)):
            pred_value = prediction[0]
        elif isinstance(prediction, dict):
            # Eğer prediction bir dictionary ise, ilk değeri al
            pred_value = list(prediction.values())[0]
        else:
            pred_value = prediction
        
        # String değerleri sayıya çevir
        if isinstance(pred_value, str):
            if pred_value.upper() == 'FAIL':
                risk_score = 0.8  # FAIL için yüksek risk
            elif pred_value.upper() == 'PASS':
                risk_score = 0.2  # PASS için düşük risk
            else:
                try:
                    risk_score = float(pred_value)
                except ValueError:
                    risk_score = 0.5  # Varsayılan değer
        else:
            try:
                risk_score = float(pred_value)
            except (ValueError, TypeError):
                risk_score = 0.5  # Varsayılan değer
        
        gauge_fig = self.visualizer.create_risk_gauge(risk_score)
        st.plotly_chart(gauge_fig, use_container_width=True, key="analysis_gauge")
        
        # Risk faktörleri analizi
        col1, col2 = st.columns(2)
        
        with col1:
            radar_fig = self.visualizer.create_parameter_radar(
                risk_factors['temperature_risk'],
                risk_factors['humidity_risk'],
                risk_factors['vibration_risk'],
                risk_factors['pressure_risk']
            )
            st.plotly_chart(radar_fig, use_container_width=True, key="analysis_radar")
        
        with col2:
            breakdown_fig = self.visualizer.create_risk_breakdown(risk_factors, risk_score)
            st.plotly_chart(breakdown_fig, use_container_width=True, key="analysis_breakdown")
        
        # Test parametreleri özeti
        st.markdown("### 📊 Test Parametreleri Özeti")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sıcaklık", f"{test_params['temperature']}°C")
        with col2:
            st.metric("Nem", f"{test_params['humidity']}%")
        with col3:
            st.metric("Titreşim", f"{test_params['vibration']}g")
        with col4:
            st.metric("Basınç", f"{test_params['pressure']}hPa")
    
    def sidebar(self):
        """Sidebar içeriği - Standart bağlantılı test senaryoları"""
        
        # Ana başlık
        st.sidebar.markdown("## TestScope AI")
        st.sidebar.markdown("---")
        
        # Model durumu - Sadece önemli bilgi
        if self.model and self.model.is_trained:
            model_info = self.model.get_model_info()
            st.sidebar.markdown("### Model")
            st.sidebar.metric("Genel Doğruluk", f"{model_info.get('accuracy', 0):.1%}", 
                             help="Modelin geçmiş testlerdeki başarı oranı")
        
        st.sidebar.markdown("---")
        
        # Test standardı seçimi - Sidebar'da da göster
        st.sidebar.markdown("### Test Standardı")
        selected_standard = st.sidebar.selectbox(
            "Standart Seçin",
            ["MIL-STD-810", "ISO 16750", "IEC 60068"],
            help="Test standardını seçin, hızlı testler buna göre güncellenecek"
        )
        
        # Seçilen standarda göre test senaryolarını al
        test_scenarios = self.get_test_scenarios(selected_standard)
        
        st.sidebar.markdown("---")
        
        # Hızlı testler - Standart bağlantılı
        st.sidebar.markdown("### Hazır Test Senaryoları")
        st.sidebar.markdown(f"*{selected_standard} standardına göre sabit değerler*")
        st.sidebar.markdown("*Tek tıkla hazır senaryo yükle*")
        
        # Test butonları - Dinamik olarak oluştur
        for test_name, scenario in test_scenarios.items():
            # Tooltip oluştur
            tooltip = self.create_visual_tooltip(
                scenario["temp"], scenario["humidity"], 
                scenario["vibration"], scenario["pressure"],
                selected_standard, scenario["method"], scenario["duration"]
            )
            
            # Hazır senaryo buton stili - Koyu tema
            button_style = f"""
            <style>
            .fixed-scenario-button-{test_name.replace(' ', '-').lower()} {{
                background: linear-gradient(135deg, {scenario['color']} 0%, {scenario['color']}80 100%);
                border: 2px solid #2C3E50;
                color: white;
                padding: 12px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                font-weight: bold;
                margin: 4px 0;
                cursor: pointer;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
                width: 100%;
                position: relative;
            }}
            .fixed-scenario-button-{test_name.replace(' ', '-').lower()}:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                border-color: #34495E;
            }}
            .fixed-scenario-button-{test_name.replace(' ', '-').lower()}::before {{
                content: "🔒";
                position: absolute;
                left: 8px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 12px;
            }}
            </style>
            """
            st.sidebar.markdown(button_style, unsafe_allow_html=True)
            
            # Risk hesaplama ve renk kodlaması
            scenario_risk = self.data_processor.calculate_risk_factors(
                scenario["temp"], scenario["humidity"], 
                scenario["vibration"], scenario["pressure"]
            )
            total_scenario_risk = (scenario_risk['temperature_risk'] + scenario_risk['humidity_risk'] +
                                   scenario_risk['vibration_risk'] + scenario_risk['pressure_risk']) / 4
            
            # Risk emoji'si
            if total_scenario_risk <= 0.3:
                risk_emoji = "🟢"
            elif total_scenario_risk <= 0.6:
                risk_emoji = "🟡"
            else:
                risk_emoji = "🔴"
            
            # Buton oluştur - Sadece risk seviyesi için emoji
            button_key = f"scenario_{test_name.replace(' ', '_').lower()}"
            if st.sidebar.button(
                f"{risk_emoji} {test_name}", 
                use_container_width=True, 
                help=tooltip,
                key=button_key
            ):
                st.session_state.temp_slider = scenario["temp"]
                st.session_state.hum_slider = scenario["humidity"]
                st.session_state.vib_slider = scenario["vibration"]
                st.session_state.pres_slider = scenario["pressure"]
                st.session_state.selected_standard = selected_standard
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Veri işlemleri - Küçük bölüm
        st.sidebar.markdown("### Veri")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Üret", help="Yeni veri oluştur"):
                self.data_generator.save_mock_data()
                st.success("✓")
        with col2:
            if st.button("🔄 Eğit", help="Modeli eğit"):
                self.train_new_model()
                st.success("✓")
    
    def risk_analysis_tab(self):
        """Risk analizi sekmesi"""
        
        st.subheader("Risk Analizi")
        
        # Test parametreleri girişi
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Test Parametreleri")
            
            # Test standardı seçimi
            test_standard = st.selectbox(
                "Test Standardı",
                ["MIL-STD-810", "ISO 16750", "IEC 60068"],
                help="Kullanılacak test standardını seçin"
            )
            
            # Test tipi seçimi
            test_type = st.selectbox(
                "Test Tipi",
                ["Yüksek Sıcaklık", "Düşük Sıcaklık", "Nem", "Titreşim", "Termal Şok"],
                help="Gerçekleştirilecek test tipini seçin"
            )
            
            # Model seçimi
            model_type = st.selectbox(
                "Model",
                ["Random Forest", "Logistic Regression"],
                help="Risk tahmini için kullanılacak model"
            )
            
            # Güven eşiği
            confidence_threshold = st.slider(
                "Güven Eşiği (%)",
                min_value=50,
                max_value=99,
                value=70,
                help="Minimum güven oranı"
            )
        
        with col2:
            st.subheader("Parametre Değerleri")
            
            # Parametre kartları
            col_temp, col_hum = st.columns(2)
            
            with col_temp:
                st.markdown("### Sıcaklık")
                temperature = st.slider(
                    "Sıcaklık (°C)",
                    min_value=-40,
                    max_value=70,
                    value=25,
                    help="Test sıcaklığı (-40°C ile +70°C arası)"
                )
                
                # Risk renk kodu
                temp_risk = self.data_processor.calculate_risk_factors(temperature, 50, 5, 1013)['temperature_risk']
                temp_color = "🟢" if temp_risk < 0.3 else "🟡" if temp_risk < 0.6 else "🔴"
                st.markdown(f"{temp_color} **Seçilen Değer:** {temperature}°C")
            
            with col_hum:
                st.markdown("### Nem")
                humidity = st.slider(
                    "Nem (%)",
                    min_value=10,
                    max_value=95,
                    value=50,
                    help="Test nem oranı (%10 ile %95 arası)"
                )
                
                # Risk renk kodu
                hum_risk = self.data_processor.calculate_risk_factors(25, humidity, 5, 1013)['humidity_risk']
                hum_color = "🟢" if hum_risk < 0.3 else "🟡" if hum_risk < 0.6 else "🔴"
                st.markdown(f"{hum_color} **Seçilen Değer:** {humidity}%")
            
            col_vib, col_pres = st.columns(2)
            
            with col_vib:
                st.markdown("### Titreşim")
                vibration = st.slider(
                    "Titreşim (g)",
                    min_value=0.1,
                    max_value=50.0,
                    value=5.0,
                    step=0.1,
                    help="Test titreşim değeri (0.1g ile 50g arası)"
                )
                
                # Risk renk kodu
                vib_risk = self.data_processor.calculate_risk_factors(25, 50, vibration, 1013)['vibration_risk']
                vib_color = "🟢" if vib_risk < 0.3 else "🟡" if vib_risk < 0.6 else "🔴"
                st.markdown(f"{vib_color} **Seçilen Değer:** {vibration}g")
            
            with col_pres:
                st.markdown("### Basınç")
                pressure = st.slider(
                    "Basınç (hPa)",
                    min_value=800,
                    max_value=1200,
                    value=1013,
                    help="Test basınç değeri (800hPa ile 1200hPa arası)"
                )
                
                # Risk renk kodu
                pres_risk = self.data_processor.calculate_risk_factors(25, 50, 5, pressure)['pressure_risk']
                pres_color = "🟢" if pres_risk < 0.3 else "🟡" if pres_risk < 0.6 else "🔴"
                st.markdown(f"{pres_color} **Seçilen Değer:** {pressure}hPa")
        
        # Risk analizi butonu
        if st.button("Risk Analizi Yap", type="primary", use_container_width=True):
            self.perform_risk_analysis(temperature, humidity, vibration, pressure)
    
    def perform_risk_analysis(self, temperature, humidity, vibration, pressure):
        """Risk analizi yapar"""
        
        if not self.model or not self.model.is_trained:
            st.error("❌ Model henüz eğitilmemiş!")
            return
        
        # Test parametrelerini doğrula
        validation = self.data_processor.validate_test_parameters(
            temperature, humidity, vibration, pressure
        )
        
        # Uyarıları göster
        if validation['warnings']:
            for warning in validation['warnings']:
                st.warning(f"⚠️ {warning}")
        
        # Risk faktörlerini hesapla
        risk_factors = self.data_processor.calculate_risk_factors(
            temperature, humidity, vibration, pressure
        )
        
        # Model tahmini
        test_data = pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity,
            'vibration': vibration,
            'pressure': pressure
        }])
        
        prediction = self.model.predict(test_data)
        
        # Sonuçları göster
        self.display_risk_results(prediction, risk_factors, {
            'temperature': temperature,
            'humidity': humidity,
            'vibration': vibration,
            'pressure': pressure
        })
    
    def display_risk_results(self, prediction, risk_factors, test_params):
        """Risk analiz sonuçlarını gösterir"""
        
        st.subheader("Risk Analiz Sonuçları")
        
        # Ana metrikler - iyileştirilmiş tasarım
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Tahmin sonucu - yumuşak kırmızı ton
            result_color = "🟢" if prediction['prediction'] == 'PASS' else "🔴"
            result_bg = "#d4edda" if prediction['prediction'] == 'PASS' else "#ffe6e6"  # Yumuşak kırmızı
            result_border = "#c3e6cb" if prediction['prediction'] == 'PASS' else "#ffcccc"  # Yumuşak kırmızı border
            result_text = "#155724" if prediction['prediction'] == 'PASS' else "#8b0000"  # Koyu kırmızı
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {result_bg}; 
                        border-radius: 10px; border: 2px solid {result_border}; margin: 5px;">
                <h3 style="margin: 0; color: {result_text}; font-size: 1.3rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                    <span style="margin-right: 8px;">{result_color}</span> {prediction['prediction']}
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk skoru - tutarlı yeşil ton
            risk_score = prediction['risk_score']
            risk_level = "Düşük" if risk_score < 0.3 else "Orta" if risk_score < 0.6 else "Yüksek"
            risk_color = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.6 else "🔴"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #dee2e6; margin: 5px;">
                <h3 style="margin: 0; color: #495057; font-size: 1.1rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                    <span style="margin-right: 8px;">{risk_color}</span> Risk Seviyesi
                </h3>
                <p style="margin: 8px 0 0 0; font-size: 1.2rem; font-weight: bold; color: #495057;">
                    {risk_level}
                </p>
                <p style="margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #28a745;">
                    {risk_score:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Tahmin güveni - mavi ton
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #dee2e6; margin: 5px;">
                <h3 style="margin: 0; color: #495057; font-size: 1.1rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                    <span style="margin-right: 8px;">🎯</span> Tahmin Güveni
                    <span style="margin-left: 5px; font-size: 0.8rem; color: #6c757d; cursor: help;" title="Bu tahmin özelinde modelin kendine güven seviyesi">ℹ️</span>
                </h3>
                <p style="margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #4682B4;">
                    {prediction['confidence']:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # PASS olasılığı - tutarlı yeşil ton
            pass_prob = prediction['pass_probability']
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #dee2e6; margin: 5px;">
                <h3 style="margin: 0; color: #495057; font-size: 1.1rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                    <span style="margin-right: 8px;">✅</span> PASS Olasılığı
                </h3>
                <p style="margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #28a745;">
                    {pass_prob:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Grafikler
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk gauge
            fig_gauge = self.visualizer.create_risk_gauge(prediction['risk_score'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Test parametreleri radar grafiği
            fig_radar = self.visualizer.create_parameter_radar(
                risk_factors['temperature_risk'],
                risk_factors['humidity_risk'], 
                risk_factors['vibration_risk'],
                risk_factors['pressure_risk']
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Risk faktörleri analizi
        fig_risk = self.visualizer.create_risk_breakdown(risk_factors, prediction['risk_score'])
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Öneriler
        recommendations = self.data_processor.get_test_recommendations(prediction['risk_score'])
        
        st.subheader("💡 Test Önerileri")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    def data_analysis_tab(self):
        """Veri analizi sekmesi"""
        
        st.subheader("📊 Veri Analizi")
        
        # Mock data yükle
        data_file = 'data/mock_data.csv'
        
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            
            # Temel istatistikler
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Toplam Test", len(df))
            
            with col2:
                pass_rate = (df['pass_fail'] == 'PASS').mean() * 100
                st.metric("PASS Oranı", f"{pass_rate:.1f}%")
            
            with col3:
                avg_risk = df['risk_score'].mean()
                st.metric("Ortalama Risk", f"{avg_risk:.3f}")
            
            # Grafikler
            col1, col2 = st.columns(2)
            
            with col1:
                # Test kategorileri
                category_counts = df['test_category'].value_counts()
                fig_cat = px.pie(values=category_counts.values, names=category_counts.index, 
                               title="Test Kategorileri Dağılımı")
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                # Risk skoru dağılımı
                fig_risk_dist = px.histogram(df, x='risk_score', nbins=30, 
                                           title="Risk Değerlendirme Dağılımı")
                st.plotly_chart(fig_risk_dist, use_container_width=True)
            
            # Test geçmişi
            if 'test_date' in df.columns:
                fig_history = self.visualizer.create_test_history_chart(df)
                st.plotly_chart(fig_history, use_container_width=True)
            
            # Parametre dağılımları
            fig_params = self.visualizer.create_parameter_distribution(df)
            st.plotly_chart(fig_params, use_container_width=True)
            
            # Veri tablosu
            st.subheader("📋 Veri Önizleme")
            st.dataframe(df.head(10))
            
        else:
            st.warning("⚠️ Mock data bulunamadı. Lütfen sidebar'dan yeni veri üretin.")
    
    def model_info_tab(self):
        """Model bilgileri sekmesi"""
        
        st.subheader("🤖 Model Bilgileri")
        
        if self.model and self.model.is_trained:
            model_info = self.model.get_model_info()
            
            # Model metrikleri
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Tipi", model_info['model_type'].replace('_', ' ').title())
            
            with col2:
                st.metric("Genel Doğruluk", f"{model_info.get('accuracy', 0):.3f}", 
                         help="Modelin geçmiş testlerdeki başarı oranı")
            
            with col3:
                st.metric("Precision", f"{model_info.get('precision', 0):.3f}")
            
            with col4:
                st.metric("Recall", f"{model_info.get('recall', 0):.3f}")
            
            # Özellik önem dereceleri
            if self.model:
                feature_importance = self.model.get_feature_importance()
                if not feature_importance.empty:
                    st.subheader("📈 Özellik Önem Dereceleri")
                    fig_importance = self.visualizer.create_feature_importance_plot(feature_importance)
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            # Model detayları
            st.subheader("🔍 Model Detayları")
            st.json(model_info)
            
        else:
            st.warning("⚠️ Model henüz eğitilmemiş!")
    
    def standards_tab(self):
        """Test standartları sekmesi"""
        
        st.subheader("📋 Test Standartları")
        
        standards_info = self.data_processor.get_test_standards_info()
        
        for standard, info in standards_info.items():
            with st.expander(f"📖 {standard}"):
                st.write(f"**Açıklama:** {info['description']}")
                st.write("**Test Metodları:**")
                
                for method, description in info['methods'].items():
                    st.write(f"• **{method}:** {description}")
        
        # Test limitleri
        st.subheader("⚡ Test Limitleri")
        
        limits_df = pd.DataFrame([
            {"Parametre": "Sıcaklık", "Min": "-40°C", "Max": "+70°C", "Birim": "°C"},
            {"Parametre": "Nem", "Min": "10%", "Max": "95%", "Birim": "%"},
            {"Parametre": "Titreşim", "Min": "0.1g", "Max": "50.0g", "Birim": "g"},
            {"Parametre": "Basınç", "Min": "800hPa", "Max": "1200hPa", "Birim": "hPa"}
        ])
        
        st.dataframe(limits_df, use_container_width=True)

def main():
    st.set_page_config(
        page_title="TestScope AI - Çevresel Test Risk Tahmini",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Global tooltip CSS - Düzeltilmiş konumlandırma
    st.markdown("""
    <style>
    /* Düzeltilmiş tooltip styling - Doğru konumlandırma */
    .stTooltip, [data-testid="tooltip"], .tooltip, div[role="tooltip"] {
        background-color: #2c3e50 !important;
        color: white !important;
        border: 2px solid #34495e !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        max-width: 320px !important;
        z-index: 99999 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        position: absolute !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* Tooltip hover efekti */
    .stTooltip:hover, [data-testid="tooltip"]:hover, .tooltip:hover, div[role="tooltip"]:hover {
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* Tooltip konumlandırma düzeltmeleri */
    [data-testid="stTooltip"] {
        position: absolute !important;
        left: auto !important;
        right: auto !important;
        top: auto !important;
        bottom: auto !important;
    }
    
    /* Tooltip içeriği için özel stil */
    .tooltip-content {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    
    .tooltip-header {
        font-weight: bold;
        font-size: 14px;
        color: white;
        border-bottom: 1px solid #34495e;
        padding-bottom: 6px;
        margin-bottom: 6px;
    }
    
    .tooltip-row {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 2px 0;
    }
    
    .tooltip-icon {
        font-size: 14px;
        min-width: 18px;
    }
    
    .tooltip-text {
        flex: 1;
        color: #ecf0f1;
        font-size: 12px;
    }
    
    .tooltip-risk {
        font-weight: bold;
        padding: 3px 6px;
        border-radius: 4px;
        font-size: 11px;
        text-align: center;
        margin-top: 6px;
    }
    
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ffcdd2;
    }
    
    .risk-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffcc02;
    }
    
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    app = TestScopeApp()
    app.main()

if __name__ == "__main__":
    main() 