"""
TestScope AI - Görselleştirme Araçları
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os

class Visualizer:
    """Görselleştirme araçları"""
    
    def __init__(self):
        # Matplotlib stil ayarları
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Profesyonel renk paleti - Risk seviyelerine göre
        self.colors = {
            'primary': '#2E8B57',      # Sea Green (ana renk)
            'secondary': '#4682B4',     # Steel Blue (ikincil)
            'success': '#28A745',       # Bootstrap Green (düşük risk)
            'warning': '#FFC107',       # Bootstrap Yellow (orta risk)
            'danger': '#DC3545',        # Bootstrap Red (yüksek risk)
            'info': '#17A2B8',          # Bootstrap Info
            'light_success': '#D4EDDA', # Açık yeşil (arka plan)
            'light_warning': '#FFF3CD', # Açık sarı (arka plan)
            'light_danger': '#F8D7DA',  # Açık kırmızı (arka plan)
            'text_success': '#155724',  # Koyu yeşil (metin)
            'text_warning': '#856404',  # Koyu sarı (metin)
            'text_danger': '#721C24'    # Koyu kırmızı (metin)
        }
    
    def create_risk_gauge(self, risk_score: float, title: str = "Risk Değerlendirme Skoru (%)") -> go.Figure:
        """Risk skoru için profesyonel gauge grafiği oluşturur"""
        
        # Risk skorunu yüzdeye çevir
        risk_percentage = risk_score * 100
        
                        # Risk seviyelerine göre renk paleti - Responsive ayarlarla
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
                         title = {
                 'text': title,
                 'font': {'size': 22, 'family': 'Arial, sans-serif', 'color': '#FFFFFF'}
             },
                         number = {
                 'font': {'size': 24, 'family': 'Roboto, Arial, sans-serif', 'color': '#FFFFFF'},
                 'suffix': '%',
                 'valueformat': '.1f'
             },
            gauge = {
                'axis': {
                    'range': [0, 100],
                    'tickfont': {'size': 12, 'family': 'Arial, sans-serif', 'color': '#FFFFFF'},
                    'tickmode': 'array',
                    'tickvals': [0, 20, 40, 60, 80, 100],
                    'ticktext': ['0%', '20%', '40%', '60%', '80%', '100%']
                },
                'bar': {'color': '#3498DB', 'thickness': 0.1},  # İnce mavi ibre
                'steps': [
                    {'range': [0, 40], 'color': '#28A745'},      # Yeşil (düşük risk)
                    {'range': [40, 60], 'color': '#FFC107'},      # Sarı (orta risk)
                    {'range': [60, 100], 'color': '#DC3545'}      # Kırmızı (yüksek risk)
                ],
                'threshold': {
                    'line': {'color': '#C0392B', 'width': 3},
                    'thickness': 0.75,
                    'value': risk_percentage  # Threshold'u risk değerine ayarla
                }
            }
                 ))
         
         # İbre ucuna marker ekle - kaldırıldı
         
        fig.update_layout(
             font={'size': 16, 'family': 'Arial, sans-serif', 'color': '#FFFFFF'},
             margin=dict(t=80, b=100, l=30, r=30),  # Alt margin legend için daha fazla alan
             paper_bgcolor='rgba(0,0,0,0)',  # Şeffaf arka plan
             plot_bgcolor='rgba(0,0,0,0)',    # Şeffaf arka plan
             # Responsive boyut ayarları
             autosize=True,  # Otomatik boyutlandırmayı aç
             width=None,  # Otomatik genişlik
             height=500,  # Legend için daha fazla alan
             # Container ayarları
             uirevision=True,  # Zoom/pan durumunu koru
             hovermode='closest',
             # X/Y eksenlerini gizle
             xaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False),
             yaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False),
             # "Toplam Risk" yazısını merkeze sabitle
             annotations=[
                 dict(
                     text="Toplam Risk",
                     x=0.5, y=0.5,  # Gauge merkezi
                     xref="paper", yref="paper",
                     showarrow=False,
                     font=dict(
                         size=16,  # Orta boyut font
                         color='rgba(255, 255, 255, 1)',  # Tam opak beyaz
                         family='Arial, sans-serif',
                         weight='bold'  # Kalın yazı
                     ),
                     bgcolor='rgba(0, 0, 0, 0)',  # Şeffaf arka plan
                     bordercolor='rgba(0, 0, 0, 0)',
                     borderwidth=0,
                     xanchor='center',
                     yanchor='middle',
                     align='center'
                 ),
                 # Legend'i gauge dışına taşı
                 dict(
                     text="🟢 Düşük (0-40%)     🟡 Orta (40-60%)     🔴 Yüksek (60-100%)",
                     x=0.5, y=-0.15,  # Gauge dışında, altında
                     xref="paper", yref="paper",
                     showarrow=False,
                     font=dict(
                         size=10,  # Küçük font
                         color='rgba(255, 255, 255, 0.9)',  # Hafif şeffaf beyaz
                         family='Arial, sans-serif',
                         weight='normal'  # Normal kalınlık
                     ),
                     bgcolor='rgba(0, 0, 0, 0)',  # Tamamen şeffaf
                     bordercolor='rgba(0, 0, 0, 0)',
                     borderwidth=0,
                     xanchor='center',
                     yanchor='middle',
                     align='center'
                 )
             ]
         )
        
        return fig
    
    def create_parameter_radar(self, temp_risk, humidity_risk, vibration_risk, pressure_risk):
        """Parametre risk değerlerini radar grafiği olarak görselleştirir"""
        # Değerleri yüzde olarak al (0-1 aralığından yüzdeye çevir)
        categories = ['Sıcaklık', 'Nem', 'Titreşim', 'Basınç']
        values = [temp_risk * 100, humidity_risk * 100, vibration_risk * 100, pressure_risk * 100]
        
        fig = go.Figure()
        
        # Ana radar grafiği - Görseldeki formata uygun
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # İlk değeri ekle
            theta=categories + [categories[0]],  # İlk kategoriyi ekle
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.2)',  # Daha şeffaf yeşil
            line_color='rgba(46, 204, 113, 1)',   # Koyu yeşil kenarlık
            line_width=2,  # Kalın çizgi
            marker=dict(
                size=8,  # Küçük marker
                color='rgba(46, 204, 113, 1)',
                line=dict(color='white', width=1),
                symbol='circle'
            ),
            name='Risk Değerleri'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                    tickvals=[0, 20, 40, 60, 80, 100],
                    tickfont=dict(size=10, color='rgba(255, 255, 255, 0.7)', weight='normal'),
                    gridcolor='rgba(255, 255, 255, 0.3)',  # Daha hafif ızgara
                    linecolor='rgba(255, 255, 255, 0.5)',  # Daha hafif çizgiler
                    tickangle=0,
                    linewidth=1
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color='white', weight='normal'),
                    gridcolor='rgba(255, 255, 255, 0.3)',  # Daha hafif ızgara
                    linecolor='rgba(255, 255, 255, 0.5)',  # Daha hafif çizgiler
                    tickmode='array',
                    ticktext=['Sıcaklık', 'Nem', 'Titreşim', 'Basınç'],
                    tickvals=[0, 1, 2, 3],
                    linewidth=1,
                    tickangle=0,
                    showticklabels=True,  # Etiketleri zorla göster
                    ticklabelstep=1,  # Her etiketi göster
                    ticks='outside'  # Etiketleri dışarıda göster
                ),
                bgcolor='rgba(0, 0, 0, 0)',  # Şeffaf arka plan
                domain=dict(x=[0.1, 0.9], y=[0.1, 0.9])  # Normal alan
            ),
            
            showlegend=False,
                         title=dict(
                 text="Test Parametreleri (Risk Değerlendirme %)",
                 font=dict(size=18, color='white', weight='normal'),
                 x=0.5,
                 y=0.98,
                 xanchor='center',
                 yanchor='top'
             ),
                        paper_bgcolor='rgba(0,0,0,0)',  # Şeffaf arka plan
            plot_bgcolor='rgba(0,0,0,0)',    # Şeffaf arka plan
            width=700,  # Daha büyük boyut
            height=600,  # Daha büyük boyut
            margin=dict(t=80, b=60, l=60, r=60)  # Daha büyük margin
        )
        
        return fig
    
    def create_risk_breakdown(self, risk_factors: Dict[str, float], model_risk_score: float = None) -> go.Figure:
        """Risk faktörlerinin dağılımını gösterir"""
        
        # Parametre isimlerini Türkçe'ye çevir
        turkish_names = {
            'temperature_risk': 'Sıcaklık Riski',
            'humidity_risk': 'Nem Riski',
            'vibration_risk': 'Titreşim Riski',
            'pressure_risk': 'Basınç Riski',
            'total_risk': 'Toplam Risk'
        }
        
        # Türkçe isimlerle yeni dictionary oluştur
        turkish_risk_factors = {}
        for key, value in risk_factors.items():
            turkish_key = turkish_names.get(key, key)
            # Eğer model risk skoru verilmişse ve bu "Toplam Risk" ise, model skorunu kullan
            if turkish_key == 'Toplam Risk' and model_risk_score is not None:
                turkish_risk_factors[turkish_key] = model_risk_score
            else:
                turkish_risk_factors[turkish_key] = value
        
        factors = list(turkish_risk_factors.keys())
        values = list(turkish_risk_factors.values())
        
        # Risk seviyelerine göre renk belirleme - Normalize edilmiş değerler (0-1 aralığı)
        colors = []
        for v in values:
            if v > 0.6:  # Yüksek risk (60-100%)
                colors.append(self.colors['danger'])
            elif v > 0.3:  # Orta risk (30-60%)
                colors.append(self.colors['warning'])
            else:  # Düşük risk (0-30%)
                colors.append(self.colors['success'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=factors,
                y=values,
                marker_color=colors,
                text=[f'{v:.1%}' for v in values],  # Yüzde gösterimi
                textposition='auto'
            )
        ])
        
        fig.update_layout(
             title=dict(
                 text="Risk Faktörleri Analizi",
                 font=dict(size=18, color='white', weight='normal')
             ),
             xaxis_title=dict(
                 text="Risk Parametreleri",
                 font=dict(size=14, color='white')
             ),
             yaxis_title=dict(
                 text="Risk Değerlendirme Skoru",
                 font=dict(size=14, color='white')
             ),
             height=400,
             showlegend=False
         )
        
        return fig
    
    def create_test_history_chart(self, df: pd.DataFrame) -> go.Figure:
        """Test geçmişi grafiği oluşturur"""
        
        if 'test_date' in df.columns and 'risk_score' in df.columns:
            df['test_date'] = pd.to_datetime(df['test_date'])
            df = df.sort_values('test_date')
            
            fig = go.Figure()
            
            # Risk skoru trendi
            fig.add_trace(go.Scatter(
                x=df['test_date'],
                y=df['risk_score'],
                mode='lines+markers',
                name='Risk Skoru',
                line=dict(color=self.colors['primary'])
            ))
            
            # Pass/Fail noktaları
            if 'pass_fail' in df.columns:
                pass_data = df[df['pass_fail'] == 'PASS']
                fail_data = df[df['pass_fail'] == 'FAIL']
                
                if not pass_data.empty:
                    fig.add_trace(go.Scatter(
                        x=pass_data['test_date'],
                        y=pass_data['risk_score'],
                        mode='markers',
                        name='PASS',
                        marker=dict(color=self.colors['success'], size=8)
                    ))
                
                if not fail_data.empty:
                    fig.add_trace(go.Scatter(
                        x=fail_data['test_date'],
                        y=fail_data['risk_score'],
                        mode='markers',
                        name='FAIL',
                        marker=dict(color=self.colors['danger'], size=8)
                    ))
            
            fig.update_layout(
                 title=dict(
                     text="Test Geçmişi - Risk Değerlendirme Trendi",
                     font=dict(size=18, color='white', weight='normal')
                 ),
                 xaxis_title=dict(
                     text="Test Tarihi",
                     font=dict(size=14, color='white')
                 ),
                 yaxis_title=dict(
                     text="Risk Değerlendirme Skoru",
                     font=dict(size=14, color='white')
                 ),
                 height=500
             )
            
            return fig
        
        return go.Figure()
    
    def create_parameter_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Test parametrelerinin dağılımını gösterir"""
        
        parameters = ['temperature', 'humidity', 'vibration', 'pressure']
        available_params = [p for p in parameters if p in df.columns]
        
        if not available_params:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=available_params,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['info'], self.colors['warning']]
        
        for i, param in enumerate(available_params):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df[param],
                    nbinsx=20,
                    name=param,
                    marker_color=colors[i % len(colors)]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
             title=dict(
                 text="Test Parametreleri Analizi",
                 font=dict(size=18, color='white', weight='normal')
             ),
             height=600,
             showlegend=False
         )
        
        return fig
    
    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray) -> go.Figure:
        """Confusion matrix grafiği oluşturur"""
        
        labels = ['PASS', 'FAIL']
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
             title=dict(
                 text="Karışıklık Matrisi",
                 font=dict(size=18, color='white', weight='normal')
             ),
             xaxis_title=dict(
                 text="Tahmin Edilen Sonuç",
                 font=dict(size=14, color='white')
             ),
             yaxis_title=dict(
                 text="Gerçek Sonuç",
                 font=dict(size=14, color='white')
             ),
             height=400
         )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance: pd.DataFrame) -> go.Figure:
        """Özellik önem dereceleri grafiği oluşturur"""
        
        fig = go.Figure(data=go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker_color=self.colors['primary']
        ))
        
        fig.update_layout(
             title=dict(
                 text="Özellik Önem Analizi",
                 font=dict(size=18, color='white', weight='normal')
             ),
             xaxis_title=dict(
                 text="Önem Derecesi",
                 font=dict(size=14, color='white')
             ),
             yaxis_title=dict(
                 text="Test Parametreleri",
                 font=dict(size=14, color='white')
             ),
             height=400
         )
        
        return fig
    
    def create_dashboard(self, test_data: Dict, risk_factors: Dict, 
                        model_info: Dict) -> go.Figure:
        """Ana dashboard grafiği oluşturur"""
        
        # Alt grafikler oluştur
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Risk Değerlendirme Skoru", "Risk Faktörleri Analizi", "Test Parametreleri Analizi", "Model Performans Metrikleri"),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatterpolar"}, {"type": "table"}]]
        )
        
        # 1. Risk skoru gauge
        risk_score = test_data.get('risk_score', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 100], 'color': "red"}]}
            ),
            row=1, col=1
        )
        
        # 2. Risk faktörleri
        factors = list(risk_factors.keys())
        values = list(risk_factors.values())
        fig.add_trace(
            go.Bar(x=factors, y=values, marker_color=self.colors['danger']),
            row=1, col=2
        )
        
        # 3. Test parametreleri radar
        test_params = {
            'Sıcaklık Riski': test_data.get('temperature', 0),
            'Nem Riski': test_data.get('humidity', 0),
            'Titreşim Riski': test_data.get('vibration', 0),
            'Basınç Riski': test_data.get('pressure', 0)
        }
        fig.add_trace(
            go.Scatterpolar(
                r=list(test_params.values()),
                theta=list(test_params.keys()),
                fill='toself',
                line_color=self.colors['primary']
            ),
            row=2, col=1
        )
        
        # 4. Model bilgileri tablosu
        model_data = [
            ['Model Algoritması', model_info.get('model_type', 'N/A')],
            ['Doğruluk Oranı', f"{model_info.get('accuracy', 0):.3f}"],
            ['F1-Skoru', f"{model_info.get('f1_score', 0):.3f}"],
            ['Kesinlik', f"{model_info.get('precision', 0):.3f}"],
            ['Duyarlılık', f"{model_info.get('recall', 0):.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Performans Metriği', 'Değer']),
                cells=dict(values=[[row[0] for row in model_data], 
                                 [row[1] for row in model_data]])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
             title=dict(
                 text="TestScope AI - Risk Değerlendirme Dashboard",
                 font=dict(size=20, color='white', weight='normal')
             ),
             height=800,
             showlegend=False
         )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, path: str = "notebooks/"):
        """Grafiği dosyaya kaydeder"""
        
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        # HTML formatında kaydet
        fig.write_html(f"{filepath}.html")
        
        # PNG formatında kaydet
        fig.write_image(f"{filepath}.png")
        
        print(f"Grafik kaydedildi: {filepath}")
    
    def create_matplotlib_plots(self, df: pd.DataFrame, save_path: str = "notebooks/"):
        """Matplotlib ile temel grafikler oluşturur"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Risk skoru dağılımı
        if 'risk_score' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['risk_score'], bins=30, alpha=0.7, color=self.colors['primary'])
            plt.axvline(x=0.5, color='red', linestyle='--', label='Risk Eşiği')
            plt.xlabel('Risk Skoru')
            plt.ylabel('Frekans')
            plt.title('Risk Skor Dağılımı')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Pass/Fail oranları
        if 'pass_fail' in df.columns:
            plt.figure(figsize=(8, 6))
            pass_fail_counts = df['pass_fail'].value_counts()
            colors = [self.colors['success'], self.colors['danger']]
            plt.pie(pass_fail_counts.values, labels=pass_fail_counts.index, 
                   colors=colors, autopct='%1.1f%%')
            plt.title('Pass/Fail Oranları')
            plt.tight_layout()
            plt.savefig(f'{save_path}pass_fail_ratio.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Test kategorileri dağılımı
        if 'test_category' in df.columns:
            plt.figure(figsize=(10, 6))
            category_counts = df['test_category'].value_counts()
            sns.barplot(x=category_counts.values, y=category_counts.index)
            plt.title('Test Kategorileri Dağılımı')
            plt.xlabel('Test Sayısı')
            plt.tight_layout()
            plt.savefig(f'{save_path}test_categories.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Matplotlib grafikleri kaydedildi: {save_path}") 