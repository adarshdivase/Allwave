import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List
import json
from datetime import datetime, timedelta
import base64
from io import BytesIO

st.set_page_config(page_title="AI Room Configurator Pro Max", page_icon="🏢", layout="wide")

# Enhanced Custom CSS with Advanced Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        font-weight: 500;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1e3c72 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        color: white !important;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-card h3, .metric-card h4, .metric-card p {
        color: white !important;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .comparison-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 2px solid #e1e5e9;
        transition: all 0.3s ease;
    }
    
    .comparison-card:hover {
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.1);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Comprehensive Product Database ---
class EnhancedProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Budget': {
                    'BenQ IRP55': {'price': 1299, 'specs': '55" 4K Interactive Display, 20-point touch', 'rating': 4.2, 'brand': 'BenQ'},
                    'LG 75UP8000PUA': {'price': 1899, 'specs': '75" 4K LED, webOS, ThinQ AI', 'rating': 4.4, 'brand': 'LG'},
                    'Samsung QB65R': {'price': 2499, 'specs': '65" 4K QLED Business Display', 'rating': 4.5, 'brand': 'Samsung'}
                },
                'Professional': {
                    'Sharp/NEC 100" 4K Display': {'price': 15999, 'specs': '100" 4K UHD, 500 nits, 24/7 Operation', 'rating': 4.7, 'brand': 'Sharp/NEC'},
                    'Sony BRAVIA FW-85BZ40H': {'price': 8999, 'specs': '85" 4K Pro Display, Android TV', 'rating': 4.6, 'brand': 'Sony'},
                    'Planar UltraRes X Series': {'price': 12999, 'specs': '86" 4K Multi-touch Display', 'rating': 4.5, 'brand': 'Planar'}
                },
                'Premium': {
                    'LG MAGNIT 136"': {'price': 75000, 'specs': 'MicroLED, 4K, AI-powered processing, Cable-less', 'rating': 4.9, 'brand': 'LG'},
                    'Samsung "The Wall" 146"': {'price': 99999, 'specs': 'MicroLED, 4K, 0.8mm Pixel Pitch, AI Upscaling', 'rating': 5.0, 'brand': 'Samsung'},
                    'Sony Crystal LED 220"': {'price': 150000, 'specs': 'Crystal LED, 4K+, Seamless Modular Design', 'rating': 5.0, 'brand': 'Sony'}
                }
            },
            'cameras': {
                'Budget': {
                    'Logitech MeetUp': {'price': 899, 'specs': '4K Ultra HD, 120° FOV, Built-in Speakers', 'rating': 4.3, 'brand': 'Logitech'},
                    'Poly Studio P5': {'price': 699, 'specs': 'HD Webcam, Automatic Group Framing', 'rating': 4.2, 'brand': 'Poly'},
                    'Jabra PanaCast': {'price': 1199, 'specs': '4K Panoramic Camera, 180° FOV', 'rating': 4.4, 'brand': 'Jabra'}
                },
                'Professional': {
                    'Logitech Rally Bar': {'price': 3999, 'specs': '4K PTZ, AI Viewfinder, RightSight Auto-Framing', 'rating': 4.8, 'brand': 'Logitech'},
                    'Poly Studio E70': {'price': 4200, 'specs': 'Dual 4K sensors, Poly DirectorAI, Speaker Tracking', 'rating': 4.9, 'brand': 'Poly'},
                    'Aver CAM520 Pro3': {'price': 2899, 'specs': '4K PTZ, 18x Optical Zoom, AI Auto-Framing', 'rating': 4.6, 'brand': 'Aver'}
                },
                'Premium': {
                    'Cisco Room Kit EQ': {'price': 19999, 'specs': 'AI-powered Quad Camera, Speaker Tracking, Codec', 'rating': 5.0, 'brand': 'Cisco'},
                    'Crestron Flex UC-MM30-Z': {'price': 15999, 'specs': 'Advanced AI Camera, 4K PTZ, Zoom Integration', 'rating': 4.9, 'brand': 'Crestron'},
                    'Polycom Studio X70': {'price': 12999, 'specs': 'Dual 4K cameras, AI-powered Director', 'rating': 4.8, 'brand': 'Polycom'}
                }
            },
            'audio': {
                'Budget': {
                    'Yamaha YVC-1000': {'price': 1299, 'specs': 'USB/Bluetooth Speakerphone, Adaptive Echo Canceller', 'rating': 4.3, 'brand': 'Yamaha'},
                    'ClearOne CHAT 50': {'price': 599, 'specs': 'USB Speakerphone, Duplex Audio', 'rating': 4.1, 'brand': 'ClearOne'},
                    'Jabra Speak 750': {'price': 399, 'specs': 'UC Speakerphone, 360° Microphone', 'rating': 4.4, 'brand': 'Jabra'}
                },
                'Professional': {
                    'QSC Core Nano': {'price': 2500, 'specs': 'Network Audio I/O, Q-SYS Ecosystem, Software DSP', 'rating': 4.7, 'brand': 'QSC'},
                    'Biamp TesiraFORTE X 400': {'price': 4500, 'specs': 'AEC, Dante/AVB, USB Audio, Launch Config', 'rating': 4.8, 'brand': 'Biamp'},
                    'ClearOne BMA 360': {'price': 3299, 'specs': 'Beamforming Mic Array, 360° Coverage', 'rating': 4.6, 'brand': 'ClearOne'}
                },
                'Premium': {
                    'Shure MXA920 Ceiling Array': {'price': 6999, 'specs': 'Automatic Coverage, Steerable Coverage, IntelliMix DSP', 'rating': 5.0, 'brand': 'Shure'},
                    'Sennheiser TeamConnect Ceiling 2': {'price': 5999, 'specs': 'AI-Enhanced Audio, Beam Steering Technology', 'rating': 4.9, 'brand': 'Sennheiser'},
                    'Audio-Technica ATUC-50CU': {'price': 4999, 'specs': 'Ceiling Array, AI Noise Reduction', 'rating': 4.8, 'brand': 'Audio-Technica'}
                }
            },
            'control_systems': {
                'Budget': {
                    'Extron TouchLink Pro 725T': {'price': 1999, 'specs': '7" Touchpanel, PoE+, Web Interface', 'rating': 4.3, 'brand': 'Extron'},
                    'AMX Modero X Series NXD-700Vi': {'price': 2299, 'specs': '7" Touch Panel, Built-in Video', 'rating': 4.4, 'brand': 'AMX'},
                    'Crestron TSW-570': {'price': 1799, 'specs': '5" Touch Screen, Wi-Fi, PoE', 'rating': 4.5, 'brand': 'Crestron'}
                },
                'Professional': {
                    'Crestron Flex UC': {'price': 3999, 'specs': 'Tabletop Touchpanel, UC Integration', 'rating': 4.6, 'brand': 'Crestron'},
                    'AMX Enova DGX': {'price': 5999, 'specs': 'Digital Matrix Switching, Control System', 'rating': 4.7, 'brand': 'AMX'},
                    'Extron DTP3 CrossPoint': {'price': 7999, 'specs': '4K60 Matrix Switching, Advanced Control', 'rating': 4.8, 'brand': 'Extron'}
                },
                'Premium': {
                    'Crestron NVX System': {'price': 15999, 'specs': 'Enterprise Control Platform, AV over IP', 'rating': 4.9, 'brand': 'Crestron'},
                    'Q-SYS Core 8 Flex': {'price': 12999, 'specs': 'Unified AV/IT Platform, Software-based', 'rating': 5.0, 'brand': 'QSC'},
                    'Biamp Vocia MS-1': {'price': 18999, 'specs': 'Networked Paging System, Enterprise Grade', 'rating': 4.9, 'brand': 'Biamp'}
                }
            },
            'lighting': {
                'Budget': {
                    'Philips Hue Pro': {'price': 899, 'specs': 'Smart LED System, App Control', 'rating': 4.2, 'brand': 'Philips'},
                    'Lutron Caseta Pro': {'price': 1299, 'specs': 'Wireless Dimming System', 'rating': 4.4, 'brand': 'Lutron'},
                    'Leviton Decora Smart': {'price': 799, 'specs': 'Wi-Fi Enabled Switches and Dimmers', 'rating': 4.1, 'brand': 'Leviton'}
                },
                'Professional': {
                    'Crestron DIN-2MC2': {'price': 2999, 'specs': '2-Channel Dimmer, 0-10V Control', 'rating': 4.6, 'brand': 'Crestron'},
                    'Lutron Quantum': {'price': 4999, 'specs': 'Total Light Management System', 'rating': 4.8, 'brand': 'Lutron'},
                    'ETC ColorSource': {'price': 3999, 'specs': 'LED Architectural Lighting', 'rating': 4.7, 'brand': 'ETC'}
                },
                'Premium': {
                    'Ketra N4 Hub': {'price': 8999, 'specs': 'Natural Light Technology, Circadian Rhythm', 'rating': 5.0, 'brand': 'Lutron/Ketra'},
                    'USAI BeveLED': {'price': 12999, 'specs': 'Architectural LED Lighting System', 'rating': 4.9, 'brand': 'USAI'},
                    'Signify Interact Pro': {'price': 15999, 'specs': 'IoT-connected Lighting Management', 'rating': 4.8, 'brand': 'Signify'}
                }
            }
        }
        
        self.room_templates = {
            'Huddle Room (2-6 people)': {
                'typical_size': (3, 4), 'capacity_range': (2, 6),
                'recommended_tier': 'Budget', 'typical_usage': 'Quick meetings, brainstorming'
            },
            'Small Conference (6-12 people)': {
                'typical_size': (4, 6), 'capacity_range': (6, 12),
                'recommended_tier': 'Professional', 'typical_usage': 'Team meetings, presentations'
            },
            'Large Conference (12-20 people)': {
                'typical_size': (6, 10), 'capacity_range': (12, 20),
                'recommended_tier': 'Professional', 'typical_usage': 'Department meetings, training'
            },
            'Boardroom (8-16 people)': {
                'typical_size': (5, 8), 'capacity_range': (8, 16),
                'recommended_tier': 'Premium', 'typical_usage': 'Executive meetings, board meetings'
            },
            'Training Room (20-50 people)': {
                'typical_size': (8, 12), 'capacity_range': (20, 50),
                'recommended_tier': 'Professional', 'typical_usage': 'Training, workshops, seminars'
            },
            'Auditorium (50+ people)': {
                'typical_size': (12, 20), 'capacity_range': (50, 200),
                'recommended_tier': 'Premium', 'typical_usage': 'Large presentations, events'
            }
        }

# --- Advanced AI Recommendation Engine ---
class MaximizedAVRecommender:
    def __init__(self):
        self.db = EnhancedProductDatabase()
    
    def get_comprehensive_recommendations(self, room_specs: Dict, user_preferences: Dict) -> Dict:
        budget_tier = user_preferences.get('budget_tier', 'Professional')
        brand_preference = user_preferences.get('preferred_brands', [])
        special_features = user_preferences.get('special_features', [])
        
        recommendations = {
            'display': self._recommend_display_advanced(room_specs, budget_tier, brand_preference),
            'camera': self._recommend_camera_advanced(room_specs, budget_tier, brand_preference),
            'audio': self._recommend_audio_advanced(room_specs, budget_tier, special_features),
            'control': self._recommend_control_advanced(room_specs, budget_tier),
            'lighting': self._recommend_lighting_advanced(room_specs, budget_tier, special_features),
            'accessories': self._recommend_accessories_advanced(room_specs, special_features),
            'alternatives': self._generate_alternatives(room_specs, budget_tier),
            'confidence_score': self._calculate_advanced_confidence(room_specs, user_preferences),
            'room_analysis': self._analyze_room_characteristics(room_specs),
            'upgrade_path': self._suggest_upgrade_path(room_specs, budget_tier)
        }
        return recommendations
    
    def _recommend_display_advanced(self, specs, tier, brands):
        room_area = specs['length'] * specs['width']
        viewing_distance = max(specs['length'], specs['width']) * 0.6
        
        # Smart display selection based on multiple factors
        if room_area < 20 and specs['capacity'] <= 8:
            size_category = 'small'
        elif room_area < 60 and specs['capacity'] <= 20:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        products = self.db.products['displays'][tier]
        
        # Filter by brand preference if specified
        if brands:
            products = {k: v for k, v in products.items() if v['brand'] in brands}
        
        if products:
            # Select based on room characteristics
            selected = list(products.items())[0]  # Default selection
            for name, product in products.items():
                if size_category == 'large' and '100"' in name or 'Wall' in name:
                    selected = (name, product)
                    break
                elif size_category == 'medium' and any(size in name for size in ['75"', '85"', '86"']):
                    selected = (name, product)
                    break
        else:
            # Fallback to any tier if brand filtering eliminated all options
            products = self.db.products['displays'][tier]
            selected = list(products.items())[0]
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['viewing_distance_optimal'] = f"{viewing_distance:.1f}m"
        result['brightness_needed'] = self._calculate_brightness_needs(specs)
        return result
    
    def _recommend_camera_advanced(self, specs, tier, brands):
        products = self.db.products['cameras'][tier]
        
        if brands:
            products = {k: v for k, v in products.items() if v['brand'] in brands}
        
        if not products:
            products = self.db.products['cameras'][tier]
        
        # Advanced camera selection logic
        room_depth = max(specs['length'], specs['width'])
        
        if specs['capacity'] <= 6 and room_depth <= 5:
            camera_type = 'fixed'
        elif specs['capacity'] <= 16:
            camera_type = 'ptz'
        else:
            camera_type = 'multi_camera'
        
        selected = list(products.items())[0]
        for name, product in products.items():
            if camera_type == 'multi_camera' and ('EQ' in name or 'Studio X' in name):
                selected = (name, product)
                break
            elif camera_type == 'ptz' and ('Rally' in name or 'E70' in name):
                selected = (name, product)
                break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['recommended_mounting'] = self._suggest_camera_mounting(specs)
        result['coverage_analysis'] = self._analyze_camera_coverage(specs, camera_type)
        return result
    
    def _recommend_audio_advanced(self, specs, tier, features):
        products = self.db.products['audio'][tier]
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        
        # Determine audio configuration based on room and features
        if 'Noise Reduction' in features:
            config_type = 'premium_processing'
        elif room_volume > 200:
            config_type = 'distributed'
        elif specs['ceiling_height'] > 3.5:
            config_type = 'ceiling_array'
        else:
            config_type = 'table_system'
        
        selected = list(products.items())[0]
        for name, product in products.items():
            if config_type == 'ceiling_array' and ('MXA920' in name or 'Ceiling' in name):
                selected = (name, product)
                break
            elif config_type == 'premium_processing' and ('TesiraFORTE' in name or 'Core' in name):
                selected = (name, product)
                break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['configuration'] = self._design_audio_config(specs, config_type)
        result['acoustic_analysis'] = self._analyze_acoustics(specs)
        return result
    
    def _recommend_control_advanced(self, specs, tier):
        products = self.db.products['control_systems'][tier]
        complexity_score = len(specs.get('special_requirements', [])) + (specs['capacity'] // 10)
        
        selected = list(products.items())[0]
        if complexity_score > 3:  # High complexity
            for name, product in products.items():
                if 'NVX' in name or 'DGX' in name or 'Core' in name:
                    selected = (name, product)
                    break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['integration_options'] = self._suggest_integrations(specs)
        return result
    
    def _recommend_lighting_advanced(self, specs, tier, features):
        products = self.db.products['lighting'][tier]
        
        # Lighting needs analysis
        window_factor = specs.get('windows', 0) / 100
        needs_daylight_sync = 'Circadian Lighting' in features or window_factor > 0.3
        
        selected = list(products.items())[0]
        if needs_daylight_sync and tier == 'Premium':
            for name, product in products.items():
                if 'Ketra' in name or 'Natural Light' in product['specs']:
                    selected = (name, product)
                    break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['lighting_analysis'] = self._analyze_lighting_needs(specs, features)
        return result
    
    def _recommend_accessories_advanced(self, specs, features):
        accessories = []
        
        # Essential accessories
        accessories.append({
            'category': 'Cable Management',
            'item': 'Under-table Cable Tray System',
            'model': 'FSR FL-500P Series',
            'price': 1299,
            'necessity': 'Essential'
        })
        
        # Feature-based accessories
        if 'Wireless Presentation' in features:
            accessories.append({
                'category': 'Wireless Presentation',
                'item': 'Professional Wireless System',
                'model': 'Barco ClickShare Conference CX-50',
                'price': 2999,
                'necessity': 'Required'
            })
        
        if 'Room Scheduling' in features:
            accessories.append({
                'category': 'Room Booking',
                'item': 'Smart Room Panel',
                'model': 'Crestron TSS-1070-B-S',
                'price': 1899,
                'necessity': 'Required'
            })
        
        if 'Digital Whiteboard' in features:
            accessories.append({
                'category': 'Interactive Display',
                'item': 'Digital Whiteboard Solution',
                'model': 'Microsoft Surface Hub 2S 85"',
                'price': 21999,
                'necessity': 'Optional'
            })
        
        # Room-size based accessories
        if specs['capacity'] > 16:
            accessories.append({
                'category': 'Power Management',
                'item': 'Intelligent Power Distribution',
                'model': 'Middle Atlantic UPS-2200R',
                'price': 1599,
                'necessity': 'Recommended'
            })
        
        # Advanced accessories
        accessories.extend([
            {'category': 'Environmental', 'item': 'Air Quality Monitor', 'model': 'Awair Omni', 'price': 299, 'necessity': 'Optional'},
            {'category': 'Security', 'item': 'Privacy Screen System', 'model': 'Smart Glass Solutions', 'price': 4999, 'necessity': 'Optional'},
            {'category': 'Furniture', 'item': 'Height-Adjustable Table', 'model': 'Herman Miller Ratio', 'price': 3999, 'necessity': 'Recommended'}
        ])
        
        return accessories
    
    def _generate_alternatives(self, specs, tier):
        alternatives = {}
        all_tiers = ['Budget', 'Professional', 'Premium']
        
        for alt_tier in all_tiers:
            if alt_tier != tier:
                alt_recs = {}
                for category in ['displays', 'cameras', 'audio']:
                    products = self.db.products[category][alt_tier]
                    alt_recs[category] = list(products.items())[0]
                alternatives[alt_tier] = alt_recs
        
        return alternatives
    
    def _analyze_room_characteristics(self, specs):
        room_area = specs['length'] * specs['width']
        room_volume = room_area * specs['ceiling_height']
        aspect_ratio = max(specs['length'], specs['width']) / min(specs['length'], specs['width'])
        
        analysis = {
            'size_category': self._categorize_room_size(room_area),
            'shape_analysis': self._analyze_room_shape(aspect_ratio),
            'acoustic_properties': self._estimate_acoustic_properties(specs),
            'lighting_challenges': self._identify_lighting_challenges(specs),
            'capacity_efficiency': self._analyze_capacity_efficiency(specs)
        }
        
        return analysis
    
    def _suggest_upgrade_path(self, specs, current_tier):
        tiers = ['Budget', 'Professional', 'Premium']
        current_index = tiers.index(current_tier)
        
        upgrade_path = []
        
        if current_index < len(tiers) - 1:
            next_tier = tiers[current_index + 1]
            upgrade_path.append({
                'phase': 'Short-term (6-12 months)',
                'tier': next_tier,
                'focus': 'Core AV upgrade',
                'estimated_cost': self._estimate_tier_cost(specs, next_tier) - self._estimate_tier_cost(specs, current_tier)
            })
        
        if current_index < len(tiers) - 2:
            ultimate_tier = tiers[-1]
            upgrade_path.append({
                'phase': 'Long-term (2-3 years)',
                'tier': ultimate_tier,
                'focus': 'Premium features & AI integration',
                'estimated_cost': self._estimate_tier_cost(specs, ultimate_tier) - self._estimate_tier_cost(specs, current_tier)
            })
        
        return upgrade_path
    
    # Helper methods
    def _calculate_brightness_needs(self, specs):
        window_factor = specs.get('windows', 0) / 100
        base_brightness = 300  # nits
        window_adjustment = window_factor * 200
        return int(base_brightness + window_adjustment)
    
    def _suggest_camera_mounting(self, specs):
        if specs['ceiling_height'] > 3.5:
            return "Ceiling mount recommended for optimal coverage"
        else:
            return "Wall mount at display location"
    
    def _analyze_camera_coverage(self, specs, camera_type):
        room_area = specs['length'] * specs['width']
        if camera_type == 'multi_camera':
            coverage = min(100, (room_area / 80) * 100)
        else:
            coverage = min(95, (room_area / 50) * 100)
        return f"{coverage:.0f}% optimal coverage"

    def _design_audio_config(self, specs, config_type):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        
        config = {
            'type': config_type,
            'coverage': f"{min(100, room_volume / 2):.0f}%",
            'microphone_count': max(2, specs['capacity'] // 4),
            'speaker_zones': max(1, specs['capacity'] // 8),
            'processing_power': 'High' if room_volume > 150 else 'Standard'
        }
        
        return config
    
    def _analyze_acoustics(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        surface_area = 2 * (specs['length'] * specs['width'] + 
                            specs['length'] * specs['ceiling_height'] + 
                            specs['width'] * specs['ceiling_height'])
        
        # Estimate reverberation time (simplified)
        rt60_estimate = 0.16 * room_volume / (0.3 * surface_area)  # Assuming moderate absorption
        
        return {
            'rt60_estimate': f"{rt60_estimate:.2f} seconds",
            'acoustic_treatment_needed': rt60_estimate > 0.8,
            'sound_masking_recommended': specs['capacity'] > 20,
            'echo_risk': 'High' if max(specs['length'], specs['width']) > 8 else 'Low'
        }
    
    def _suggest_integrations(self, specs):
        integrations = ['Microsoft Teams', 'Zoom', 'Google Meet']
        
        if specs['capacity'] > 16:
            integrations.extend(['Cisco Webex', 'BlueJeans'])
        
        if 'VTC' in specs.get('special_requirements', []):
            integrations.append('Polycom RealPresence')
        
        return integrations
    
    def _analyze_lighting_needs(self, specs, features):
        window_factor = specs.get('windows', 0) / 100
        room_area = specs['length'] * specs['width']
        
        analysis = {
            'natural_light_factor': f"{window_factor * 100:.0f}%",
            'artificial_light_zones': max(1, room_area // 20),
            'dimming_required': True,
            'color_temperature_control': 'Circadian Lighting' in features,
            'daylight_harvesting': window_factor > 0.2
        }
        
        return analysis
    
    def _calculate_advanced_confidence(self, specs, preferences):
        base_confidence = 0.85
        
        # Adjust based on room complexity
        if len(specs.get('special_requirements', [])) > 3:
            base_confidence -= 0.1
        
        # Adjust based on brand constraints
        if preferences.get('preferred_brands'):
            base_confidence += 0.05
        
        # Adjust based on budget tier
        if preferences.get('budget_tier') == 'Premium':
            base_confidence += 0.1
        
        return min(0.99, max(0.70, base_confidence))
    
    def _categorize_room_size(self, area):
        if area < 20:
            return "Small (Huddle)"
        elif area < 50:
            return "Medium (Conference)"
        elif area < 100:
            return "Large (Training)"
        else:
            return "Extra Large (Auditorium)"
    
    def _analyze_room_shape(self, aspect_ratio):
        if aspect_ratio < 1.3:
            return "Square - Good for collaboration"
        elif aspect_ratio < 2.0:
            return "Rectangular - Versatile layout"
        else:
            return "Long/Narrow - Challenging for AV"
    
    def _estimate_acoustic_properties(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        
        properties = {
            'reverb_category': 'High' if room_volume > 200 else 'Moderate' if room_volume > 80 else 'Low',
            'treatment_needed': room_volume > 150,
            'echo_potential': max(specs['length'], specs['width']) > 10
        }
        
        return properties
    
    def _identify_lighting_challenges(self, specs):
        challenges = []
        
        if specs.get('windows', 0) > 30:
            challenges.append("High natural light - glare control needed")
        
        if specs['ceiling_height'] > 4:
            challenges.append("High ceiling - requires powerful fixtures")
        
        if specs['capacity'] > 20:
            challenges.append("Large audience - uniform lighting critical")
        
        return challenges if challenges else ["Standard lighting requirements"]
    
    def _analyze_capacity_efficiency(self, specs):
        room_area = specs['length'] * specs['width']
        efficiency = specs['capacity'] / room_area
        
        if efficiency > 0.5:
            return "High density - space optimization good"
        elif efficiency > 0.3:
            return "Moderate density - balanced layout"
        else:
            return "Low density - spacious environment"
    
    def _estimate_tier_cost(self, specs, tier):
        base_costs = {'Budget': 15000, 'Professional': 45000, 'Premium': 120000}
        room_multiplier = 1 + (specs['capacity'] / 50)
        return int(base_costs[tier] * room_multiplier)


# --- Advanced Visualization Components ---
class EnhancedVisualizationEngine:
    @staticmethod
    def create_3d_room_visualization(room_specs, recommendations):
        fig = go.Figure()
        
        # Room dimensions
        length, width, height = room_specs['length'], room_specs['width'], room_specs['ceiling_height']
        
        # Room wireframe
        room_x = [0, length, length, 0, 0, 0, length, length, 0, 0, length, length, length, length, 0, 0]
        room_y = [0, 0, width, width, 0, 0, 0, width, width, 0, 0, width, 0, width, width, 0]
        room_z = [0, 0, 0, 0, 0, height, height, height, height, height, height, height, height, height, height, height]
        
        fig.add_trace(go.Scatter3d(
            x=room_x, y=room_y, z=room_z,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.8)', width=2),
            name='Room Structure'
        ))
        
        # Display position
        display_x = [length * 0.05, length * 0.05]
        display_y = [width * 0.2, width * 0.8]
        display_z = [height * 0.4, height * 0.4]
        
        fig.add_trace(go.Scatter3d(
            x=display_x, y=display_y, z=display_z,
            mode='markers+lines',
            marker=dict(size=15, color='red'),
            line=dict(color='red', width=8),
            name='Display Wall'
        ))
        
        # Camera position
        fig.add_trace(go.Scatter3d(
            x=[length * 0.95], y=[width * 0.5], z=[height * 0.8],
            mode='markers',
            marker=dict(size=12, color='blue', symbol='diamond'),
            name='Camera'
        ))
        
        # Seating arrangement
        seats_x = []
        seats_y = []
        seats_z = []
        
        rows = max(2, room_specs['capacity'] // 8)
        cols = room_specs['capacity'] // rows
        
        for row in range(rows):
            for col in range(cols):
                seats_x.append(length * (0.3 + 0.5 * row / rows))
                seats_y.append(width * (0.2 + 0.6 * col / cols))
                seats_z.append(0.8)
        
        fig.add_trace(go.Scatter3d(
            x=seats_x, y=seats_y, z=seats_z,
            mode='markers',
            marker=dict(size=6, color='green'),
            name='Seating'
        ))
        
        fig.update_layout(
            title="3D Room Layout Visualization",
            scene=dict(
                xaxis_title=f"Length ({length}m)",
                yaxis_title=f"Width ({width}m)",
                zaxis_title=f"Height ({height}m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_cost_breakdown_chart(recommendations):
        categories = ['Display', 'Camera', 'Audio', 'Control', 'Lighting']
        costs = [
            recommendations['display']['price'],
            recommendations['camera']['price'],
            recommendations['audio']['price'],
            recommendations['control']['price'],
            recommendations['lighting']['price']
        ]
        
        # Add accessories cost
        if 'accessories' in recommendations:
            accessories_cost = sum(item['price'] for item in recommendations['accessories'][:5])
            categories.append('Accessories')
            costs.append(accessories_cost)
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=costs,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
                text=[f"${cost:,.0f}" for cost in costs],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Cost Breakdown by Category",
            xaxis_title="Equipment Category",
            yaxis_title="Cost (USD)",
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_feature_comparison_radar(recommendations, alternatives):
        categories = ['Performance', 'Features', 'Reliability', 'Integration', 'Value']
        
        # Current recommendation scores
        current_scores = [4.5, 4.2, 4.6, 4.4, 4.3]  # Example scores
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=current_scores + [current_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Recommendation',
            line_color='rgb(102, 126, 234)'
        ))
        
        # Add alternative tier comparison if available
        if alternatives and 'Professional' in alternatives:
            alt_scores = [4.0, 3.8, 4.2, 4.0, 4.8]
            fig.add_trace(go.Scatterpolar(
                r=alt_scores + [alt_scores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Professional Alternative',
                line_color='rgb(255, 107, 107)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )),
            showlegend=True,
            title="Feature Comparison Analysis",
            height=500
        )
        
        return fig


# --- Main Application ---
def main():
    st.title("🏢 AI Room Configurator Pro Max")
    st.markdown("### Transform Your Space with Intelligent AV Design")
    
    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    if 'room_specs' not in st.session_state:
        st.session_state.room_specs = None
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="premium-card"><h2>🎛️ Room Configuration</h2></div>', unsafe_allow_html=True)
        
        # Room Template Selection
        template = st.selectbox(
            "Room Template",
            options=list(EnhancedProductDatabase().room_templates.keys()),
            help="Choose a template that best matches your intended use"
        )
        
        template_info = EnhancedProductDatabase().room_templates[template]
        
        # Room Dimensions
        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            length = st.slider("Length (m)", 2.0, 20.0, float(template_info['typical_size'][0]), 0.5)
        with col2:
            width = st.slider("Width (m)", 2.0, 20.0, float(template_info['typical_size'][1]), 0.5)
        
        ceiling_height = st.slider("Ceiling Height (m)", 2.4, 6.0, 3.0, 0.1)
        capacity = st.slider("Capacity", 2, 100, template_info['capacity_range'][1])
        
        # Environment Factors
        st.subheader("🌟 Environment")
        windows = st.slider("Windows (%)", 0, 80, 20, 5)
        ambient_light = st.select_slider("Ambient Light", 
                                        options=['Low', 'Medium', 'High'], 
                                        value='Medium')
        
        # Budget & Brand Preferences
        st.subheader("💰 Budget & Brands")
        budget_tier = st.selectbox("Budget Tier", 
                                    options=['Budget', 'Professional', 'Premium'],
                                    index=1)
        
        preferred_brands = st.multiselect("Preferred Brands",
                                        options=['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'],
                                        help="Leave empty for best overall recommendations")
        
        # Special Features
        st.subheader("✨ Special Features")
        special_features = st.multiselect("Required Features",
                                        options=['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 
                                                'Noise Reduction', 'Circadian Lighting', 'AI Analytics', 'Cloud Integration'])
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            special_requirements = st.multiselect("Special Requirements",
                                                options=['Government Security', 'Medical Compliance', 'Broadcast Quality',
                                                        'Multi-language Support', '24/7 Operation'])
            
            integration_priority = st.select_slider("Integration Priority",
                                                options=['Cost', 'Performance', 'Future-proof'],
                                                value='Performance')
        
        # Generate Button
        if st.button("🚀 Generate AI Recommendation", type="primary", use_container_width=True):
            room_specs = {
                'template': template,
                'length': length,
                'width': width,
                'ceiling_height': ceiling_height,
                'capacity': capacity,
                'windows': windows,
                'ambient_light': ambient_light,
                'special_requirements': special_requirements
            }
            
            user_preferences = {
                'budget_tier': budget_tier,
                'preferred_brands': preferred_brands,
                'special_features': special_features,
                'integration_priority': integration_priority
            }
            
            recommender = MaximizedAVRecommender()
            recommendations = recommender.get_comprehensive_recommendations(room_specs, user_preferences)
            
            st.session_state.recommendations = recommendations
            st.session_state.room_specs = room_specs
            st.success("✅ AI Analysis Complete!")
    
    # Main Content Area
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        room_specs = st.session_state.room_specs
        
        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_cost = (recommendations['display']['price'] + 
                      recommendations['camera']['price'] + 
                      recommendations['audio']['price'] + 
                      recommendations['control']['price'] + 
                      recommendations['lighting']['price'])
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>${total_cost:,.0f}</h3>
                <p>Total Investment</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{recommendations['confidence_score']:.0%}</h3>
                <p>AI Confidence</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{room_specs['length']}m × {room_specs['width']}m</h3>
                <p>Room Size</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{room_specs['capacity']}</h3>
                <p>Capacity</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Tab Interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📊 Analysis", "🎨 Visualization", "🔄 Alternatives", "📋 Report"])
        
        with tab1:
            st.subheader("AI-Powered Equipment Recommendations")
            
            # Equipment cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📺 Display System")
                display_rec = recommendations['display']
                st.markdown(f'''
                <div class="feature-card">
                    <h4>{display_rec['model']}</h4>
                    <p><strong>Price:</strong> ${display_rec['price']:,}</p>
                    <p><strong>Specs:</strong> {display_rec['specs']}</p>
                    <p><strong>Rating:</strong> ⭐ {display_rec['rating']}/5.0</p>
                    <p><strong>Optimal Viewing:</strong> {display_rec['viewing_distance_optimal']}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("#### 🎥 Camera System")
                camera_rec = recommendations['camera']
                st.markdown(f'''
                <div class="feature-card">
                    <h4>{camera_rec['model']}</h4>
                    <p><strong>Price:</strong> ${camera_rec['price']:,}</p>
                    <p><strong>Specs:</strong> {camera_rec['specs']}</p>
                    <p><strong>Rating:</strong> ⭐ {camera_rec['rating']}/5.0</p>
                    <p><strong>Coverage:</strong> {camera_rec['coverage_analysis']}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("#### 🔊 Audio System")
                audio_rec = recommendations['audio']
                st.markdown(f'''
                <div class="feature-card">
                    <h4>{audio_rec['model']}</h4>
                    <p><strong>Price:</strong> ${audio_rec['price']:,}</p>
                    <p><strong>Specs:</strong> {audio_rec['specs']}</p>
                    <p><strong>Rating:</strong> ⭐ {audio_rec['rating']}/5.0</p>
                    <p><strong>Configuration:</strong> {audio_rec['configuration']['type']}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎛️ Control System")
                control_rec = recommendations['control']
                st.markdown(f'''
                <div class="feature-card">
                    <h4>{control_rec['model']}</h4>
                    <p><strong>Price:</strong> ${control_rec['price']:,}</p>
                    <p><strong>Specs:</strong> {control_rec['specs']}</p>
                    <p><strong>Rating:</strong> ⭐ {control_rec['rating']}/5.0</p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("#### 💡 Lighting System")
                lighting_rec = recommendations['lighting']
                st.markdown(f'''
                <div class="feature-card">
                    <h4>{lighting_rec['model']}</h4>
                    <p><strong>Price:</strong> ${lighting_rec['price']:,}</p>
                    <p><strong>Specs:</strong> {lighting_rec['specs']}</p>
                    <p><strong>Rating:</strong> ⭐ {lighting_rec['rating']}/5.0</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Accessories
                if recommendations['accessories']:
                    st.markdown("#### 🔧 Essential Accessories")
                    for acc in recommendations['accessories'][:3]:
                        st.markdown(f'''
                        <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #667eea;">
                            <strong>{acc['item']}</strong><br>
                            Model: {acc['model']}<br>
                            Price: ${acc['price']:,} ({acc['necessity']})
                        </div>
                        ''', unsafe_allow_html=True)
        
        with tab2:
            st.subheader("📊 Room Analysis & Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Room Characteristics")
                analysis = recommendations['room_analysis']
                
                st.markdown(f'''
                <div class="comparison-card">
                    <h5>Size Analysis</h5>
                    <p><strong>Category:</strong> {analysis['size_category']}</p>
                    <p><strong>Shape:</strong> {analysis['shape_analysis']}</p>
                    <p><strong>Capacity Efficiency:</strong> {analysis['capacity_efficiency']}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="comparison-card">
                    <h5>Acoustic Properties</h5>
                    <p><strong>Reverb Category:</strong> {analysis['acoustic_properties']['reverb_category']}</p>
                    <p><strong>Treatment Needed:</strong> {'Yes' if analysis['acoustic_properties']['treatment_needed'] else 'No'}</p>
                    <p><strong>Echo Risk:</strong> {'High' if analysis['acoustic_properties']['echo_potential'] else 'Low'}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Performance Visualization")
                st.plotly_chart(EnhancedVisualizationEngine.create_cost_breakdown_chart(recommendations), 
                                use_container_width=True)
        
        with tab3:
            st.subheader("🎨 3D Room Visualization")
            
            # 3D Room Layout
            fig_3d = EnhancedVisualizationEngine.create_3d_room_visualization(room_specs, recommendations)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Feature Comparison Radar
            col1, col2 = st.columns(2)
            with col1:
                fig_radar = EnhancedVisualizationEngine.create_feature_comparison_radar(recommendations, 
                                                                                        recommendations.get('alternatives', {}))
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Installation Timeline
                st.markdown("#### 📅 Installation Timeline")
                timeline_data = [
                    {'Phase': 'Planning & Design', 'Duration': '2-3 weeks', 'Status': 'Ready'},
                    {'Phase': 'Equipment Procurement', 'Duration': '3-6 weeks', 'Status': 'Pending'},
                    {'Phase': 'Installation', 'Duration': '1-2 weeks', 'Status': 'Pending'},
                    {'Phase': 'Testing & Training', 'Duration': '1 week', 'Status': 'Pending'}
                ]
                
                for item in timeline_data:
                    status_color = '#11998e' if item['Status'] == 'Ready' else '#f093fb'
                    st.markdown(f'''
                    <div style="background: {status_color}; color: white; padding: 10px; margin: 5px 0; border-radius: 8px;">
                        <strong>{item['Phase']}</strong><br>
                        Duration: {item['Duration']} | Status: {item['Status']}
                    </div>
                    ''', unsafe_allow_html=True)
        
        with tab4:
            st.subheader("🔄 Alternative Configurations")
            
            if recommendations.get('alternatives'):
                for tier_name, alt_config in recommendations['alternatives'].items():
                    st.markdown(f"#### {tier_name} Alternative")
                    
                    col1, col2, col3 = st.columns(3)
                    categories = ['displays', 'cameras', 'audio']
                    columns = [col1, col2, col3]
                    
                    for i, category in enumerate(categories):
                        with columns[i]:
                            if category in alt_config:
                                product_name, product_info = alt_config[category]
                                st.markdown(f'''
                                <div class="comparison-card">
                                    <h6>{category.title()}</h6>
                                    <p><strong>{product_name}</strong></p>
                                    <p>${product_info['price']:,}</p>
                                    <p>⭐ {product_info['rating']}/5.0</p>
                                </div>
                                ''', unsafe_allow_html=True)
            
            # Upgrade Path
            if recommendations.get('upgrade_path'):
                st.markdown("#### 🚀 Upgrade Roadmap")
                for upgrade in recommendations['upgrade_path']:
                    st.markdown(f'''
                    <div class="alert-success">
                        <h5>{upgrade['phase']}</h5>
                        <p><strong>Tier:</strong> {upgrade['tier']}</p>
                        <p><strong>Focus:</strong> {upgrade['focus']}</p>
                        <p><strong>Additional Investment:</strong> ${upgrade['estimated_cost']:,}</p>
                    </div>
                    ''', unsafe_allow_html=True)
        
        with tab5:
            st.subheader("📋 Professional Report")
            
            # Executive Summary
            st.markdown(f'''
            <div class="premium-card">
                <h3>Executive Summary</h3>
                <p>This AI-generated recommendation provides a comprehensive AV solution for a {room_specs['template']} 
                ({room_specs['length']}m × {room_specs['width']}m) accommodating {room_specs['capacity']} people.</p>
                
                <p><strong>Total Investment:</strong> ${total_cost:,}</p>
                <p><strong>Confidence Level:</strong> {recommendations['confidence_score']:.0%}</p>
                <p><strong>Recommended Tier:</strong> {budget_tier}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Detailed Specifications
            st.markdown("#### Detailed Equipment Specifications")
            
            specs_data = []
            for category in ['display', 'camera', 'audio', 'control', 'lighting']:
                item = recommendations[category]
                specs_data.append({
                    'Category': category.title(),
                    'Model': item['model'],
                    'Price': f"${item['price']:,}",
                    'Rating': f"{item['rating']}/5.0",
                    'Brand': item['brand']
                })
            
            df_specs = pd.DataFrame(specs_data)
            st.dataframe(df_specs, use_container_width=True)
            
            # Export Options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📄 Export PDF Report"):
                    st.info("PDF export functionality would be implemented here")
            
            with col2:
                if st.button("📊 Export Excel"):
                    st.info("Excel export functionality would be implemented here")
            
            with col3:
                if st.button("📧 Email Report"):
                    st.info("Email functionality would be implemented here")
    
    else:
        # Welcome Screen
        st.markdown('''
        <div class="premium-card" style="text-align: center; padding: 50px;">
            <h2>🚀 Welcome to AI Room Configurator Pro Max</h2>
            <p style="font-size: 18px;">Configure your room parameters in the sidebar to get started with AI-powered recommendations.</p>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 30px;">
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px;">
                    <h4>🎯 AI-Powered</h4>
                    <p>Advanced algorithms analyze your space and requirements</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px;">
                    <h4>📊 Data-Driven</h4>
                    <p>Recommendations based on extensive product database</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px;">
                    <h4>🔮 Future-Ready</h4>
                    <p>Scalable solutions with upgrade pathways</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div class="feature-card">
                <h4>🏢 Room Templates</h4>
                <p>Pre-configured templates for different room types and use cases</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="feature-card">
                <h4>🎨 3D Visualization</h4>
                <p>Interactive 3D room layouts with equipment placement</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="feature-card">
                <h4>💰 Cost Analysis</h4>
                <p>Detailed cost breakdowns with alternative options</p>
            </div>
            ''', unsafe_allow_html=True)


# CSS Styling
st.markdown('''
<style>
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 28px;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .comparison-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
