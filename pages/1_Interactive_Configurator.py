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

# Updated & Centralized Custom CSS with high-contrast colors for readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #4A90E2; /* Professional Blue */
        --success-color: #50C878; /* Emerald Green */
        --background-color: #F4F7F9; /* Light Gray */
        --dark-background: #2E3A4B;
        --text-color: #333745; /* Charcoal */
        --card-shadow: 0 10px 30px rgba(0,0,0,0.07);
        --border-radius-lg: 20px;
        --border-radius-md: 12px;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .main > div {
        background: white;
        border-radius: var(--border-radius-lg);
        padding: 25px;
        margin: 15px;
        box-shadow: var(--card-shadow);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(90deg, var(--dark-background) 0%, #4A5568 100%);
        padding: 10px;
        border-radius: var(--border-radius-md);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        font-weight: 500;
        padding: 10px 18px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
    }
    
    /* Updated for better readability */
    .premium-card {
        background-color: var(--dark-background);
        padding: 25px;
        border-radius: var(--border-radius-lg);
        color: white;
        margin: 15px 0;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    /* Updated for better readability */
    .metric-card {
        background-color: var(--primary-color);
        padding: 20px;
        border-radius: var(--border-radius-md);
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.3);
        color: white !important;
        text-align: center;
        margin: 10px 0;
    }

    .metric-card h3 {
        color: white !important;
        margin: 0;
        font-size: 28px;
    }

    .metric-card p {
        color: white !important;
        margin: 5px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: var(--border-radius-md);
        margin: 10px 0;
        border-left: 5px solid var(--primary-color);
        box-shadow: var(--card-shadow);
        color: var(--text-color);
    }
    
    .comparison-card {
        background: white;
        padding: 20px;
        border-radius: var(--border-radius-md);
        margin: 10px 0;
        border: 1px solid #e1e5e9;
        transition: all 0.3s ease;
    }
    
    .comparison-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 10px 30px rgba(74, 144, 226, 0.1);
    }
    
    /* Updated for better readability */
    .alert-success {
        background-color: var(--success-color);
        color: white;
        padding: 15px;
        border-radius: var(--border-radius-md);
        margin: 10px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #50E3C2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--dark-background);
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
        
        if room_area < 20 and specs['capacity'] <= 8:
            size_category = 'small'
        elif room_area < 60 and specs['capacity'] <= 20:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        products = self.db.products['displays'][tier]
        
        if brands:
            products = {k: v for k, v in products.items() if v['brand'] in brands}
        
        if products:
            selected = list(products.items())[0]
            for name, product in products.items():
                if size_category == 'large' and '100"' in name or 'Wall' in name:
                    selected = (name, product)
                    break
                elif size_category == 'medium' and any(size in name for size in ['75"', '85"', '86"']):
                    selected = (name, product)
                    break
        else:
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
        if complexity_score > 3:
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
        
        accessories.append({'category': 'Cable Management', 'item': 'Under-table Cable Tray System', 'model': 'FSR FL-500P Series', 'price': 1299, 'necessity': 'Essential'})
        if 'Wireless Presentation' in features:
            accessories.append({'category': 'Wireless Presentation', 'item': 'Professional Wireless System', 'model': 'Barco ClickShare Conference CX-50', 'price': 2999, 'necessity': 'Required'})
        if 'Room Scheduling' in features:
            accessories.append({'category': 'Room Booking', 'item': 'Smart Room Panel', 'model': 'Crestron TSS-1070-B-S', 'price': 1899, 'necessity': 'Required'})
        if 'Digital Whiteboard' in features:
            accessories.append({'category': 'Interactive Display', 'item': 'Digital Whiteboard Solution', 'model': 'Microsoft Surface Hub 2S 85"', 'price': 21999, 'necessity': 'Optional'})
        if specs['capacity'] > 16:
            accessories.append({'category': 'Power Management', 'item': 'Intelligent Power Distribution', 'model': 'Middle Atlantic UPS-2200R', 'price': 1599, 'necessity': 'Recommended'})
        
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
        aspect_ratio = max(specs['length'], specs['width']) / min(specs['length'], specs['width'])
        
        return {
            'size_category': self._categorize_room_size(room_area),
            'shape_analysis': self._analyze_room_shape(aspect_ratio),
            'acoustic_properties': self._estimate_acoustic_properties(specs),
            'lighting_challenges': self._identify_lighting_challenges(specs),
            'capacity_efficiency': self._analyze_capacity_efficiency(specs)
        }
    
    def _suggest_upgrade_path(self, specs, current_tier):
        tiers = ['Budget', 'Professional', 'Premium']
        current_index = tiers.index(current_tier)
        upgrade_path = []
        
        if current_index < len(tiers) - 1:
            next_tier = tiers[current_index + 1]
            upgrade_path.append({
                'phase': 'Short-term (6-12 months)', 'tier': next_tier, 'focus': 'Core AV upgrade',
                'estimated_cost': self._estimate_tier_cost(specs, next_tier) - self._estimate_tier_cost(specs, current_tier)
            })
        
        if current_index < len(tiers) - 2:
            ultimate_tier = tiers[-1]
            upgrade_path.append({
                'phase': 'Long-term (2-3 years)', 'tier': ultimate_tier, 'focus': 'Premium features & AI integration',
                'estimated_cost': self._estimate_tier_cost(specs, ultimate_tier) - self._estimate_tier_cost(specs, current_tier)
            })
        
        return upgrade_path
    
    def _calculate_brightness_needs(self, specs):
        window_factor = specs.get('windows', 0) / 100
        return int(300 + (window_factor * 200))
    
    def _suggest_camera_mounting(self, specs):
        return "Ceiling mount recommended for optimal coverage" if specs['ceiling_height'] > 3.5 else "Wall mount at display location"
    
    def _analyze_camera_coverage(self, specs, camera_type):
        room_area = specs['length'] * specs['width']
        coverage_factor = 80 if camera_type == 'multi_camera' else 50
        coverage = min(100, (room_area / coverage_factor) * 100)
        return f"{coverage:.0f}% optimal coverage"

    def _design_audio_config(self, specs, config_type):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        return {
            'type': config_type, 'coverage': f"{min(100, room_volume / 2):.0f}%",
            'microphone_count': max(2, specs['capacity'] // 4), 'speaker_zones': max(1, specs['capacity'] // 8),
            'processing_power': 'High' if room_volume > 150 else 'Standard'
        }
    
    def _analyze_acoustics(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        surface_area = 2 * (specs['length'] * specs['width'] + specs['length'] * specs['ceiling_height'] + specs['width'] * specs['ceiling_height'])
        rt60_estimate = 0.16 * room_volume / (0.3 * surface_area)
        return {
            'rt60_estimate': f"{rt60_estimate:.2f} seconds", 'acoustic_treatment_needed': rt60_estimate > 0.8,
            'sound_masking_recommended': specs['capacity'] > 20, 'echo_risk': 'High' if max(specs['length'], specs['width']) > 8 else 'Low'
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
        return {
            'natural_light_factor': f"{window_factor * 100:.0f}%", 'artificial_light_zones': max(1, (specs['length'] * specs['width']) // 20),
            'dimming_required': True, 'color_temperature_control': 'Circadian Lighting' in features, 'daylight_harvesting': window_factor > 0.2
        }
    
    def _calculate_advanced_confidence(self, specs, preferences):
        base_confidence = 0.85
        if len(specs.get('special_requirements', [])) > 3: base_confidence -= 0.1
        if preferences.get('preferred_brands'): base_confidence += 0.05
        if preferences.get('budget_tier') == 'Premium': base_confidence += 0.1
        return min(0.99, max(0.70, base_confidence))
    
    def _categorize_room_size(self, area):
        if area < 20: return "Small (Huddle)"
        elif area < 50: return "Medium (Conference)"
        elif area < 100: return "Large (Training)"
        else: return "Extra Large (Auditorium)"
    
    def _analyze_room_shape(self, aspect_ratio):
        if aspect_ratio < 1.3: return "Square - Good for collaboration"
        elif aspect_ratio < 2.0: return "Rectangular - Versatile layout"
        else: return "Long/Narrow - Challenging for AV"
    
    def _estimate_acoustic_properties(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        return {
            'reverb_category': 'High' if room_volume > 200 else 'Moderate' if room_volume > 80 else 'Low',
            'treatment_needed': room_volume > 150, 'echo_potential': max(specs['length'], specs['width']) > 10
        }
    
    def _identify_lighting_challenges(self, specs):
        challenges = []
        if specs.get('windows', 0) > 30: challenges.append("High natural light - glare control needed")
        if specs['ceiling_height'] > 4: challenges.append("High ceiling - requires powerful fixtures")
        if specs['capacity'] > 20: challenges.append("Large audience - uniform lighting critical")
        return challenges if challenges else ["Standard lighting requirements"]
    
    def _analyze_capacity_efficiency(self, specs):
        efficiency = specs['capacity'] / (specs['length'] * specs['width'])
        if efficiency > 0.5: return "High density - space optimization good"
        elif efficiency > 0.3: return "Moderate density - balanced layout"
        else: return "Low density - spacious environment"
    
    def _estimate_tier_cost(self, specs, tier):
        base_costs = {'Budget': 15000, 'Professional': 45000, 'Premium': 120000}
        return int(base_costs[tier] * (1 + (specs['capacity'] / 50)))

# --- Simplified 3D Room Visualization Engine ---
class EnhancedVisualizationEngine:
    @staticmethod
    def create_3d_room_visualization(room_specs, recommendations):
        fig = go.Figure()
        
        # Room dimensions
        length, width, height = room_specs['length'], room_specs['width'], room_specs['ceiling_height']
        
        # Create room wireframe outline
        # Floor outline
        floor_x = [0, length, length, 0, 0]
        floor_y = [0, 0, width, width, 0]
        floor_z = [0, 0, 0, 0, 0]
        
        fig.add_trace(go.Scatter3d(
            x=floor_x, y=floor_y, z=floor_z,
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.8)', width=4),
            name='Room Floor',
            showlegend=False
        ))
        
        # Ceiling outline
        ceiling_x = [0, length, length, 0, 0]
        ceiling_y = [0, 0, width, width, 0]
        ceiling_z = [height, height, height, height, height]
        
        fig.add_trace(go.Scatter3d(
            x=ceiling_x, y=ceiling_y, z=ceiling_z,
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.6)', width=3),
            name='Room Ceiling',
            showlegend=False
        ))
        
        # Vertical edges
        for corner_x, corner_y in [(0, 0), (length, 0), (length, width), (0, width)]:
            fig.add_trace(go.Scatter3d(
                x=[corner_x, corner_x], y=[corner_y, corner_y], z=[0, height],
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.6)', width=3),
                showlegend=False
            ))
        
        # Display Screen (as a rectangle)
        screen_width = min(3, width * 0.6)
        screen_height = screen_width * 9/16
        screen_y_center = width / 2
        screen_z_center = height * 0.4
        
        # Screen corners
        screen_corners_x = [0.05, 0.05, 0.05, 0.05, 0.05]
        screen_corners_y = [
            screen_y_center - screen_width/2,
            screen_y_center + screen_width/2,
            screen_y_center + screen_width/2,
            screen_y_center - screen_width/2,
            screen_y_center - screen_width/2
        ]
        screen_corners_z = [
            screen_z_center - screen_height/2,
            screen_z_center - screen_height/2,
            screen_z_center + screen_height/2,
            screen_z_center + screen_height/2,
            screen_z_center - screen_height/2
        ]
        
        fig.add_trace(go.Scatter3d(
            x=screen_corners_x, y=screen_corners_y, z=screen_corners_z,
            mode='lines+markers',
            line=dict(color='black', width=8),
            marker=dict(size=6, color='black'),
            name='Display Screen'
        ))
        
        # Conference Table (as a rectangle)
        table_length = min(length * 0.7, 4)
        table_width = min(width * 0.4, 1.5)
        table_height = 0.75
        table_x_center = length * 0.6
        table_y_center = width * 0.5
        
        # Table outline
        table_x = [
            table_x_center - table_length/2,
            table_x_center + table_length/2,
            table_x_center + table_length/2,
            table_x_center - table_length/2,
            table_x_center - table_length/2
        ]
        table_y = [
            table_y_center - table_width/2,
            table_y_center - table_width/2,
            table_y_center + table_width/2,
            table_y_center + table_width/2,
            table_y_center - table_width/2
        ]
        table_z = [table_height] * 5
        
        fig.add_trace(go.Scatter3d(
            x=table_x, y=table_y, z=table_z,
            mode='lines+markers',
            line=dict(color='rgb(139, 115, 85)', width=6),
            marker=dict(size=4, color='rgb(139, 115, 85)'),
            name='Conference Table'
        ))
        
        # Camera System (sleek bar above screen)
        camera_width = screen_width * 0.8
        camera_y_start = screen_y_center - camera_width/2
        camera_y_end = screen_y_center + camera_width/2
        camera_z = screen_z_center + screen_height/2 + 0.2
        camera_x = 0.08
        
        fig.add_trace(go.Scatter3d(
            x=[camera_x, camera_x],
            y=[camera_y_start, camera_y_end],
            z=[camera_z, camera_z],
            mode='lines+markers',
            line=dict(color='rgb(60, 60, 60)', width=10),
            marker=dict(size=8, color='rgb(60, 60, 60)', symbol='square'),
            name='Camera System'
        ))
        
        # Seating arrangement
        capacity = min(room_specs['capacity'], 12)
        chairs_per_side = min(6, capacity // 2)
        
        chair_x_positions = []
        chair_y_positions = []
        chair_z_positions = []
        
        # Chairs along table
        for i in range(chairs_per_side):
            chair_x = table_x_center - table_length/2 + (i + 1) * table_length / (chairs_per_side + 1)
            
            # Left side
            chair_x_positions.append(chair_x)
            chair_y_positions.append(table_y_center - table_width/2 - 0.4)
            chair_z_positions.append(0.85)
            
            # Right side
            if len(chair_x_positions) < capacity:
                chair_x_positions.append(chair_x)
                chair_y_positions.append(table_y_center + table_width/2 + 0.4)
                chair_z_positions.append(0.85)
        
        fig.add_trace(go.Scatter3d(
            x=chair_x_positions,
            y=chair_y_positions,
            z=chair_z_positions,
            mode='markers',
            marker=dict(
                size=10,
                color='rgb(70, 130, 180)',
                symbol='square'
            ),
            name=f'Seating ({len(chair_x_positions)} chairs)'
        ))
        
        # Ceiling Speakers
        speaker_positions = [
            (length * 0.25, width * 0.25, height - 0.1),
            (length * 0.75, width * 0.25, height - 0.1),
            (length * 0.25, width * 0.75, height - 0.1),
            (length * 0.75, width * 0.75, height - 0.1)
        ]
        
        speaker_x = [pos[0] for pos in speaker_positions]
        speaker_y = [pos[1] for pos in speaker_positions]
        speaker_z = [pos[2] for pos in speaker_positions]
        
        fig.add_trace(go.Scatter3d(
            x=speaker_x,
            y=speaker_y,
            z=speaker_z,
            mode='markers',
            marker=dict(
                size=8,
                color='rgb(220, 220, 220)',
                symbol='circle',
                line=dict(color='rgb(100, 100, 100)', width=1)
            ),
            name='Ceiling Speakers'
        ))
        
        # LED Lighting
        light_positions = []
        for i in range(2):
            for j in range(3):
                light_positions.append((
                    length * (i + 1) / 3,
                    width * (j + 1) / 4,
                    height - 0.05
                ))
        
        light_x = [pos[0] for pos in light_positions]
        light_y = [pos[1] for pos in light_positions]
        light_z = [pos[2] for pos in light_positions]
        
        fig.add_trace(go.Scatter3d(
            x=light_x,
            y=light_y,
            z=light_z,
            mode='markers',
            marker=dict(
                size=6,
                color='rgb(255, 255, 150)',
                symbol='circle'
            ),
            name='LED Lighting'
        ))
        
        # Touch Control Panel
        fig.add_trace(go.Scatter3d(
            x=[length - 0.1],
            y=[width * 0.85],
            z=[height * 0.35],
            mode='markers',
            marker=dict(
                size=12,
                color='rgb(240, 240, 240)',
                symbol='square',
                line=dict(color='rgb(150, 150, 150)', width=2)
            ),
            name='Control Panel'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="3D Conference Room Layout",
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            scene=dict(
                xaxis=dict(
                    title=f"Length ({length:.1f}m)",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    range=[0, length + 0.5]
                ),
                yaxis=dict(
                    title=f"Width ({width:.1f}m)",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    range=[0, width + 0.5]
                ),
                zaxis=dict(
                    title=f"Height ({height:.1f}m)",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    range=[0, height + 0.2]
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.8, z=1.2),
                    center=dict(x=0, y=0, z=0.3)
                ),
                bgcolor='rgb(250, 250, 250)',
                aspectmode='cube'
            ),
            height=600,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.8,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1
            ),
            margin=dict(l=0, r=120, t=50, b=0),
            paper_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_equipment_layout_2d(room_specs, recommendations):
        """Create a clean 2D floor plan view"""
        fig = go.Figure()
        
        length, width = room_specs['length'], room_specs['width']
        
        # Room outline
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=length, y1=width,
            line=dict(color="rgb(100, 100, 100)", width=3),
            fillcolor="rgba(245, 245, 245, 0.3)"
        )
        
        # Display wall
        fig.add_shape(
            type="line",
            x0=0, y0=width*0.2, x1=0, y1=width*0.8,
            line=dict(color="red", width=8)
        )
        
        # Conference table
        table_length = min(length * 0.7, 4)
        table_width = min(width * 0.4, 1.5)
        table_x_center = length * 0.6
        table_y_center = width * 0.5
        
        fig.add_shape(
            type="rect",
            x0=table_x_center - table_length/2,
            y0=table_y_center - table_width/2,
            x1=table_x_center + table_length/2,
            y1=table_y_center + table_width/2,
            line=dict(color="brown", width=2),
            fillcolor="rgba(139, 115, 85, 0.4)"
        )
        
        # Add text annotations for equipment
        fig.add_annotation(
            x=0.1, y=width*0.5,
            text="Display<br>System",
            showarrow=True,
            arrowcolor="red",
            bgcolor="white",
            bordercolor="red"
        )
        
        fig.add_annotation(
            x=length-0.3, y=width*0.85,
            text="Control<br>Panel",
            showarrow=True,
            arrowcolor="gray",
            bgcolor="white",
            bordercolor="gray"
        )
        
        fig.update_layout(
            title="Equipment Layout - Floor Plan View",
            xaxis=dict(
                title=f"Length ({length:.1f}m)", 
                scaleanchor="y", 
                scaleratio=1,
                range=[0, length + 0.5]
            ),
            yaxis=dict(
                title=f"Width ({width:.1f}m)",
                range=[0, width + 0.5]
            ),
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
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
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=costs,
                marker_color=colors,
                text=[f"${cost:,.0f}" for cost in costs],
                textposition='auto',
                textfont=dict(color='white', size=12)
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="Investment Breakdown by Category",
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title="Equipment Category",
            yaxis_title="Investment (USD)",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(tickformat='$,.0f')
        )
        
        return fig
    
    @staticmethod
    def create_feature_comparison_radar(recommendations, alternatives):
        categories = ['Performance', 'Features', 'Reliability', 'Integration', 'Value']
        current_scores = [4.5, 4.2, 4.6, 4.4, 4.3]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=current_scores + [current_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Recommended Solution',
            line_color='#3498db',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                )
            ),
            showlegend=True,
            title=dict(
                text="Solution Performance Analysis",
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            height=400,
            paper_bgcolor='white'
        )
        
        return fig

# --- Main Application ---
def main():
    st.title("🏢 AI Room Configurator Pro Max")
    st.markdown("### Transform Your Space with Intelligent AV Design")
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'room_specs' not in st.session_state:
        st.session_state.room_specs = None
    
    with st.sidebar:
        st.markdown('<div class="premium-card" style="margin-top: -50px;"><h2>🎛️ Room Configuration</h2></div>', unsafe_allow_html=True)
        
        template = st.selectbox("Room Template", list(EnhancedProductDatabase().room_templates.keys()), help="Choose a template to start.")
        template_info = EnhancedProductDatabase().room_templates[template]
        
        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        length = col1.slider("Length (m)", 2.0, 20.0, float(template_info['typical_size'][0]), 0.5)
        width = col2.slider("Width (m)", 2.0, 20.0, float(template_info['typical_size'][1]), 0.5)
        ceiling_height = st.slider("Ceiling Height (m)", 2.4, 6.0, 3.0, 0.1)
        capacity = st.slider("Capacity", 2, 100, template_info['capacity_range'][1])
        
        st.subheader("🌟 Environment")
        windows = st.slider("Windows (%)", 0, 80, 20, 5, help="Percentage of wall space with windows.")
        
        st.subheader("💰 Budget & Brands")
        budget_tier = st.selectbox("Budget Tier", ['Budget', 'Professional', 'Premium'], index=1)
        preferred_brands = st.multiselect("Preferred Brands", ['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'], help="Leave empty for best overall recommendations")
        
        st.subheader("✨ Special Features")
        special_features = st.multiselect("Required Features", ['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 'Noise Reduction', 'Circadian Lighting', 'AI Analytics'])
        
        if st.button("🚀 Generate AI Recommendation"):
            room_specs = {
                'template': template, 'length': length, 'width': width, 'ceiling_height': ceiling_height,
                'capacity': capacity, 'windows': windows, 'special_requirements': [] 
            }
            user_preferences = {
                'budget_tier': budget_tier, 'preferred_brands': preferred_brands, 'special_features': special_features
            }
            recommender = MaximizedAVRecommender()
            st.session_state.recommendations = recommender.get_comprehensive_recommendations(room_specs, user_preferences)
            st.session_state.room_specs = room_specs
            st.session_state.budget_tier = budget_tier # Save budget tier for report tab
            st.success("✅ AI Analysis Complete!")
    
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        room_specs = st.session_state.room_specs
        
        total_cost = sum(recommendations[cat]['price'] for cat in ['display', 'camera', 'audio', 'control', 'lighting'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="metric-card"><h3>${total_cost:,.0f}</h3><p>Total Investment</p></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><h3>{recommendations["confidence_score"]:.0%}</h3><p>AI Confidence</p></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><h3>{room_specs["length"]}m × {room_specs["width"]}m</h3><p>Room Size</p></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><h3>{room_specs["capacity"]}</h3><p>Capacity</p></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📊 Analysis", "🎨 Visualization", "🔄 Alternatives", "📋 Report"])
        
        with tab1:
            st.subheader("AI-Powered Equipment Recommendations")
            col1, col2 = st.columns(2)
            with col1:
                for cat, icon in [('display', '📺'), ('camera', '🎥'), ('audio', '🔊')]:
                    rec = recommendations[cat]
                    st.markdown(f"#### {icon} {cat.title()} System")
                    st.markdown(f"""<div class="feature-card">
                        <h4>{rec['model']}</h4>
                        <p><strong>Price:</strong> ${rec['price']:,} | <strong>Rating:</strong> ⭐ {rec['rating']}/5.0</p>
                        <p><strong>Specs:</strong> {rec['specs']}</p>
                    </div>""", unsafe_allow_html=True)
            with col2:
                for cat, icon in [('control', '🎛️'), ('lighting', '💡')]:
                    rec = recommendations[cat]
                    st.markdown(f"#### {icon} {cat.title()} System")
                    st.markdown(f"""<div class="feature-card">
                        <h4>{rec['model']}</h4>
                        <p><strong>Price:</strong> ${rec['price']:,} | <strong>Rating:</strong> ⭐ {rec['rating']}/5.0</p>
                        <p><strong>Specs:</strong> {rec['specs']}</p>
                    </div>""", unsafe_allow_html=True)
                if recommendations['accessories']:
                    st.markdown("#### 🔧 Essential Accessories")
                    for acc in recommendations['accessories'][:2]:
                        st.markdown(f"<div class='feature-card'><strong>{acc['item']}</strong> ({acc['model']})<br>Price: ${acc['price']:,} ({acc['necessity']})</div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("Room Analysis & Performance Metrics")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Room Characteristics")
                analysis = recommendations['room_analysis']
                st.markdown(f"""<div class="comparison-card" style="color: black;">
                    <p><strong>Category:</strong> {analysis['size_category']}</p>
                    <p><strong>Shape:</strong> {analysis['shape_analysis']}</p>
                    <p><strong>Acoustics:</strong> Reverb is {analysis['acoustic_properties']['reverb_category']}, Treatment needed: {'Yes' if analysis['acoustic_properties']['treatment_needed'] else 'No'}</p>
                    <p><strong>Lighting Challenges:</strong> {', '.join(analysis['lighting_challenges'])}</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("#### Investment & Performance")
                st.plotly_chart(EnhancedVisualizationEngine.create_cost_breakdown_chart(recommendations), use_container_width=True)
                st.plotly_chart(EnhancedVisualizationEngine.create_feature_comparison_radar(recommendations, recommendations.get('alternatives', {})), use_container_width=True)

        with tab3:
            st.subheader("Interactive Room Visualization")
            st.plotly_chart(EnhancedVisualizationEngine.create_3d_room_visualization(room_specs, recommendations), use_container_width=True)
            st.plotly_chart(EnhancedVisualizationEngine.create_equipment_layout_2d(room_specs, recommendations), use_container_width=True)

        with tab4:
            st.subheader("Alternative Configurations & Upgrade Path")
            col1, col2 = st.columns(2)
            with col1:
                if recommendations.get('alternatives'):
                    for tier_name, alt_config in recommendations['alternatives'].items():
                        st.markdown(f"#### {tier_name} Alternative")
                        for cat in ['displays', 'cameras', 'audio']:
                            if cat in alt_config:
                                name, info = alt_config[cat]
                                st.markdown(f"""<div class="comparison-card" style="color: black;">
                                    <strong>{cat.title()}:</strong> {name}<br>
                                    ${info['price']:,} | ⭐ {info['rating']}/5.0
                                </div>""", unsafe_allow_html=True)
            with col2:
                 if recommendations.get('upgrade_path'):
                    st.markdown("#### 🚀 Upgrade Roadmap")
                    for upgrade in recommendations['upgrade_path']:
                        st.markdown(f"""<div class="alert-success">
                            <h5>{upgrade['phase']} to {upgrade['tier']}</h5>
                            <p><strong>Focus:</strong> {upgrade['focus']}</p>
                            <p><strong>Add. Investment:</strong> ${upgrade['estimated_cost']:,}</p>
                        </div>""", unsafe_allow_html=True)
        
        with tab5:
            st.subheader("Professional Report Summary")
            st.markdown(f"""<div class="premium-card">
                <h3>Executive Summary</h3>
                <p>AI-generated AV solution for a <strong>{room_specs['template']}</strong> ({room_specs['length']}m × {room_specs['width']}m) for <strong>{room_specs['capacity']} people</strong>.</p>
                <p><strong>Total Investment:</strong> ${total_cost:,} | <strong>Confidence:</strong> {recommendations['confidence_score']:.0%} | <strong>Recommended Tier:</strong> {st.session_state.budget_tier}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("#### Detailed Equipment Specifications")
            specs_data = [{'Category': cat.title(), 'Model': recommendations[cat]['model'], 'Price': f"${recommendations[cat]['price']:,}", 'Rating': f"{recommendations[cat]['rating']}/5.0", 'Brand': recommendations[cat]['brand']} for cat in ['display', 'camera', 'audio', 'control', 'lighting']]
            st.dataframe(pd.DataFrame(specs_data), use_container_width=True)

    else:
        st.markdown('''<div class="premium-card" style="text-align: center; padding: 50px;">
            <h2>🚀 Welcome to AI Room Configurator Pro Max</h2>
            <p style="font-size: 18px;">Configure your room in the sidebar to generate an intelligent AV design.</p>
        </div>''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
