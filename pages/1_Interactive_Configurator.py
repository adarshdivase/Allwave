import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any
import colorsys

st.set_page_config(page_title="AI Room Configurator Pro Max", page_icon="🏢", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --primary-color: #2563eb;
        --success-color: #22c55e;
        --background-color: #f6f8fa;
        --dark-background: #1e293b;
        --sidebar-bg: #1e293b;
        --sidebar-text: #f3f4f6;
        --text-color: #1e293b;
        --body-text-color: #334155;
        --card-shadow: 0 6px 24px rgba(30,41,59,0.08);
        --border-radius-lg: 18px;
        --border-radius-md: 12px;
        --accent-color: #e0e7ef;
        --feature-bg: #fff;
        --feature-border: #2563eb;
        --metric-bg: #2563eb;
        --metric-text: #fff;
    }
    body, .stApp, .main, .main > div {
        color: var(--text-color) !important;
        background-color: var(--background-color) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .main > div {
        background: var(--feature-bg) !important;
        border-radius: var(--border-radius-lg);
        padding: 28px;
        margin: 18px;
        box-shadow: var(--card-shadow);
        color: var(--text-color) !important;
    }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: var(--text-color) !important;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .main p, .main li, .main div, .main span, .main label {
        color: var(--body-text-color) !important;
        font-size: 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
        background: linear-gradient(90deg, var(--dark-background) 0%, #334155 100%);
        padding: 12px;
        border-radius: var(--border-radius-md);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        color: #fff !important;
        font-weight: 600;
        padding: 12px 22px;
        transition: all 0.2s ease;
        font-size: 16px;
        letter-spacing: 0.01em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.18);
        transform: translateY(-2px);
        color: #fff !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: #fff !important;
        box-shadow: 0 4px 18px rgba(37,99,235,0.18);
    }
    .premium-card {
        background: var(--dark-background) !important;
        padding: 28px;
        border-radius: var(--border-radius-lg);
        color: #fff !important;
        margin: 18px 0;
        box-shadow: 0 12px 32px rgba(30,41,59,0.18);
        border: 1px solid #334155;
    }
    .metric-card {
        background: var(--metric-bg) !important;
        padding: 22px;
        border-radius: var(--border-radius-md);
        box-shadow: 0 8px 25px rgba(37,99,235,0.13);
        color: var(--metric-text) !important;
        text-align: center;
        margin: 12px 0;
        border: none;
    }
    .metric-card h3 {
        color: var(--metric-text) !important;
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .metric-card p {
        color: var(--metric-text) !important;
        margin: 7px 0 0 0;
        font-size: 15px;
        opacity: 0.92;
        font-weight: 600;
    }
    .feature-card {
        background: var(--feature-bg) !important;
        padding: 22px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        border-left: 6px solid var(--feature-border);
        box-shadow: 0 2px 8px rgba(37,99,235,0.05);
        color: var(--text-color) !important;
    }
    .comparison-card {
        background: var(--feature-bg);
        padding: 22px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        border: 1px solid var(--accent-color);
        color: var(--text-color);
        transition: all 0.2s ease;
    }
    .comparison-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 8px 24px rgba(37,99,235,0.09);
    }
    .alert-success {
        background: var(--success-color);
        color: #fff !important;
        padding: 16px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(34,197,94,0.12);
    }
    .alert-success h5, .alert-success p {
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #38bdf8 100%);
        color: #fff;
        border: none;
        padding: 14px 28px;
        border-radius: 28px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.2s ease;
        width: 100%;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(37,99,235,0.09);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 32px rgba(37,99,235,0.18);
        background: linear-gradient(135deg, #1e40af 0%, #38bdf8 100%);
    }
    .css-1d391kg {
        background: var(--sidebar-bg);
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
            'lighting': self._recommend_lighting_advanced(room_specs, budget_tier, user_preferences, special_features),
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
        acoustic_features = specs.get('environment', {}).get('acoustic_features', [])
        
        config_type = 'table_system' # Default
        if 'Sound Absorption Needed' in acoustic_features or 'Echo Control Required' in acoustic_features:
            config_type = 'premium_processing'
        elif room_volume > 200 or specs.get('environment', {}).get('ceiling_type') == "Open Plenum":
            config_type = 'distributed'
        elif specs['ceiling_height'] > 3.5:
            config_type = 'ceiling_array'
        
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
        complexity_score += len(specs.get('environment', {}).get('env_controls', []))

        selected = list(products.items())[0]
        if complexity_score > 4:
            for name, product in products.items():
                if 'NVX' in name or 'DGX' in name or 'Core' in name:
                    selected = (name, product)
                    break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['integration_options'] = self._suggest_integrations(specs)
        return result
    
    def _recommend_lighting_advanced(self, specs, tier, user_prefs, features):
        products = self.db.products['lighting'][tier]
        env_config = specs.get('environment', {})
        
        needs_daylight_sync = 'Circadian Lighting' in features or env_config.get('natural_light') in ['High', 'Very High']
        
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
        accessibility_needs = specs.get('environment', {}).get('accessibility', [])
        
        accessories.append({'category': 'Cable Management', 'item': 'Under-table Cable Tray System', 'model': 'FSR FL-500P Series', 'price': 1299, 'necessity': 'Essential'})
        if 'Wireless Presentation' in features:
            accessories.append({'category': 'Wireless Presentation', 'item': 'Professional Wireless System', 'model': 'Barco ClickShare Conference CX-50', 'price': 2999, 'necessity': 'Required'})
        if 'Room Scheduling' in features:
            accessories.append({'category': 'Room Booking', 'item': 'Smart Room Panel', 'model': 'Crestron TSS-1070-B-S', 'price': 1899, 'necessity': 'Required'})
        if 'Hearing Loop System' in accessibility_needs:
             accessories.append({'category': 'Accessibility', 'item': 'Inductive Loop System', 'model': 'Williams AV PLR BP1', 'price': 1500, 'necessity': 'Required'})

        if specs['capacity'] > 16:
            accessories.append({'category': 'Power Management', 'item': 'Intelligent Power Distribution', 'model': 'Middle Atlantic UPS-2200R', 'price': 1599, 'necessity': 'Recommended'})
        
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

    def _generate_smart_upgrade_plan(self, specs, current_tier, estimated_cost):
        """Generate a detailed phased upgrade plan"""
        phases = {
            'Immediate (0-3 months)': {
                'priorities': [
                    'Essential software upgrades and licensing',
                    'Control system programming optimization',
                    'Staff training on current systems'
                ],
                'cost_percentage': 0.15,
                'focus': 'Maximizing current infrastructure'
            },
            'Phase 1 (3-6 months)': {
                'priorities': [
                    'Display system upgrade',
                    'Camera system enhancement',
                    'Basic audio improvements'
                ],
                'cost_percentage': 0.35,
                'focus': 'Core AV capabilities'
            },
            'Phase 2 (6-9 months)': {
                'priorities': [
                    'Advanced audio processing implementation',
                    'Lighting control system upgrade',
                    'Room automation integration'
                ],
                'cost_percentage': 0.30,
                'focus': 'Enhanced functionality'
            },
            'Final Phase (9-12 months)': {
                'priorities': [
                    'Premium features activation',
                    'AI analytics integration',
                    'Complete system optimization'
                ],
                'cost_percentage': 0.20,
                'focus': 'Premium capabilities'
            }
        }

        # Calculate phase-wise budgets
        for phase in phases.values():
            phase['budget'] = estimated_cost * phase['cost_percentage']
            
        # Add ROI metrics
        roi_metrics = {
            'Productivity Gain': '15-20%',
            'Energy Savings': '10-15%',
            'Maintenance Cost Reduction': '25-30%',
            'System Downtime Reduction': '40-50%'
        }

        return {
            'phases': phases,
            'roi_metrics': roi_metrics,
            'total_investment': estimated_cost,
            'monthly_investment': estimated_cost / 12
        }
    
    def _calculate_brightness_needs(self, specs):
        env = specs.get('environment', {})
        light_levels = {"Very Low": -50, "Low": -25, "Moderate": 0, "High": 50, "Very High": 100}
        natural_light_adjust = light_levels.get(env.get('natural_light', 'Moderate'), 0)
        return int(350 + natural_light_adjust)
    
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
        env = specs.get('environment', {})
        wall_absorb = {"Drywall": 0.1, "Glass": 0.03, "Concrete": 0.02, "Wood Panels": 0.15, "Acoustic Panels": 0.8}
        absorption_coeff = wall_absorb.get(env.get('wall_material', 'Drywall'), 0.1)
        
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        surface_area = 2 * (specs['length'] * specs['width'] + specs['length'] * specs['ceiling_height'] + specs['width'] * specs['ceiling_height'])
        rt60_estimate = 0.161 * room_volume / (absorption_coeff * surface_area)
        return {
            'rt60_estimate': f"{rt60_estimate:.2f} seconds", 'acoustic_treatment_needed': rt60_estimate > 0.8,
            'sound_masking_recommended': 'Speech Privacy Important' in env.get('acoustic_features', []),
            'echo_risk': 'High' if env.get('wall_material') in ['Glass', 'Concrete'] else 'Low'
        }
    
    def _suggest_integrations(self, specs):
        integrations = ['Microsoft Teams', 'Zoom', 'Google Meet']
        if specs['capacity'] > 16:
            integrations.extend(['Cisco Webex', 'BlueJeans'])
        if 'VTC' in specs.get('special_requirements', []):
            integrations.append('Polycom RealPresence')
        return integrations
    
    def _analyze_lighting_needs(self, specs, features):
        env = specs.get('environment', {})
        natural_light = env.get('natural_light', 'Moderate')
        return {
            'natural_light_factor': f"{natural_light}",
            'artificial_light_zones': max(1, (specs['length'] * specs['width']) // 20),
            'dimming_required': True,
            'color_temperature_control': 'Circadian Lighting' in features or env.get('color_scheme') != 'Neutral',
            'daylight_harvesting': 'Daylight Harvesting' in env.get('env_controls', [])
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
        if specs.get('environment', {}).get('natural_light') in ['High', 'Very High']: challenges.append("High natural light - glare control needed")
        if specs['ceiling_height'] > 4: challenges.append("High ceiling - requires powerful fixtures")
        if 'Presentations' in specs.get('environment', {}).get('room_purpose', []): challenges.append("Presentation mode requires zoned lighting")
        return challenges if challenges else ["Standard lighting requirements"]
    
    def _analyze_capacity_efficiency(self, specs):
        efficiency = specs['capacity'] / (specs['length'] * specs['width'])
        if efficiency > 0.5: return "High density - space optimization good"
        elif efficiency > 0.3: return "Moderate density - balanced layout"
        else: return "Low density - spacious environment"
    
    def _estimate_tier_cost(self, specs, tier):
        base_costs = {'Budget': 15000, 'Professional': 45000, 'Premium': 120000}
        return int(base_costs[tier] * (1 + (specs['capacity'] / 50)))

# --- NEW Visualization Engine with Material/Lighting Sim ---
class EnhancedMaterials:
    """Material definitions for realistic rendering"""
    @staticmethod
    def get_material_presets():
        return {
            'wood': {'base_color': '#8B4513', 'roughness': 0.7, 'metallic': 0.0, 'normal_strength': 1.0, 'ambient_occlusion': 0.8, 'grain_scale': 0.5},
            'metal': {'base_color': '#B8B8B8', 'roughness': 0.2, 'metallic': 0.9, 'normal_strength': 0.5, 'ambient_occlusion': 0.3},
            'glass': {'base_color': '#FFFFFF', 'roughness': 0.05, 'metallic': 0.9, 'normal_strength': 0.1, 'ambient_occlusion': 0.1, 'opacity': 0.3},
            'fabric': {'base_color': '#303030', 'roughness': 0.9, 'metallic': 0.0, 'normal_strength': 0.8, 'ambient_occlusion': 0.7},
            'display': {'base_color': '#000000', 'roughness': 0.1, 'metallic': 0.5, 'normal_strength': 0.2, 'ambient_occlusion': 0.2, 'emission': 0.2}
        }

class EnhancedLighting:
    """Advanced lighting calculations"""
    def __init__(self, room_specs: Dict[str, Any]):
        self.room_specs = room_specs
        self.ambient_intensity = 0.3
        self.direct_intensity = 0.7
        self.shadow_softness = 0.5

    def calculate_lighting(self, position: np.ndarray, normal: np.ndarray) -> float:
        """Calculate lighting at a point"""
        ambient = self.ambient_intensity
        diffuse = self.calculate_diffuse(position, normal)
        specular = self.calculate_specular(position, normal)
        shadow = self.calculate_shadow(position)
        return (ambient + diffuse * shadow) * (1 + specular)

    def calculate_diffuse(self, position: np.ndarray, normal: np.ndarray) -> float:
        """Calculate diffuse lighting"""
        light_positions = self.get_light_positions()
        total_diffuse = 0.0
        for light_pos in light_positions:
            light_dir = light_pos - position
            light_dir = light_dir / np.linalg.norm(light_dir)
            diffuse = max(0, np.dot(normal, light_dir))
            distance = np.linalg.norm(light_pos - position)
            attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance)
            total_diffuse += diffuse * attenuation
        return min(1.0, total_diffuse * self.direct_intensity)

    def calculate_specular(self, position: np.ndarray, normal: np.ndarray) -> float:
        """Calculate specular highlights"""
        camera_pos = np.array([self.room_specs['length'] * 1.5, self.room_specs['width'] * 1.5, self.room_specs['ceiling_height'] * 1.2])
        view_dir = camera_pos - position
        view_dir = view_dir / np.linalg.norm(view_dir)
        total_specular = 0.0
        for light_pos in self.get_light_positions():
            light_dir = light_pos - position
            light_dir = light_dir / np.linalg.norm(light_dir)
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            spec = max(0, np.dot(view_dir, reflect_dir)) ** 32
            total_specular += spec
        return min(1.0, total_specular * 0.3)

    def calculate_shadow(self, position: np.ndarray) -> float:
        """Calculate soft shadows"""
        shadow = 1.0
        for light_pos in self.get_light_positions():
            shadow_ray = light_pos - position
            distance = np.linalg.norm(shadow_ray)
            shadow *= 1.0 - self.shadow_softness / (1.0 + distance)
        return max(0.2, shadow)

    def get_light_positions(self) -> List[np.ndarray]:
        """Get light positions in the room"""
        length, width, height = self.room_specs['length'], self.room_specs['width'], self.room_specs['ceiling_height']
        return [
            np.array([length * 0.25, width * 0.25, height - 0.1]),
            np.array([length * 0.75, width * 0.25, height - 0.1]),
            np.array([length * 0.25, width * 0.75, height - 0.1]),
            np.array([length * 0.75, width * 0.75, height - 0.1])
        ]

class TextureGenerator:
    """Generate realistic textures for materials"""
    @staticmethod
    def create_wood_texture(size: tuple, grain_scale: float = 0.5) -> np.ndarray:
        x, y = np.linspace(0, size[0] * grain_scale, size[0]), np.linspace(0, size[1] * grain_scale, size[1])
        X, Y = np.meshgrid(x, y)
        noise = np.random.normal(0, 1, size) * 0.1
        grain = np.sin(X * 10) * 0.5 + 0.5 + np.sin(X * 20 + Y * 5) * 0.2 + noise
        return np.clip(grain, 0, 1)

    @staticmethod
    def create_fabric_texture(size: tuple, scale: float = 1.0) -> np.ndarray:
        x, y = np.linspace(0, size[0] * scale, size[0]), np.linspace(0, size[1] * scale, size[1])
        X, Y = np.meshgrid(x, y)
        weave_x, weave_y = np.sin(X * 20) * 0.5 + 0.5, np.sin(Y * 20) * 0.5 + 0.5
        texture = weave_x * weave_y + np.random.normal(0, 1, size) * 0.1
        return np.clip(texture, 0, 1)

class EnhancedVisualizationEngine:
    def __init__(self):
        self.color_schemes = {
            'professional': {'wall': '#F0F2F5','floor': '#E5E9F0','accent': '#4A90E2','wood': '#8B5E3C','screen': '#1A1A1A','metal': '#B8B8B8','glass': '#FFFFFF'},
            'modern': {'wall': '#FFFFFF','floor': '#F8F9FA','accent': '#2563EB','wood': '#A0522D','screen': '#000000','metal': '#C0C0C0','glass': '#E8F0FE'},
            'classic': {'wall': '#F5F5DC','floor': '#DEB887','accent': '#800000','wood': '#8B4513','screen': '#2F4F4F','metal': '#CD853F','glass': '#F0F8FF'},
            'warm': {'wall': '#FEF3E0','floor': '#F3D1A3','accent': '#E53E3E','wood': '#BF5B32','screen': '#2D3748','metal': '#D69E2E','glass': '#FFF5EB'},
            'cool': {'wall': '#E6FFFA','floor': '#B2F5EA','accent': '#38B2AC','wood': '#718096','screen': '#1A202C','metal': '#A0AEC0','glass': '#EBF8FF'}
        }
        self.lighting_modes = {
            'day': {'ambient': 0.8, 'intensity': 1.0, 'color': 'rgb(255, 255, 224)'},
            'evening': {'ambient': 0.4, 'intensity': 0.7, 'color': 'rgb(255, 228, 181)'},
            'presentation': {'ambient': 0.3, 'intensity': 0.5, 'color': 'rgb(200, 200, 255)'},
            'video conference': {'ambient': 0.6, 'intensity': 0.8, 'color': 'rgb(220, 220, 255)'}
        }

    def create_3d_room_visualization(self, room_specs, recommendations, viz_config):
        fig = go.Figure()
        colors = self.color_schemes[viz_config['style_options']['color_scheme']]
        lighting = self.lighting_modes[viz_config['style_options']['lighting_mode']]
        
        # Add room structure
        self._add_room_structure(fig, room_specs, colors, lighting)
        
        # Add core elements based on configuration
        if viz_config['room_elements']['show_table']:
            self._add_table(fig, room_specs, colors, viz_config['style_options']['table_style'])
        
        if viz_config['room_elements']['show_chairs']:
            self._add_seating(fig, room_specs, colors, viz_config['style_options']['chair_style'])
        
        if viz_config['room_elements']['show_displays']:
            self._add_display_system(fig, room_specs, colors, recommendations)
        
        if viz_config['room_elements']['show_cameras']:
            self._add_camera_system(fig, room_specs, colors, recommendations)
        
        if viz_config['room_elements']['show_lighting']:
            self._add_lighting_system(fig, room_specs, colors, lighting)
        
        if viz_config['room_elements']['show_speakers']:
            self._add_audio_system(fig, room_specs, colors, recommendations)

        if viz_config['room_elements']['show_control']:
            self._add_control_system(fig, room_specs, colors)
        
        # Add additional elements
        if viz_config['room_elements']['show_whiteboard']:
            self._add_whiteboard(fig, room_specs, colors)
        
        if viz_config['room_elements']['show_credenza']:
            self._add_credenza(fig, room_specs, colors)
        
        # Add advanced features
        if viz_config['advanced_features']['show_measurements']:
            self._add_measurements(fig, room_specs)
        
        if viz_config['advanced_features']['show_zones']:
            self._add_coverage_zones(fig, room_specs)
        
        if viz_config['advanced_features']['show_cable_paths']:
            self._add_cable_management(fig, room_specs, colors)
        
        if viz_config['advanced_features']['show_network']:
            self._add_network_points(fig, room_specs)
        
        # Update camera view and layout
        self._update_camera_view(fig, room_specs, viz_config['style_options']['view_angle'])
        self._update_layout(fig, room_specs)
        
        return fig

    def _add_room_structure(self, fig, specs, colors, lighting):
        self._add_walls(fig, specs, colors, lighting)
        fig.add_trace(go.Surface(x=[[0, specs['length']], [0, specs['length']]], y=[[0, 0], [specs['width'], specs['width']]], z=[[0, 0], [0, 0]], colorscale=[[0, colors['floor']], [1, colors['floor']]], showscale=False, name='Floor', lighting=lighting))

    def _add_walls(self, fig, specs, colors, lighting):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        fig.add_trace(go.Surface(x=[[0, 0], [0, 0]], y=[[0, width], [0, width]], z=[[0, 0], [height, height]], colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, opacity=0.9, lighting=lighting, name="Back Wall"))
        fig.add_trace(go.Surface(x=[[0, length], [0, length]], y=[[0, 0], [0, 0]], z=[[0, 0], [height, height]], colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, opacity=0.7, lighting=lighting, name="Left Wall"))
        fig.add_trace(go.Surface(x=[[0, length], [0, length]], y=[[width, width], [width, width]], z=[[0, 0], [height, height]], colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, opacity=0.7, lighting=lighting, name="Right Wall"))

    def _add_table(self, fig, specs, colors, table_style):
        length, width = specs['length'], specs['width']
        table_height, table_x_center, table_y_center = 0.75, length * 0.55, width * 0.5
        table_length, table_width = min(length * 0.6, 5), min(width * 0.4, 2)
        
        if table_style in ['rectangular', 'boat-shaped', 'modular']:
            x, y = np.meshgrid(np.linspace(table_x_center - table_length/2, table_x_center + table_length/2, 2), np.linspace(table_y_center - table_width/2, table_y_center + table_width/2, 2))
            z = np.full_like(x, table_height)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Table'))
        elif table_style == 'oval':
            theta = np.linspace(0, 2 * np.pi, 50)
            x, y = table_x_center + (table_length / 2) * np.cos(theta), table_y_center + (table_width / 2) * np.sin(theta)
            z = np.full_like(x, table_height)
            fig.add_trace(go.Scatter3d(x=np.append(x, x[0]), y=np.append(y, y[0]), z=np.append(z, z[0]), mode='lines', line=dict(color=colors['wood'], width=5), fill='toself', fillcolor=colors['wood'], name='Table'))

    def _add_seating(self, fig, specs, colors, chair_style):
        length, width, capacity = specs['length'], specs['width'], specs['capacity']
        table_length, table_width = min(length * 0.6, 5), min(width * 0.4, 2)
        table_x_center, table_y_center = length * 0.55, width * 0.5
        chairs_per_side = min(6, capacity // 2)

        for i in range(chairs_per_side):
            x_pos = table_x_center - table_length/2 + ((i + 0.5) * table_length / chairs_per_side)
            for offset in [-1, 1]:
                y_pos = table_y_center + offset * (table_width/2 + 0.5)
                if chair_style in ['modern', 'casual']:
                    fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[0.4], mode='markers', marker=dict(size=10, symbol='square', color=colors['accent']), name='Chair'))
                elif chair_style in ['executive', 'training']:
                    fig.add_trace(go.Mesh3d(x=[x_pos-0.2, x_pos+0.2, x_pos+0.2, x_pos-0.2], y=[y_pos, y_pos, y_pos, y_pos], z=[0.4, 0.4, 1.0, 1.0], i=[0,0],j=[1,2],k=[2,3], color=colors['metal'], name='Chair'))

    def _add_display_system(self, fig, specs, colors, recommendations):
        width, height = specs['width'], specs['ceiling_height']
        screen_width, screen_height = min(3.5, width * 0.6), min(3.5, width * 0.6) * 9/16
        screen_y_center, screen_z_center = width / 2, height * 0.5
        fig.add_trace(go.Mesh3d(x=[0.05]*4, y=[screen_y_center-screen_width/2, screen_y_center+screen_width/2, screen_y_center+screen_width/2, screen_y_center-screen_width/2], z=[screen_z_center-screen_height/2, screen_z_center-screen_height/2, screen_z_center+screen_height/2, screen_z_center+screen_height/2], i=[0,0],j=[1,2],k=[2,3], color=colors['screen'], name='Display'))

    def _add_camera_system(self, fig, specs, colors, recommendations):
        width, height = specs['width'], specs['ceiling_height']
        screen_height = min(3.5, width * 0.6) * 9 / 16
        camera_z = height * 0.5 + screen_height / 2 + 0.1
        fig.add_trace(go.Scatter3d(x=[0.08], y=[width/2], z=[camera_z], mode='markers', marker=dict(size=8, symbol='diamond', color=colors['screen']), name='Camera'))

    def _add_lighting_system(self, fig, specs, colors, lighting):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        light_color = lighting['color']
        for i in range(2):
            for j in range(3):
                fig.add_trace(go.Scatter3d(x=[length * (i + 1)/3], y=[width * (j + 1)/4], z=[height-0.05], mode='markers', marker=dict(size=7, color=light_color, symbol='circle'), name='Lighting'))

    def _add_audio_system(self, fig, specs, colors, recommendations):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        for i in [0.25, 0.75]:
            for j in [0.25, 0.75]:
                fig.add_trace(go.Scatter3d(x=[length*i], y=[width*j], z=[height-0.1], mode='markers', marker=dict(size=6, color=colors['metal'], symbol='circle-open'), name='Speaker'))
                
    def _add_control_system(self, fig, specs, colors):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        fig.add_trace(go.Scatter3d(x=[length - 0.1], y=[width * 0.8], z=[height * 0.4], mode='markers', marker=dict(size=10, symbol='square', color=colors['accent']), name='Control Panel'))
    
    def _add_whiteboard(self, fig, specs, colors):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        wb_width, wb_height = min(2, width * 0.4), min(2, width * 0.4) * 3/4
        fig.add_trace(go.Mesh3d(x=[length-0.05]*4, y=[width/2-wb_width/2, width/2+wb_width/2, width/2+wb_width/2, width/2-wb_width/2], z=[height*0.3, height*0.3, height*0.3+wb_height, height*0.3+wb_height], i=[0,0],j=[1,2],k=[2,3], color=colors['glass'], opacity=0.8, name='Whiteboard'))
    
    def _add_credenza(self, fig, specs, colors):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        c_len, c_depth, c_height = min(2.5, width * 0.5), 0.5, 0.8
        x, y = np.meshgrid(np.linspace(0.1, 0.1+c_depth, 2), np.linspace(width/2-c_len/2, width/2+c_len/2, 2))
        z = np.full_like(x, c_height)
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Credenza'))
    
    def _add_measurements(self, fig, specs):
        length, width = specs['length'], specs['width']
        fig.add_trace(go.Scatter3d(x=[0, length], y=[width*1.1, width*1.1], z=[0,0], mode='lines+text', text=["", f"{length}m"], line=dict(color='black', width=2), textposition="top right", name="Length"))
        fig.add_trace(go.Scatter3d(x=[length*1.1, length*1.1], y=[0, width], z=[0,0], mode='lines+text', text=["", f"{width}m"], line=dict(color='black', width=2), textposition="middle right", name="Width"))

    def _add_coverage_zones(self, fig, specs):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        x, y, z = [0.08, length*0.8, length*0.8], [width/2, width*0.1, width*0.9], [height*0.6, 0, 0]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color='blue', opacity=0.15, name='Camera Zone'))
    
    def _add_cable_management(self, fig, specs, colors):
        length, width = specs['length'], specs['width']
        table_x_center, table_y_center = length * 0.55, width * 0.5
        fig.add_trace(go.Scatter3d(x=[0.1, table_x_center], y=[width/2, table_y_center], z=[0.1, 0.1], mode='lines', line=dict(color=colors['accent'], width=4, dash='dot'), name='Cabling'))

    def _add_network_points(self, fig, specs):
        length, width = specs['length'], specs['width']
        fig.add_trace(go.Scatter3d(x=[0.1, length-0.1], y=[0.1, width-0.1], z=[0.2, 0.2], mode='markers', marker=dict(size=8, symbol='cross', color='green'), name='Network Point'))

    def _update_camera_view(self, fig, specs, view_angle):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        camera_angles = {
            'perspective': dict(eye=dict(x=length * -0.8, y=width * -1.2, z=height * 0.8), center=dict(x=length/2, y=width/2, z=height/3)),
            'top': dict(eye=dict(x=length/2, y=width/2, z=height + 2), center=dict(x=length/2, y=width/2, z=0)),
            'front': dict(eye=dict(x=length + 2, y=width/2, z=height*0.5), center=dict(x=length/2, y=width/2, z=height/2)),
            'side': dict(eye=dict(x=length/2, y=width + 2, z=height*0.5), center=dict(x=length/2, y=width/2, z=height/2)),
            'corner': dict(eye=dict(x=length * 1.5, y=width * 1.5, z=height * 1.2), center=dict(x=length/2, y=width/2, z=height/3))
        }
        fig.update_layout(scene_camera=camera_angles[view_angle])

    def _update_layout(self, fig, specs):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title="Length (m)", yaxis_title="Width (m)", zaxis_title="Height (m)",
                xaxis=dict(range=[0, length], showbackground=False),
                yaxis=dict(range=[0, width], showbackground=False),
                zaxis=dict(range=[0, height], showbackground=False)
            ),
            title="Interactive Conference Room Design",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255, 255, 255, 0.9)'),
            margin=dict(l=0, r=0, t=30, b=0)
        )

    @staticmethod
    def create_equipment_layout_2d(room_specs, recommendations):
        """Create an enhanced 2D floor plan with more details"""
        fig = go.Figure()
        
        length, width = room_specs['length'], room_specs['width']
        
        # Enhanced Room outline with shadow effect
        fig.add_shape(type="rect", x0=-0.2, y0=-0.2, x1=length+0.2, y1=width+0.2, line=dict(color="rgba(200,200,200,0.5)", width=2), fillcolor="rgba(240,240,240,0.3)", layer='below')
        
        # Main room outline
        fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color="rgb(70,70,70)", width=3), fillcolor="rgba(250,250,250,1)")
        
        # Add windows if specified
        if room_specs.get('environment', {}).get('windows', 0) > 0:
            window_sections = int(room_specs['environment']['windows'] / 20)
            window_width_section = width / (window_sections * 2) 
            for i in range(window_sections):
                y_start = (width / 2) - (window_sections * window_width_section / 2) + (i * window_width_section * 2)
                fig.add_shape(type="rect", x0=length-0.1, y0=y_start, x1=length, y1=y_start + window_width_section, line=dict(color="rgb(150,200,255)", width=2), fillcolor="rgba(200,230,255,0.7)")

        # Display wall with screen
        screen_width = min(width * 0.6, 3.5)
        screen_start = (width - screen_width) / 2
        fig.add_shape(type="rect", x0=0, y0=screen_start, x1=0.15, y1=screen_start + screen_width, line=dict(color="rgb(50,50,50)", width=2), fillcolor="rgb(80,80,80)")
        
        # Conference table with realistic shape
        table_length, table_width = min(length * 0.7, 4.5), min(width * 0.4, 1.5)
        table_x, table_y = length * 0.6, width * 0.5
        
        # Table shadow
        fig.add_shape(type="rect", x0=table_x - table_length/2 + 0.1, y0=table_y - table_width/2 + 0.1, x1=table_x + table_length/2 + 0.1, y1=table_y + table_width/2 + 0.1, line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(0,0,0,0.1)")
        
        # Main table
        fig.add_shape(type="rect", x0=table_x - table_length/2, y0=table_y - table_width/2, x1=table_x + table_length/2, y1=table_y + table_width/2, line=dict(color="rgb(120,85,60)", width=2), fillcolor="rgb(139,115,85)")
        
        # Add chairs based on capacity
        capacity = min(room_specs['capacity'], 12)
        chairs_per_side = min(6, capacity // 2)
        chair_positions = []
        for i in range(chairs_per_side):
            x_pos = table_x - table_length/2 + ((i + 1) * table_length/(chairs_per_side + 1))
            chair_positions.extend([(x_pos, table_y - table_width/2 - 0.4), (x_pos, table_y + table_width/2 + 0.4)])
        
        for x, y in chair_positions[:capacity]:
            fig.add_shape(type="circle", x0=x-0.25+0.05, y0=y-0.25+0.05, x1=x+0.25+0.05, y1=y+0.25+0.05, line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(0,0,0,0.1)")
            fig.add_shape(type="circle", x0=x-0.25, y0=y-0.25, x1=x+0.25, y1=y+0.25, line=dict(color="rgb(70,130,180)"), fillcolor="rgb(100,149,237)")
        
        # Add equipment zones and annotations
        fig.add_shape(type="rect", x0=0, y0=width*0.2, x1=0.8, y1=width*0.8, line=dict(color="rgba(255,100,100,0.3)", width=2), fillcolor="rgba(255,100,100,0.1)")
        
        camera_points = [[0.1, width*0.45], [0.1, width*0.55], [length*0.8, width*0.2], [length*0.8, width*0.8]]
        fig.add_shape(type="path", path=f"M {camera_points[0][0]},{camera_points[0][1]} L {camera_points[1][0]},{camera_points[1][1]} L {camera_points[3][0]},{camera_points[3][1]} L {camera_points[2][0]},{camera_points[2][1]} Z", line=dict(color="rgba(100,200,100,0.3)", width=1), fillcolor="rgba(100,200,100,0.1)")
        
        speaker_positions = [(length*0.25, width*0.25), (length*0.75, width*0.25), (length*0.25, width*0.75), (length*0.75, width*0.75)]
        for x, y in speaker_positions:
            fig.add_shape(type="circle", x0=x-0.15, y0=y-0.15, x1=x+0.15, y1=y+0.15, line=dict(color="rgba(100,100,255,0.3)"), fillcolor="rgba(100,100,255,0.1)")
            fig.add_shape(type="circle", x0=x-1.5, y0=y-1.5, x1=x+1.5, y1=y+1.5, line=dict(color="rgba(100,100,255,0.1)"), fillcolor="rgba(100,100,255,0.05)")
        
        # Add measurements and annotations
        fig.add_annotation(x=length/2, y=-0.5, text=f"{length:.1f}m", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=-0.5, y=width/2, text=f"{width:.1f}m", textangle=-90, showarrow=False, font=dict(size=10))
        annotations = [
            dict(x=0.1, y=width*0.5, text="Display System", showarrow=True, arrowcolor="rgb(255,100,100)", bgcolor="white", bordercolor="rgb(255,100,100)", borderwidth=2),
            dict(x=length-0.3, y=width*0.85, text="Control Panel", showarrow=True, arrowcolor="rgb(100,100,100)", bgcolor="white", bordercolor="rgb(100,100,100)", borderwidth=2),
            dict(x=length*0.5, y=width*0.1, text="Camera Coverage Zone", showarrow=False, font=dict(size=10, color="rgb(100,200,100)")),
            dict(x=length*0.8, y=width*0.5, text="Speaker Coverage", showarrow=False, font=dict(size=10, color="rgb(100,100,255)"))
        ]
        
        fig.update_layout(title=dict(text="Enhanced Floor Plan with Equipment Layout", y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=16, color='rgb(50,50,50)')), xaxis=dict(title="Length (m)", range=[-1, length+1], showgrid=False, zeroline=False, scaleanchor="y", scaleratio=1), yaxis=dict(title="Width (m)", range=[-1, width+1], showgrid=False, zeroline=False), height=600, showlegend=False, annotations=annotations, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=100, b=50, l=50, r=50))
        
        return fig
    
    @staticmethod
    def create_cost_breakdown_chart(recommendations):
        categories = ['Display', 'Camera', 'Audio', 'Control', 'Lighting']
        costs = [recommendations['display']['price'], recommendations['camera']['price'], recommendations['audio']['price'], recommendations['control']['price'], recommendations['lighting']['price']]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        fig = go.Figure(data=[go.Bar(x=categories, y=costs, marker_color=colors, text=[f"${cost:,.0f}" for cost in costs], textposition='auto', textfont=dict(color='white', size=12))])
        
        fig.update_layout(title=dict(text="Investment Breakdown by Category", x=0.5, font=dict(size=16, color='#2c3e50')), xaxis_title="Equipment Category", yaxis_title="Investment (USD)", height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white', yaxis=dict(tickformat='$,.0f'))
        
        return fig
    
    @staticmethod
    def create_feature_comparison_radar(recommendations, alternatives):
        categories = ['Performance', 'Features', 'Reliability', 'Integration', 'Value']
        current_scores = [4.5, 4.2, 4.6, 4.4, 4.3]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=current_scores + [current_scores[0]], theta=categories + [categories[0]], fill='toself', name='Recommended Solution', line_color='#3498db', fillcolor='rgba(52, 152, 219, 0.2)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5], tickmode='linear', tick0=0, dtick=1)), showlegend=True, title=dict(text="Solution Performance Analysis", x=0.5, font=dict(size=16, color='#2c3e50')), height=400, paper_bgcolor='white')
        
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
        
        st.markdown("---")
        st.subheader("🌟 Environment & Atmosphere")

        env_col1, env_col2 = st.columns(2)
        with env_col1:
            windows = st.slider("Windows (%)", 0, 80, 20, 5, help="Percentage of wall space with windows")
            natural_light = st.select_slider("Natural Light Level", options=["Very Low", "Low", "Moderate", "High", "Very High"], value="Moderate", help="Amount of natural light entering the room")
        
        with env_col2:
            ceiling_type = st.selectbox("Ceiling Type", ["Standard", "Drop Ceiling", "Open Plenum", "Acoustic Tiles"], help="Type of ceiling construction")
            wall_material = st.selectbox("Wall Material", ["Drywall", "Glass", "Concrete", "Wood Panels", "Acoustic Panels"], help="Primary wall material")

        st.markdown("##### 🎯 Room Purpose & Acoustics")
        room_purpose = st.multiselect("Primary Activities", ["Video Conferencing", "Presentations", "Training", "Board Meetings", "Collaborative Work", "Hybrid Meetings", "Social Events"], default=["Video Conferencing", "Presentations"], help="Select all typical activities")
        acoustic_features = st.multiselect("Acoustic Considerations", ["Sound Absorption Needed", "Echo Control Required", "External Noise Issues", "Speech Privacy Important", "Music Playback Required"], help="Select acoustic challenges to address")

        st.markdown("##### 🎛️ Environmental Controls")
        env_controls = st.multiselect("Control Systems", ["Automated Lighting", "Motorized Shades", "Climate Control", "Air Quality Monitoring", "Occupancy Sensors", "Daylight Harvesting"], help="Select desired environmental control features")

        st.markdown("##### 🎨 Ambiance & Design")
        color_scheme_temp = st.select_slider("Color Temperature", options=["Warm", "Neutral Warm", "Neutral", "Neutral Cool", "Cool"], value="Neutral", help="Preferred lighting color temperature")
        design_style = st.selectbox("Interior Design Style", ["Modern Corporate", "Executive", "Creative/Tech", "Traditional", "Industrial", "Minimalist"], help="Overall design aesthetic")

        st.markdown("##### ♿ Accessibility Features")
        accessibility = st.multiselect("Accessibility Requirements", ["Wheelchair Access", "Hearing Loop System", "High Contrast Displays", "Voice Control", "Adjustable Furniture", "Braille Signage"], help="Select required accessibility features")

        st.markdown("---")
        st.subheader("💰 Budget & Brands")
        budget_tier = st.selectbox("Budget Tier", ['Budget', 'Professional', 'Premium'], index=1)
        preferred_brands = st.multiselect("Preferred Brands", ['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'], help="Leave empty for best overall recommendations")
        
        st.subheader("✨ Special Features")
        special_features = st.multiselect("Required Features", ['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 'Noise Reduction', 'Circadian Lighting', 'AI Analytics'])

        # --- Visualization Options in Sidebar ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Visualization Options")
        
        expander_room = st.sidebar.expander("Room Elements", expanded=True)
        with expander_room:
            room_elements_config = {
                'show_chairs': st.checkbox("Show Chairs", value=True),
                'show_displays': st.checkbox("Show Displays", value=True),
                'show_cameras': st.checkbox("Show Cameras", value=True),
                'show_speakers': st.checkbox("Show Speakers", value=True),
                'show_lighting': st.checkbox("Show Lighting", value=True),
                'show_control': st.checkbox("Show Control Panel", value=True),
                'show_table': st.checkbox("Show Table", value=True),
                'show_whiteboard': st.checkbox("Digital Whiteboard", value=False),
                'show_credenza': st.checkbox("Credenza/Storage", value=False)
            }
        
        expander_style = st.sidebar.expander("Style Options", expanded=False)
        with expander_style:
            style_config = {
                'chair_style': st.selectbox("Chair Style", ['modern', 'executive', 'training', 'casual']),
                'table_style': st.selectbox("Table Style", ['rectangular', 'oval', 'boat-shaped', 'modular']),
                'color_scheme': st.selectbox("Color Scheme", ['professional', 'modern', 'classic', 'warm', 'cool']),
                'lighting_mode': st.selectbox("Lighting Mode", ['day', 'evening', 'presentation', 'video conference']),
                'view_angle': st.selectbox("View Angle", ['perspective', 'top', 'front', 'side', 'corner'])
            }

        expander_advanced = st.sidebar.expander("Advanced Features", expanded=False)
        with expander_advanced:
            advanced_config = {
                'show_measurements': st.checkbox("Show Measurements", value=False),
                'show_zones': st.checkbox("Show Audio/Video Zones", value=False),
                'show_cable_paths': st.checkbox("Show Cable Management", value=False),
                'show_network': st.checkbox("Show Network Points", value=False),
                'quality_level': st.slider("Rendering Quality", 1, 5, 3)
            }
        
        if st.button("🚀 Generate AI Recommendation"):
            environment_config = {
                'windows': windows, 'natural_light': natural_light, 'ceiling_type': ceiling_type,
                'wall_material': wall_material, 'room_purpose': room_purpose,
                'acoustic_features': acoustic_features, 'env_controls': env_controls,
                'color_scheme': color_scheme_temp, 'design_style': design_style, 'accessibility': accessibility
            }
            
            room_specs = {
                'template': template, 'length': length, 'width': width, 'ceiling_height': ceiling_height,
                'capacity': capacity, 'environment': environment_config, 'special_requirements': []
            }
            user_preferences = {
                'budget_tier': budget_tier, 'preferred_brands': preferred_brands, 'special_features': special_features,
                'design_style': design_style, 'color_scheme': color_scheme_temp
            }
            
            recommender = MaximizedAVRecommender()
            st.session_state.recommendations = recommender.get_comprehensive_recommendations(room_specs, user_preferences)
            st.session_state.room_specs = room_specs
            st.session_state.budget_tier = budget_tier
            st.success("✅ AI Analysis Complete!")

    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        room_specs = st.session_state.room_specs
        recommender = MaximizedAVRecommender()
        
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
                    st.markdown(f"""<div class="feature-card"><h4>{rec['model']}</h4><p><strong>Price:</strong> ${rec['price']:,} | <strong>Rating:</strong> ⭐ {rec['rating']}/5.0</p><p><strong>Specs:</strong> {rec['specs']}</p></div>""", unsafe_allow_html=True)
            with col2:
                for cat, icon in [('control', '🎛️'), ('lighting', '💡')]:
                    rec = recommendations[cat]
                    st.markdown(f"#### {icon} {cat.title()} System")
                    st.markdown(f"""<div class="feature-card"><h4>{rec['model']}</h4><p><strong>Price:</strong> ${rec['price']:,} | <strong>Rating:</strong> ⭐ {rec['rating']}/5.0</p><p><strong>Specs:</strong> {rec['specs']}</p></div>""", unsafe_allow_html=True)
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
                st.markdown(f"""<div class="comparison-card" style="border: 1px solid #dee2e6;">
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
            viz_config = {'room_elements': room_elements_config, 'style_options': style_config, 'advanced_features': advanced_config}
            viz_engine = EnhancedVisualizationEngine()
            fig_3d = viz_engine.create_3d_room_visualization(room_specs, recommendations, viz_config)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.plotly_chart(EnhancedVisualizationEngine.create_equipment_layout_2d(room_specs, recommendations), use_container_width=True)

        with tab4:
            st.subheader("Alternative Configurations & Smart Upgrade Planner")
            if recommendations.get('alternatives'):
                st.markdown("#### Alternative Configurations")
                for tier_name, alt_config in recommendations['alternatives'].items():
                    st.markdown(f"##### {tier_name} Tier")
                    col1, col2, col3 = st.columns(3)
                    cols = [col1, col2, col3]
                    for i, cat in enumerate(['displays', 'cameras', 'audio']):
                        if cat in alt_config:
                            with cols[i]:
                                name, info = alt_config[cat]
                                st.markdown(f"""<div class="comparison-card" style="border: 1px solid #dee2e6;"><strong>{cat.title()}:</strong> {name}<br>${info['price']:,} | ⭐ {info['rating']}/5.0</div>""", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)

            if recommendations.get('upgrade_path'):
                upgrade = recommendations['upgrade_path'][0] 
                smart_plan = recommender._generate_smart_upgrade_plan(room_specs, st.session_state.budget_tier, upgrade['estimated_cost'])
                st.markdown("""<div class="premium-card"> ... </div>""", unsafe_allow_html=True) # Content omitted for brevity

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
