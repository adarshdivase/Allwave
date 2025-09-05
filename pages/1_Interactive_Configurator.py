import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Room Configurator Pro Max",
    page_icon="🏢",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        /* Primary Brand Colors */
        --primary-blue: #2563eb;
        --primary-blue-hover: #1d4ed8;
        --primary-blue-light: #3b82f6;
        --primary-blue-dark: #1e40af;
        
        /* Secondary Colors */
        --success-green: #10b981;
        --error-red: #ef4444;
        
        /* Neutral Colors */
        --white: #ffffff;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
        
        /* Background Colors */
        --background-primary: var(--gray-50);
        --background-sidebar: var(--gray-900);
        
        /* Text Colors */
        --text-primary: var(--gray-900);
        --text-secondary: var(--gray-600);
        --text-white: var(--white);
        --text-white-secondary: rgba(255,255,255,0.9);
        
        /* Component Colors */
        --card-background: var(--white);
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        --card-shadow-hover: 0 8px 25px rgba(0, 0, 0, 0.1), 0 4px 12px rgba(0, 0, 0, 0.05);
        --border-color: var(--gray-200);
        --border-color-focus: var(--primary-blue);
        
        /* Radius and Spacing */
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        
        /* Specific Component Colors */
        --metric-bg: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-light) 100%);
        --premium-card-bg: var(--background-dark);
        --feature-card-accent: var(--primary-blue);
        --comparison-card-border: var(--gray-300);
    }

    /* Base App Styling */
    .stApp {
        background-color: var(--background-primary) !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Main Content Area */
    .main .block-container {
        padding: 2rem 1rem !important;
        max-width: 1200px !important;
    }

    /* General Content Cards */
    .main > div > div {
        background: var(--card-background) !important;
        border-radius: var(--radius-lg) !important;
        padding: 2rem !important;
        margin: 1rem 0 !important;
        box-shadow: var(--card-shadow) !important;
        border: 1px solid var(--border-color) !important;
        transition: box-shadow 0.2s ease !important;
    }

    .main > div > div:hover {
        box-shadow: var(--card-shadow-hover) !important;
    }

    /* Typography */
    .main h1, .main h2, .main h3, .main h4 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .main p, .main li {
        color: var(--text-secondary) !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--background-dark) !important;
        border-radius: var(--radius-md) !important;
        padding: 8px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1) !important;
        color: var(--text-white-secondary) !important;
        border-radius: var(--radius-sm) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-blue) !important;
        color: var(--text-white) !important;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: var(--metric-bg) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.5rem !important;
    }

    div[data-testid="metric-container"] label {
        color: var(--text-white-secondary) !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--text-white) !important;
        font-size: 2rem !important;
    }

    /* Custom CSS Classes */
    .premium-card {
        background: var(--premium-card-bg) !important;
        color: var(--text-white) !important;
        border-color: var(--gray-700) !important;
    }
    .premium-card h2, .premium-card p {
        color: var(--text-white) !important;
    }
    
    .feature-card {
        border-left: 4px solid var(--feature-card-accent) !important;
    }
    
    .comparison-card {
        border: 2px solid var(--comparison-card-border) !important;
    }
    
    .comparison-card:hover {
        border-color: var(--primary-blue) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--metric-bg) !important;
        color: var(--text-white) !important;
        border: none !important;
        border-radius: var(--radius-xl) !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(37,99,235,0.3) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--background-sidebar) !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: var(--text-white) !important;
    }

    [data-testid="stSidebar"] p {
        color: var(--text-white-secondary) !important;
    }

</style>
""", unsafe_allow_html=True)


# --- DATA MODELS ---

class EnhancedProductDatabase:
    """A comprehensive database of AV equipment and room templates."""
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

class BudgetManager:
    """Manages and validates equipment costs against budget tier limits."""
    def __init__(self, budget_tier: str):
        self.tier_limits = {
            'Budget': {'min': 5000, 'max': 25000},
            'Professional': {'min': 25000, 'max': 75000},
            'Premium': {'min': 75000, 'max': 500000}
        }
        self.current_tier = budget_tier
        self.running_total = 0
    
    def validate_item(self, item_cost: float):
        """Checks if adding an item exceeds the budget. Updates total if valid."""
        new_total = self.running_total + item_cost
        tier_limit = self.tier_limits[self.current_tier]['max']
        
        if new_total > tier_limit:
            return False, f"Budget exceeded for {self.current_tier} tier (${tier_limit:,})"
        
        self.running_total = new_total
        return True, None

# --- LOGIC CLASSES ---

class MaximizedAVRecommender:
    """The core engine for generating intelligent AV recommendations."""
    def __init__(self):
        self.db = EnhancedProductDatabase()
    
    def get_comprehensive_recommendations(
        self,
        room_specs: Dict,
        user_preferences: Dict,
        budget_manager: BudgetManager
    ) -> Dict:
        """
        Generates a full set of recommendations and analyses for a given room.
        
        Args:
            room_specs: Dictionary of room dimensions and environmental details.
            user_preferences: Dictionary of user's budget, brand, and feature choices.
            budget_manager: The BudgetManager instance to track costs.

        Returns:
            A dictionary containing all recommendations and analytical results.
        """
        budget_tier = user_preferences.get('budget_tier', 'Professional')
        brand_preference = user_preferences.get('preferred_brands', [])
        special_features = user_preferences.get('special_features', [])
        
        recommendations = {}
        
        # Sequentially recommend and validate budget for each category
        recommendations['display'] = self._recommend_display_advanced(room_specs, budget_tier, brand_preference)
        budget_manager.validate_item(recommendations['display']['price'])
        
        recommendations['camera'] = self._recommend_camera_advanced(room_specs, budget_tier, brand_preference)
        budget_manager.validate_item(recommendations['camera']['price'])
        
        recommendations['audio'] = self._recommend_audio_advanced(room_specs, budget_tier, special_features)
        budget_manager.validate_item(recommendations['audio']['price'])
        
        recommendations['control'] = self._recommend_control_advanced(room_specs, budget_tier)
        budget_manager.validate_item(recommendations['control']['price'])
        
        recommendations['lighting'] = self._recommend_lighting_advanced(room_specs, budget_tier, user_preferences, special_features)
        budget_manager.validate_item(recommendations['lighting']['price'])
        
        recommendations['accessories'] = self._recommend_accessories_advanced(room_specs, special_features)
        for acc in recommendations['accessories']:
            budget_manager.validate_item(acc['price'])
        
        # Add supplementary analysis
        recommendations['alternatives'] = self._generate_alternatives(room_specs, budget_tier)
        recommendations['confidence_score'] = self._calculate_advanced_confidence(room_specs, user_preferences)
        recommendations['room_analysis'] = self._analyze_room_characteristics(room_specs)
        recommendations['upgrade_path'] = self._suggest_upgrade_path(room_specs, budget_tier)
        
        return recommendations
    
    def _recommend_display_advanced(self, specs: Dict, tier: str, brands: List[str]) -> Dict:
        """Recommends a display based on room size and viewing distance."""
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
        
        if not products:
            products = self.db.products['displays'][tier] # Fallback if no brand match

        selected = list(products.items())[0] # Default selection
        for name, product in products.items():
            if size_category == 'large' and ('100"' in name or 'Wall' in name):
                selected = (name, product)
                break
            elif size_category == 'medium' and any(size in name for size in ['75"', '85"', '86"']):
                selected = (name, product)
                break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['viewing_distance_optimal'] = f"{viewing_distance:.1f}m"
        result['brightness_needed'] = self._calculate_brightness_needs(specs)
        return result
    
    # ... Other recommendation helper methods (_recommend_camera_advanced, etc.) ...
    # (These methods follow similar logic to the one above, using room specs to
    # filter and select the most appropriate product from the database)
    
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
        
        config_type = 'table_system'
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
                if any(k in name for k in ['NVX', 'DGX', 'Core']):
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
        aspect_ratio = max(specs['length'], specs['width']) / min(specs['length'], specs['width']) if min(specs['length'], specs['width']) > 0 else 1
        
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
        """Generate a detailed phased upgrade plan."""
        phases = {
            'Immediate (0-3 months)': {
                'priorities': [
                    'Essential software upgrades and licensing',
                    'Control system programming optimization',
                    'Staff training on current systems'
                ], 'cost_percentage': 0.15, 'focus': 'Maximizing current infrastructure'
            },
            'Phase 1 (3-6 months)': {
                'priorities': ['Display system upgrade','Camera system enhancement','Basic audio improvements'],
                'cost_percentage': 0.35, 'focus': 'Core AV capabilities'
            },
            'Phase 2 (6-9 months)': {
                'priorities': ['Advanced audio processing implementation','Lighting control system upgrade','Room automation integration'],
                'cost_percentage': 0.30, 'focus': 'Enhanced functionality'
            },
            'Final Phase (9-12 months)': {
                'priorities': ['Premium features activation','AI analytics integration','Complete system optimization'],
                'cost_percentage': 0.20, 'focus': 'Premium capabilities'
            }
        }
        for phase in phases.values():
            phase['budget'] = estimated_cost * phase['cost_percentage']
        
        roi_metrics = {
            'Productivity Gain': '15-20%', 'Energy Savings': '10-15%',
            'Maintenance Cost Reduction': '25-30%', 'System Downtime Reduction': '40-50%'
        }
        return {
            'phases': phases, 
            'roi_metrics': roi_metrics, 
            'total_investment': estimated_cost, 
            'monthly_investment': estimated_cost / 12 if estimated_cost > 0 else 0
        }
    
    # ... Other helper methods for analysis (_calculate_brightness_needs, etc.) ...
    
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
            'type': config_type, 
            'coverage': f"{min(100, room_volume / 2):.0f}%",
            'microphone_count': max(2, specs['capacity'] // 4), 
            'speaker_zones': max(1, specs['capacity'] // 8),
            'processing_power': 'High' if room_volume > 150 else 'Standard'
        }

    def _analyze_acoustics(self, specs):
        env = specs.get('environment', {})
        wall_absorb = {"Drywall": 0.1, "Glass": 0.03, "Concrete": 0.02, "Wood Panels": 0.15, "Acoustic Panels": 0.8}
        absorption_coeff = wall_absorb.get(env.get('wall_material', 'Drywall'), 0.1)
        
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        surface_area = 2 * (specs['length'] * specs['width'] + specs['length'] * specs['ceiling_height'] + specs['width'] * specs['ceiling_height'])
        
        if absorption_coeff > 0:
            rt60_estimate = 0.161 * room_volume / (absorption_coeff * surface_area)
        else:
            rt60_estimate = float('inf')
            
        return {
            'rt60_estimate': f"{rt60_estimate:.2f} seconds", 
            'acoustic_treatment_needed': rt60_estimate > 0.8,
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
        if len(specs.get('special_requirements', [])) > 3:
            base_confidence -= 0.1
        if preferences.get('preferred_brands'):
            base_confidence += 0.05
        if preferences.get('budget_tier') == 'Premium':
            base_confidence += 0.1
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
            'treatment_needed': room_volume > 150, 
            'echo_potential': max(specs['length'], specs['width']) > 10
        }

    def _identify_lighting_challenges(self, specs):
        challenges = []
        if specs.get('environment', {}).get('natural_light') in ['High', 'Very High']:
            challenges.append("High natural light - glare control needed")
        if specs['ceiling_height'] > 4:
            challenges.append("High ceiling - requires powerful fixtures")
        if 'Presentations' in specs.get('environment', {}).get('room_purpose', []):
            challenges.append("Presentation mode requires zoned lighting")
        return challenges if challenges else ["Standard lighting requirements"]

    def _analyze_capacity_efficiency(self, specs):
        area = specs['length'] * specs['width']
        efficiency = specs['capacity'] / area if area > 0 else 0
        
        if efficiency > 0.5: return "High density - space optimization good"
        elif efficiency > 0.3: return "Moderate density - balanced layout"
        else: return "Low density - spacious environment"

    def _estimate_tier_cost(self, specs, tier):
        base_costs = {'Budget': 15000, 'Professional': 45000, 'Premium': 120000}
        return int(base_costs[tier] * (1 + (specs['capacity'] / 50)))

# --- VISUALIZATION ENGINE ---

class EnhancedVisualizationEngine:
    """Handles the creation of all data visualizations."""
    
    def calculate_table_requirements(self, room_specs: Dict) -> Dict:
        """Calculates the optimal conference table size based on room capacity."""
        capacity = room_specs['capacity']
        space_per_person = 0.75
        
        min_table_length = max(capacity * 0.6, 2.0)
        min_table_width = max(capacity * 0.3, 1.0)
        
        max_length = room_specs['length'] * 0.7
        max_width = room_specs['width'] * 0.4
        
        table_length = min(min_table_length, max_length)
        table_width = min(min_table_width, max_width)
        
        perimeter = (table_length * 2) + (table_width * 2)
        perimeter_seats = int(perimeter / space_per_person)
        
        return {
            'length': table_length,
            'width': table_width,
            'area': table_length * table_width,
            'seats': min(capacity, perimeter_seats)
        }

    def create_3d_room_visualization(self, room_specs: Dict, recommendations: Dict, viz_config: Dict) -> go.Figure:
        """Creates an interactive 3D model of the room with equipment."""
        fig = go.Figure()
        length, width, height = room_specs['length'], room_specs['width'], room_specs['ceiling_height']

        # Floor, Walls, and Ceiling
        # ... (Code for creating surfaces remains the same) ...

        # Display Screen
        # ... (Code for creating display screen remains the same) ...
        
        return fig # Placeholder for full implementation

    @staticmethod
    def create_equipment_layout_2d(room_specs: Dict, recommendations: Dict) -> go.Figure:
        """Creates a top-down 2D floor plan with equipment and coverage zones."""
        fig = go.Figure()
        length, width = room_specs['length'], room_specs['width']

        # Room Outline
        fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color="black", width=2), fillcolor="lightgrey")
        
        # ... (Code for adding equipment shapes and coverage zones) ...
        
        fig.update_layout(
            title="2D Equipment Layout",
            xaxis_title="Length (m)",
            yaxis_title="Width (m)",
            height=600
        )
        return fig # Placeholder for full implementation

    @staticmethod
    def create_cost_breakdown_chart(recommendations: Dict) -> go.Figure:
        """Creates a bar chart showing the cost breakdown by category."""
        categories = ['Display', 'Camera', 'Audio', 'Control', 'Lighting']
        costs = [
            recommendations['display']['price'],
            recommendations['camera']['price'],
            recommendations['audio']['price'],
            recommendations['control']['price'],
            recommendations['lighting']['price']
        ]
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=costs,
            text=[f"${c:,.0f}" for c in costs],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Investment Breakdown by Category",
            xaxis_title="Equipment Category",
            yaxis_title="Cost (USD)"
        )
        return fig

    @staticmethod
    def create_feature_comparison_radar(recommendations: Dict, alternatives: Dict) -> go.Figure:
        """Creates a radar chart for comparing solution features."""
        categories = ['Performance', 'Features', 'Reliability', 'Integration', 'Value']
        current_scores = [4.5, 4.2, 4.6, 4.4, 4.3] # Simulated scores
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=current_scores,
            theta=categories,
            fill='toself',
            name='Recommended Solution'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title_text="Solution Performance Analysis"
        )
        return fig

# --- UTILITY FUNCTIONS ---

def validate_configuration(room_specs: Dict, budget_manager: BudgetManager) -> (List[str], List[str]):
    """Validates the user's configuration for logical errors and warnings."""
    warnings = []
    errors = []
    
    min_area_per_person = 1.5
    room_area = room_specs['length'] * room_specs['width']
    required_area = room_specs['capacity'] * min_area_per_person
    
    if room_area < required_area:
        errors.append(f"Room is too small for {room_specs['capacity']} people. Minimum {required_area:.1f}m² required.")
    
    if room_specs['width'] > 0:
        aspect_ratio = room_specs['length'] / room_specs['width']
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            warnings.append("Room's aspect ratio may be challenging for AV equipment placement.")
            
    return warnings, errors

# --- MAIN APPLICATION ---

def main():
    """Main function to run the Streamlit application."""
    
    # Initialize session state
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            'active_features': {'table_space', 'budget_tracking', 'camera_angles'},
            'table_config': None
        }
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'room_specs' not in st.session_state:
        st.session_state.room_specs = None

    # --- SIDEBAR UI ---
    with st.sidebar:
        st.markdown(
            '<div class="premium-card" style="margin-top: -50px;">'
            '<h2>🎛️ Room Configuration</h2></div>', 
            unsafe_allow_html=True
        )
        
        db = EnhancedProductDatabase()
        template = st.selectbox(
            "Room Template", 
            list(db.room_templates.keys()), 
            help="Choose a template to pre-fill common values."
        )
        template_info = db.room_templates[template]
        
        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        length = col1.slider("Length (m)", 2.0, 20.0, float(template_info['typical_size'][0]), 0.5)
        width = col2.slider("Width (m)", 2.0, 20.0, float(template_info['typical_size'][1]), 0.5)
        ceiling_height = col1.slider("Ceiling Height (m)", 2.4, 6.0, 3.0, 0.1)
        capacity = col2.slider("Capacity", 2, 100, template_info['capacity_range'][1])
        
        st.divider()
        st.subheader("🌟 Environment & Atmosphere")
        # ... (Additional sidebar inputs for environment, budget, features) ...
        env_col1, env_col2 = st.columns(2)
        with env_col1:
            windows = st.slider("Windows (%)", 0, 80, 20, 5, help="Percentage of wall space with windows")
            natural_light = st.select_slider("Natural Light Level", options=["Very Low", "Low", "Moderate", "High", "Very High"], value="Moderate")
        with env_col2:
            ceiling_type = st.selectbox("Ceiling Type", ["Standard", "Drop Ceiling", "Open Plenum", "Acoustic Tiles"])
            wall_material = st.selectbox("Wall Material", ["Drywall", "Glass", "Concrete", "Wood Panels", "Acoustic Panels"])
        
        st.markdown("##### 🎯 Room Purpose & Acoustics")
        room_purpose = st.multiselect("Primary Activities", ["Video Conferencing", "Presentations", "Training", "Board Meetings"], default=["Video Conferencing", "Presentations"])
        acoustic_features = st.multiselect("Acoustic Considerations", ["Sound Absorption", "Echo Control", "Noise Issues"])
        
        st.divider()
        st.subheader("💰 Budget & Brands")
        budget_tier = st.selectbox("Budget Tier", ['Budget', 'Professional', 'Premium'], index=1)
        preferred_brands = st.multiselect("Preferred Brands", ['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'])
        
        st.subheader("✨ Special Features")
        special_features = st.multiselect("Required Features", ['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 'AI Analytics'])

    # --- MAIN PAGE UI ---
    st.title("🏢 AI Room Configurator Pro Max")
    st.markdown("### Transform Your Space with Intelligent AV Design")

    if st.button("🚀 Generate AI Recommendation"):
        environment_config = {
            'windows': windows, 'natural_light': natural_light, 'ceiling_type': ceiling_type,
            'wall_material': wall_material, 'room_purpose': room_purpose,
            'acoustic_features': acoustic_features
        }
        room_specs = {
            'template': template, 'length': length, 'width': width, 'ceiling_height': ceiling_height,
            'capacity': capacity, 'environment': environment_config, 'special_requirements': []
        }
        user_preferences = {
            'budget_tier': budget_tier, 'preferred_brands': preferred_brands, 'special_features': special_features
        }
        
        budget_manager = BudgetManager(budget_tier)
        warnings, errors = validate_configuration(room_specs, budget_manager)
        
        if errors:
            st.error(f"🚨 Please correct the following errors: \n\n* " + "\n* ".join(errors))
        else:
            if warnings:
                st.warning(f"⚠️ Consider these design warnings: \n\n* " + "\n* ".join(warnings))
            
            recommender = MaximizedAVRecommender()
            viz_engine = EnhancedVisualizationEngine()
            
            st.session_state.app_state['table_config'] = viz_engine.calculate_table_requirements(room_specs)
            
            recommendations = recommender.get_comprehensive_recommendations(
                room_specs, user_preferences, budget_manager=budget_manager
            )
            
            if budget_manager.running_total > budget_manager.tier_limits[budget_tier]['max']:
                st.error(f"Cost of ${budget_manager.running_total:,.0f} exceeds the {budget_tier} tier limit.")
            else:
                st.session_state.recommendations = recommendations
                st.session_state.room_specs = room_specs
                st.session_state.budget_tier = budget_tier
                st.success("✅ AI Analysis Complete!")

    # Display results if they exist in session state
    if st.session_state.recommendations:
        recs = st.session_state.recommendations
        specs = st.session_state.room_specs
        
        total_cost = sum(item['price'] for item in recs.values() if isinstance(item, dict) and 'price' in item)
        total_cost += sum(acc['price'] for acc in recs.get('accessories', []))

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Investment", f"${total_cost:,.0f}")
        col2.metric("AI Confidence", f"{recs['confidence_score']:.0%}")
        col3.metric("Room Size", f"{specs['length']}m × {specs['width']}m")
        col4.metric("Capacity", f"{specs['capacity']}")
        
        # Tabs for detailed results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📊 Analysis", "🎨 Visualization", "🔄 Alternatives", "📋 Report"])
        
        with tab1:
            st.header("AI-Powered Equipment Recommendations")
            # ... (Display recommended products in feature cards) ...
            
        with tab2:
            st.header("Room Analysis & Performance Metrics")
            # ... (Display room characteristics and performance charts) ...
            
        with tab3:
            st.header("Interactive Room Visualization")
            # viz_engine = EnhancedVisualizationEngine()
            # fig_3d = viz_engine.create_3d_room_visualization(specs, recs, {})
            # st.plotly_chart(fig_3d, use_container_width=True)
            # st.plotly_chart(viz_engine.create_equipment_layout_2d(specs, recs), use_container_width=True)

        with tab4:
            st.header("Alternative Configurations & Upgrade Planner")
            # ... (Display alternatives and upgrade path) ...
            
        with tab5:
            st.header("Professional Report Summary")
            # ... (Display executive summary and detailed specs table) ...

    else:
        st.markdown(
            '''<div class="premium-card" style="text-align: center; padding: 50px;">
                <h2>🚀 Welcome to AI Room Configurator Pro Max</h2>
                <p style="font-size: 18px;">Configure your room in the sidebar to generate an intelligent AV design.</p>
            </div>''',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
