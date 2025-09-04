import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any
import random

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
        --sidebar-bg: #0f172a;
        --sidebar-text: #f3f4f6;
        --text-color: #0b1220;
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
    .stSidebar .css-1d391kg, .css-1d391kg {
        background: linear-gradient(180deg, var(--sidebar-bg), #0b1220) !important;
        color: var(--sidebar-text) !important;
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
</style>
""", unsafe_allow_html=True)


# --- Comprehensive Product Database ---
class EnhancedProductDatabase:
    def __init__(self):
        # (product lists omitted for brevity in this file preview — kept in memory)
        self.products = {
            'displays': {
                'Budget': {
                    'BenQ IRP55': {'price': 1299, 'specs': '55" 4K Interactive Display, 20-point touch', 'rating': 4.2, 'brand': 'BenQ'},
                    'LG 75UP8000PUA': {'price': 1899, 'specs': '75" 4K LED, webOS, ThinQ AI', 'rating': 4.4, 'brand': 'LG'},
                    'Samsung QB65R': {'price': 2499, 'specs': '65" 4K QLED Business Display', 'rating': 4.5, 'brand': 'Samsung'}
                },
                'Professional': {
                    'Sharp/NEC 100\" 4K Display': {'price': 15999, 'specs': '100" 4K UHD, 500 nits, 24/7 Operation', 'rating': 4.7, 'brand': 'Sharp/NEC'},
                    'Sony BRAVIA FW-85BZ40H': {'price': 8999, 'specs': '85" 4K Pro Display, Android TV', 'rating': 4.6, 'brand': 'Sony'},
                    'Planar UltraRes X Series': {'price': 12999, 'specs': '86" 4K Multi-touch Display', 'rating': 4.5, 'brand': 'Planar'}
                },
                'Premium': {
                    'LG MAGNIT 136\"': {'price': 75000, 'specs': 'MicroLED, 4K, AI-powered processing, Cable-less', 'rating': 4.9, 'brand': 'LG'},
                    'Samsung The Wall 146\"': {'price': 99999, 'specs': 'MicroLED, 4K, 0.8mm Pixel Pitch, AI Upscaling', 'rating': 5.0, 'brand': 'Samsung'},
                    'Sony Crystal LED 220\"': {'price': 150000, 'specs': 'Crystal LED, 4K+, Seamless Modular Design', 'rating': 5.0, 'brand': 'Sony'}
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
            'Huddle Room (2-6 people)': {'typical_size': (3, 4), 'capacity_range': (2, 6), 'recommended_tier': 'Budget', 'typical_usage': 'Quick meetings, brainstorming'},
            'Small Conference (6-12 people)': {'typical_size': (4, 6), 'capacity_range': (6, 12), 'recommended_tier': 'Professional', 'typical_usage': 'Team meetings, presentations'},
            'Large Conference (12-20 people)': {'typical_size': (6, 10), 'capacity_range': (12, 20), 'recommended_tier': 'Professional', 'typical_usage': 'Department meetings, training'},
            'Boardroom (8-16 people)': {'typical_size': (5, 8), 'capacity_range': (8, 16), 'recommended_tier': 'Premium', 'typical_usage': 'Executive meetings, board meetings'},
            'Training Room (20-50 people)': {'typical_size': (8, 12), 'capacity_range': (20, 50), 'recommended_tier': 'Professional', 'typical_usage': 'Training, workshops, seminars'},
            'Auditorium (50+ people)': {'typical_size': (12, 20), 'capacity_range': (50, 200), 'recommended_tier': 'Premium', 'typical_usage': 'Large presentations, events'}
        }


# --- Simple Recommender implementation so app runs ---
class MaximizedAVRecommender:
    def __init__(self):
        self.db = EnhancedProductDatabase()

    def _pick_product(self, category: str, tier: str):
        items = self.db.products.get(category + 's' if category != 'control' else 'control_systems', {})
        # Normalized tier keys in db sometimes differ; try both
        tier = tier if tier in items else tier
        tier_dict = items.get(tier, {})
        if not tier_dict:
            # fallback: choose any
            for t in items:
                tier_dict = items[t]
                break
        if not tier_dict:
            return {'model': f'Generic {category}', 'price': 1000, 'specs': '', 'rating': 4.0, 'brand': 'Generic'}
        name = random.choice(list(tier_dict.keys()))
        info = tier_dict[name]
        return {'model': name, 'price': info['price'], 'specs': info['specs'], 'rating': info['rating'], 'brand': info.get('brand', 'Unknown')}

    def get_comprehensive_recommendations(self, room_specs: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        tier = user_preferences.get('budget_tier', 'Professional')
        display = self._pick_product('display', tier)
        camera = self._pick_product('camera', tier)
        audio = self._pick_product('audio', tier)
        control = self._pick_product('control', tier)
        lighting = self._pick_product('lighting', tier)

        accessories = []
        if 'Digital Whiteboard' in user_preferences.get('special_features', []):
            accessories.append({'item': 'Digital Whiteboard', 'model': 'Basic WB 65"', 'price': 2499, 'necessity': 'Optional'})

        # Simple analysis
        room_analysis = {
            'size_category': 'Medium' if room_specs['length'] * room_specs['width'] < 40 else 'Large',
            'shape_analysis': 'Rectangular',
            'acoustic_properties': {'reverb_category': 'Moderate', 'treatment_needed': True},
            'lighting_challenges': ['Glare from windows'] if room_specs.get('environment', {}).get('natural_light') in ['High', 'Very High'] else []
        }

        alternatives = {}
        # Provide simple alternatives for display/camera/audio
        alternatives['Professional'] = {'displays': (list(self.db.products['displays']['Professional'].keys())[0], list(self.db.products['displays']['Professional'].values())[0]),
                                       'cameras': (list(self.db.products['cameras']['Professional'].keys())[0], list(self.db.products['cameras']['Professional'].values())[0]),
                                       'audio': (list(self.db.products['audio']['Professional'].keys())[0], list(self.db.products['audio']['Professional'].values())[0])}

        upgrade_path = [{'tier': 'Premium', 'estimated_cost': sum([display['price'], camera['price'], audio['price']]) * 0.5}]

        confidence_score = 0.85

        return {
            'display': display, 'camera': camera, 'audio': audio, 'control': control, 'lighting': lighting,
            'accessories': accessories, 'room_analysis': room_analysis, 'alternatives': alternatives,
            'upgrade_path': upgrade_path, 'confidence_score': confidence_score
        }

    def _generate_smart_upgrade_plan(self, room_specs, current_tier, estimated_cost):
        # Simple phased plan
        phases = {
            'Phase 1': {'budget': estimated_cost * 0.4, 'focus': 'Core AV (display, camera)', 'priorities': ['Upgrade display', 'Upgrade camera']},
            'Phase 2': {'budget': estimated_cost * 0.35, 'focus': 'Audio & Control', 'priorities': ['Ceiling mics', 'Control system']},
            'Phase 3': {'budget': estimated_cost * 0.25, 'focus': 'Lighting & Finishing', 'priorities': ['Lighting control', 'Acoustic panels']}
        }
        total = sum(p['budget'] for p in phases.values())
        return {'phases': phases, 'total_investment': total, 'monthly_investment': total / 12}


# --- Visualization Engine (robust and validated) ---
class EnhancedVisualizationEngine:
    def __init__(self):
        self.color_schemes = {
            'professional': {'wall': '#F0F2F5', 'floor': '#E5E9F0', 'accent': '#4A90E2', 'wood': '#8B5E3C', 'screen': '#1A1A1A', 'metal': '#B8B8B8', 'glass': '#FFFFFF'},
            'modern': {'wall': '#FFFFFF', 'floor': '#F8F9FA', 'accent': '#2563EB', 'wood': '#A0522D', 'screen': '#000000', 'metal': '#C0C0C0', 'glass': '#E8F0FE'},
            'classic': {'wall': '#F5F5DC', 'floor': '#DEB887', 'accent': '#800000', 'wood': '#8B4513', 'screen': '#2F4F4F', 'metal': '#CD853F', 'glass': '#F0F8FF'},
            'warm': {'wall': '#FEF3E0', 'floor': '#F3D1A3', 'accent': '#E53E3E', 'wood': '#BF5B32', 'screen': '#2D3748', 'metal': '#D69E2E', 'glass': '#FFF5EB'},
            'cool': {'wall': '#E6FFFA', 'floor': '#B2F5EA', 'accent': '#38B2AC', 'wood': '#718096', 'screen': '#1A202C', 'metal': '#A0AEC0', 'glass': '#EBF8FF'}
        }
        self.lighting_modes = {
            'day': {'ambient': 0.8, 'diffuse': 1.0, 'color': 'rgb(255, 255, 224)'},
            'evening': {'ambient': 0.4, 'diffuse': 0.7, 'color': 'rgb(255, 228, 181)'},
            'presentation': {'ambient': 0.3, 'diffuse': 0.5, 'color': 'rgb(200, 200, 255)'},
            'video conference': {'ambient': 0.6, 'diffuse': 0.8, 'color': 'rgb(220, 220, 255)'}
        }

    def create_3d_room_visualization(self, room_specs, recommendations, viz_config):
        fig = go.Figure()
        colors = self.color_schemes.get(viz_config['style_options']['color_scheme'], self.color_schemes['professional'])
        lighting = self.lighting_modes.get(viz_config['style_options']['lighting_mode'], self.lighting_modes['day'])

        # build scene
        self._add_room_structure(fig, room_specs, colors, lighting)

        # elements
        if viz_config['room_elements']['show_table']: self._add_table(fig, room_specs, colors, viz_config['style_options']['table_style'])
        if viz_config['room_elements']['show_chairs']: self._add_seating(fig, room_specs, colors, viz_config['style_options']['chair_style'])
        if viz_config['room_elements']['show_displays']: self._add_display_system(fig, room_specs, colors, recommendations)
        if viz_config['room_elements']['show_cameras']: self._add_camera_system(fig, room_specs, colors, recommendations)
        if viz_config['room_elements']['show_lighting']: self._add_lighting_system(fig, room_specs, colors, lighting)
        if viz_config['room_elements']['show_speakers']: self._add_audio_system(fig, room_specs, colors, recommendations)
        if viz_config['room_elements']['show_control']: self._add_control_system(fig, room_specs, colors)
        if viz_config['room_elements']['show_whiteboard']: self._add_whiteboard(fig, room_specs, colors)
        if viz_config['room_elements']['show_credenza']: self._add_credenza(fig, room_specs, colors)
        if viz_config['advanced_features']['show_measurements']: self._add_measurements(fig, room_specs)
        if viz_config['advanced_features']['show_zones']: self._add_coverage_zones(fig, room_specs)
        if viz_config['advanced_features']['show_cable_paths']: self._add_cable_management(fig, room_specs, colors)
        if viz_config['advanced_features']['show_network']: self._add_network_points(fig, room_specs)

        # camera & layout
        self._update_camera_view(fig, room_specs, viz_config['style_options']['view_angle'])
        self._update_layout(fig, room_specs)

        return fig

    def _add_room_structure(self, fig, specs, colors, lighting):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        # lighting dict for surface - valid numeric floats
        lighting_conf = dict(ambient=float(lighting.get('ambient', 0.5)), diffuse=float(lighting.get('diffuse', 0.6)), specular=0.3, roughness=0.8, fresnel=0.2)

        # floor
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[0.0, 0.0], [width, width]])
        z = np.array([[0.0, 0.0], [0.0, 0.0]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['floor']], [1, colors['floor']]], showscale=False, lighting=lighting_conf, name='Floor'))

        # back wall (x=0)
        x = np.array([[0.0, 0.0], [0.0, 0.0]])
        y = np.array([[0.0, width], [0.0, width]])
        z = np.array([[0.0, 0.0], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.95, name='Back Wall'))

        # left wall (y=0)
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[0.0, 0.0], [0.0, 0.0]])
        z = np.array([[0.0, 0.0], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.9, name='Left Wall'))

        # right wall (y=width)
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[width, width], [width, width]])
        z = np.array([[0.0, 0.0], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.9, name='Right Wall'))

        # ceiling
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[0.0, 0.0], [width, width]])
        z = np.array([[height, height], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.95, name='Ceiling'))

    def _add_table(self, fig, specs, colors, table_style):
        length, width = float(specs['length']), float(specs['width'])
        t_h = 0.75
        tx = length * 0.55
        ty = width * 0.5
        t_len = min(length * 0.6, 5.0)
        t_w = min(width * 0.4, 2.0)
        # simple rectangular top as surface
        x, y = np.meshgrid(np.linspace(tx - t_len / 2, tx + t_len / 2, 2), np.linspace(ty - t_w / 2, ty + t_w / 2, 2))
        z = np.full_like(x, t_h)
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Table'))

    def _add_seating(self, fig, specs, colors, chair_style):
        length, width, capacity = float(specs['length']), float(specs['width']), int(specs.get('capacity', 6))
        t_len = min(length * 0.6, 5.0)
        t_w = min(width * 0.4, 2.0)
        tcx = length * 0.55
        tcy = width * 0.5
        chairs_per_side = min(6, capacity // 2) or 1
        for i in range(chairs_per_side):
            x_pos = tcx - t_len / 2 + ((i + 0.5) * t_len / chairs_per_side)
            for offset in [-1, 1]:
                y_pos = tcy + offset * (t_w / 2 + 0.45)
                # lightweight marker to represent chair
                fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[0.45], mode='markers', marker=dict(size=8, color=colors['accent']), name='Chair'))

    def _add_display_system(self, fig, specs, colors, recommendations):
        width, height = float(specs['width']), float(specs['ceiling_height'])
        screen_w = min(3.5, width * 0.6)
        screen_h = screen_w * 9 / 16
        sy = width / 2
        sz = height * 0.5
        # simple panel as mesh
        x = [0.05, 0.05, 0.05, 0.05]
        y = [sy - screen_w / 2, sy + screen_w / 2, sy + screen_w / 2, sy - screen_w / 2]
        z = [sz - screen_h / 2, sz - screen_h / 2, sz + screen_h / 2, sz + screen_h / 2]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color=colors['screen'], name='Display', opacity=1.0))

    def _add_camera_system(self, fig, specs, colors, recommendations):
        width, height = float(specs['width']), float(specs['ceiling_height'])
        cam_z = height * 0.5
        fig.add_trace(go.Scatter3d(x=[0.08], y=[width / 2], z=[cam_z], mode='markers', marker=dict(size=6, color=colors['screen'], symbol='diamond'), name='Camera'))

    def _add_lighting_system(self, fig, specs, colors, lighting):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        light_color = lighting.get('color', 'rgb(255,255,224)')
        for i in range(2):
            for j in range(3):
                x = length * (i + 1) / 3
                y = width * (j + 1) / 4
                z = height - 0.05
                fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(size=6, color=light_color), name='Ceiling Light'))

    def _add_audio_system(self, fig, specs, colors, recommendations):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        for i in [0.25, 0.75]:
            for j in [0.25, 0.75]:
                fig.add_trace(go.Scatter3d(x=[length * i], y=[width * j], z=[height - 0.1], mode='markers', marker=dict(size=6, color=colors['metal']), name='Speaker'))

    def _add_control_system(self, fig, specs, colors):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        fig.add_trace(go.Scatter3d(x=[length - 0.3], y=[width * 0.8], z=[1.0], mode='markers', marker=dict(size=8, color=colors['accent']), name='Control Panel'))

    def _add_whiteboard(self, fig, specs, colors):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        wb_w = min(2.0, width * 0.4)
        wb_h = wb_w * 0.7
        x = [length - 0.05] * 4
        y = [width / 2 - wb_w / 2, width / 2 + wb_w / 2, width / 2 + wb_w / 2, width / 2 - wb_w / 2]
        z = [height * 0.3, height * 0.3, height * 0.3 + wb_h, height * 0.3 + wb_h]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color=colors['glass'], opacity=0.85, name='Whiteboard'))

    def _add_credenza(self, fig, specs, colors):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        c_len = min(2.5, width * 0.5)
        c_depth = 0.45
        c_h = 0.8
        x = np.array([[0.1, 0.1 + c_depth], [0.1, 0.1 + c_depth]])
        y = np.array([[width / 2 - c_len / 2, width / 2 - c_len / 2], [width / 2 + c_len / 2, width / 2 + c_len / 2]])
        z = np.full_like(x, c_h)
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Credenza'))

    def _add_measurements(self, fig, specs):
        length, width = float(specs['length']), float(specs['width'])
        fig.add_trace(go.Scatter3d(x=[0, length], y=[width * 1.05, width * 1.05], z=[0, 0], mode='lines+text', text=["", f"{length:.1f}m"], line=dict(color='black', width=2), textposition="top right", name="Length"))
        fig.add_trace(go.Scatter3d(x=[length * 1.05, length * 1.05], y=[0, width], z=[0, 0], mode='lines+text', text=["", f"{width:.1f}m"], line=dict(color='black', width=2), textposition="middle right", name="Width"))

    def _add_coverage_zones(self, fig, specs):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        x = [0.08, length * 0.8, length * 0.8]
        y = [width / 2, width * 0.1, width * 0.9]
        z = [height * 0.6, 0, 0]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color='rgba(52,152,219,0.6)', opacity=0.12, name='Coverage'))

    def _add_cable_management(self, fig, specs, colors):
        length, width = float(specs['length']), float(specs['width'])
        tx = length * 0.55
        ty = width * 0.5
        fig.add_trace(go.Scatter3d(x=[0.08, tx], y=[width / 2, ty], z=[0.1, 0.1], mode='lines', line=dict(color=colors['accent'], width=3, dash='dash'), name='Cable'))

    def _add_network_points(self, fig, specs):
        length, width = float(specs['length']), float(specs['width'])
        fig.add_trace(go.Scatter3d(x=[0.2, length - 0.2], y=[0.2, width - 0.2], z=[0.2, 0.2], mode='markers', marker=dict(size=6, color='green'), name='Network'))

    def _update_camera_view(self, fig, specs, view_angle):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        camera_angles = {
            'perspective': dict(eye=dict(x=length * 1.2, y= - width * 1.2, z=height), center=dict(x=length/3, y=width/3, z=0)),
            'top': dict(eye=dict(x=length/2, y=width/2, z=height + 3), center=dict(x=length/2, y=width/2, z=0)),
            'front': dict(eye=dict(x=length + 3, y=width/2, z=height * 0.6), center=dict(x=length/2, y=width/2, z=height/2)),
            'side': dict(eye=dict(x=length/2, y=width + 3, z=height * 0.6), center=dict(x=length/2, y=width/2, z=height/2)),
            'corner': dict(eye=dict(x=length * 1.5, y=width * 1.5, z=height * 1.2), center=dict(x=length/2, y=width/2, z=height/3))
        }
        cam = camera_angles.get(view_angle, camera_angles['corner'])
        fig.update_layout(scene=dict(camera=cam))

    def _update_layout(self, fig, specs):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        fig.update_layout(
            autosize=True,
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=(width / length) if length != 0 else 1, z=0.6),
                xaxis=dict(range=[-0.2, length + 0.2], title="Length (m)", showgrid=False),
                yaxis=dict(range=[-0.2, width + 0.2], title="Width (m)", showgrid=False),
                zaxis=dict(range=[0, height + 0.5], title="Height (m)", showgrid=False),
                bgcolor='rgba(245,247,250,1)'
            ),
            title=dict(text="Interactive Conference Room Design", x=0.5),
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )

    @staticmethod
    def create_equipment_layout_2d(room_specs, recommendations):
        fig = go.Figure()
        length, width = float(room_specs['length']), float(room_specs['width'])

        fig.add_shape(type="rect", x0=-0.2, y0=-0.2, x1=length + 0.2, y1=width + 0.2,
                      line=dict(color="rgba(150,150,150,0.4)", width=2),
                      fillcolor="rgba(240,240,240,0.3)", layer='below')
        fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width,
                      line=dict(color="rgb(70,70,70)", width=2), fillcolor="rgba(255,255,255,1)")

        # windows
        windows_pct = room_specs.get('environment', {}).get('windows', 0)
        if windows_pct:
            sections = max(1, int(windows_pct / 20))
            w_h = width / (sections * 2)
            for i in range(sections):
                y0 = (i * 2) * w_h + 0.1
                fig.add_shape(type="rect", x0=length - 0.01, y0=y0, x1=length, y1=y0 + w_h * 0.8,
                              line=dict(color="rgba(120,160,255,0.9)", width=1), fillcolor="rgba(200,230,255,0.6)")

        # screen
        screen_w = min(width * 0.6, 3.5)
        s0 = (width - screen_w) / 2
        fig.add_shape(type="rect", x0=0, y0=s0, x1=0.12, y1=s0 + screen_w, line=dict(color="rgb(50,50,50)", width=2),
                      fillcolor="rgb(60,60,60)")

        # table
        table_len = min(length * 0.7, 4.5)
        table_w = min(width * 0.4, 1.5)
        tx, ty = length * 0.6, width * 0.5
        fig.add_shape(type="rect", x0=tx - table_len / 2, y0=ty - table_w / 2, x1=tx + table_len / 2, y1=ty + table_w / 2,
                      line=dict(color="rgb(120,85,60)", width=2), fillcolor="rgb(139,115,85)")

        # chairs
        cap = min(int(room_specs.get('capacity', 6)), 12)
        chairs_side = min(6, cap // 2) or 1
        chairs = []
        for i in range(chairs_side):
            x_pos = tx - table_len / 2 + ((i + 1) * table_len / (chairs_side + 1))
            chairs.extend([(x_pos, ty - table_w / 2 - 0.35), (x_pos, ty + table_w / 2 + 0.35)])

        for x, y in chairs[:cap]:
            fig.add_shape(type="circle", x0=x - 0.22, y0=y - 0.22, x1=x + 0.22, y1=y + 0.22,
                          line=dict(color="rgba(0,0,0,0.15)"), fillcolor="rgba(70,130,180,0.9)")

        # annotations and layout
        fig.update_xaxes(showgrid=False, zeroline=False, range=[-1, length + 1], title_text="Length (m)")
        fig.update_yaxes(showgrid=False, zeroline=False, range=[-1, width + 1], title_text="Width (m)", scaleanchor="x", scaleratio=1)
        fig.update_layout(height=600, title="Enhanced Floor Plan", plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                          margin=dict(t=80, b=40, l=40, r=40))
        fig.add_annotation(x=length / 2, y=-0.6, text=f"{length:.1f} m", showarrow=False)
        fig.add_annotation(x=-0.6, y=width / 2, text=f"{width:.1f} m", showarrow=False, textangle=-90)

        return fig

    @staticmethod
    def create_cost_breakdown_chart(recommendations):
        categories = ['Display', 'Camera', 'Audio', 'Control', 'Lighting']
        costs = [recommendations['display']['price'], recommendations['camera']['price'], recommendations['audio']['price'], recommendations['control']['price'], recommendations['lighting']['price']]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        fig = go.Figure(data=[go.Bar(x=categories, y=costs, marker_color=colors, text=[f"${cost:,.0f}" for cost in costs], textposition='auto')])
        fig.update_layout(title_text="Investment Breakdown by Category", height=380, paper_bgcolor='white', plot_bgcolor='white')
        return fig

    @staticmethod
    def create_feature_comparison_radar(recommendations, alternatives):
        categories = ['Performance', 'Features', 'Reliability', 'Integration', 'Value']
        current_scores = [4.5, 4.2, 4.6, 4.4, 4.3]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=current_scores + [current_scores[0]], theta=categories + [categories[0]], fill='toself', name='Recommended', line_color='#3498db', fillcolor='rgba(52,152,219,0.2)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), height=380, paper_bgcolor='white', plot_bgcolor='white')
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
        st.markdown('<div class="premium-card" style="margin-top: -30px;"><h2>🎛️ Room Configuration</h2></div>', unsafe_allow_html=True)

        db = EnhancedProductDatabase()
        template = st.selectbox("Room Template", list(db.room_templates.keys()), help="Choose a template to start.")
        template_info = db.room_templates[template]

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
        room_purpose = st.multiselect("Primary Activities", ["Video Conferencing", "Presentations", "Training", "Board Meetings", "Collaborative Work", "Hybrid Meetings", "Social Events"], default=["Video Conferencing", "Presentations"])
        acoustic_features = st.multiselect("Acoustic Considerations", ["Sound Absorption Needed", "Echo Control Required", "External Noise Issues", "Speech Privacy Important", "Music Playback Required"])

        st.markdown("##### 🎛️ Environmental Controls")
        env_controls = st.multiselect("Control Systems", ["Automated Lighting", "Motorized Shades", "Climate Control", "Air Quality Monitoring", "Occupancy Sensors", "Daylight Harvesting"])

        st.markdown("##### 🎨 Ambiance & Design")
        color_scheme_temp = st.select_slider("Color Temperature", options=["Warm", "Neutral Warm", "Neutral", "Neutral Cool", "Cool"], value="Neutral")
        design_style = st.selectbox("Interior Design Style", ["Modern Corporate", "Executive", "Creative/Tech", "Traditional", "Industrial", "Minimalist"])

        st.markdown("##### ♿ Accessibility Features")
        accessibility = st.multiselect("Accessibility Requirements", ["Wheelchair Access", "Hearing Loop System", "High Contrast Displays", "Voice Control", "Adjustable Furniture", "Braille Signage"])

        st.markdown("---")
        st.subheader("💰 Budget & Brands")
        budget_tier = st.selectbox("Budget Tier", ['Budget', 'Professional', 'Premium'], index=1)
        preferred_brands = st.multiselect("Preferred Brands", ['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'])

        st.subheader("✨ Special Features")
        special_features = st.multiselect("Required Features", ['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 'Noise Reduction', 'Circadian Lighting', 'AI Analytics'])

        st.markdown("---")
        st.sidebar.markdown("### 🎨 Visualization Options")

        with st.sidebar.expander("Room Elements", expanded=True):
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

        with st.sidebar.expander("Style Options", expanded=False):
            style_config = {
                'chair_style': st.selectbox("Chair Style", ['modern', 'executive', 'training', 'casual']),
                'table_style': st.selectbox("Table Style", ['rectangular', 'oval', 'boat-shaped', 'modular']),
                'color_scheme': st.selectbox("Color Scheme", ['professional', 'modern', 'classic', 'warm', 'cool']),
                'lighting_mode': st.selectbox("Lighting Mode", ['day', 'evening', 'presentation', 'video conference']),
                'view_angle': st.selectbox("View Angle", ['perspective', 'top', 'front', 'side', 'corner'])
            }

        with st.sidebar.expander("Advanced Features", expanded=False):
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
                'capacity': capacity, 'environment': environment_config, 'special_requirements': special_features
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
        with col1:
            st.markdown(f'<div class="metric-card"><h3>${total_cost:,.0f}</h3><p>Total Investment</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{int(recommendations["confidence_score"]*100)}%</h3><p>AI Confidence</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{room_specs["length"]}m × {room_specs["width"]}m</h3><p>Room Size</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{room_specs["capacity"]}</h3><p>Capacity</p></div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📊 Analysis", "🎨 Visualization", "🔄 Alternatives", "📋 Report"])

        with tab1:
            st.subheader("AI-Powered Equipment Recommendations")
            colA, colB = st.columns(2)
            with colA:
                for cat, icon in [('display', '📺'), ('camera', '🎥'), ('audio', '🔊')]:
                    rec = recommendations[cat]
                    st.markdown(f"#### {icon} {cat.title()} System")
                    st.markdown(f"""<div class="feature-card"><h4>{rec['model']}</h4><p><strong>Price:</strong> ${rec['price']:,} | <strong>Rating:</strong> ⭐ {rec['rating']}/5.0</p><p><strong>Specs:</strong> {rec['specs']}</p></div>""", unsafe_allow_html=True)
            with colB:
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
            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("#### Room Characteristics")
                analysis = recommendations['room_analysis']
                st.markdown(f"""<div class="comparison-card"><p><strong>Category:</strong> {analysis['size_category']}</p><p><strong>Shape:</strong> {analysis['shape_analysis']}</p><p><strong>Acoustics:</strong> Reverb is {analysis['acoustic_properties']['reverb_category']}, Treatment needed: {'Yes' if analysis['acoustic_properties']['treatment_needed'] else 'No'}</p><p><strong>Lighting Challenges:</strong> {', '.join(analysis.get('lighting_challenges', []))}</p></div>""", unsafe_allow_html=True)
            with colB:
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
                    colA, colB, colC = st.columns(3)
                    cols = [colA, colB, colC]
                    for i, cat in enumerate(['displays', 'cameras', 'audio']):
                        if cat in alt_config:
                            with cols[i]:
                                name, info = alt_config[cat]
                                st.markdown(f"""<div class="comparison-card"><strong>{cat.title()}:</strong> {name}<br>${info['price']:,} | ⭐ {info['rating']}/5.0</div>""", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            if recommendations.get('upgrade_path'):
                upgrade = recommendations['upgrade_path'][0]
                smart_plan = recommender._generate_smart_upgrade_plan(room_specs, st.session_state.budget_tier, upgrade['estimated_cost'])
                st.markdown(f"""
                <div class="premium-card">
                    <h3>💡 Upgrade Strategy Overview to {upgrade['tier']} Tier</h3>
                    <p>A structured approach to achieving premium AV capabilities while maintaining operational continuity.</p>
                    <p><strong>Total Add. Investment:</strong> ${smart_plan['total_investment']:,.0f} | <strong>Est. Monthly:</strong> ${smart_plan['monthly_investment']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                cols = st.columns(len(smart_plan['phases']))
                for i, (phase_name, phase_info) in enumerate(smart_plan['phases'].items()):
                    with cols[i]:
                        st.markdown(f"""<div class="feature-card"><h4>{phase_name}</h4><p><strong>Budget:</strong> ${phase_info['budget']:,.0f}</p><p><strong>Focus:</strong> {phase_info['focus']}</p><ul style="font-size:0.9em;">{''.join([f'<li>{p}</li>' for p in phase_info['priorities']])}</ul></div>""", unsafe_allow_html=True)

        with tab5:
            st.subheader("Professional Report Summary")
            st.markdown(f"""<div class="premium-card"><h3>Executive Summary</h3><p>AI-generated AV solution for a <strong>{room_specs['template']}</strong> ({room_specs['length']}m × {room_specs['width']}m) for <strong>{room_specs['capacity']} people</strong>.</p><p><strong>Total Investment:</strong> ${total_cost:,} | <strong>Confidence:</strong> {int(recommendations['confidence_score']*100)}% | <strong>Recommended Tier:</strong> {st.session_state.budget_tier}</p></div>""", unsafe_allow_html=True)
            st.markdown("#### Detailed Equipment Specifications")
            specs_data = [{'Category': cat.title(), 'Model': recommendations[cat]['model'], 'Price': f"${recommendations[cat]['price']:,}", 'Rating': f"{recommendations[cat]['rating']}/5.0", 'Brand': recommendations[cat].get('brand', '')} for cat in ['display', 'camera', 'audio', 'control', 'lighting']]
            st.dataframe(pd.DataFrame(specs_data), use_container_width=True)

    else:
        st.markdown('''<div class="premium-card" style="text-align: center; padding: 50px;"><h2>🚀 Welcome to AI Room Configurator Pro Max</h2><p style="font-size: 18px;">Configure your room in the sidebar to generate an intelligent AV design.</p></div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
```# filepath: c:\Users\Adarsh\Desktop\allwave\pages\1_Interactive_Configurator.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any
import random

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
        --sidebar-bg: #0f172a;
        --sidebar-text: #f3f4f6;
        --text-color: #0b1220;
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
    .stSidebar .css-1d391kg, .css-1d391kg {
        background: linear-gradient(180deg, var(--sidebar-bg), #0b1220) !important;
        color: var(--sidebar-text) !important;
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
</style>
""", unsafe_allow_html=True)


# --- Comprehensive Product Database ---
class EnhancedProductDatabase:
    def __init__(self):
        # (product lists omitted for brevity in this file preview — kept in memory)
        self.products = {
            'displays': {
                'Budget': {
                    'BenQ IRP55': {'price': 1299, 'specs': '55" 4K Interactive Display, 20-point touch', 'rating': 4.2, 'brand': 'BenQ'},
                    'LG 75UP8000PUA': {'price': 1899, 'specs': '75" 4K LED, webOS, ThinQ AI', 'rating': 4.4, 'brand': 'LG'},
                    'Samsung QB65R': {'price': 2499, 'specs': '65" 4K QLED Business Display', 'rating': 4.5, 'brand': 'Samsung'}
                },
                'Professional': {
                    'Sharp/NEC 100\" 4K Display': {'price': 15999, 'specs': '100" 4K UHD, 500 nits, 24/7 Operation', 'rating': 4.7, 'brand': 'Sharp/NEC'},
                    'Sony BRAVIA FW-85BZ40H': {'price': 8999, 'specs': '85" 4K Pro Display, Android TV', 'rating': 4.6, 'brand': 'Sony'},
                    'Planar UltraRes X Series': {'price': 12999, 'specs': '86" 4K Multi-touch Display', 'rating': 4.5, 'brand': 'Planar'}
                },
                'Premium': {
                    'LG MAGNIT 136\"': {'price': 75000, 'specs': 'MicroLED, 4K, AI-powered processing, Cable-less', 'rating': 4.9, 'brand': 'LG'},
                    'Samsung The Wall 146\"': {'price': 99999, 'specs': 'MicroLED, 4K, 0.8mm Pixel Pitch, AI Upscaling', 'rating': 5.0, 'brand': 'Samsung'},
                    'Sony Crystal LED 220\"': {'price': 150000, 'specs': 'Crystal LED, 4K+, Seamless Modular Design', 'rating': 5.0, 'brand': 'Sony'}
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
            'Huddle Room (2-6 people)': {'typical_size': (3, 4), 'capacity_range': (2, 6), 'recommended_tier': 'Budget', 'typical_usage': 'Quick meetings, brainstorming'},
            'Small Conference (6-12 people)': {'typical_size': (4, 6), 'capacity_range': (6, 12), 'recommended_tier': 'Professional', 'typical_usage': 'Team meetings, presentations'},
            'Large Conference (12-20 people)': {'typical_size': (6, 10), 'capacity_range': (12, 20), 'recommended_tier': 'Professional', 'typical_usage': 'Department meetings, training'},
            'Boardroom (8-16 people)': {'typical_size': (5, 8), 'capacity_range': (8, 16), 'recommended_tier': 'Premium', 'typical_usage': 'Executive meetings, board meetings'},
            'Training Room (20-50 people)': {'typical_size': (8, 12), 'capacity_range': (20, 50), 'recommended_tier': 'Professional', 'typical_usage': 'Training, workshops, seminars'},
            'Auditorium (50+ people)': {'typical_size': (12, 20), 'capacity_range': (50, 200), 'recommended_tier': 'Premium', 'typical_usage': 'Large presentations, events'}
        }


# --- Simple Recommender implementation so app runs ---
class MaximizedAVRecommender:
    def __init__(self):
        self.db = EnhancedProductDatabase()

    def _pick_product(self, category: str, tier: str):
        items = self.db.products.get(category + 's' if category != 'control' else 'control_systems', {})
        # Normalized tier keys in db sometimes differ; try both
        tier = tier if tier in items else tier
        tier_dict = items.get(tier, {})
        if not tier_dict:
            # fallback: choose any
            for t in items:
                tier_dict = items[t]
                break
        if not tier_dict:
            return {'model': f'Generic {category}', 'price': 1000, 'specs': '', 'rating': 4.0, 'brand': 'Generic'}
        name = random.choice(list(tier_dict.keys()))
        info = tier_dict[name]
        return {'model': name, 'price': info['price'], 'specs': info['specs'], 'rating': info['rating'], 'brand': info.get('brand', 'Unknown')}

    def get_comprehensive_recommendations(self, room_specs: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        tier = user_preferences.get('budget_tier', 'Professional')
        display = self._pick_product('display', tier)
        camera = self._pick_product('camera', tier)
        audio = self._pick_product('audio', tier)
        control = self._pick_product('control', tier)
        lighting = self._pick_product('lighting', tier)

        accessories = []
        if 'Digital Whiteboard' in user_preferences.get('special_features', []):
            accessories.append({'item': 'Digital Whiteboard', 'model': 'Basic WB 65"', 'price': 2499, 'necessity': 'Optional'})

        # Simple analysis
        room_analysis = {
            'size_category': 'Medium' if room_specs['length'] * room_specs['width'] < 40 else 'Large',
            'shape_analysis': 'Rectangular',
            'acoustic_properties': {'reverb_category': 'Moderate', 'treatment_needed': True},
            'lighting_challenges': ['Glare from windows'] if room_specs.get('environment', {}).get('natural_light') in ['High', 'Very High'] else []
        }

        alternatives = {}
        # Provide simple alternatives for display/camera/audio
        alternatives['Professional'] = {'displays': (list(self.db.products['displays']['Professional'].keys())[0], list(self.db.products['displays']['Professional'].values())[0]),
                                       'cameras': (list(self.db.products['cameras']['Professional'].keys())[0], list(self.db.products['cameras']['Professional'].values())[0]),
                                       'audio': (list(self.db.products['audio']['Professional'].keys())[0], list(self.db.products['audio']['Professional'].values())[0])}

        upgrade_path = [{'tier': 'Premium', 'estimated_cost': sum([display['price'], camera['price'], audio['price']]) * 0.5}]

        confidence_score = 0.85

        return {
            'display': display, 'camera': camera, 'audio': audio, 'control': control, 'lighting': lighting,
            'accessories': accessories, 'room_analysis': room_analysis, 'alternatives': alternatives,
            'upgrade_path': upgrade_path, 'confidence_score': confidence_score
        }

    def _generate_smart_upgrade_plan(self, room_specs, current_tier, estimated_cost):
        # Simple phased plan
        phases = {
            'Phase 1': {'budget': estimated_cost * 0.4, 'focus': 'Core AV (display, camera)', 'priorities': ['Upgrade display', 'Upgrade camera']},
            'Phase 2': {'budget': estimated_cost * 0.35, 'focus': 'Audio & Control', 'priorities': ['Ceiling mics', 'Control system']},
            'Phase 3': {'budget': estimated_cost * 0.25, 'focus': 'Lighting & Finishing', 'priorities': ['Lighting control', 'Acoustic panels']}
        }
        total = sum(p['budget'] for p in phases.values())
        return {'phases': phases, 'total_investment': total, 'monthly_investment': total / 12}


# --- Visualization Engine (robust and validated) ---
class EnhancedVisualizationEngine:
    def __init__(self):
        self.color_schemes = {
            'professional': {'wall': '#F0F2F5', 'floor': '#E5E9F0', 'accent': '#4A90E2', 'wood': '#8B5E3C', 'screen': '#1A1A1A', 'metal': '#B8B8B8', 'glass': '#FFFFFF'},
            'modern': {'wall': '#FFFFFF', 'floor': '#F8F9FA', 'accent': '#2563EB', 'wood': '#A0522D', 'screen': '#000000', 'metal': '#C0C0C0', 'glass': '#E8F0FE'},
            'classic': {'wall': '#F5F5DC', 'floor': '#DEB887', 'accent': '#800000', 'wood': '#8B4513', 'screen': '#2F4F4F', 'metal': '#CD853F', 'glass': '#F0F8FF'},
            'warm': {'wall': '#FEF3E0', 'floor': '#F3D1A3', 'accent': '#E53E3E', 'wood': '#BF5B32', 'screen': '#2D3748', 'metal': '#D69E2E', 'glass': '#FFF5EB'},
            'cool': {'wall': '#E6FFFA', 'floor': '#B2F5EA', 'accent': '#38B2AC', 'wood': '#718096', 'screen': '#1A202C', 'metal': '#A0AEC0', 'glass': '#EBF8FF'}
        }
        self.lighting_modes = {
            'day': {'ambient': 0.8, 'diffuse': 1.0, 'color': 'rgb(255, 255, 224)'},
            'evening': {'ambient': 0.4, 'diffuse': 0.7, 'color': 'rgb(255, 228, 181)'},
            'presentation': {'ambient': 0.3, 'diffuse': 0.5, 'color': 'rgb(200, 200, 255)'},
            'video conference': {'ambient': 0.6, 'diffuse': 0.8, 'color': 'rgb(220, 220, 255)'}
        }

    def create_3d_room_visualization(self, room_specs, recommendations, viz_config):
        fig = go.Figure()
        colors = self.color_schemes.get(viz_config['style_options']['color_scheme'], self.color_schemes['professional'])
        lighting = self.lighting_modes.get(viz_config['style_options']['lighting_mode'], self.lighting_modes['day'])

        # build scene
        self._add_room_structure(fig, room_specs, colors, lighting)

        # elements
        if viz_config['room_elements']['show_table']: self._add_table(fig, room_specs, colors, viz_config['style_options']['table_style'])
        if viz_config['room_elements']['show_chairs']: self._add_seating(fig, room_specs, colors, viz_config['style_options']['chair_style'])
        if viz_config['room_elements']['show_displays']: self._add_display_system(fig, room_specs, colors, recommendations)
        if viz_config['room_elements']['show_cameras']: self._add_camera_system(fig, room_specs, colors, recommendations)
        if viz_config['room_elements']['show_lighting']: self._add_lighting_system(fig, room_specs, colors, lighting)
        if viz_config['room_elements']['show_speakers']: self._add_audio_system(fig, room_specs, colors, recommendations)
        if viz_config['room_elements']['show_control']: self._add_control_system(fig, room_specs, colors)
        if viz_config['room_elements']['show_whiteboard']: self._add_whiteboard(fig, room_specs, colors)
        if viz_config['room_elements']['show_credenza']: self._add_credenza(fig, room_specs, colors)
        if viz_config['advanced_features']['show_measurements']: self._add_measurements(fig, room_specs)
        if viz_config['advanced_features']['show_zones']: self._add_coverage_zones(fig, room_specs)
        if viz_config['advanced_features']['show_cable_paths']: self._add_cable_management(fig, room_specs, colors)
        if viz_config['advanced_features']['show_network']: self._add_network_points(fig, room_specs)

        # camera & layout
        self._update_camera_view(fig, room_specs, viz_config['style_options']['view_angle'])
        self._update_layout(fig, room_specs)

        return fig

    def _add_room_structure(self, fig, specs, colors, lighting):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        # lighting dict for surface - valid numeric floats
        lighting_conf = dict(ambient=float(lighting.get('ambient', 0.5)), diffuse=float(lighting.get('diffuse', 0.6)), specular=0.3, roughness=0.8, fresnel=0.2)

        # floor
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[0.0, 0.0], [width, width]])
        z = np.array([[0.0, 0.0], [0.0, 0.0]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['floor']], [1, colors['floor']]], showscale=False, lighting=lighting_conf, name='Floor'))

        # back wall (x=0)
        x = np.array([[0.0, 0.0], [0.0, 0.0]])
        y = np.array([[0.0, width], [0.0, width]])
        z = np.array([[0.0, 0.0], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.95, name='Back Wall'))

        # left wall (y=0)
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[0.0, 0.0], [0.0, 0.0]])
        z = np.array([[0.0, 0.0], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.9, name='Left Wall'))

        # right wall (y=width)
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[width, width], [width, width]])
        z = np.array([[0.0, 0.0], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.9, name='Right Wall'))

        # ceiling
        x = np.array([[0.0, length], [0.0, length]])
        y = np.array([[0.0, 0.0], [width, width]])
        z = np.array([[height, height], [height, height]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, lighting=lighting_conf, opacity=0.95, name='Ceiling'))

    def _add_table(self, fig, specs, colors, table_style):
        length, width = float(specs['length']), float(specs['width'])
        t_h = 0.75
        tx = length * 0.55
        ty = width * 0.5
        t_len = min(length * 0.6, 5.0)
        t_w = min(width * 0.4, 2.0)
        # simple rectangular top as surface
        x, y = np.meshgrid(np.linspace(tx - t_len / 2, tx + t_len / 2, 2), np.linspace(ty - t_w / 2, ty + t_w / 2, 2))
        z = np.full_like(x, t_h)
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Table'))

    def _add_seating(self, fig, specs, colors, chair_style):
        length, width, capacity = float(specs['length']), float(specs['width']), int(specs.get('capacity', 6))
        t_len = min(length * 0.6, 5.0)
        t_w = min(width * 0.4, 2.0)
        tcx = length * 0.55
        tcy = width * 0.5
        chairs_per_side = min(6, capacity // 2) or 1
        for i in range(chairs_per_side):
            x_pos = tcx - t_len / 2 + ((i + 0.5) * t_len / chairs_per_side)
            for offset in [-1, 1]:
                y_pos = tcy + offset * (t_w / 2 + 0.45)
                # lightweight marker to represent chair
                fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[0.45], mode='markers', marker=dict(size=8, color=colors['accent']), name='Chair'))

    def _add_display_system(self, fig, specs, colors, recommendations):
        width, height = float(specs['width']), float(specs['ceiling_height'])
        screen_w = min(3.5, width * 0.6)
        screen_h = screen_w * 9 / 16
        sy = width / 2
        sz = height * 0.5
        # simple panel as mesh
        x = [0.05, 0.05, 0.05, 0.05]
        y = [sy - screen_w / 2, sy + screen_w / 2, sy + screen_w / 2, sy - screen_w / 2]
        z = [sz - screen_h / 2, sz - screen_h / 2, sz + screen_h / 2, sz + screen_h / 2]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color=colors['screen'], name='Display', opacity=1.0))

    def _add_camera_system(self, fig, specs, colors, recommendations):
        width, height = float(specs['width']), float(specs['ceiling_height'])
        cam_z = height * 0.5
        fig.add_trace(go.Scatter3d(x=[0.08], y=[width / 2], z=[cam_z], mode='markers', marker=dict(size=6, color=colors['screen'], symbol='diamond'), name='Camera'))

    def _add_lighting_system(self, fig, specs, colors, lighting):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        light_color = lighting.get('color', 'rgb(255,255,224)')
        for i in range(2):
            for j in range(3):
                x = length * (i + 1) / 3
                y = width * (j + 1) / 4
                z = height - 0.05
                fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(size=6, color=light_color), name='Ceiling Light'))

    def _add_audio_system(self, fig, specs, colors, recommendations):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        for i in [0.25, 0.75]:
            for j in [0.25, 0.75]:
                fig.add_trace(go.Scatter3d(x=[length * i], y=[width * j], z=[height - 0.1], mode='markers', marker=dict(size=6, color=colors['metal']), name='Speaker'))

    def _add_control_system(self, fig, specs, colors):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        fig.add_trace(go.Scatter3d(x=[length - 0.3], y=[width * 0.8], z=[1.0], mode='markers', marker=dict(size=8, color=colors['accent']), name='Control Panel'))

    def _add_whiteboard(self, fig, specs, colors):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        wb_w = min(2.0, width * 0.4)
        wb_h = wb_w * 0.7
        x = [length - 0.05] * 4
        y = [width / 2 - wb_w / 2, width / 2 + wb_w / 2, width / 2 + wb_w / 2, width / 2 - wb_w / 2]
        z = [height * 0.3, height * 0.3, height * 0.3 + wb_h, height * 0.3 + wb_h]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color=colors['glass'], opacity=0.85, name='Whiteboard'))

    def _add_credenza(self, fig, specs, colors):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        c_len = min(2.5, width * 0.5)
        c_depth = 0.45
        c_h = 0.8
        x = np.array([[0.1, 0.1 + c_depth], [0.1, 0.1 + c_depth]])
        y = np.array([[width / 2 - c_len / 2, width / 2 - c_len / 2], [width / 2 + c_len / 2, width / 2 + c_len / 2]])
        z = np.full_like(x, c_h)
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Credenza'))

    def _add_measurements(self, fig, specs):
        length, width = float(specs['length']), float(specs['width'])
        fig.add_trace(go.Scatter3d(x=[0, length], y=[width * 1.05, width * 1.05], z=[0, 0], mode='lines+text', text=["", f"{length:.1f}m"], line=dict(color='black', width=2), textposition="top right", name="Length"))
        fig.add_trace(go.Scatter3d(x=[length * 1.05, length * 1.05], y=[0, width], z=[0, 0], mode='lines+text', text=["", f"{width:.1f}m"], line=dict(color='black', width=2), textposition="middle right", name="Width"))

    def _add_coverage_zones(self, fig, specs):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        x = [0.08, length * 0.8, length * 0.8]
        y = [width / 2, width * 0.1, width * 0.9]
        z = [height * 0.6, 0, 0]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0], j=[1], k=[2], color='rgba(52,152,219,0.6)', opacity=0.12, name='Coverage'))

    def _add_cable_management(self, fig, specs, colors):
        length, width = float(specs['length']), float(specs['width'])
        tx = length * 0.55
        ty = width * 0.5
        fig.add_trace(go.Scatter3d(x=[0.08, tx], y=[width / 2, ty], z=[0.1, 0.1], mode='lines', line=dict(color=colors['accent'], width=3, dash='dash'), name='Cable'))

    def _add_network_points(self, fig, specs):
        length, width = float(specs['length']), float(specs['width'])
        fig.add_trace(go.Scatter3d(x=[0.2, length - 0.2], y=[0.2, width - 0.2], z=[0.2, 0.2], mode='markers', marker=dict(size=6, color='green'), name='Network'))

    def _update_camera_view(self, fig, specs, view_angle):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        camera_angles = {
            'perspective': dict(eye=dict(x=length * 1.2, y= - width * 1.2, z=height), center=dict(x=length/3, y=width/3, z=0)),
            'top': dict(eye=dict(x=length/2, y=width/2, z=height + 3), center=dict(x=length/2, y=width/2, z=0)),
            'front': dict(eye=dict(x=length + 3, y=width/2, z=height * 0.6), center=dict(x=length/2, y=width/2, z=height/2)),
            'side': dict(eye=dict(x=length/2, y=width + 3, z=height * 0.6), center=dict(x=length/2, y=width/2, z=height/2)),
            'corner': dict(eye=dict(x=length * 1.5, y=width * 1.5, z=height * 1.2), center=dict(x=length/2, y=width/2, z=height/3))
        }
        cam = camera_angles.get(view_angle, camera_angles['corner'])
        fig.update_layout(scene=dict(camera=cam))

    def _update_layout(self, fig, specs):
        length, width, height = float(specs['length']), float(specs['width']), float(specs['ceiling_height'])
        fig.update_layout(
            autosize=True,
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=(width / length) if length != 0 else 1, z=0.6),
                xaxis=dict(range=[-0.2, length + 0.2], title="Length (m)", showgrid=False),
                yaxis=dict(range=[-0.2, width + 0.2], title="Width (m)", showgrid=False),
                zaxis=dict(range=[0, height + 0.5], title="Height (m)", showgrid=False),
                bgcolor='rgba(245,247,250,1)'
            ),
            title=dict(text="Interactive Conference Room Design", x=0.5),
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )

    @staticmethod
    def create_equipment_layout_2d(room_specs, recommendations):
        fig = go.Figure()
        length, width = float(room_specs['length']), float(room_specs['width'])

        fig.add_shape(type="rect", x0=-0.2, y0=-0.2, x1=length + 0.2, y1=width + 0.2,
                      line=dict(color="rgba(150,150,150,0.4)", width=2),
                      fillcolor="rgba(240,240,240,0.3)", layer='below')
        fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width,
                      line=dict(color="rgb(70,70,70)", width=2), fillcolor="rgba(255,255,255,1)")

        # windows
        windows_pct = room_specs.get('environment', {}).get('windows', 0)
        if windows_pct:
            sections = max(1, int(windows_pct / 20))
            w_h = width / (sections * 2)
            for i in range(sections):
                y0 = (i * 2) * w_h + 0.1
                fig.add_shape(type="rect", x0=length - 0.01, y0=y0, x1=length, y1=y0 + w_h * 0.8,
                              line=dict(color="rgba(120,160,255,0.9)", width=1), fillcolor="rgba(200,230,255,0.6)")

        # screen
        screen_w = min(width * 0.6, 3.5)
        s0 = (width - screen_w) / 2
        fig.add_shape(type="rect", x0=0, y0=s0, x1=0.12, y1=s0 + screen_w, line=dict(color="rgb(50,50,50)", width=2),
                      fillcolor="rgb(60,60,60)")

        # table
        table_len = min(length * 0.7, 4.5)
        table_w = min(width * 0.4, 1.5)
        tx, ty = length * 0.6, width * 0.5
        fig.add_shape(type="rect", x0=tx - table_len / 2, y0=ty - table_w / 2, x1=tx + table_len / 2, y1=ty + table_w / 2,
                      line=dict(color="rgb(120,85,60)", width=2), fillcolor="rgb(139,115,85)")

        # chairs
        cap = min(int(room_specs.get('capacity', 6)), 12)
        chairs_side = min(6, cap // 2) or 1
        chairs = []
        for i in range(chairs_side):
            x_pos = tx - table_len / 2 + ((i + 1) * table_len / (chairs_side + 1))
            chairs.extend([(x_pos, ty - table_w / 2 - 0.35), (x_pos, ty + table_w / 2 + 0.35)])

        for x, y in chairs[:cap]:
            fig.add_shape(type="circle", x0=x - 0.22, y0=y - 0.22, x1=x + 0.22, y1=y + 0.22,
                          line=dict(color="rgba(0,0,0,0.15)"), fillcolor="rgba(70,130,180,0.9)")

        # annotations and layout
        fig.update_xaxes(showgrid=False, zeroline=False, range=[-1, length + 1], title_text="Length (m)")
        fig.update_yaxes(showgrid=False, zeroline=False, range=[-1, width + 1], title_text="Width (m)", scaleanchor="x", scaleratio=1)
        fig.update_layout(height=600, title="Enhanced Floor Plan", plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                          margin=dict(t=80, b=40, l=40, r=40))
        fig.add_annotation(x=length / 2, y=-0.6, text=f"{length:.1f} m", showarrow=False)
        fig.add_annotation(x=-0.6, y=width / 2, text=f"{width:.1f} m", showarrow=False, textangle=-90)

        return fig

    @staticmethod
    def create_cost_breakdown_chart(recommendations):
        categories = ['Display', 'Camera', 'Audio', 'Control', 'Lighting']
        costs = [recommendations['display']['price'], recommendations['camera']['price'], recommendations['audio']['price'], recommendations['control']['price'], recommendations['lighting']['price']]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        fig = go.Figure(data=[go.Bar(x=categories, y=costs, marker_color=colors, text=[f"${cost:,.0f}" for cost in costs], textposition='auto')])
        fig.update_layout(title_text="Investment Breakdown by Category", height=380, paper_bgcolor='white', plot_bgcolor='white')
        return fig

    @staticmethod
    def create_feature_comparison_radar(recommendations, alternatives):
        categories = ['Performance', 'Features', 'Reliability', 'Integration', 'Value']
        current_scores = [4.5, 4.2, 4.6, 4.4, 4.3]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=current_scores + [current_scores[0]], theta=categories + [categories[0]], fill='toself', name='Recommended', line_color='#3498db', fillcolor='rgba(52,152,219,0.2)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), height=380, paper_bgcolor='white', plot_bgcolor='white')
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
        st.markdown('<div class="premium-card" style="margin-top: -30px;"><h2>🎛️ Room Configuration</h2></div>', unsafe_allow_html=True)

        db = EnhancedProductDatabase()
        template = st.selectbox("Room Template", list(db.room_templates.keys()), help="Choose a template to start.")
        template_info = db.room_templates[template]

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
        room_purpose = st.multiselect("Primary Activities", ["Video Conferencing", "Presentations", "Training", "Board Meetings", "Collaborative Work", "Hybrid Meetings", "Social Events"], default=["Video Conferencing", "Presentations"])
        acoustic_features = st.multiselect("Acoustic Considerations", ["Sound Absorption Needed", "Echo Control Required", "External Noise Issues", "Speech Privacy Important", "Music Playback Required"])

        st.markdown("##### 🎛️ Environmental Controls")
        env_controls = st.multiselect("Control Systems", ["Automated Lighting", "Motorized Shades", "Climate Control", "Air Quality Monitoring", "Occupancy Sensors", "Daylight Harvesting"])

        st.markdown("##### 🎨 Ambiance & Design")
        color_scheme_temp = st.select_slider("Color Temperature", options=["Warm", "Neutral Warm", "Neutral", "Neutral Cool", "Cool"], value="Neutral")
        design_style = st.selectbox("Interior Design Style", ["Modern Corporate", "Executive", "Creative/Tech", "Traditional", "Industrial", "Minimalist"])

        st.markdown("##### ♿ Accessibility Features")
        accessibility = st.multiselect("Accessibility Requirements", ["Wheelchair Access", "Hearing Loop System", "High Contrast Displays", "Voice Control", "Adjustable Furniture", "Braille Signage"])

        st.markdown("---")
        st.subheader("💰 Budget & Brands")
        budget_tier = st.selectbox("Budget Tier", ['Budget', 'Professional', 'Premium'], index=1)
        preferred_brands = st.multiselect("Preferred Brands", ['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'])

        st.subheader("✨ Special Features")
        special_features = st.multiselect("Required Features", ['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 'Noise Reduction', 'Circadian Lighting', 'AI Analytics'])

        st.markdown("---")
        st.sidebar.markdown("### 🎨 Visualization Options")

        with st.sidebar.expander("Room Elements", expanded=True):
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

        with st.sidebar.expander("Style Options", expanded=False):
            style_config = {
                'chair_style': st.selectbox("Chair Style", ['modern', 'executive', 'training', 'casual']),
                'table_style': st.selectbox("Table Style", ['rectangular', 'oval', 'boat-shaped', 'modular']),
                'color_scheme': st.selectbox("Color Scheme", ['professional', 'modern', 'classic', 'warm', 'cool']),
                'lighting_mode': st.selectbox("Lighting Mode", ['day', 'evening', 'presentation', 'video conference']),
                'view_angle': st.selectbox("View Angle", ['perspective', 'top', 'front', 'side', 'corner'])
            }

        with st.sidebar.expander("Advanced Features", expanded=False):
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
                'capacity': capacity, 'environment': environment_config, 'special_requirements': special_features
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
        with col1:
            st.markdown(f'<div class="metric-card"><h3>${total_cost:,.0f}</h3><p>Total Investment</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{int(recommendations["confidence_score"]*100)}%</h3><p>AI Confidence</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{room_specs["length"]}m × {room_specs["width"]}m</h3><p>Room Size</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{room_specs["capacity"]}</h3><p>Capacity</p></div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📊 Analysis", "🎨 Visualization", "🔄 Alternatives", "📋 Report"])

        with tab1:
            st.subheader("AI-Powered Equipment Recommendations")
            colA, colB = st.columns(2)
            with colA:
                for cat, icon in [('display', '📺'), ('camera', '🎥'), ('audio', '🔊')]:
                    rec = recommendations[cat]
                    st.markdown(f"#### {icon} {cat.title()} System")
                    st.markdown(f"""<div class="feature-card"><h4>{rec['model']}</h4><p><strong>Price:</strong> ${rec['price']:,} | <strong>Rating:</strong> ⭐ {rec['rating']}/5.0</p><p><strong>Specs:</strong> {rec['specs']}</p></div>""", unsafe_allow_html=True)
            with colB:
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
            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("#### Room Characteristics")
                analysis = recommendations['room_analysis']
                st.markdown(f"""<div class="comparison-card"><p><strong>Category:</strong> {analysis['size_category']}</p><p><strong>Shape:</strong> {analysis['shape_analysis']}</p><p><strong>Acoustics:</strong> Reverb is {analysis['acoustic_properties']['reverb_category']}, Treatment needed: {'Yes' if analysis['acoustic_properties']['treatment_needed'] else 'No'}</p><p><strong>Lighting Challenges:</strong> {', '.join(analysis.get('lighting_challenges', []))}</p></div>""", unsafe_allow_html=True)
            with colB:
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
                    st.markdown(f"##### {tier_name} T…
