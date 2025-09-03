import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO

st.set_page_config(page_title="AI Room Configurator Pro", page_icon="🏢", layout="wide")

# Custom CSS for premium styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e3c72 !important;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: black !important; 
    }
    .metric-card h4 {
        color: black !important; 
    }
    .metric-card p {
        color: black !important; 
    }
    .product-image {
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        object-fit: contain;
        height: 150px;
        width: 100%;
    }
    .review-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #ffd700;
        color: black !important; /* Set review text color to black */
    }
</style>
""", unsafe_allow_html=True)


# --- Enhanced Product Database with Reviews & Image URLs ---
class ProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Samsung QMR Series 85"': {
                    'price': 7999, 'image_url': 'https://image-us.samsung.com/SamsungUS/samsungbusiness/products/displays/4k-uhd/qm-r-series/85-qm85r/LH85QMRBGC-GO-GALLERY-600x600.jpg',
                    'specs': '4K UHD, 500 nits, 24/7 operation, Tizen OS', 'rating': 4.7,
                    'reviews': [{'user': 'IT Manager', 'rating': 5, 'text': 'Solid performance for our standard meeting rooms.'}]
                },
                'Samsung QMR Series 98"': {
                    'price': 12999, 'image_url': 'https://image-us.samsung.com/SamsungUS/samsungbusiness/products/displays/4k-uhd/qm-r-series/98-qm98r/LH98QMRBGCGO-03-600x600.jpg',
                    'specs': '4K UHD, 500 nits, 24/7 operation, MagicINFO', 'rating': 4.8,
                    'reviews': [
                        {'user': 'Tech Director, Fortune 500', 'rating': 5, 'text': 'Crystal clear image quality, perfect for our boardroom'},
                        {'user': 'AV Manager', 'rating': 4.5, 'text': 'Excellent display, easy integration with control systems'}
                    ]
                },
                'LG LAEC Series 136" LED': {
                    'price': 45999, 'image_url': 'https://www.lg.com/us/business/images/commercial-tvs/md_laec015-bu_136_all-in-one_dvled_display_e_d_v1/gallery/medium01.jpg',
                    'specs': 'All-in-One LED, 1.2mm pixel pitch, HDR10', 'rating': 4.9,
                    'reviews': [
                        {'user': 'Corporate AV Lead', 'rating': 5, 'text': 'Stunning visuals, no bezels, worth every penny'},
                        {'user': 'Integration Specialist', 'rating': 4.8, 'text': 'Best LED wall solution we\'ve deployed'}
                    ]
                },
                'Sony Crystal LED': {
                    'price': 89999, 'image_url': 'https://pro.sony/s3/2019/09/17154508/c-series-main-image-1.png?w=640',
                    'specs': 'MicroLED, 1.0mm pitch, 1800 nits, HDR', 'rating': 5.0,
                    'reviews': [
                        {'user': 'Executive Board', 'rating': 5, 'text': 'Absolutely breathtaking quality'},
                        {'user': 'CTO', 'rating': 5, 'text': 'The future of display technology'}
                    ]
                }
            },
            'cameras': {
                 'Logitech Rally Bar Mini': {
                    'price': 3999, 'image_url': 'https://www.logitech.com/content/dam/logitech/en/products/video-conferencing/rally-bar-mini/gallery/rally-bar-mini-gallery-1-graphite.png',
                    'specs': '4K PTZ, 4x digital zoom, AI Viewfinder, integrated audio', 'rating': 4.6,
                    'reviews': [{'user': 'Small Business Owner', 'rating': 4.5, 'text': 'Perfect all-in-one for our huddle rooms.'}]
                },
                'Logitech Rally Plus': {
                    'price': 5999, 'image_url': 'https://www.logitech.com/content/dam/logitech/en/products/video-conferencing/rally/rally-ultra-hd-gallery-1.png',
                    'specs': '4K PTZ, 15x zoom, AI auto-framing, dual speakers', 'rating': 4.7,
                    'reviews': [
                        {'user': 'IT Director', 'rating': 5, 'text': 'Best video quality in hybrid meetings'},
                        {'user': 'Facility Manager', 'rating': 4.5, 'text': 'Easy setup, great tracking'}
                    ]
                },
                'Poly Studio X70': {
                    'price': 8999, 'image_url': 'https://www.poly.com/content/dam/www/products/video/studio/studio-x70/poly-studio-x70-callouts-8-1-desktop.png.thumb.1280.1280.png',
                    'specs': 'Dual 4K cameras, 120° FOV, NoiseBlockAI', 'rating': 4.8,
                    'reviews': [
                        {'user': 'VP Engineering', 'rating': 5, 'text': 'Exceptional AI director mode'},
                        {'user': 'Meeting Room Admin', 'rating': 4.6, 'text': 'Crystal clear even in large rooms'}
                    ]
                },
                'Cisco Room Kit Pro': {
                    'price': 15999, 'image_url': 'https://www.cisco.com/c/en/us/products/collaboration-endpoints/webex-room-series/room-kit-pro/jcr:content/Grid/category_content/layout_0_1/layout-0-0/anchor_2/image.img.png/1632345582372.png',
                    'specs': 'Triple camera, 5K video, speaker tracking', 'rating': 4.9,
                    'reviews': [
                        {'user': 'Enterprise Architect', 'rating': 5, 'text': 'Enterprise-grade reliability'},
                        {'user': 'AV Consultant', 'rating': 4.8, 'text': 'Best-in-class for large spaces'}
                    ]
                }
            },
            'audio': {
                'Biamp Parlé': {
                    'price': 4999, 'image_url': 'https://www.biamp.com/assets/2/15/Main-Images/Parle_VBC_2500_Front_1.png',
                    'specs': 'Beamforming mic bar, AEC, AGC, Dante', 'rating': 4.7,
                    'reviews': [
                        {'user': 'Systems Integrator', 'rating': 4.8, 'text': 'Excellent DSP, clean audio'},
                        {'user': 'Tech Lead', 'rating': 4.6, 'text': 'Great for medium rooms'}
                    ]
                },
                'Shure MXA920': {
                    'price': 6999, 'image_url': 'https://www.shure.com/images/pr/press_release_mxa920_square_and_round_form_factors_white_grey/image_large_transparent',
                    'specs': 'Ceiling array, steerable coverage, IntelliMix DSP', 'rating': 4.9,
                    'reviews': [
                        {'user': 'Audio Engineer', 'rating': 5, 'text': 'Invisible yet perfect audio capture'},
                        {'user': 'Consultant', 'rating': 4.8, 'text': 'Game-changer for ceiling installations'}
                    ]
                },
                 'Bose Videobar VB-S': {
                    'price': 1199, 'image_url': 'https://assets.bose.com/content/dam/Bose_DAM/Web/pro/global/products/videobars/videobar_vb-s/product_silo_images/vb-s-q-silo-01-sq.psd/jcr:content/renditions/cq5dam.web.1280.1280.png',
                    'specs': 'All-in-one soundbar with 4K camera and mics', 'rating': 4.5,
                    'reviews': [{'user': 'Startup Office Manager', 'rating': 4.7, 'text': 'Incredible value and performance for the price.'}]
                },
                'QSC Q-SYS': {
                    'price': 12999, 'image_url': 'https://www.qsc.com/resource-files/productimages/sys-cor-nv32h-f-m.png',
                    'specs': 'Full ecosystem, network audio, advanced DSP', 'rating': 4.8,
                    'reviews': [
                        {'user': 'AV Director', 'rating': 5, 'text': 'Most flexible platform available'},
                        {'user': 'Integrator', 'rating': 4.6, 'text': 'Powerful but requires expertise'}
                    ]
                }
            }
        }


# --- Advanced AI Recommendation Engine ---
class AdvancedAVRecommender:
    def __init__(self):
        self.db = ProductDatabase()
        self.compatibility_matrix = self._build_compatibility_matrix()

    def _build_compatibility_matrix(self):
        """Build product compatibility scoring"""
        return {
            ('Logitech', 'Logitech'): 1.2,  # Same brand bonus
            ('Poly', 'Poly'): 1.2,
            ('Cisco', 'Cisco'): 1.2,
            ('Shure', 'Biamp'): 1.1,  # Known good combinations
            ('QSC', 'Samsung'): 1.1,
        }

    def get_ai_recommendations(self, room_specs: Dict) -> Dict:
        """Generate comprehensive AI-powered recommendations"""
        recommendations = {
            'display': self._recommend_display(room_specs),
            'camera': self._recommend_camera(room_specs),
            'audio': self._recommend_audio(room_specs),
            'control': self._recommend_control(room_specs),
            'accessories': self._recommend_accessories(room_specs),
            'confidence_score': self._calculate_confidence(room_specs)
        }
        return recommendations

    def _recommend_display(self, specs):
        viewing_distance = specs['length'] * 0.9
        optimal_diagonal = (viewing_distance / 4) * 39.37 # 4:1 rule for analytical viewing

        if optimal_diagonal < 90:
            selected = 'Samsung QMR Series 85"'
            tech_type = "4K Display"
        elif optimal_diagonal < 120:
            selected = 'Samsung QMR Series 98"'
            tech_type = "Large 4K Display"
        elif optimal_diagonal < 160:
            selected = 'LG LAEC Series 136" LED'
            tech_type = "LED Wall"
        else:
            selected = 'Sony Crystal LED'
            tech_type = "MicroLED Wall"
        
        product_info = self.db.products['displays'][selected]

        return {
            'primary': selected, 'technology': tech_type,
            'size': f"{int(optimal_diagonal)} inches (calculated)",
            'price': product_info['price'], 'specs': product_info['specs'],
            'rating': product_info['rating'], 'reviews': product_info['reviews'],
            'image_url': product_info['image_url'],
            'reasoning': f"Optimal for {viewing_distance:.1f}m viewing distance with {specs['capacity']} viewers"
        }

    def _recommend_camera(self, specs):
        capacity = specs['capacity']

        if capacity <= 6:
            selected = 'Logitech Rally Bar Mini'
            features = "All-in-one bar for small rooms"
        elif capacity <= 12:
            selected = 'Logitech Rally Plus'
            features = "Auto-framing for medium groups"
        elif capacity <= 20:
            selected = 'Poly Studio X70'
            features = "Dual cameras with AI director"
        else:
            selected = 'Cisco Room Kit Pro'
            features = "Triple camera system with speaker tracking"
        
        product_info = self.db.products['cameras'][selected]

        return {
            'primary': selected, 'features': features,
            'price': product_info['price'], 'specs': product_info['specs'],
            'rating': product_info['rating'], 'reviews': product_info['reviews'],
            'image_url': product_info['image_url'],
            'fov_coverage': self._calculate_fov_coverage(specs),
            'ai_features': ['Auto-framing', 'Speaker tracking', 'People counting', 'Whiteboard mode']
        }

    def _recommend_audio(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']

        if room_volume < 60:
            selected = 'Bose Videobar VB-S'
            config = "All-in-one soundbar"
        elif room_volume < 100:
            selected = 'Biamp Parlé'
            config = "Single beamforming bar"
        elif room_volume < 180:
            selected = 'Shure MXA920'
            config = "Ceiling array with zones"
        else:
            selected = 'QSC Q-SYS'
            config = "Distributed audio system"

        product_info = self.db.products['audio'][selected]

        return {
            'primary': selected, 'configuration': config,
            'price': product_info['price'], 'specs': product_info['specs'],
            'rating': product_info['rating'], 'reviews': product_info['reviews'],
            'image_url': product_info['image_url'],
            'coverage_zones': self._calculate_audio_zones(specs),
            'acoustic_treatment': self._recommend_acoustic_treatment(specs)
        }

    def _recommend_control(self, specs):
        complexity = specs.get('complexity_score', 3)
        if complexity <= 3:
            return {
                'primary': 'Crestron Flex UC', 'type': 'Tabletop touchpanel',
                'price': 3999, 'rating': 4.6,
                'features': ['One-touch join', 'Room scheduling', 'Preset scenes'],
                'image_url': 'https://www.crestron.com/Images/product/UC-MM30-T_Angle_1_1920.jpg'
            }
        else:
            return {
                'primary': 'Crestron NVX System', 'type': 'Enterprise control platform',
                'price': 15999, 'rating': 4.9,
                'features': ['Full automation', 'Network AV', 'API integration', 'Analytics'],
                'image_url': 'https://www.crestron.com/Images/product/DM-NVX-E760_Front_1_1920.jpg'
            }
            
    def _recommend_accessories(self, specs):
        accessories = []
        if specs['capacity'] > 10:
            accessories.append({'item': 'Wireless presentation system', 'model': 'Barco ClickShare CX-50', 'price': 1999})
        if 'Recording' in specs.get('special_requirements', []):
            accessories.append({'item': 'Recording appliance', 'model': 'Epiphan Pearl Nexus', 'price': 8999})
        accessories.append({'item': 'Cable management', 'model': 'FSR Floor boxes and raceways', 'price': 999})
        return accessories

    def _calculate_fov_coverage(self, specs): return min(100, (120 / (specs['length'] * specs['width'])) * 100)
    def _calculate_audio_zones(self, specs): return max(1, int((specs['length'] * specs['width']) / 25))
    
    def _recommend_acoustic_treatment(self, specs):
        treatments = []
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        if room_volume > 100: treatments.append("Acoustic ceiling tiles (NRC 0.85+)")
        if specs['width'] > 8: treatments.append("Wall absorption panels (40% coverage)")
        if specs['length'] > 10: treatments.append("Rear wall diffusion")
        return treatments if treatments else ["Minimal treatment needed."]

    def _calculate_confidence(self, specs):
        base_score = 85
        ratio = specs['length'] / specs['width']
        if 1.2 <= ratio <= 1.8: base_score += 10
        area_per_person = (specs['length'] * specs['width']) / specs['capacity']
        if 2.5 <= area_per_person <= 4: base_score += 5
        return min(100, int(base_score + random.uniform(-2, 2)))

# --- Helper for creating 3D mesh objects ---
def create_cuboid(center, size, color='grey', name='cuboid'):
    """Creates a Mesh3d cuboid trace."""
    dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2
    x, y, z = center
    
    return go.Mesh3d(
        x=[x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx],
        y=[y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy],
        z=[z-dz, z-dz, z-dz, z-dz, z+dz, z+dz, z+dz, z+dz],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color,
        opacity=1.0,
        flatshading=True,
        name=name,
        showlegend=False
    )

def create_chair(position, rotation_deg=0, color='saddlebrown'):
    """Creates a list of traces for a realistic chair."""
    traces = []
    x, y, z = position
    
    # Convert rotation to radians
    theta = np.deg2rad(rotation_deg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def rotate(px, py):
        rotated = np.dot(rotation_matrix, [px, py])
        return rotated[0], rotated[1]

    # Chair components (relative positions)
    seat_center_rel = (0, 0, 0.4)
    back_center_rel = (0, -0.2, 0.8)
    leg_dims = (0.05, 0.05, 0.35)
    
    # Seat
    seat_rot_x, seat_rot_y = rotate(seat_center_rel[0], seat_center_rel[1])
    traces.append(create_cuboid((x + seat_rot_x, y + seat_rot_y, z + seat_center_rel[2]), (0.4, 0.4, 0.05), color))

    # Backrest
    back_rot_x, back_rot_y = rotate(back_center_rel[0], back_center_rel[1])
    traces.append(create_cuboid((x + back_rot_x, y + back_rot_y, z + back_center_rel[2]), (0.4, 0.05, 0.4), color))

    # Legs
    leg_positions_rel = [(0.18, 0.18, 0.175), (-0.18, 0.18, 0.175), (0.18, -0.18, 0.175), (-0.18, -0.18, 0.175)]
    for lx, ly, lz in leg_positions_rel:
        leg_rot_x, leg_rot_y = rotate(lx, ly)
        traces.append(create_cuboid((x + leg_rot_x, y + leg_rot_y, z + lz), leg_dims, color))
        
    return traces

# --- Premium 3D Visualization ---
def create_photorealistic_3d_room(specs, recommendations):
    """Create photorealistic 3D room with actual furniture models"""
    fig = go.Figure()
    room_w, room_l, room_h = specs['width'], specs['length'], specs['ceiling_height']

    # Floor
    fig.add_trace(go.Surface(x=np.array([[0, room_w]]), y=np.array([[0, 0], [room_l, room_l]]), z=np.zeros((2,2)),
                             colorscale='ylorbr', showscale=False, name='Floor', lighting=dict(ambient=0.7)))
    # Back Wall
    fig.add_trace(go.Surface(x=np.array([[0, room_w]]), y=np.array([[room_l, room_l]]), z=np.array([[0, 0], [room_h, room_h]]),
                             colorscale=[[0, 'lightgrey'], [1, 'lightgrey']], showscale=False, name='Back Wall'))
    # Left Wall
    fig.add_trace(go.Surface(x=np.array([[0, 0]]), y=np.array([[0, room_l]]), z=np.array([[0, 0], [room_h, room_h]]),
                             colorscale=[[0, 'whitesmoke'], [1, 'whitesmoke']], showscale=False, name='Left Wall'))

    # Conference Table
    table_l, table_w, table_h = room_l * 0.6, room_w * 0.4, 0.75
    table_x, table_y = room_w / 2, room_l * 0.45
    fig.add_trace(create_cuboid((table_x, table_y, table_h - 0.05), (table_w, table_l, 0.1), color='rgb(139, 69, 19)', name='Tabletop'))
    fig.add_trace(create_cuboid((table_x, table_y, table_h / 2 - 0.05), (table_w * 0.5, table_l * 0.5, table_h - 0.1), color='rgb(105, 105, 105)', name='Table Base'))
    
    # Add Chairs
    capacity = specs['capacity']
    chairs_per_side = (capacity - (capacity % 2)) // 2
    chair_y_spacing = table_l / (chairs_per_side + 0.5)
    
    for i in range(chairs_per_side):
        y_pos = (table_y - table_l/2) + (i + 0.75) * chair_y_spacing
        # Left side chairs
        for trace in create_chair((table_x - table_w/2 - 0.6, y_pos, 0), rotation_deg=90): fig.add_trace(trace)
        # Right side chairs
        for trace in create_chair((table_x + table_w/2 + 0.6, y_pos, 0), rotation_deg=-90): fig.add_trace(trace)

    if capacity % 2 == 1: # Head of table
        for trace in create_chair((table_x, table_y - table_l/2 - 0.6, 0), rotation_deg=0): fig.add_trace(trace)
    if capacity % 2 == 0 and capacity > 2: # Two heads
         for trace in create_chair((table_x, table_y - table_l/2 - 0.6, 0), rotation_deg=0): fig.add_trace(trace)
         for trace in create_chair((table_x, table_y + table_l/2 + 0.6, 0), rotation_deg=180): fig.add_trace(trace)


    # Display
    display_w, display_h = room_w * 0.7, room_w * 0.7 * (9/16)
    fig.add_trace(create_cuboid((room_w/2, room_l-0.05, 1.5), (display_w, 0.05, display_h), color='black', name='Display'))
    fig.add_trace(create_cuboid((room_w/2, room_l-0.04, 1.5), (display_w*0.95, 0.05, display_h*0.9), color='#313689', name='Screen'))

    # Camera
    fig.add_trace(create_cuboid((room_w/2, room_l - 0.2, display_h + 1.6), (0.2, 0.15, 0.15), color='darkgrey', name='Camera'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Width (m)", range=[0, room_w], showbackground=False, zeroline=False),
            yaxis=dict(title="Length (m)", range=[0, room_l], showbackground=False, zeroline=False),
            zaxis=dict(title="Height (m)", range=[0, room_h], showbackground=False, zeroline=False),
            camera=dict(eye=dict(x=-1.5, y=-2.5, z=1.8)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=room_l/room_w, z=room_h/room_w),
            bgcolor='rgb(240, 240, 245)'
        ),
        showlegend=False, height=700, margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


# --- Cost Calculator with ROI ---
class CostCalculator:
    def __init__(self):
        self.labor_rates = {'basic': 150, 'standard': 200, 'premium': 300}

    def calculate_total_cost(self, recommendations, specs):
        equipment_cost = 0
        for category in ['display', 'camera', 'audio', 'control']:
            if category in recommendations:
                equipment_cost += recommendations[category].get('price', 0)
        if 'accessories' in recommendations:
            for item in recommendations['accessories']:
                equipment_cost += item.get('price', 0)

        complexity = specs.get('complexity_score', 3)
        labor_hours = 16 + (complexity * 4) + (specs['capacity'] * 0.5)
        labor_rate = self.labor_rates['premium'] if equipment_cost > 30000 else self.labor_rates['standard']
        installation_cost = labor_hours * labor_rate
        infrastructure = self._calculate_infrastructure(specs)
        training = 2000 if specs['capacity'] > 12 else 1000
        support_year1 = equipment_cost * 0.1
        total = equipment_cost + installation_cost + infrastructure + training + support_year1
        return {'equipment': equipment_cost, 'installation': installation_cost, 'infrastructure': infrastructure,
                'training': training, 'support_year1': support_year1, 'total': total}

    def _calculate_infrastructure(self, specs):
        base_cost = 5000  # Network, power, mounting
        area_multiplier = max(1.0, (specs['length'] * specs['width']) / 50)
        special_cost = 3000 if 'Recording' in specs.get('special_requirements', []) else 0
        return int(base_cost * area_multiplier + special_cost)

    def calculate_roi(self, cost_breakdown, specs):
        total_investment = cost_breakdown['total']
        meeting_hours_per_week = specs['capacity'] * 4
        hourly_rate = 75  # Average loaded cost per person
        av_efficiency_gain = 0.15
        annual_savings = (meeting_hours_per_week * 52 * hourly_rate) * av_efficiency_gain
        annual_travel_budget = specs['capacity'] * 5000
        travel_savings = annual_travel_budget * 0.3
        total_annual_savings = annual_savings + travel_savings
        payback_period = total_investment / total_annual_savings if total_annual_savings > 0 else float('inf')
        roi_3_years = ((total_annual_savings * 3 - total_investment) / total_investment) * 100 if total_investment > 0 else float('inf')
        return {'annual_savings': total_annual_savings, 'payback_months': payback_period * 12, 'roi_3_years': roi_3_years}


# --- Environmental Analysis ---
def analyze_room_environment(specs):
    return {
        'lighting': analyze_lighting_conditions(specs),
        'acoustics': analyze_acoustic_properties(specs),
        'thermal': analyze_thermal_considerations(specs),
        'network': analyze_network_requirements(specs)
    }

def analyze_lighting_conditions(specs):
    room_area = specs['length'] * specs['width']
    window_area_percent = specs.get('windows', 0) / 100
    if window_area_percent > 0.2:
        return {'challenge_level': "High", 'recommendations': ["Install motorized blinds", "Use high-brightness display (>700 nits)"]}
    elif window_area_percent > 0.1:
        return {'challenge_level': "Medium", 'recommendations': ["Adjustable window treatments", "Dimmable LED lighting"]}
    else:
        return {'challenge_level': "Low", 'recommendations': ["Standard LED lighting with scene control"]}

def analyze_acoustic_properties(specs):
    room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
    surface_area = 2 * (specs['length']*specs['width'] + specs['length']*specs['ceiling_height'] + specs['width']*specs['ceiling_height'])
    estimated_rt60 = room_volume / (surface_area * 0.15)
    if estimated_rt60 > 0.8:
        return {'estimated_rt60': f"{estimated_rt60:.2f}s", 'challenge_level': "High", 'treatments': ["Acoustic ceiling tiles", "Wall absorption panels"]}
    elif estimated_rt60 > 0.5:
        return {'estimated_rt60': f"{estimated_rt60:.2f}s", 'challenge_level': "Medium", 'treatments': ["Partial acoustic treatment", "Soft furnishings"]}
    else:
        return {'estimated_rt60': f"{estimated_rt60:.2f}s", 'challenge_level': "Low", 'treatments': ["Minimal treatment needed"]}

def analyze_thermal_considerations(specs):
    equipment_load, people_load, lighting_load = 2500, specs['capacity'] * 100, (specs['length']*specs['width']) * 15
    total_load = equipment_load + people_load + lighting_load
    return {'estimated_heat_load': f"{total_load} Watts", 'cooling_requirement': f"{total_load * 3.41:.1f} BTU/hr"}

def analyze_network_requirements(specs):
    video_streams = 4 if specs['capacity'] > 8 else 2
    total_bandwidth = video_streams * 10 + 20
    return {'recommended_bandwidth': f"{total_bandwidth} Mbps", 'network_requirements': ["Dedicated VLAN for AV", "QoS policies", "PoE+ switches"]}


# --- Advanced Visualization Functions ---
def create_acoustic_analysis_chart(analysis):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('RT60 Analysis', 'Noise Floor Simulation'))
    frequencies, rt60_values = [125, 250, 500, 1000, 2000, 4000], [0.9, 0.8, 0.7, 0.6, 0.5, 0.5]
    fig.add_trace(go.Scatter(x=frequencies, y=rt60_values, name="Current RT60", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=frequencies, y=[0.5]*len(frequencies), name="Target RT60", line=dict(color='green', dash='dash')), row=1, col=1)
    
    time_points = np.arange(0, 24, 1)
    noise_floor = 35 + 5 * np.sin(time_points * np.pi / 12) + np.random.normal(0, 1.5, len(time_points))
    fig.add_trace(go.Scatter(x=time_points, y=noise_floor, name="Ambient Noise", line=dict(color='orange')), row=1, col=2)
    
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_yaxes(title_text="RT60 (s)", row=1, col=1)
    fig.update_yaxes(title_text="Noise Level (dBA)", row=1, col=2)
    fig.update_layout(height=400, showlegend=True, title_text="Acoustic Analysis")
    return fig

def create_cost_breakdown_chart(cost_data, roi_analysis):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]], subplot_titles=('Cost Breakdown', 'Payback Period Scenarios'))
    categories, values = list(cost_data.keys())[:-1], [cost_data[cat] for cat in list(cost_data.keys())[:-1]]
    fig.add_trace(go.Pie(labels=categories, values=values, hole=0.4), row=1, col=1)
    
    scenarios = ['Conservative', 'Realistic', 'Optimistic']
    months = [roi_analysis['payback_months'] * 1.2, roi_analysis['payback_months'], roi_analysis['payback_months'] * 0.8]
    fig.add_trace(go.Bar(x=scenarios, y=months, name="Payback"), row=1, col=2)
    
    fig.update_yaxes(title_text="Months", row=1, col=2)
    fig.update_layout(height=400, showlegend=False, title_text="Financial Analysis")
    return fig

# --- Main Application Interface ---
def main():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #1e3c72; font-size: 3em; margin-bottom: 0;">🏢 AI Room Configurator Pro</h1>
            <p style="color: #666; font-size: 1.2em; margin-top: 0;">Enterprise-Grade AV Solutions Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

    if 'recommendations' not in st.session_state: st.session_state.recommendations = None
    if 'room_specs' not in st.session_state: st.session_state.room_specs = None

    with st.sidebar:
        st.markdown("### 📏 Room Specifications")
        length = st.slider("Room Length (m)", 3.0, 20.0, 8.0, 0.5)
        width = st.slider("Room Width (m)", 3.0, 15.0, 6.0, 0.5)
        ceiling_height = st.slider("Ceiling Height (m)", 2.5, 5.0, 3.0, 0.1)
        capacity = st.slider("Seating Capacity", 4, 50, 12)
        st.markdown("### 🏗️ Room Characteristics")
        room_type = st.selectbox("Room Type", ['Conference Room', 'Boardroom', 'Training Room', 'Huddle Room'])
        windows = st.slider("Window Area (%)", 0, 50, 20)
        st.markdown("### ⚙️ Special Requirements")
        special_req = st.multiselect("Additional Features", ['Recording', 'Live Streaming', 'Wireless Presentation'])

        if st.button("🚀 Generate AI Recommendations", type="primary"):
            room_specs = {'length': length, 'width': width, 'ceiling_height': ceiling_height, 'capacity': capacity,
                          'room_type': room_type, 'windows': windows, 'special_requirements': special_req,
                          'complexity_score': len(special_req) + (1 if capacity > 16 else 0)}
            st.session_state.room_specs = room_specs
            recommender = AdvancedAVRecommender()
            st.session_state.recommendations = recommender.get_ai_recommendations(room_specs)
            st.success("✅ Recommendations generated successfully!")

    if st.session_state.recommendations:
        recommendations, room_specs = st.session_state.recommendations, st.session_state.room_specs
        tabs = st.tabs(["🎯 AI Recommendations", "📐 3D Visualization", "💰 Cost Analysis", "🔊 Environmental Analysis", "📋 Project Summary"])

        with tabs[0]:
            st.header("AI-Powered Equipment Recommendations")
            for category in ['display', 'camera', 'audio', 'control']:
                if category in recommendations:
                    rec = recommendations[category]
                    st.markdown(f"---")
                    st.subheader(f"{category.title()} Recommendation: {rec['primary']}")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(rec['image_url'], use_column_width=True, caption=rec['primary'])
                    with col2:
                        rating = rec.get('rating', 0)
                        st.markdown(f"**Price:** `${rec['price']:,}` | **Rating:** {'⭐' * int(rating)} ({rating}/5.0)")
                        st.markdown(f"**Key Specs:** {rec.get('specs', 'N/A')}")
                        if 'reasoning' in rec:
                            st.info(f"💡 **AI Reasoning:** {rec['reasoning']}", icon="🤖")

                    if 'reviews' in rec:
                        with st.expander("View Customer Reviews"):
                            for review in rec['reviews']:
                                st.markdown(f"""
                                <div class="review-card">
                                    <strong>{review['user']}</strong> - {'⭐' * int(review['rating'])}
                                    <br><em>"{review['text']}"</em>
                                </div>
                                """, unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("### 🏗️ Photorealistic 3D Room Preview")
            with st.spinner("Rendering 3D model..."):
                try:
                    fig_3d = create_photorealistic_3d_room(room_specs, recommendations)
                    st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render 3D model: {e}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Room Area", f"{room_specs['length'] * room_specs['width']:.1f} m²")
            c2.metric("Volume", f"{room_specs['length'] * room_specs['width'] * room_specs['ceiling_height']:.1f} m³")
            c3.metric("Area/Person", f"{(room_specs['length'] * room_specs['width']) / room_specs['capacity']:.1f} m²")
            c4.metric("Viewing Distance", f"{room_specs['length'] * 0.9:.1f} m")

        with tabs[2]:
            st.markdown("### 💰 Comprehensive Cost & ROI Analysis")
            calculator = CostCalculator()
            cost_breakdown = calculator.calculate_total_cost(recommendations, room_specs)
            roi_analysis = calculator.calculate_roi(cost_breakdown, room_specs)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Equipment Cost", f"${cost_breakdown['equipment']:,}")
            c2.metric("Services & Install", f"${cost_breakdown['installation'] + cost_breakdown['training']:,}")
            c3.metric("Total Investment", f"${cost_breakdown['total']:,}")
            c4.metric("Payback Period", f"{roi_analysis['payback_months']:.1f} months", delta=f"{roi_analysis['roi_3_years']:.0f}% 3-yr ROI", delta_color="off")
            
            st.plotly_chart(create_cost_breakdown_chart(cost_breakdown, roi_analysis), use_container_width=True)

        with tabs[3]:
            st.markdown("### 🔊 Environmental & Performance Analysis")
            env_analysis = analyze_room_environment(room_specs)
            
            st.plotly_chart(create_acoustic_analysis_chart(env_analysis['acoustics']), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"**Lighting Challenge:** {env_analysis['lighting']['challenge_level']}")
                for rec in env_analysis['lighting']['recommendations']: st.write(f"• {rec}")
            with c2:
                st.info(f"**Acoustic Challenge:** {env_analysis['acoustics']['challenge_level']}")
                for treatment in env_analysis['acoustics']['treatments']: st.write(f"• {treatment}")
            with c3:
                st.info(f"**Est. Heat Load:** {env_analysis['thermal']['estimated_heat_load']}")
                st.write(f"• **Cooling Needed:** {env_analysis['thermal']['cooling_requirement']}")
                st.write(f"• **Bandwidth:** {env_analysis['network']['recommended_bandwidth']}")

        with tabs[4]:
            st.markdown("### 📋 Executive Project Summary")
            cost_breakdown = CostCalculator().calculate_total_cost(recommendations, room_specs)
            roi_analysis = CostCalculator().calculate_roi(cost_breakdown, room_specs)
            
            st.markdown(f"""
            This AI-generated proposal outlines a state-of-the-art AV solution for a **{room_specs['room_type']}** with a capacity of **{room_specs['capacity']} people**.
            The total estimated investment is **${cost_breakdown['total']:,}**, with an expected payback period of **{roi_analysis['payback_months']:.1f} months** and a 3-year ROI of **{roi_analysis['roi_3_years']:.0f}%**.
            """)
            
            st.markdown("#### 🗓️ Implementation Timeline")
            timeline_df = pd.DataFrame({
                'Phase': ['Planning & Design', 'Procurement', 'Installation & Commissioning', 'Training & Go-Live'],
                'Duration (weeks)': [2, 4, 2, 1]
            })
            st.dataframe(timeline_df, use_container_width=True)
            
            st.markdown("#### 🚀 Next Steps")
            st.markdown("""
            1. **Review Recommendations:** Evaluate the proposed solution with your team.
            2. **Schedule Consultation:** Book a technical discussion with our AV experts.
            3. **Proposal Refinement:** Customize the solution based on detailed feedback.
            """)
            if st.button("📄 Generate PDF Report", type="primary"):
                st.success("✅ PDF report generated! (Feature in development)")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>🎯 Welcome to the AI Room Configurator Pro</h2>
            <p style="font-size: 1.2em; color: #666;">
                Use the sidebar to input your room's details, then click "Generate AI Recommendations" 
                to receive a custom AV solution with 3D models and financial analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
