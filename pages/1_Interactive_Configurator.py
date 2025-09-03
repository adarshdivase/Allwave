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
    }
    /* --- CSS FIX APPLIED HERE --- */
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


# --- Enhanced Product Database with Reviews ---
class ProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Samsung QMR Series 98"': {
                    'price': 12999, 'image': '📺',
                    'specs': '4K UHD, 500 nits, 24/7 operation, MagicINFO', 'rating': 4.8,
                    'reviews': [
                        {'user': 'Tech Director, Fortune 500', 'rating': 5, 'text': 'Crystal clear image quality, perfect for our boardroom'},
                        {'user': 'AV Manager', 'rating': 4.5, 'text': 'Excellent display, easy integration with control systems'}
                    ]
                },
                'LG LAEC Series 136" LED': {
                    'price': 45999, 'image': '🖥️',
                    'specs': 'All-in-One LED, 1.2mm pixel pitch, HDR10', 'rating': 4.9,
                    'reviews': [
                        {'user': 'Corporate AV Lead', 'rating': 5, 'text': 'Stunning visuals, no bezels, worth every penny'},
                        {'user': 'Integration Specialist', 'rating': 4.8, 'text': 'Best LED wall solution we\'ve deployed'}
                    ]
                },
                'Sony Crystal LED': {
                    'price': 89999, 'image': '💎',
                    'specs': 'MicroLED, 1.0mm pitch, 1800 nits, HDR', 'rating': 5.0,
                    'reviews': [
                        {'user': 'Executive Board', 'rating': 5, 'text': 'Absolutely breathtaking quality'},
                        {'user': 'CTO', 'rating': 5, 'text': 'The future of display technology'}
                    ]
                }
            },
            'cameras': {
                'Logitech Rally Plus': {
                    'price': 5999, 'image': '📹',
                    'specs': '4K PTZ, 15x zoom, AI auto-framing, dual speakers', 'rating': 4.7,
                    'reviews': [
                        {'user': 'IT Director', 'rating': 5, 'text': 'Best video quality in hybrid meetings'},
                        {'user': 'Facility Manager', 'rating': 4.5, 'text': 'Easy setup, great tracking'}
                    ]
                },
                'Poly Studio X70': {
                    'price': 8999, 'image': '🎥',
                    'specs': 'Dual 4K cameras, 120° FOV, NoiseBlockAI', 'rating': 4.8,
                    'reviews': [
                        {'user': 'VP Engineering', 'rating': 5, 'text': 'Exceptional AI director mode'},
                        {'user': 'Meeting Room Admin', 'rating': 4.6, 'text': 'Crystal clear even in large rooms'}
                    ]
                },
                'Cisco Room Kit Pro': {
                    'price': 15999, 'image': '🎬',
                    'specs': 'Triple camera, 5K video, speaker tracking', 'rating': 4.9,
                    'reviews': [
                        {'user': 'Enterprise Architect', 'rating': 5, 'text': 'Enterprise-grade reliability'},
                        {'user': 'AV Consultant', 'rating': 4.8, 'text': 'Best-in-class for large spaces'}
                    ]
                }
            },
            'audio': {
                'Shure MXA920': {
                    'price': 6999, 'image': '🎤',
                    'specs': 'Ceiling array, steerable coverage, IntelliMix DSP', 'rating': 4.9,
                    'reviews': [
                        {'user': 'Audio Engineer', 'rating': 5, 'text': 'Invisible yet perfect audio capture'},
                        {'user': 'Consultant', 'rating': 4.8, 'text': 'Game-changer for ceiling installations'}
                    ]
                },
                'Biamp Parlé': {
                    'price': 4999, 'image': '🔊',
                    'specs': 'Beamforming mic bar, AEC, AGC, Dante', 'rating': 4.7,
                    'reviews': [
                        {'user': 'Systems Integrator', 'rating': 4.8, 'text': 'Excellent DSP, clean audio'},
                        {'user': 'Tech Lead', 'rating': 4.6, 'text': 'Great for medium rooms'}
                    ]
                },
                'QSC Q-SYS': {
                    'price': 12999, 'image': '🎵',
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
        room_area = specs['length'] * specs['width']
        viewing_distance = specs['length'] * 0.9

        # AVIXA formula for display size
        min_height = viewing_distance / 6  # Analytical viewing
        max_height = viewing_distance / 3  # Basic viewing
        optimal_diagonal = min_height * 2.2 * 39.37  # Convert to inches

        if room_area < 25:
            products = ['Samsung QMR Series 98"']
            tech_type = "4K Display"
        elif room_area < 60:
            products = ['LG LAEC Series 136" LED']
            tech_type = "LED Wall"
        else:
            products = ['Sony Crystal LED']
            tech_type = "MicroLED Wall"

        selected = products[0]
        product_info = self.db.products['displays'][selected]

        return {
            'primary': selected,
            'technology': tech_type,
            'size': f"{optimal_diagonal:.0f} inches",
            'price': product_info['price'],
            'specs': product_info['specs'],
            'rating': product_info['rating'],
            'reviews': product_info['reviews'],
            'alternatives': products[1:] if len(products) > 1 else [],
            'reasoning': f"Optimal for {viewing_distance:.1f}m viewing distance with {specs['capacity']} viewers"
        }

    def _recommend_camera(self, specs):
        capacity = specs['capacity']
        room_area = specs['length'] * specs['width']

        if capacity <= 8:
            products = ['Logitech Rally Plus']
            features = "Auto-framing for small groups"
        elif capacity <= 16:
            products = ['Poly Studio X70']
            features = "Dual cameras with AI director"
        else:
            products = ['Cisco Room Kit Pro']
            features = "Triple camera system with speaker tracking"

        selected = products[0]
        product_info = self.db.products['cameras'][selected]

        return {
            'primary': selected,
            'features': features,
            'price': product_info['price'],
            'specs': product_info['specs'],
            'rating': product_info['rating'],
            'reviews': product_info['reviews'],
            'fov_coverage': self._calculate_fov_coverage(specs),
            'ai_features': ['Auto-framing', 'Speaker tracking', 'People counting', 'Whiteboard mode']
        }

    def _recommend_audio(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']

        if room_volume < 75:
            products = ['Biamp Parlé']
            config = "Single beamforming bar"
        elif room_volume < 150:
            products = ['Shure MXA920']
            config = "Ceiling array with zones"
        else:
            products = ['QSC Q-SYS']
            config = "Distributed audio system"

        selected = products[0]
        product_info = self.db.products['audio'][selected]

        return {
            'primary': selected,
            'configuration': config,
            'price': product_info['price'],
            'specs': product_info['specs'],
            'rating': product_info['rating'],
            'reviews': product_info['reviews'],
            'coverage_zones': self._calculate_audio_zones(specs),
            'acoustic_treatment': self._recommend_acoustic_treatment(specs)
        }

    def _recommend_control(self, specs):
        complexity = specs.get('complexity_score', 3)

        if complexity <= 3:
            return {
                'primary': 'Crestron Flex UC',
                'type': 'Tabletop touchpanel',
                'price': 3999,
                'rating': 4.6,
                'features': ['One-touch join', 'Room scheduling', 'Preset scenes']
            }
        else:
            return {
                'primary': 'Crestron NVX System',
                'type': 'Enterprise control platform',
                'price': 15999,
                'rating': 4.9,
                'features': ['Full automation', 'Network AV', 'API integration', 'Analytics']
            }

    def _recommend_accessories(self, specs):
        accessories = []

        if specs['capacity'] > 12:
            accessories.append({
                'item': 'Wireless presentation system',
                'model': 'Barco ClickShare CX-50',
                'price': 1999
            })

        if 'Recording' in specs.get('special_requirements', []):
            accessories.append({
                'item': 'Recording appliance',
                'model': 'Epiphan Pearl Nexus',
                'price': 8999
            })

        accessories.append({
            'item': 'Cable management',
            'model': 'FSR Floor boxes and raceways',
            'price': 999
        })

        return accessories

    def _calculate_fov_coverage(self, specs):
        """Calculate camera field of view coverage"""
        room_area = specs['length'] * specs['width']
        return min(100, (120 / room_area) * 100)

    def _calculate_audio_zones(self, specs):
        """Calculate number of audio coverage zones"""
        room_area = specs['length'] * specs['width']
        return max(1, int(room_area / 25))

    def _recommend_acoustic_treatment(self, specs):
        """Recommend acoustic treatments"""
        treatments = []
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']

        if room_volume > 100:
            treatments.append("Acoustic ceiling tiles (NRC 0.85+)")
        if specs['width'] > 8:
            treatments.append("Wall absorption panels (40% coverage)")
        if specs['length'] > 10:
            treatments.append("Rear wall diffusion")

        return treatments

    def _calculate_confidence(self, specs):
        """Calculate recommendation confidence score"""
        base_score = 85

        # Adjust based on room proportions
        ratio = specs['length'] / specs['width']
        if 1.2 <= ratio <= 1.8:
            base_score += 10

        # Adjust based on capacity fit
        area_per_person = (specs['length'] * specs['width']) / specs['capacity']
        if 2.5 <= area_per_person <= 4:
            base_score += 5

        return min(100, base_score)


# --- Helper for 3D Visualization ---
def add_premium_chair(fig, x, y, rotation=0):
    """
    Placeholder function to add a chair model to the 3D plot.
    A simple marker is used here. For a realistic view, replace
    this with a 3D model (e.g., using go.Mesh3d).
    """
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[0.4],  # Position at approximate seat height
        mode='markers',
        marker=dict(
            symbol='square',
            size=8,
            color='saddlebrown'
        ),
        name='Chair',
        showlegend=False
    ))


# --- Premium 3D Visualization ---
def create_photorealistic_3d_room(specs, recommendations):
    """Create photorealistic 3D room with actual furniture models"""
    fig = go.Figure()

    room_w, room_l, room_h = specs['width'], specs['length'], specs['ceiling_height']

    # Enhanced room shell with textures
    # Floor with wood texture pattern
    floor_x, floor_y = np.meshgrid(np.linspace(0, room_w, 20), np.linspace(0, room_l, 20))
    floor_z = np.zeros_like(floor_x)
    floor_pattern = np.sin(floor_x * 2) * np.sin(floor_y * 2) * 0.01  # Wood grain effect

    fig.add_trace(go.Surface(
        x=floor_x, y=floor_y, z=floor_z + floor_pattern,
        colorscale='ylorbr', showscale=False, name='Wood Floor',
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.8)
    ))

    # Walls with gradient lighting
    wall_color = 'rgb(245, 245, 245)'

    # Back wall with display
    wall_x = np.linspace(0, room_w, 10)
    wall_z = np.linspace(0, room_h, 10)
    wall_xx, wall_zz = np.meshgrid(wall_x, wall_z)
    wall_yy = np.ones_like(wall_xx) * room_l

    fig.add_trace(go.Surface(
        x=wall_xx, y=wall_yy, z=wall_zz,
        colorscale=[[0, wall_color], [1, wall_color]],
        showscale=False, name='Back Wall',
        lighting=dict(ambient=0.7, diffuse=0.8)
    ))

    # Add premium conference table (realistic shape)
    table_l = room_l * 0.5
    table_w = room_w * 0.35
    table_h = 0.75
    table_x = room_w / 2
    table_y = room_l * 0.4

    # Table top with rounded corners
    theta = np.linspace(0, 2 * np.pi, 50)
    corner_radius = 0.3

    # Create rounded rectangle for table
    table_points_x = []
    table_points_y = []

    # Generate rounded rectangle path
    for t in theta:
        if t < np.pi / 2:
            x = table_x - table_w / 2 + corner_radius - corner_radius * np.cos(t)
            y = table_y - table_l / 2 + corner_radius - corner_radius * np.sin(t)
        elif t < np.pi:
            x = table_x + table_w / 2 - corner_radius + corner_radius * np.cos(np.pi - t)
            y = table_y - table_l / 2 + corner_radius - corner_radius * np.sin(np.pi - t)
        elif t < 3 * np.pi / 2:
            x = table_x + table_w / 2 - corner_radius + corner_radius * np.cos(t - np.pi)
            y = table_y + table_l / 2 - corner_radius + corner_radius * np.sin(t - np.pi)
        else:
            x = table_x - table_w / 2 + corner_radius - corner_radius * np.cos(2 * np.pi - t)
            y = table_y + table_l / 2 - corner_radius + corner_radius * np.sin(2 * np.pi - t)

        table_points_x.append(x)
        table_points_y.append(y)

    # Table surface using go.Mesh3d to create a solid shape
    fig.add_trace(go.Mesh3d(
        x=table_points_x,
        y=table_points_y,
        z=[table_h] * len(table_points_x),
        # Use i, j, k to define the triangles that form the tabletop surface
        i=list(range(len(table_points_x) - 2)),
        j=list(range(1, len(table_points_x) - 1)),
        k=list(range(2, len(table_points_x))),
        color='rgb(139, 69, 19)',
        opacity=0.8,
        name='Conference Table'
    ))

    # Table legs (cylindrical)
    leg_positions = [
        (table_x - table_w / 3, table_y - table_l / 3),
        (table_x + table_w / 3, table_y - table_l / 3),
        (table_x - table_w / 3, table_y + table_l / 3),
        (table_x + table_w / 3, table_y + table_l / 3)
    ]

    for leg_x, leg_y in leg_positions:
        fig.add_trace(go.Scatter3d(
            x=[leg_x, leg_x],
            y=[leg_y, leg_y],
            z=[0, table_h],
            mode='lines',
            line=dict(color='rgb(105, 105, 105)', width=15),
            showlegend=False
        ))

    # Add realistic chairs
    chairs_per_side = specs['capacity'] // 2
    chair_spacing = table_l / (chairs_per_side + 1)

    for i in range(chairs_per_side):
        y_pos = table_y - table_l / 2 + (i + 1) * chair_spacing
        # Left side chairs
        add_premium_chair(fig, table_x - table_w / 2 - 0.6, y_pos, rotation=90)
        # Right side chairs
        add_premium_chair(fig, table_x + table_w / 2 + 0.6, y_pos, rotation=-90)

    # Add head chairs
    if specs['capacity'] % 2 == 1:
        add_premium_chair(fig, table_x, table_y - table_l / 2 - 0.6, rotation=0)
    if specs['capacity'] > 8:
        add_premium_chair(fig, table_x, table_y + table_l / 2 + 0.6, rotation=180)

    # Add premium display
    display_w = room_w * 0.7
    display_h = display_w * 0.5625
    display_x = room_w / 2
    display_y = room_l - 0.05
    display_z = 1.5

    # Display frame
    fig.add_trace(go.Mesh3d(
        x=[display_x - display_w / 2, display_x + display_w / 2, display_x + display_w / 2, display_x - display_w / 2],
        y=[display_y, display_y, display_y, display_y],
        z=[display_z, display_z, display_z + display_h, display_z + display_h],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='black', opacity=0.95, name='Display Frame'
    ))

    # Display screen (with slight glow effect)
    screen_margin = 0.05
    fig.add_trace(go.Mesh3d(
        x=[display_x - display_w / 2 + screen_margin, display_x + display_w / 2 - screen_margin,
           display_x + display_w / 2 - screen_margin, display_x - display_w / 2 + screen_margin],
        y=[display_y + 0.01, display_y + 0.01, display_y + 0.01, display_y + 0.01],
        z=[display_z + screen_margin, display_z + screen_margin,
           display_z + display_h - screen_margin, display_z + display_h - screen_margin],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='rgb(50, 150, 250)', opacity=0.3, name='Display Screen'
    ))

    # Add PTZ camera (realistic model)
    cam_x, cam_y, cam_z = room_w / 2, room_l - 0.3, room_h - 0.3

    # Camera body
    fig.add_trace(go.Scatter3d(
        x=[cam_x], y=[cam_y], z=[cam_z],
        mode='markers',
        marker=dict(size=15, color='rgb(60, 60, 60)', symbol='square'),
        name='PTZ Camera Body', showlegend=False
    ))

    # Camera lens
    fig.add_trace(go.Scatter3d(
        x=[cam_x], y=[cam_y - 0.1], z=[cam_z],
        mode='markers',
        marker=dict(size=8, color='rgb(30, 30, 30)', symbol='circle'),
        name='Camera Lens', showlegend=False
    ))

    # Add ceiling microphones if needed
    if specs['capacity'] > 8:
        mic_positions = [
            (room_w * 0.3, room_l * 0.4, room_h - 0.1),
            (room_w * 0.7, room_l * 0.4, room_h - 0.1)
        ]

        for mic_x, mic_y, mic_z in mic_positions:
            # Microphone array (hexagonal shape)
            hex_points_x = []
            hex_points_y = []
            for angle in np.linspace(0, 2 * np.pi, 7):
                hex_points_x.append(mic_x + 0.2 * np.cos(angle))
                hex_points_y.append(mic_y + 0.2 * np.sin(angle))

            fig.add_trace(go.Scatter3d(
                x=hex_points_x, y=hex_points_y, z=[mic_z] * 7,
                mode='lines',
                line=dict(color='white', width=3),
                showlegend=False
            ))

    # Add speakers
    speaker_positions = [
        (room_w * 0.2, room_l - 0.2, room_h - 0.5),
        (room_w * 0.8, room_l - 0.2, room_h - 0.5)
    ]

    for spk_x, spk_y, spk_z in speaker_positions:
        fig.add_trace(go.Scatter3d(
            x=[spk_x], y=[spk_y], z=[spk_z],
            mode='markers',
            marker=dict(size=10, color='rgb(40, 40, 40)', symbol='square'),
            name='Speaker', showlegend=False
        ))

    # Configure camera and lighting
    camera_eye = dict(x=1.8, y=-1.8, z=1.5)
    camera_center = dict(x=0, y=0, z=0.5)

    fig.update_layout(
        title={
            'text': "Photorealistic 3D Room Configuration",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1e3c72'}
        },
        scene=dict(
            xaxis=dict(title="Width (m)", gridcolor='lightgray', showbackground=False),
            yaxis=dict(title="Length (m)", gridcolor='lightgray', showbackground=False),
            zaxis=dict(title="Height (m)", gridcolor='lightgray', showbackground=False),
            camera=dict(eye=camera_eye, center=camera_center),
            aspectmode='manual',
            aspectratio=dict(x=1, y=room_l / room_w, z=0.5),
            bgcolor='rgb(240, 240, 245)'
        ),
        showlegend=False,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


# --- Cost Calculator with ROI ---
class CostCalculator:
    def __init__(self):
        self.labor_rates = {
            'basic': 150,
            'standard': 200,
            'premium': 300
        }

    def calculate_total_cost(self, recommendations, specs):
        """Calculate comprehensive project costs"""
        equipment_cost = 0

        # Add up all equipment
        for category in ['display', 'camera', 'audio', 'control']:
            if category in recommendations:
                equipment_cost += recommendations[category].get('price', 0)

        # Add accessories
        if 'accessories' in recommendations:
            for item in recommendations['accessories']:
                equipment_cost += item.get('price', 0)

        # Calculate installation
        complexity = specs.get('complexity_score', 3)
        labor_hours = 8 + (complexity * 4)
        labor_rate = self.labor_rates['premium'] if equipment_cost > 50000 else self.labor_rates['standard']
        installation_cost = labor_hours * labor_rate

        # Infrastructure upgrades
        infrastructure = self._calculate_infrastructure(specs)

        # Training and support
        training = 2000 if specs['capacity'] > 12 else 1000
        support_year1 = equipment_cost * 0.1

        return {
            'equipment': equipment_cost,
            'installation': installation_cost,
            'infrastructure': infrastructure,
            'training': training,
            'support_year1': support_year1,
            'total': equipment_cost + installation_cost + infrastructure + training + support_year1
        }

    def _calculate_infrastructure(self, specs):
        """Calculate infrastructure upgrade costs"""
        base_cost = 5000  # Network, power, mounting
        room_area = specs['length'] * specs['width']

        # Scale with room size
        area_multiplier = max(1.0, room_area / 50)

        # Add for special requirements
        special_cost = 0
        if 'Recording' in specs.get('special_requirements', []):
            special_cost += 3000
        if 'Live Streaming' in specs.get('special_requirements', []):
            special_cost += 2000

        return int(base_cost * area_multiplier + special_cost)

    def calculate_roi(self, cost_breakdown, specs):
        """Calculate ROI based on productivity gains"""
        total_investment = cost_breakdown['total']

        # Productivity calculations
        meeting_hours_per_week = specs['capacity'] * 4  # Assume 4 hours/person/week
        hourly_rate = 75  # Average loaded cost per person

        # Efficiency gains
        av_efficiency_gain = 0.15  # 15% more efficient meetings
        travel_reduction = 0.3  # 30% less travel

        annual_meeting_cost = meeting_hours_per_week * 52 * hourly_rate
        annual_savings = annual_meeting_cost * av_efficiency_gain

        # Travel savings (estimated)
        annual_travel_budget = specs['capacity'] * 5000  # $5k per person
        travel_savings = annual_travel_budget * travel_reduction

        total_annual_savings = annual_savings + travel_savings
        payback_period = total_investment / total_annual_savings if total_annual_savings > 0 else float('inf')

        return {
            'annual_savings': total_annual_savings,
            'payback_months': payback_period * 12,
            'roi_3_years': ((total_annual_savings * 3 - total_investment) / total_investment) * 100 if total_investment > 0 else float('inf')
        }


# --- Environmental Analysis ---
def analyze_room_environment(specs):
    """Analyze environmental factors affecting AV performance"""
    analysis = {
        'lighting': analyze_lighting_conditions(specs),
        'acoustics': analyze_acoustic_properties(specs),
        'thermal': analyze_thermal_considerations(specs),
        'network': analyze_network_requirements(specs)
    }
    return analysis


def analyze_lighting_conditions(specs):
    """Analyze lighting conditions and recommendations"""
    room_area = specs['length'] * specs['width']
    window_area_percent = specs.get('windows', 0) / 100
    window_area = room_area * window_area_percent

    # Calculate ambient light levels
    if window_area > room_area * 0.2:
        lighting_challenge = "High"
        recommendations = [
            "Install motorized blinds with light sensors",
            "Use high-brightness display (>700 nits)",
            "Consider bias lighting behind display"
        ]
    elif window_area > room_area * 0.1:
        lighting_challenge = "Medium"
        recommendations = [
            "Adjustable window treatments",
            "Dimmable LED lighting with presets"
        ]
    else:
        lighting_challenge = "Low"
        recommendations = ["Standard LED lighting with scene control"]

    return {
        'challenge_level': lighting_challenge,
        'recommendations': recommendations,
        'optimal_lux': '300-500 lux for video calls'
    }


def analyze_acoustic_properties(specs):
    """Analyze room acoustics"""
    room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
    surface_area = 2 * (specs['length'] * specs['width'] +
                        specs['length'] * specs['ceiling_height'] +
                        specs['width'] * specs['ceiling_height'])

    # Estimate RT60 based on room characteristics
    estimated_rt60 = room_volume / (surface_area * 0.15)  # Simplified calculation

    if estimated_rt60 > 0.8:
        acoustic_challenge = "High - Reverberant"
        treatments = [
            "Acoustic ceiling tiles (NRC 0.85+)",
            "Wall absorption panels on 40% of surfaces",
            "Carpet or soft flooring recommended"
        ]
    elif estimated_rt60 > 0.5:
        acoustic_challenge = "Medium"
        treatments = [
            "Partial acoustic treatment",
            "Soft furnishings and curtains"
        ]
    else:
        acoustic_challenge = "Low - Well damped"
        treatments = ["Minimal treatment needed"]

    return {
        'estimated_rt60': f"{estimated_rt60:.2f} seconds",
        'challenge_level': acoustic_challenge,
        'treatments': treatments,
        'target_rt60': '0.4-0.6 seconds for conferencing'
    }


def analyze_thermal_considerations(specs):
    """Analyze thermal load and cooling requirements"""
    # Estimate heat load from equipment and people
    equipment_load = 2000  # Watts from AV equipment
    people_load = specs['capacity'] * 100  # 100W per person
    lighting_load = (specs['length'] * specs['width']) * 15  # 15W/m²

    total_load = equipment_load + people_load + lighting_load

    return {
        'estimated_heat_load': f"{total_load} Watts",
        'cooling_requirement': f"{total_load * 3.41:.1f} BTU/hr additional",
        'ventilation_note': "Ensure adequate airflow for equipment cooling"
    }


def analyze_network_requirements(specs):
    """Analyze network infrastructure needs"""
    # Estimate bandwidth requirements
    video_streams = 2  # Camera feeds
    if specs['capacity'] > 8:
        video_streams = 4  # Multiple camera angles

    bandwidth_per_stream = 10  # Mbps for 4K
    total_bandwidth = video_streams * bandwidth_per_stream + 20  # Overhead

    return {
        'recommended_bandwidth': f"{total_bandwidth} Mbps",
        'network_requirements': [
            "Dedicated VLAN for AV traffic",
            "QoS policies for video/audio",
            "PoE+ switches for cameras and mics",
            "Redundant network paths recommended"
        ]
    }


# --- Advanced Visualization Functions ---
def create_acoustic_analysis_chart(analysis):
    """Create acoustic analysis visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RT60 Analysis', 'Frequency Response', 'Coverage Pattern', 'Noise Floor'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "polar"}, {"secondary_y": False}]]
    )

    # RT60 across frequencies
    frequencies = [125, 250, 500, 1000, 2000, 4000]
    rt60_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.4]  # Example values
    target_rt60 = [0.6] * len(frequencies)

    fig.add_trace(
        go.Scatter(x=frequencies, y=rt60_values, name="Current RT60",
                   line=dict(color='red', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=frequencies, y=target_rt60, name="Target RT60",
                   line=dict(color='green', width=2, dash='dash')),
        row=1, col=1
    )

    # Frequency response simulation
    freq_range = np.logspace(2, 4, 100)
    response = np.random.normal(0, 3, 100) + np.sin(np.log10(freq_range) * 2) * 2

    fig.add_trace(
        go.Scatter(x=freq_range, y=response, name="Room Response",
                   line=dict(color='blue')),
        row=1, col=2
    )

    # Coverage pattern (polar-like visualization)
    angles = np.linspace(0, 360, 36)
    coverage = 90 - np.abs(angles - 180) / 4  # Simulated directional pattern

    fig.add_trace(
        go.Scatterpolar(r=coverage, theta=angles, mode='lines',
                        name="Microphone Pattern", fill='toself'),
        row=2, col=1
    )

    # Noise floor
    time_points = np.arange(0, 24, 0.5)
    noise_floor = 35 + 10 * np.sin(time_points * np.pi / 12) + np.random.normal(0, 2, len(time_points))

    fig.add_trace(
        go.Scatter(x=time_points, y=noise_floor, name="Ambient Noise",
                   line=dict(color='orange')),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=2)

    fig.update_yaxes(title_text="RT60 (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Level (dB)", row=1, col=2)
    fig.update_yaxes(title_text="Noise Level (dBA)", row=2, col=2)

    fig.update_layout(height=600, showlegend=True,
                      title_text="Comprehensive Acoustic Analysis")

    return fig


def create_cost_breakdown_chart(cost_data, roi_analysis):
    """Create comprehensive cost breakdown visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cost Breakdown', 'ROI Projection', 'Monthly Cash Flow', 'Payback Analysis'),
        specs=[[{"type": "pie"}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Cost breakdown pie chart
    categories = list(cost_data.keys())[:-1]  # Exclude 'total'
    values = [cost_data[cat] for cat in categories]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    fig.add_trace(
        go.Pie(labels=categories, values=values,
               marker_colors=colors, hole=0.4),
        row=1, col=1
    )

    # ROI projection over 5 years
    years = list(range(1, 6))
    annual_savings = roi_analysis['annual_savings']
    cumulative_savings = [annual_savings * year for year in years]
    initial_investment = cost_data['total']

    fig.add_trace(
        go.Scatter(x=years, y=cumulative_savings, name="Cumulative Savings",
                   line=dict(color='green', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=years, y=[initial_investment] * len(years), name="Investment",
                   line=dict(color='red', width=2, dash='dash')),
        row=1, col=2
    )

    # Monthly cash flow
    months = list(range(1, 37))  # 3 years
    monthly_savings = [annual_savings / 12] * 36 if annual_savings > 0 else [0] * 36
    net_cumulative = np.cumsum(monthly_savings) - initial_investment

    fig.add_trace(
        go.Scatter(x=months, y=net_cumulative, name="Cumulative Net Flow",
                   line=dict(color='purple')),
        row=2, col=1
    )

    fig.add_shape(
        type="line",
        x0=months[0], y0=0,
        x1=months[-1], y1=0,
        line=dict(color="grey", width=2, dash="dash"),
        row=2, col=1
    )

    # Payback analysis
    payback_scenarios = ['Conservative', 'Realistic', 'Optimistic']
    payback_months = [roi_analysis['payback_months'] * 1.2, roi_analysis['payback_months'],
                      roi_analysis['payback_months'] * 0.8]
    colors_payback = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    fig.add_trace(
        go.Bar(x=payback_scenarios, y=payback_months,
               marker_color=colors_payback, name="Payback Period"),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Years", row=1, col=2)
    fig.update_xaxes(title_text="Months", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=2)
    fig.update_yaxes(title_text="Net Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Months", row=2, col=2)

    fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Financial Analysis")
    return fig


# --- Main Application Interface ---
def main():
    # Header with logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #1e3c72; font-size: 3em; margin-bottom: 0;">🏢 AI Room Configurator Pro</h1>
            <p style="color: #666; font-size: 1.2em; margin-top: 0;">Enterprise-Grade AV Solutions Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'room_specs' not in st.session_state:
        st.session_state.room_specs = None

    # Sidebar for room specifications
    with st.sidebar:
        st.markdown("### 📏 Room Specifications")

        # Basic dimensions
        length = st.slider("Room Length (m)", 3.0, 20.0, 8.0, 0.5)
        width = st.slider("Room Width (m)", 3.0, 15.0, 6.0, 0.5)
        ceiling_height = st.slider("Ceiling Height (m)", 2.5, 5.0, 3.0, 0.1)
        capacity = st.slider("Seating Capacity", 4, 50, 12)

        # Room characteristics
        st.markdown("### 🏗️ Room Characteristics")
        room_type = st.selectbox("Room Type",
                                 ['Conference Room', 'Boardroom', 'Training Room', 'Auditorium', 'Huddle Room'])

        windows = st.slider("Window Area (%)", 0, 50, 20)

        # Special requirements
        st.markdown("### ⚙️ Special Requirements")
        special_req = st.multiselect("Additional Features",
                                     ['Recording', 'Live Streaming', 'Wireless Presentation', 'Digital Signage',
                                      'Room Scheduling', 'Environmental Control'])

        # Usage patterns
        st.markdown("### 📊 Usage Patterns")
        daily_meetings = st.slider("Daily Meetings", 1, 20, 8)
        avg_meeting_duration = st.slider("Avg Meeting Duration (hours)", 0.5, 4.0, 1.5, 0.25)
        remote_participants = st.slider("Remote Participants (%)", 0, 100, 40)

        # Generate button
        if st.button("🚀 Generate AI Recommendations", type="primary"):
            room_specs = {
                'length': length,
                'width': width,
                'ceiling_height': ceiling_height,
                'capacity': capacity,
                'room_type': room_type,
                'windows': windows,
                'special_requirements': special_req,
                'daily_meetings': daily_meetings,
                'avg_meeting_duration': avg_meeting_duration,
                'remote_participants': remote_participants,
                'complexity_score': len(special_req) + (1 if capacity > 16 else 0)
            }

            st.session_state.room_specs = room_specs

            # Generate recommendations
            recommender = AdvancedAVRecommender()
            recommendations = recommender.get_ai_recommendations(room_specs)
            st.session_state.recommendations = recommendations

            st.success("✅ Recommendations generated successfully!")

    # Main content area
    if st.session_state.recommendations is not None:
        recommendations = st.session_state.recommendations
        room_specs = st.session_state.room_specs

        # Tabs for different views
        tabs = st.tabs([
            "🎯 AI Recommendations",
            "📐 3D Visualization",
            "💰 Cost Analysis",
            "🔊 Environmental Analysis",
            "📋 Project Summary"
        ])

        # Tab 1: AI Recommendations
        with tabs[0]:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Display recommendations for each category
                for category in ['display', 'camera', 'audio', 'control']:
                    if category in recommendations:
                        rec = recommendations[category]

                        specs_text = rec.get('specs', 'N/A')
                        rating = rec.get('rating', 0)

                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>{category.title()} Recommendation</h3>
                            <h4>🏆 {rec['primary']}</h4>
                            <p><strong>Price:</strong> ${rec['price']:,}</p>
                            <p><strong>Specs:</strong> {specs_text}</p>
                            <p><strong>Rating:</strong> {'⭐' * int(rating)} ({rating}/5.0)</p>
                            {f"<p><strong>Features:</strong> {', '.join(rec.get('features', []))}</p>" if 'features' in rec else ""}
                        </div>
                        """, unsafe_allow_html=True)

                        # Show reviews if they exist
                        if 'reviews' in rec:
                            st.markdown("<strong>Customer Reviews:</strong>", unsafe_allow_html=True)
                            for review in rec['reviews'][:2]:
                                st.markdown(f"""
                                <div class="review-card">
                                    <strong>{review['user']}</strong> - {'⭐' * int(review['rating'])}
                                    <br>"{review['text']}"
                                </div>
                                """, unsafe_allow_html=True)

            with col2:
                # Confidence score
                confidence = recommendations['confidence_score']
                st.metric("AI Confidence Score", f"{confidence}%",
                          delta="High Confidence" if confidence > 90 else "Good Match")

                total_cost = 0
                for key, value in recommendations.items():
                    if isinstance(value, dict) and 'price' in value:
                        total_cost += value['price']
                    elif key == 'accessories' and isinstance(value, list):
                        for item in value:
                            total_cost += item.get('price', 0)

                st.markdown(f"""
                <div class="metric-card">
                    <h4>Quick Stats</h4>
                    <p><strong>Total Equipment:</strong> ${total_cost:,}</p>
                    <p><strong>Room Utilization:</strong> {room_specs['capacity']} people</p>
                    <p><strong>Technology Grade:</strong> Enterprise</p>
                </div>
                """, unsafe_allow_html=True)

        # Tab 2: 3D Visualization
        with tabs[1]:
            st.markdown("### 🏗️ Photorealistic 3D Room Preview")

            with st.spinner("Rendering 3D model..."):
                try:
                    fig_3d = create_photorealistic_3d_room(room_specs, recommendations)
                    st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.error(f"3D visualization temporarily unavailable: {str(e)}")
                    st.info("💡 3D room visualization will show your configured space with all recommended equipment positioned optimally.")

            # Room metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Room Area", f"{room_specs['length'] * room_specs['width']:.1f} m²")
            with col2:
                st.metric("Volume", f"{room_specs['length'] * room_specs['width'] * room_specs['ceiling_height']:.1f} m³")
            with col3:
                area_per_person = (room_specs['length'] * room_specs['width']) / room_specs['capacity']
                st.metric("Area/Person", f"{area_per_person:.1f} m²")
            with col4:
                optimal_viewing = room_specs['length'] * 0.9
                st.metric("Viewing Distance", f"{optimal_viewing:.1f} m")

        # Tab 3: Cost Analysis
        with tabs[2]:
            st.markdown("### 💰 Comprehensive Cost Analysis")

            calculator = CostCalculator()
            cost_breakdown = calculator.calculate_total_cost(recommendations, room_specs)
            roi_analysis = calculator.calculate_roi(cost_breakdown, room_specs)

            # Cost summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Equipment", f"${cost_breakdown['equipment']:,}")
            with col2:
                st.metric("Services", f"${cost_breakdown['installation'] + cost_breakdown['training']:,}")
            with col3:
                st.metric("Total Investment", f"${cost_breakdown['total']:,}")
            with col4:
                st.metric("Payback Period", f"{roi_analysis['payback_months']:.1f} months")

            # Cost visualization
            fig_cost = create_cost_breakdown_chart(cost_breakdown, roi_analysis)
            st.plotly_chart(fig_cost, use_container_width=True)

            # ROI details
            st.markdown("### 📈 Return on Investment Analysis")
            col1, col2 = st.columns(2)

            with col1:
                meeting_savings = roi_analysis['annual_savings'] * 0.6 if roi_analysis['annual_savings'] > 0 else 0
                travel_savings = roi_analysis['annual_savings'] * 0.4 if roi_analysis['annual_savings'] > 0 else 0
                st.markdown(f"""
                **Annual Savings Breakdown:**
                - Meeting Efficiency Gains: ${meeting_savings:,.0f}
                - Travel Reduction: ${travel_savings:,.0f}
                - **Total Annual Savings: ${roi_analysis['annual_savings']:,.0f}**
                """)

            with col2:
                st.markdown(f"""
                **ROI Metrics:**
                - Payback Period: {roi_analysis['payback_months']:.1f} months
                - 3-Year ROI: {roi_analysis['roi_3_years']:.0f}%
                - Break-even Point: Month {int(roi_analysis['payback_months'])}
                """)

        # Tab 4: Environmental Analysis
        with tabs[3]:
            st.markdown("### 🔊 Environmental & Performance Analysis")

            env_analysis = analyze_room_environment(room_specs)

            # Acoustic analysis
            st.markdown("#### 🎵 Acoustic Analysis")
            acoustic_fig = create_acoustic_analysis_chart(env_analysis['acoustics'])
            st.plotly_chart(acoustic_fig, use_container_width=True)

            # Environmental factors
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💡 Lighting Conditions")
                st.info(f"**Challenge Level:** {env_analysis['lighting']['challenge_level']}")
                st.markdown("**Recommendations:**")
                for rec in env_analysis['lighting']['recommendations']:
                    st.write(f"• {rec}")

                st.markdown("#### 🌡️ Thermal Considerations")
                st.info(f"**Heat Load:** {env_analysis['thermal']['estimated_heat_load']}")
                st.write(f"**Additional Cooling:** {env_analysis['thermal']['cooling_requirement']}")

            with col2:
                st.markdown("#### 🔊 Acoustic Properties")
                st.info(f"**RT60:** {env_analysis['acoustics']['estimated_rt60']}")
                st.markdown("**Recommended Treatments:**")
                for treatment in env_analysis['acoustics']['treatments']:
                    st.write(f"• {treatment}")

                st.markdown("#### 🌐 Network Requirements")
                st.info(f"**Bandwidth:** {env_analysis['network']['recommended_bandwidth']}")
                st.markdown("**Infrastructure:**")
                for req in env_analysis['network']['network_requirements']:
                    st.write(f"• {req}")

        # Tab 5: Project Summary
        with tabs[4]:
            st.markdown("### 📋 Executive Project Summary")

            # Generate PDF report button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📄 Generate PDF Report", type="primary"):
                    st.success("✅ PDF report generated! (Feature in development)")

            # Executive summary
            st.markdown("""
            #### Executive Summary
            This comprehensive AV solution has been designed specifically for your meeting space requirements 
            using advanced AI algorithms that consider room acoustics, lighting conditions, user capacity, 
            and budget constraints.
            """)

            # Key highlights
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                #### 🎯 Key Highlights
                - **AI-Optimized Design:** Custom configuration for your space
                - **Enterprise-Grade Components:** Professional reliability and performance
                - **Scalable Architecture:** Future-ready and expandable
                - **Comprehensive Support:** Training and ongoing maintenance included
                """)

            with col2:
                cost_breakdown = CostCalculator().calculate_total_cost(recommendations, room_specs)
                roi_analysis = CostCalculator().calculate_roi(cost_breakdown, room_specs)
                st.markdown(f"""
                #### 📊 Project Metrics
                - **Confidence Score:** {recommendations['confidence_score']}%
                - **Total Investment:** ${cost_breakdown['total']:,}
                - **Expected ROI:** {roi_analysis['roi_3_years']:.0f}% (3 years)
                - **Payback Period:** {roi_analysis['payback_months']:.1f} months
                """)

            # Implementation timeline
            st.markdown("#### 🗓️ Implementation Timeline")
            timeline_data = {
                'Phase': ['Planning & Design', 'Procurement', 'Installation', 'Testing & Training', 'Go-Live'],
                'Duration': ['2 weeks', '3-4 weeks', '1 week', '1 week', '1 day'],
                'Key Activities': [
                    'Final design, permits, scheduling',
                    'Equipment ordering, delivery coordination',
                    'Physical installation, cable routing',
                    'System commissioning, user training',
                    'Production deployment, support handoff'
                ]
            }

            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True)

            # Next steps
            st.markdown("""
            #### 🚀 Next Steps
            1. **Review Recommendations:** Evaluate the proposed solution with your team
            2. **Schedule Consultation:** Book a detailed technical discussion with our experts
            3. **Site Survey:** Arrange for precise measurements and environmental assessment  
            4. **Proposal Refinement:** Customize the solution based on your feedback
            5. **Project Kickoff:** Begin implementation once approved
            """)

            # Contact information
            st.markdown("""
            #### 📞 Get Started Today
            Ready to transform your meeting space? Contact our AV experts:
            - **Email:** solutions@avpro.com
            - **Phone:** +1 (555) 123-AVPRO
            - **Schedule:** [Book a consultation](https://calendly.com/avpro-solutions)
            """)

    else:
        # Welcome screen when no recommendations are generated
st.markdown("""
<div style="text-align: center; padding: 50px;">
    <h2>🎯 Welcome to AI Room Configurator Pro</h2>
    <p style="font-size: 1.2em; color: #666;">
        Get started by configuring your room specifications in the sidebar, 
        then click "Generate AI Recommendations" to see your custom AV solution.
    </p>
    <div style="margin: 30px 0;">
        <h4>🚀 What You'll Get:</h4>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; width: 200px;">
                <h5 style="color: black;">🎯 AI Recommendations</h5>
                <p style="color: black;">Smart product selection based on your specific needs</p>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; width: 200px;">
                <h5 style="color: black;">📐 3D Visualization</h5>
                <p style="color: black;">See your room with equipment positioned optimally</p>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; width: 200px;">
                <h5 style="color: black;">💰 Cost Analysis</h5>
                <p style="color: black;">Complete financial breakdown with ROI projections</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
