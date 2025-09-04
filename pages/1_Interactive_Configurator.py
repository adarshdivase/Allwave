import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any

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


# --- Comprehensive Product Database (Omitted for brevity) ---
class EnhancedProductDatabase:
    # ... (Content remains the same)
    pass

# --- Advanced AI Recommendation Engine (Omitted for brevity) ---
class MaximizedAVRecommender:
    # ... (Content remains the same)
    pass

# --- NEW Visualization Engine with Material/Lighting Sim ---
class EnhancedMaterials:
    @staticmethod
    def get_material_presets():
        return {
            'wood': {'base_color': '#8B4513','roughness': 0.7,'metallic': 0.0,'normal_strength': 1.0,'ambient_occlusion': 0.8,'grain_scale': 0.5},
            'metal': {'base_color': '#B8B8B8','roughness': 0.2,'metallic': 0.9,'normal_strength': 0.5,'ambient_occlusion': 0.3},
            'glass': {'base_color': '#FFFFFF','roughness': 0.05,'metallic': 0.9,'normal_strength': 0.1,'ambient_occlusion': 0.1,'opacity': 0.3},
            'fabric': {'base_color': '#303030','roughness': 0.9,'metallic': 0.0,'normal_strength': 0.8,'ambient_occlusion': 0.7},
            'display': {'base_color': '#000000','roughness': 0.1,'metallic': 0.5,'normal_strength': 0.2,'ambient_occlusion': 0.2,'emission': 0.2}
        }

class EnhancedLighting:
    def __init__(self, room_specs: Dict[str, Any]):
        self.room_specs = room_specs
        self.ambient_intensity = 0.3
        self.direct_intensity = 0.7
        self.shadow_softness = 0.5

    def get_light_positions(self) -> List[np.ndarray]:
        length, width, height = (self.room_specs['length'], self.room_specs['width'], self.room_specs['ceiling_height'])
        return [
            np.array([length * 0.25, width * 0.25, height - 0.1]), np.array([length * 0.75, width * 0.25, height - 0.1]),
            np.array([length * 0.25, width * 0.75, height - 0.1]), np.array([length * 0.75, width * 0.75, height - 0.1])
        ]

class TextureGenerator:
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

    @staticmethod
    def create_metal_texture(size: tuple, roughness: float = 0.2) -> np.ndarray:
        texture = np.ones(size) * 0.8
        num_scratches = int(size[0] * size[1] * 0.01)
        for _ in range(num_scratches):
            x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            length, angle = np.random.randint(5, 20), np.random.random() * np.pi
            for i in range(length):
                xi, yi = int(x + i * np.cos(angle)), int(y + i * np.sin(angle))
                if 0 <= xi < size[0] and 0 <= yi < size[1]:
                    texture[yi, xi] = 1.0 - roughness * np.random.random()
        return texture

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
            'day': {'ambient': 0.8, 'diffuse': 1.0, 'color': 'rgb(255, 255, 224)'},
            'evening': {'ambient': 0.4, 'diffuse': 0.7, 'color': 'rgb(255, 228, 181)'},
            'presentation': {'ambient': 0.3, 'diffuse': 0.5, 'color': 'rgb(200, 200, 255)'},
            'video conference': {'ambient': 0.6, 'diffuse': 0.8, 'color': 'rgb(220, 220, 255)'}
        }

    def create_3d_room_visualization(self, room_specs, recommendations, viz_config):
        fig = go.Figure()
        colors = self.color_schemes[viz_config['style_options']['color_scheme']]
        lighting = self.lighting_modes[viz_config['style_options']['lighting_mode']]
        
        self._add_room_structure(fig, room_specs, colors, lighting)
        
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
        
        self._update_camera_view(fig, room_specs, viz_config['style_options']['view_angle'])
        self._update_layout(fig, room_specs)
        
        return fig

    def _add_room_structure(self, fig, specs, colors, lighting):
        self._add_walls(fig, specs, colors, lighting)
        fig.add_trace(go.Surface(x=[[0, specs['length']], [0, specs['length']]], y=[[0, 0], [specs['width'], specs['width']]], z=[[0, 0], [0, 0]], colorscale=[[0, colors['floor']], [1, colors['floor']]], showscale=False, name='Floor', lighting=lighting))

    def _add_walls(self, fig, specs, colors, lighting):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        lighting_config = {'ambient': lighting.get('ambient', 0.5), 'diffuse': lighting.get('diffuse', 0.8)}
        
        fig.add_trace(go.Surface(x=[[0, 0], [0, 0]], y=[[0, width], [0, width]], z=[[0, 0], [height, height]], colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, opacity=0.9, lighting=lighting_config, name="Back Wall"))
        fig.add_trace(go.Surface(x=[[0, length], [0, length]], y=[[0, 0], [0, 0]], z=[[0, 0], [height, height]], colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, opacity=0.7, lighting=lighting_config, name="Left Wall"))
        fig.add_trace(go.Surface(x=[[0, length], [0, length]], y=[[width, width], [width, width]], z=[[0, 0], [height, height]], colorscale=[[0, colors['wall']], [1, colors['wall']]], showscale=False, opacity=0.7, lighting=lighting_config, name="Right Wall"))

    def _add_table(self, fig, specs, colors, table_style):
        length, width = specs['length'], specs['width']
        table_height, table_x_center, table_y_center = 0.75, length * 0.55, width * 0.5
        table_length, table_width = min(length * 0.6, 5), min(width * 0.4, 2)
        
        if table_style in ['rectangular', 'boat-shaped', 'modular']:
            x, y = np.meshgrid(np.linspace(table_x_center - table_length/2, table_x_center + table_length/2, 2), np.linspace(table_y_center - table_width/2, table_y_center + table_width/2, 2))
            z = np.full_like(x, table_height)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, colors['wood']], [1, colors['wood']]], showscale=False, name='Table'))
        elif table_style == 'oval':
            # CORRECTED: Use go.Mesh3d for a filled oval shape
            theta = np.linspace(0, 2 * np.pi, 50)
            x_outline = table_x_center + (table_length / 2) * np.cos(theta)
            y_outline = table_y_center + (table_width / 2) * np.sin(theta)
            
            # Vertices: center point + outline points
            x_verts = np.concatenate(([table_x_center], x_outline))
            y_verts = np.concatenate(([table_y_center], y_outline))
            z_verts = np.full_like(x_verts, table_height)
            
            # Faces (triangles fanning out from the center)
            i_faces = [0] * (len(x_outline) - 1)
            j_faces = list(range(1, len(x_outline)))
            k_faces = list(range(2, len(x_outline) + 1))
            
            fig.add_trace(go.Mesh3d(x=x_verts, y=y_verts, z=z_verts, i=i_faces, j=j_faces, k=k_faces, color=colors['wood'], name='Table'))

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
            'perspective': dict(eye=dict(x=length * -1.5, y=width * -1.5, z=height * 1.5), center=dict(x=length/3, y=width/3, z=0)),
            'top': dict(eye=dict(x=length/2, y=width/2, z=height + 2), center=dict(x=length/2, y=width/2, z=0)),
            'front': dict(eye=dict(x=length + 2, y=width/2, z=height*0.5), center=dict(x=length/2, y=width/2, z=height/2)),
            'side': dict(eye=dict(x=length/2, y=width + 2, z=height*0.5), center=dict(x=length/2, y=width/2, z=height/2)),
            'corner': dict(eye=dict(x=length * 1.5, y=width * 1.5, z=height * 1.2), center=dict(x=length/2, y=width/2, z=height/3))
        }
        fig.update_layout(scene_camera=camera_angles[view_angle])

    def _update_layout(self, fig, specs):
        length, width, height = specs['length'], specs['width'], specs['ceiling_height']
        fig.update_layout(
            autosize=True,
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=width/length, z=0.7),
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
        
        fig.add_shape(type="rect", x0=-0.2, y0=-0.2, x1=length+0.2, y1=width+0.2, line=dict(color="rgba(200,200,200,0.5)", width=2), fillcolor="rgba(240,240,240,0.3)", layer='below')
        fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color="rgb(70,70,70)", width=3), fillcolor="rgba(250,250,250,1)")
        
        if room_specs.get('environment', {}).get('windows', 0) > 0:
            window_sections = int(room_specs['environment']['windows'] / 20)
            window_width_section = width / (window_sections * 2) if window_sections > 0 else 0
            for i in range(window_sections):
                y_start = (width / 2) - (window_sections * window_width_section / 2) + (i * window_width_section * 2)
                fig.add_shape(type="rect", x0=length-0.1, y0=y_start, x1=length, y1=y_start + window_width_section, line=dict(color="rgb(150,200,255)", width=2), fillcolor="rgba(200,230,255,0.7)")

        screen_width_2d = min(width * 0.6, 3.5)
        screen_start = (width - screen_width_2d) / 2
        fig.add_shape(type="rect", x0=0, y0=screen_start, x1=0.15, y1=screen_start + screen_width_2d, line=dict(color="rgb(50,50,50)", width=2), fillcolor="rgb(80,80,80)")
        
        table_length, table_width = min(length * 0.7, 4.5), min(width * 0.4, 1.5)
        table_x, table_y = length * 0.6, width * 0.5
        
        fig.add_shape(type="rect", x0=table_x - table_length/2 + 0.1, y0=table_y - table_width/2 + 0.1, x1=table_x + table_length/2 + 0.1, y1=table_y + table_width/2 + 0.1, line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(0,0,0,0.1)")
        fig.add_shape(type="rect", x0=table_x - table_length/2, y0=table_y - table_width/2, x1=table_x + table_length/2, y1=table_y + table_width/2, line=dict(color="rgb(120,85,60)", width=2), fillcolor="rgb(139,115,85)")
        
        capacity = min(room_specs['capacity'], 12)
        chairs_per_side = min(6, capacity // 2)
        chair_positions = []
        for i in range(chairs_per_side):
            x_pos = table_x - table_length/2 + ((i + 1) * table_length/(chairs_per_side + 1))
            chair_positions.extend([(x_pos, table_y - table_width/2 - 0.4), (x_pos, table_y + table_width/2 + 0.4)])
        
        for x, y in chair_positions[:capacity]:
            fig.add_shape(type="circle", x0=x-0.25+0.05, y0=y-0.25+0.05, x1=x+0.25+0.05, y1=y+0.25+0.05, line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(0,0,0,0.1)")
            fig.add_shape(type="circle", x0=x-0.25, y0=y-0.25, x1=x+0.25, y1=y+0.25, line=dict(color="rgb(70,130,180)"), fillcolor="rgb(100,149,237)")
        
        fig.add_shape(type="rect", x0=0, y0=width*0.2, x1=0.8, y1=width*0.8, line=dict(color="rgba(255,100,100,0.3)", width=2), fillcolor="rgba(255,100,100,0.1)")
        
        camera_points = [[0.1, width*0.45], [0.1, width*0.55], [length*0.8, width*0.2], [length*0.8, width*0.8]]
        fig.add_shape(type="path", path=f"M {camera_points[0][0]},{camera_points[0][1]} L {camera_points[1][0]},{camera_points[1][1]} L {camera_points[3][0]},{camera_points[3][1]} L {camera_points[2][0]},{camera_points[2][1]} Z", line=dict(color="rgba(100,200,100,0.3)", width=1), fillcolor="rgba(100,200,100,0.1)")
        
        speaker_positions = [(length*0.25, width*0.25), (length*0.75, width*0.25), (length*0.25, width*0.75), (length*0.75, width*0.75)]
        for x, y in speaker_positions:
            fig.add_shape(type="circle", x0=x-0.15, y0=y-0.15, x1=x+0.15, y1=y+0.15, line=dict(color="rgba(100,100,255,0.3)"), fillcolor="rgba(100,100,255,0.1)")
            fig.add_shape(type="circle", x0=x-1.5, y0=y-1.5, x1=x+1.5, y1=y+1.5, line=dict(color="rgba(100,100,255,0.1)"), fillcolor="rgba(100,100,255,0.05)")
        
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

        st.markdown("---")
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
                st.markdown("""
                <div class="premium-card">
                    <h3>💡 Upgrade Strategy Overview to {up_tier} Tier</h3>
                    <p>A structured approach to achieving premium AV capabilities while maintaining operational continuity.</p>
                    <p><strong>Total Add. Investment:</strong> ${total:,.0f} | <strong>Est. Monthly:</strong> ${monthly:,.0f}</p>
                </div>
                """.format(up_tier=upgrade['tier'], total=smart_plan['total_investment'], monthly=smart_plan['monthly_investment']), unsafe_allow_html=True)

                cols = st.columns(4)
                for i, (phase_name, phase_info) in enumerate(smart_plan['phases'].items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="feature-card">
                            <h4>{phase_name}</h4>
                            <p><strong>Budget:</strong> ${phase_info['budget']:,.0f}</p>
                            <p><strong>Focus:</strong> {phase_info['focus']}</p>
                            <ul style="font-size: 0.9em; padding-left: 15px;">
                                {''.join([f'<li>{p}</li>' for p in phase_info['priorities']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

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
