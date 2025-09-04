import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List

st.set_page_config(page_title="AI Room Configurator Pro Max", page_icon="🏢", layout="wide")

# --- Modern Logitech-style CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    :root {
        --primary-color: #7C3AED;
        --accent-color: #F4F7FA;
        --sidebar-bg: #fff;
        --sidebar-text: #22223B;
        --main-bg: #F4F7FA;
        --card-bg: #fff;
        --card-shadow: 0 8px 32px rgba(60,60,60,0.08);
        --border-radius-lg: 24px;
        --border-radius-md: 14px;
        --table-bg: #F8FAFC;
        --table-border: #E2E8F0;
        --feature-border: #7C3AED;
        --success-color: #50C878;
    }
    body, .stApp {
        font-family: 'Inter', sans-serif !important;
        background: var(--main-bg) !important;
        color: var(--sidebar-text) !important;
    }
    .main > div {
        background: var(--card-bg) !important;
        border-radius: var(--border-radius-lg);
        padding: 32px;
        margin: 18px;
        box-shadow: var(--card-shadow);
        color: var(--sidebar-text) !important;
    }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: var(--sidebar-text) !important;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .main p, .main li, .main div, .main span, .main label {
        color: #22223B !important;
        font-size: 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: var(--accent-color);
        padding: 14px;
        border-radius: var(--border-radius-md);
    }
    .stTabs [data-baseweb="tab"] {
        background: #fff;
        border-radius: 12px;
        color: #7C3AED !important;
        font-weight: 600;
        padding: 14px 28px;
        transition: all 0.2s ease;
        font-size: 16px;
        letter-spacing: 0.01em;
        border: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #F3F0FF;
        border: 2px solid #7C3AED;
        color: #7C3AED !important;
    }
    .stTabs [aria-selected="true"] {
        background: #7C3AED !important;
        color: #fff !important;
        border: 2px solid #7C3AED;
        box-shadow: 0 4px 18px rgba(124,58,237,0.12);
    }
    .premium-card {
        background: #fff !important;
        padding: 32px;
        border-radius: var(--border-radius-lg);
        color: #22223B !important;
        margin: 18px 0;
        box-shadow: var(--card-shadow);
        border: 1px solid #E2E8F0;
    }
    .metric-card {
        background: #7C3AED !important;
        padding: 24px;
        border-radius: var(--border-radius-md);
        box-shadow: var(--card-shadow);
        color: #fff !important;
        text-align: center;
        margin: 12px 0;
        border: none;
    }
    .metric-card h3 {
        color: #fff !important;
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .metric-card p {
        color: #fff !important;
        margin: 7px 0 0 0;
        font-size: 15px;
        opacity: 0.92;
        font-weight: 600;
    }
    .feature-card {
        background: #fff !important;
        padding: 24px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        border-left: 6px solid var(--feature-border);
        box-shadow: var(--card-shadow);
        color: #22223B !important;
        border-top: 1px solid #E2E8F0;
    }
    .comparison-card {
        background: #fff !important;
        padding: 24px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        border: 1px solid #E2E8F0;
        color: #22223B !important;
        transition: all 0.2s ease;
    }
    .comparison-card:hover {
        border-color: #7C3AED;
        box-shadow: 0 8px 24px rgba(124,58,237,0.09);
    }
    .alert-success {
        background: var(--success-color) !important;
        color: #fff !important;
        padding: 18px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(80,200,120,0.12);
    }
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%) !important;
        color: #fff !important;
        border: none;
        padding: 16px 32px;
        border-radius: 32px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.2s ease;
        width: 100%;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(124,58,237,0.09);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 32px rgba(124,58,237,0.18);
        background: linear-gradient(135deg, #5B21B6 0%, #A78BFA 100%) !important;
    }
    .css-1d391kg, .stSidebar, .stSidebarContent {
        background: var(--sidebar-bg) !important;
        color: var(--sidebar-text) !important;
    }
    .stSidebar label, .stSidebar span, .stSidebar p, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: var(--sidebar-text) !important;
    }
    .stSelectbox label, .stMultiSelect label {
        color: var(--sidebar-text) !important;
        font-weight: 600;
        font-size: 16px;
    }
    .stDataFrame th, .stDataFrame td {
        color: var(--sidebar-text) !important;
        font-size: 15px;
        font-weight: 600;
        background: var(--table-bg) !important;
        border-color: var(--table-border) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Product Database and Recommendation Engine ---
# ...existing EnhancedProductDatabase and MaximizedAVRecommender classes...

# --- Logitech-style 3D Visualization ---
class EnhancedVisualizationEngine:
    @staticmethod
    def create_3d_room_visualization(room_specs, recommendations):
        length, width, height = room_specs['length'], room_specs['width'], room_specs['ceiling_height']
        table_length = min(length * 0.7, 4)
        table_width = min(width * 0.4, 1.5)
        table_height = 0.75
        table_x_center = length * 0.6
        table_y_center = width * 0.5

        fig = go.Figure()

        # Walls (Logitech style: muted green/gray)
        wall_color = 'rgb(180, 200, 195)'
        fig.add_trace(go.Mesh3d(
            x=[0, length, length, 0, 0, length, length, 0],
            y=[0, 0, width, width, 0, 0, width, width],
            z=[0, 0, 0, 0, height, height, height, height],
            i=[0, 1, 2, 3, 4, 5],
            j=[1, 2, 3, 0, 5, 6],
            k=[4, 5, 6, 7, 0, 1],
            color=wall_color,
            opacity=0.18,
            name="Walls",
            showscale=False
        ))

        # Floor (light gray)
        fig.add_trace(go.Surface(
            x=[[0, length], [0, length]],
            y=[[0, 0], [width, width]],
            z=[[0, 0], [0, 0]],
            colorscale=[[0, '#F4F7FA'], [1, '#F4F7FA']],
            showscale=False,
            name='Floor',
            opacity=1
        ))

        # Display wall panel (wood accent)
        panel_width = min(3, width * 0.6)
        panel_height = panel_width * 9/16
        panel_y_start = (width - panel_width) / 2
        panel_z_start = height * 0.3
        fig.add_trace(go.Surface(
            x=[[0.01, 0.01], [0.01, 0.01]],
            y=[[panel_y_start, panel_y_start + panel_width], [panel_y_start, panel_y_start + panel_width]],
            z=[[panel_z_start, panel_z_start], [panel_z_start + panel_height, panel_z_start + panel_height]],
            colorscale=[[0, '#E2C8A1'], [1, '#E2C8A1']],
            showscale=False,
            name='Wall Panel',
            opacity=1
        ))

        # Display screen (black rectangle)
        fig.add_trace(go.Surface(
            x=[[0.02, 0.02], [0.02, 0.02]],
            y=[[panel_y_start + 0.2, panel_y_start + panel_width - 0.2], [panel_y_start + 0.2, panel_y_start + panel_width - 0.2]],
            z=[[panel_z_start + 0.2, panel_z_start + 0.2], [panel_z_start + panel_height - 0.2, panel_z_start + panel_height - 0.2]],
            colorscale=[[0, '#22223B'], [1, '#22223B']],
            showscale=False,
            name='Display',
            opacity=1
        ))

        # Conference table (rounded rectangle, white)
        fig.add_trace(go.Surface(
            x=[[table_x_center - table_length/2, table_x_center + table_length/2], [table_x_center - table_length/2, table_x_center + table_length/2]],
            y=[[table_y_center - table_width/2, table_y_center - table_width/2], [table_y_center + table_width/2, table_y_center + table_width/2]],
            z=[[table_height, table_height], [table_height, table_height]],
            colorscale=[[0, '#F8FAFC'], [1, '#F8FAFC']],
            showscale=False,
            name='Table',
            opacity=1
        ))

        # Chairs (light blue, Logitech style)
        capacity = min(room_specs['capacity'], 12)
        chairs_per_side = min(6, capacity // 2)
        chair_x_positions = []
        chair_y_positions = []
        chair_z_positions = []
        for i in range(chairs_per_side):
            chair_x = table_x_center - table_length/2 + (i + 1) * table_length / (chairs_per_side + 1)
            # Left side
            chair_x_positions.append(chair_x)
            chair_y_positions.append(table_y_center - table_width/2 - 0.35)
            chair_z_positions.append(table_height + 0.05)
            # Right side
            if len(chair_x_positions) < capacity:
                chair_x_positions.append(chair_x)
                chair_y_positions.append(table_y_center + table_width/2 + 0.35)
                chair_z_positions.append(table_height + 0.05)
        fig.add_trace(go.Scatter3d(
            x=chair_x_positions,
            y=chair_y_positions,
            z=chair_z_positions,
            mode='markers',
            marker=dict(
                size=14,
                color='#A7C7E7',
                symbol='circle',
                line=dict(color='#22223B', width=1)
            ),
            name='Chairs'
        ))

        # Camera bar (black bar above display)
        camera_width = panel_width * 0.7
        camera_y_center = width / 2
        camera_z = panel_z_start + panel_height + 0.1
        fig.add_trace(go.Scatter3d(
            x=[0.04, 0.04],
            y=[camera_y_center - camera_width/2, camera_y_center + camera_width/2],
            z=[camera_z, camera_z],
            mode='lines',
            line=dict(color='#22223B', width=12),
            name='Camera Bar'
        ))

        # Lighting (ceiling spots, yellow)
        for i in range(2):
            for j in range(3):
                fig.add_trace(go.Scatter3d(
                    x=[length * (i + 1) / 3],
                    y=[width * (j + 1) / 4],
                    z=[height - 0.05],
                    mode='markers',
                    marker=dict(size=8, color='#FFE066', symbol='circle'),
                    name='Ceiling Light'
                ))

        # Update layout for Logitech look
        fig.update_layout(
            title=dict(
                text="Conference Room Visualization",
                x=0.5,
                font=dict(size=18, color='#22223B', family='Inter')
            ),
            scene=dict(
                xaxis=dict(
                    title=f"Length ({length:.1f}m)",
                    showgrid=False,
                    showbackground=False,
                    showline=False,
                    showticklabels=False,
                    range=[-0.5, length + 0.5]
                ),
                yaxis=dict(
                    title=f"Width ({width:.1f}m)",
                    showgrid=False,
                    showbackground=False,
                    showline=False,
                    showticklabels=False,
                    range=[-0.5, width + 0.5]
                ),
                zaxis=dict(
                    title=f"Height ({height:.1f}m)",
                    showgrid=False,
                    showbackground=False,
                    showline=False,
                    showticklabels=False,
                    range=[0, height + 0.2]
                ),
                camera=dict(
                    eye=dict(x=2.2, y=2.2, z=1.2),
                    center=dict(x=0, y=0, z=0.2),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='#F4F7FA',
                aspectmode='cube'
            ),
            height=520,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='#F4F7FA',
            plot_bgcolor='#F4F7FA'
        )
        return fig

    # ...existing create_equipment_layout_2d, create_cost_breakdown_chart, create_feature_comparison_radar...

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
            st.session_state.budget_tier = budget_tier
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
                   # filepath: untitled:Untitled-2
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List

st.set_page_config(page_title="AI Room Configurator Pro Max", page_icon="🏢", layout="wide")

# --- Modern Logitech-style CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    :root {
        --primary-color: #7C3AED;
        --accent-color: #F4F7FA;
        --sidebar-bg: #fff;
        --sidebar-text: #22223B;
        --main-bg: #F4F7FA;
        --card-bg: #fff;
        --card-shadow: 0 8px 32px rgba(60,60,60,0.08);
        --border-radius-lg: 24px;
        --border-radius-md: 14px;
        --table-bg: #F8FAFC;
        --table-border: #E2E8F0;
        --feature-border: #7C3AED;
        --success-color: #50C878;
    }
    body, .stApp {
        font-family: 'Inter', sans-serif !important;
        background: var(--main-bg) !important;
        color: var(--sidebar-text) !important;
    }
    .main > div {
        background: var(--card-bg) !important;
        border-radius: var(--border-radius-lg);
        padding: 32px;
        margin: 18px;
        box-shadow: var(--card-shadow);
        color: var(--sidebar-text) !important;
    }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: var(--sidebar-text) !important;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .main p, .main li, .main div, .main span, .main label {
        color: #22223B !important;
        font-size: 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: var(--accent-color);
        padding: 14px;
        border-radius: var(--border-radius-md);
    }
    .stTabs [data-baseweb="tab"] {
        background: #fff;
        border-radius: 12px;
        color: #7C3AED !important;
        font-weight: 600;
        padding: 14px 28px;
        transition: all 0.2s ease;
        font-size: 16px;
        letter-spacing: 0.01em;
        border: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #F3F0FF;
        border: 2px solid #7C3AED;
        color: #7C3AED !important;
    }
    .stTabs [aria-selected="true"] {
        background: #7C3AED !important;
        color: #fff !important;
        border: 2px solid #7C3AED;
        box-shadow: 0 4px 18px rgba(124,58,237,0.12);
    }
    .premium-card {
        background: #fff !important;
        padding: 32px;
        border-radius: var(--border-radius-lg);
        color: #22223B !important;
        margin: 18px 0;
        box-shadow: var(--card-shadow);
        border: 1px solid #E2E8F0;
    }
    .metric-card {
        background: #7C3AED !important;
        padding: 24px;
        border-radius: var(--border-radius-md);
        box-shadow: var(--card-shadow);
        color: #fff !important;
        text-align: center;
        margin: 12px 0;
        border: none;
    }
    .metric-card h3 {
        color: #fff !important;
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .metric-card p {
        color: #fff !important;
        margin: 7px 0 0 0;
        font-size: 15px;
        opacity: 0.92;
        font-weight: 600;
    }
    .feature-card {
        background: #fff !important;
        padding: 24px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        border-left: 6px solid var(--feature-border);
        box-shadow: var(--card-shadow);
        color: #22223B !important;
        border-top: 1px solid #E2E8F0;
    }
    .comparison-card {
        background: #fff !important;
        padding: 24px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        border: 1px solid #E2E8F0;
        color: #22223B !important;
        transition: all 0.2s ease;
    }
    .comparison-card:hover {
        border-color: #7C3AED;
        box-shadow: 0 8px 24px rgba(124,58,237,0.09);
    }
    .alert-success {
        background: var(--success-color) !important;
        color: #fff !important;
        padding: 18px;
        border-radius: var(--border-radius-md);
        margin: 12px 0;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(80,200,120,0.12);
    }
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%) !important;
        color: #fff !important;
        border: none;
        padding: 16px 32px;
        border-radius: 32px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.2s ease;
        width: 100%;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(124,58,237,0.09);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 32px rgba(124,58,237,0.18);
        background: linear-gradient(135deg, #5B21B6 0%, #A78BFA 100%) !important;
    }
    .css-1d391kg, .stSidebar, .stSidebarContent {
        background: var(--sidebar-bg) !important;
        color: var(--sidebar-text) !important;
    }
    .stSidebar label, .stSidebar span, .stSidebar p, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: var(--sidebar-text) !important;
    }
    .stSelectbox label, .stMultiSelect label {
        color: var(--sidebar-text) !important;
        font-weight: 600;
        font-size: 16px;
    }
    .stDataFrame th, .stDataFrame td {
        color: var(--sidebar-text) !important;
        font-size: 15px;
        font-weight: 600;
        background: var(--table-bg) !important;
        border-color: var(--table-border) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Product Database and Recommendation Engine ---
# ...existing EnhancedProductDatabase and MaximizedAVRecommender classes...

# --- Logitech-style 3D Visualization ---
class EnhancedVisualizationEngine:
    @staticmethod
    def create_3d_room_visualization(room_specs, recommendations):
        length, width, height = room_specs['length'], room_specs['width'], room_specs['ceiling_height']
        table_length = min(length * 0.7, 4)
        table_width = min(width * 0.4, 1.5)
        table_height = 0.75
        table_x_center = length * 0.6
        table_y_center = width * 0.5

        fig = go.Figure()

        # Walls (Logitech style: muted green/gray)
        wall_color = 'rgb(180, 200, 195)'
        fig.add_trace(go.Mesh3d(
            x=[0, length, length, 0, 0, length, length, 0],
            y=[0, 0, width, width, 0, 0, width, width],
            z=[0, 0, 0, 0, height, height, height, height],
            i=[0, 1, 2, 3, 4, 5],
            j=[1, 2, 3, 0, 5, 6],
            k=[4, 5, 6, 7, 0, 1],
            color=wall_color,
            opacity=0.18,
            name="Walls",
            showscale=False
        ))

        # Floor (light gray)
        fig.add_trace(go.Surface(
            x=[[0, length], [0, length]],
            y=[[0, 0], [width, width]],
            z=[[0, 0], [0, 0]],
            colorscale=[[0, '#F4F7FA'], [1, '#F4F7FA']],
            showscale=False,
            name='Floor',
            opacity=1
        ))

        # Display wall panel (wood accent)
        panel_width = min(3, width * 0.6)
        panel_height = panel_width * 9/16
        panel_y_start = (width - panel_width) / 2
        panel_z_start = height * 0.3
        fig.add_trace(go.Surface(
            x=[[0.01, 0.01], [0.01, 0.01]],
            y=[[panel_y_start, panel_y_start + panel_width], [panel_y_start, panel_y_start + panel_width]],
            z=[[panel_z_start, panel_z_start], [panel_z_start + panel_height, panel_z_start + panel_height]],
            colorscale=[[0, '#E2C8A1'], [1, '#E2C8A1']],
            showscale=False,
            name='Wall Panel',
            opacity=1
        ))

        # Display screen (black rectangle)
        fig.add_trace(go.Surface(
            x=[[0.02, 0.02], [0.02, 0.02]],
            y=[[panel_y_start + 0.2, panel_y_start + panel_width - 0.2], [panel_y_start + 0.2, panel_y_start + panel_width - 0.2]],
            z=[[panel_z_start + 0.2, panel_z_start + 0.2], [panel_z_start + panel_height - 0.2, panel_z_start + panel_height - 0.2]],
            colorscale=[[0, '#22223B'], [1, '#22223B']],
            showscale=False,
            name='Display',
            opacity=1
        ))

        # Conference table (rounded rectangle, white)
        fig.add_trace(go.Surface(
            x=[[table_x_center - table_length/2, table_x_center + table_length/2], [table_x_center - table_length/2, table_x_center + table_length/2]],
            y=[[table_y_center - table_width/2, table_y_center - table_width/2], [table_y_center + table_width/2, table_y_center + table_width/2]],
            z=[[table_height, table_height], [table_height, table_height]],
            colorscale=[[0, '#F8FAFC'], [1, '#F8FAFC']],
            showscale=False,
            name='Table',
            opacity=1
        ))

        # Chairs (light blue, Logitech style)
        capacity = min(room_specs['capacity'], 12)
        chairs_per_side = min(6, capacity // 2)
        chair_x_positions = []
        chair_y_positions = []
        chair_z_positions = []
        for i in range(chairs_per_side):
            chair_x = table_x_center - table_length/2 + (i + 1) * table_length / (chairs_per_side + 1)
            # Left side
            chair_x_positions.append(chair_x)
            chair_y_positions.append(table_y_center - table_width/2 - 0.35)
            chair_z_positions.append(table_height + 0.05)
            # Right side
            if len(chair_x_positions) < capacity:
                chair_x_positions.append(chair_x)
                chair_y_positions.append(table_y_center + table_width/2 + 0.35)
                chair_z_positions.append(table_height + 0.05)
        fig.add_trace(go.Scatter3d(
            x=chair_x_positions,
            y=chair_y_positions,
            z=chair_z_positions,
            mode='markers',
            marker=dict(
                size=14,
                color='#A7C7E7',
                symbol='circle',
                line=dict(color='#22223B', width=1)
            ),
            name='Chairs'
        ))

        # Camera bar (black bar above display)
        camera_width = panel_width * 0.7
        camera_y_center = width / 2
        camera_z = panel_z_start + panel_height + 0.1
        fig.add_trace(go.Scatter3d(
            x=[0.04, 0.04],
            y=[camera_y_center - camera_width/2, camera_y_center + camera_width/2],
            z=[camera_z, camera_z],
            mode='lines',
            line=dict(color='#22223B', width=12),
            name='Camera Bar'
        ))

        # Lighting (ceiling spots, yellow)
        for i in range(2):
            for j in range(3):
                fig.add_trace(go.Scatter3d(
                    x=[length * (i + 1) / 3],
                    y=[width * (j + 1) / 4],
                    z=[height - 0.05],
                    mode='markers',
                    marker=dict(size=8, color='#FFE066', symbol='circle'),
                    name='Ceiling Light'
                ))

        # Update layout for Logitech look
        fig.update_layout(
            title=dict(
                text="Conference Room Visualization",
                x=0.5,
                font=dict(size=18, color='#22223B', family='Inter')
            ),
            scene=dict(
                xaxis=dict(
                    title=f"Length ({length:.1f}m)",
                    showgrid=False,
                    showbackground=False,
                    showline=False,
                    showticklabels=False,
                    range=[-0.5, length + 0.5]
                ),
                yaxis=dict(
                    title=f"Width ({width:.1f}m)",
                    showgrid=False,
                    showbackground=False,
                    showline=False,
                    showticklabels=False,
                    range=[-0.5, width + 0.5]
                ),
                zaxis=dict(
                    title=f"Height ({height:.1f}m)",
                    showgrid=False,
                    showbackground=False,
                    showline=False,
                    showticklabels=False,
                    range=[0, height + 0.2]
                ),
                camera=dict(
                    eye=dict(x=2.2, y=2.2, z=1.2),
                    center=dict(x=0, y=0, z=0.2),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='#F4F7FA',
                aspectmode='cube'
            ),
            height=520,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='#F4F7FA',
            plot_bgcolor='#F4F7FA'
        )
        return fig

    # ...existing create_equipment_layout_2d, create_cost_breakdown_chart, create_feature_comparison_radar...

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
            st.session_state.budget_tier = budget_tier
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
                    st.markdown(f"#### {icon}
