import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List

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
    .metric-card h4, .metric-card p {
        color: black !important;
    }
    .review-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #ffd700;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Enhanced Product Database with Reviews ---
class ProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Sharp/NEC 100" 4K Display': {
                    'price': 15999, 'specs': '4K UHD, 500 nits, 24/7 Operation', 'rating': 4.7,
                    'reviews': [{'user': 'IT Director', 'rating': 4.5, 'text': 'Reliable and sharp display for our main conference room.'}]
                },
                'LG MAGNIT 136"': {
                    'price': 75000, 'specs': 'MicroLED, 4K, AI-powered processing, Cable-less', 'rating': 4.9,
                    'reviews': [{'user': 'AV Integrator', 'rating': 5, 'text': 'Stunning image quality and surprisingly easy to install.'}]
                },
                'Samsung "The Wall" 146"': {
                    'price': 99999, 'specs': 'MicroLED, 4K, 0.8mm Pixel Pitch, AI Upscaling', 'rating': 5.0,
                    'reviews': [{'user': 'CEO, Tech Corp', 'rating': 5, 'text': 'Absolutely transformative for our boardroom. A true centerpiece.'}]
                }
            },
            'cameras': {
                'Logitech Rally Bar': {
                    'price': 3999, 'specs': '4K PTZ, AI Viewfinder, RightSight Auto-Framing, Integrated Speakers', 'rating': 4.8,
                    'reviews': [{'user': 'Facilities Manager', 'rating': 5, 'text': 'The all-in-one design simplified our huddle room setup.'}]
                },
                'Poly Studio E70': {
                    'price': 4200, 'specs': 'Dual lenses with 4K sensors, Poly DirectorAI, Speaker Tracking', 'rating': 4.9,
                    'reviews': [{'user': 'CTO', 'rating': 5, 'text': 'The AI-powered camera transitions are seamless. It feels like a real director is in the room.'}]
                },
                'Cisco Room Kit EQ': {
                    'price': 19999, 'specs': 'AI-powered Quad Camera, Speaker & Presenter Tracking, Codec Included', 'rating': 5.0,
                    'reviews': [{'user': 'Enterprise Architect', 'rating': 5, 'text': 'The benchmark for large, integrated video conferencing spaces.'}]
                }
            },
            'audio': {
                'QSC Core Nano': {
                    'price': 2500, 'specs': 'Network Audio I/O, Q-SYS Ecosystem, Software-based DSP', 'rating': 4.7,
                    'reviews': [{'user': 'AV Consultant', 'rating': 4.5, 'text': 'Incredibly powerful and scalable DSP in a small form factor.'}]
                },
                'Biamp TesiraFORTE X 400': {
                    'price': 4500, 'specs': 'AEC, Dante/AVB, USB Audio, Launch Auto-Configuration', 'rating': 4.8,
                    'reviews': [{'user': 'System Integrator', 'rating': 5, 'text': 'The Launch feature saves hours of commissioning time.'}]
                },
                'Shure MXA920 Ceiling Array': {
                    'price': 6999, 'specs': 'Automatic Coverage Technology, Steerable Coverage, IntelliMix DSP', 'rating': 5.0,
                    'reviews': [{'user': 'Audio Engineer', 'rating': 5, 'text': 'Set it and forget it. The audio quality is pristine.'}]
                }
            }
        }


# --- Advanced AI Recommendation Engine ---
class AdvancedAVRecommender:
    def get_ai_recommendations(self, room_specs: Dict) -> Dict:
        db = ProductDatabase()
        recommendations = {
            'display': self._recommend_display(room_specs, db),
            'camera': self._recommend_camera(room_specs, db),
            'audio': self._recommend_audio(room_specs, db),
            'control': self._recommend_control(room_specs),
            'accessories': self._recommend_accessories(room_specs),
            'confidence_score': self._calculate_confidence(room_specs)
        }
        return recommendations

    def _recommend_display(self, specs, db):
        room_area = specs['length'] * specs['width']
        if room_area < 25:
            product_name = 'Sharp/NEC 100" 4K Display'
        elif room_area < 60:
            product_name = 'LG MAGNIT 136"'
        else:
            product_name = 'Samsung "The Wall" 146"'
        return db.products['displays'][product_name]

    def _recommend_camera(self, specs, db):
        if specs['capacity'] <= 8:
            product_name = 'Logitech Rally Bar'
        elif specs['capacity'] <= 16:
            product_name = 'Poly Studio E70'
        else:
            product_name = 'Cisco Room Kit EQ'
        return db.products['cameras'][product_name]

    def _recommend_audio(self, specs, db):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        if room_volume < 75:
            product_name = 'QSC Core Nano'
            config = "In-Ceiling Speakers + Table Mics"
        elif room_volume < 150:
            product_name = 'Biamp TesiraFORTE X 400'
            config = "Distributed Audio + Beamforming Mics"
        else:
            product_name = 'Shure MXA920 Ceiling Array'
            config = "Steerable Ceiling Microphone Array"
        
        rec = db.products['audio'][product_name].copy()
        rec['configuration'] = config
        return rec

    def _recommend_control(self, specs):
        if len(specs.get('special_requirements', [])) < 2 and specs['capacity'] <= 12:
            return {'primary': 'Crestron Flex UC', 'type': 'Tabletop Touchpanel', 'price': 3999, 'rating': 4.6}
        else:
            return {'primary': 'Crestron NVX System', 'type': 'Enterprise Control Platform', 'price': 15999, 'rating': 4.9}

    def _recommend_accessories(self, specs):
        accessories = []
        if 'Wireless Presentation' in specs.get('special_requirements', []):
            accessories.append({'item': 'Wireless Presentation', 'model': 'Barco ClickShare CX-50', 'price': 1999})
        if 'Room Scheduling' in specs.get('special_requirements', []):
            accessories.append({'item': 'Room Scheduling Panel', 'model': 'Crestron TSS-770', 'price': 1500})
        accessories.append({'item': 'Cable Management', 'model': 'FSR Floor/Table Boxes', 'price': 999})
        return accessories

    def _calculate_confidence(self, specs):
        score = 80
        if 1.2 <= specs['length'] / specs['width'] <= 1.8: score += 10
        area_per_person = (specs['length'] * specs['width']) / specs['capacity']
        if 2.0 <= area_per_person <= 4.0: score += 10
        return min(100, score)

# --- Cost, ROI, and Environmental Analysis ---
class CostCalculator:
    def calculate_total_cost(self, recommendations, specs):
        equipment_cost = sum(rec.get('price', 0) for rec in recommendations.values() if isinstance(rec, dict))
        equipment_cost += sum(item['price'] for item in recommendations.get('accessories', []))
        
        complexity = len(specs.get('special_requirements', [])) + (1 if specs['capacity'] > 16 else 0)
        installation_cost = (8 + complexity * 4) * (300 if equipment_cost > 50000 else 200)
        infrastructure_cost = int(5000 * max(1.0, (specs['length'] * specs['width']) / 50))
        training_cost = 2000 if specs['capacity'] > 12 else 1000
        support_cost = equipment_cost * 0.1
        
        total = equipment_cost + installation_cost + infrastructure_cost + training_cost + support_cost
        return {'equipment': equipment_cost, 'installation': installation_cost, 'infrastructure': infrastructure_cost, 'training': training_cost, 'support_year1': support_cost, 'total': total}

    def calculate_roi(self, cost_breakdown, specs):
        total_investment = cost_breakdown['total']
        annual_savings = (specs['capacity'] * 4 * 52 * 75 * 0.15) + (specs['capacity'] * 5000 * 0.3)
        payback_months = (total_investment / annual_savings * 12) if annual_savings > 0 else float('inf')
        roi_3_years = ((annual_savings * 3 - total_investment) / total_investment * 100) if total_investment > 0 else float('inf')
        return {'annual_savings': annual_savings, 'payback_months': payback_months, 'roi_3_years': roi_3_years}

def analyze_room_environment(specs):
    room_area = specs['length'] * specs['width']
    window_area = room_area * (specs.get('windows', 0) / 100)
    if window_area > room_area * 0.2:
        lighting = {'challenge': "High", 'recs': ["Motorized blinds", "High-brightness display (>700 nits)"]}
    else:
        lighting = {'challenge': "Low", 'recs': ["Standard LED lighting with scene control"]}
    
    rt60 = (specs['length'] * specs['width'] * specs['ceiling_height']) / (2 * (room_area + specs['length'] * specs['ceiling_height'] + specs['width'] * specs['ceiling_height']) * 0.15)
    if rt60 > 0.8:
        acoustics = {'rt60': f"{rt60:.2f}s", 'recs': ["Acoustic ceiling tiles", "Wall absorption panels"]}
    else:
        acoustics = {'rt60': f"{rt60:.2f}s", 'recs': ["Minimal treatment needed"]}
        
    return {'lighting': lighting, 'acoustics': acoustics}

# --- Advanced Charting Functions ---
def create_cost_breakdown_chart(cost_data, roi_analysis):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]], subplot_titles=('Cost Breakdown', 'Payback Period Scenarios'))
    categories = list(cost_data.keys())[:-1]
    values = [cost_data[cat] for cat in categories]
    fig.add_trace(go.Pie(labels=categories, values=values, hole=0.4, marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']), row=1, col=1)
    
    scenarios = ['Conservative', 'Realistic', 'Optimistic']
    payback = [roi_analysis['payback_months'] * 1.2, roi_analysis['payback_months'], roi_analysis['payback_months'] * 0.8]
    fig.add_trace(go.Bar(x=scenarios, y=payback, marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']), row=1, col=2)
    fig.update_yaxes(title_text="Months to Payback", row=1, col=2)
    fig.update_layout(height=400, showlegend=False, title_text="Comprehensive Financial Analysis")
    return fig

# --- Helper functions for 3D Visualization ---
def get_rotation_matrix(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

def create_cuboid(center, size, rotation_y_deg=0):
    dx, dy, dz = size
    cx, cy, cz = center
    x = np.array([-dx, dx, dx, -dx, -dx, dx, dx, -dx]) / 2
    y = np.array([-dy, -dy, dy, dy, -dy, -dy, dy, dy]) / 2
    rot_mat = get_rotation_matrix(rotation_y_deg)
    rotated_coords = rot_mat @ np.vstack([x, y])
    return go.Mesh3d(x=rotated_coords[0, :] + cx, y=rotated_coords[1, :] + cy, z=np.array([-dz, -dz, -dz, -dz, dz, dz, dz, dz]) / 2 + cz,
                     i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

def add_detailed_chair(fig, position, rotation_y_deg=0):
    base_color, cushion_color = 'rgb(50, 50, 50)', 'rgb(80, 80, 80)'
    traces = []
    leg_positions = [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)]
    for lx, ly in leg_positions:
        traces.append(create_cuboid((lx, ly, 0.2), (0.05, 0.05, 0.4)))
    traces.append(create_cuboid((0, 0, 0.45), (0.5, 0.5, 0.1)))
    traces.append(create_cuboid((0, 0.2, 0.8), (0.5, 0.1, 0.6)))
    
    rot_mat = get_rotation_matrix(rotation_y_deg)
    for i, trace in enumerate(traces):
        trace.update(color=base_color if i < 4 else cushion_color, lighting=dict(diffuse=0.8), showlegend=False)
        original_coords = np.vstack([trace.x, trace.y])
        rotated_coords = rot_mat @ original_coords
        trace.x = rotated_coords[0, :] + position[0]
        trace.y = rotated_coords[1, :] + position[1]
        trace.z += position[2]
        fig.add_trace(trace)

# --- Maxed-Out Plotly 3D Visualization ---
def create_photorealistic_3d_room(specs, recommendations):
    fig = go.Figure()
    room_w, room_l, room_h = specs['width'], specs['length'], specs['ceiling_height']
    center_x, center_y = room_w / 2, room_l / 2

    # Floor with wood plank texture
    x_grid, y_grid = np.meshgrid(np.linspace(0, room_w, 10), np.linspace(0, room_l, 20))
    floor_texture = (np.sin(y_grid * np.pi * 2) > 0).astype(int) * 0.5 + 0.2
    fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=np.zeros_like(x_grid), surfacecolor=floor_texture, colorscale='YlOrBr', showscale=False, lighting=dict(ambient=0.7, diffuse=0.5)))
    
    # Walls
    wall_color = 'rgb(220, 220, 215)'
    x_wall, z_wall = np.meshgrid(np.linspace(0, room_w, 2), np.linspace(0, room_h, 2))
    fig.add_trace(go.Surface(x=x_wall, y=np.full_like(x_wall, room_l), z=z_wall, colorscale=[[0, wall_color], [1, wall_color]], showscale=False, lighting=dict(ambient=0.8, diffuse=1.0)))
    y_wall, z_wall = np.meshgrid(np.linspace(0, room_l, 2), np.linspace(0, room_h, 2))
    fig.add_trace(go.Surface(x=np.zeros_like(y_wall), y=y_wall, z=z_wall, colorscale=[[0, wall_color], [1, wall_color]], showscale=False, lighting=dict(ambient=0.8, diffuse=1.0)))
    fig.add_trace(go.Surface(x=np.full_like(y_wall, room_w), y=y_wall, z=z_wall, colorscale=[[0, wall_color], [1, wall_color]], showscale=False, lighting=dict(ambient=0.8, diffuse=1.0)))

    # Conference Table
    table_l, table_w, table_h = max(2.5, room_l * 0.5), max(1.4, room_w * 0.4), 0.75
    fig.add_trace(create_cuboid((center_x, center_y, table_h - 0.05), (table_w, table_l, 0.1)).update(color='rgb(139, 69, 19)', lighting=dict(ambient=0.2, diffuse=0.8, specular=0.9, roughness=0.3)))
    fig.add_trace(create_cuboid((center_x, center_y, (table_h - 0.1)/2), (table_w * 0.5, table_l * 0.5, table_h - 0.1)).update(color='rgb(60, 60, 60)', lighting=dict(ambient=0.4, diffuse=0.5)))
    
    # Place Chairs
    chairs_per_side = (specs['capacity'] // 2)
    y_positions = np.linspace(-table_l/2 * 0.8, table_l/2 * 0.8, chairs_per_side)
    for y_pos in y_positions:
        add_detailed_chair(fig, (center_x - table_w/2 - 0.6, center_y + y_pos, 0), 90)
        add_detailed_chair(fig, (center_x + table_w/2 + 0.6, center_y + y_pos, 0), -90)

    # Display and Camera
    display_w, display_h = room_w * 0.6, room_w * 0.6 * (9/16)
    fig.add_trace(create_cuboid((center_x, room_l-0.1, 1.4), (display_w, 0.05, display_h)).update(color='black'))
    fig.add_trace(create_cuboid((center_x, room_l-0.09, 1.4), (display_w*0.95, 0.02, display_h*0.9)).update(color='rgb(10,10,40)'))
    fig.add_trace(create_cuboid((center_x, room_l-0.1, 1.4 + display_h/2 + 0.05), (0.3, 0.15, 0.1)).update(color='rgb(80,80,80)'))

    # Ceiling Mics
    if 'array' in recommendations['audio'].get('configuration', ''):
        fig.add_trace(create_cuboid((center_x, center_y, room_h - 0.05), (0.6, 0.6, 0.05)).update(color='white'))

    fig.update_layout(title={'text': "Enhanced 3D Room Visualization", 'x': 0.5}, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=dict(eye=dict(x=0.1, y=-2.5, z=1.5), up=dict(x=0, y=0, z=1)), aspectmode='data', bgcolor='rgb(240, 240, 245)'), height=600, margin=dict(l=0, r=0, b=0, t=40), showlegend=False)
    return fig


# --- Main Application Interface ---
def main():
    col1, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🏢 AI Room Configurator Pro</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Enterprise-Grade AV Solutions Powered by AI</p>", unsafe_allow_html=True)

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    with st.sidebar:
        st.markdown("### 📏 Room Specifications")
        length = st.slider("Room Length (m)", 3.0, 20.0, 8.0, 0.5)
        width = st.slider("Room Width (m)", 3.0, 15.0, 6.0, 0.5)
        ceiling_height = st.slider("Ceiling Height (m)", 2.5, 5.0, 3.0, 0.1)
        capacity = st.slider("Seating Capacity", 2, 50, 10, 2)
        st.markdown("### 🏗️ Room Characteristics")
        windows = st.slider("Window Area (%)", 0, 50, 20)
        st.markdown("### ⚙️ Special Requirements")
        special_req = st.multiselect("Additional Features", ['Wireless Presentation', 'Room Scheduling'])
        
        if st.button("🚀 Generate AI Recommendations", type="primary"):
            st.session_state.room_specs = {'length': length, 'width': width, 'ceiling_height': ceiling_height, 'capacity': capacity, 'windows': windows, 'special_requirements': special_req}
            recommender = AdvancedAVRecommender()
            st.session_state.recommendations = recommender.get_ai_recommendations(st.session_state.room_specs)

    if st.session_state.recommendations:
        recs = st.session_state.recommendations
        specs = st.session_state.room_specs
        calculator = CostCalculator()
        cost_breakdown = calculator.calculate_total_cost(recs, specs)
        roi_analysis = calculator.calculate_roi(cost_breakdown, specs)
        env_analysis = analyze_room_environment(specs)

        tabs = st.tabs(["🎯 AI Recommendations", "📐 3D Visualization", "💰 Cost Analysis", "🔊 Environmental Analysis", "📋 Project Summary"])
        
        with tabs[0]:
            col1, col2 = st.columns([2, 1])
            with col1:
                for category, rec in recs.items():
                    if isinstance(rec, dict) and 'price' in rec:
                        st.markdown(f"""<div class="recommendation-card">
                                        <h3>{category.title()} Recommendation: 🏆 {rec.get('primary', list(rec.keys())[0])}</h3>
                                        <p><strong>Price:</strong> ${rec['price']:,}</p>
                                        <p><strong>Specs:</strong> {rec.get('specs', 'N/A')}</p>
                                        <p><strong>Rating:</strong> {'⭐' * int(rec.get('rating', 0))} ({rec.get('rating', 0)}/5.0)</p>
                                        </div>""", unsafe_allow_html=True)
                        if 'reviews' in rec:
                            for review in rec['reviews']:
                                st.markdown(f"""<div class="review-card"><strong>{review['user']}</strong> - {'⭐' * int(review['rating'])}<br>"{review['text']}"</div>""", unsafe_allow_html=True)

            with col2:
                st.metric("AI Confidence Score", f"{recs['confidence_score']}%", "High Confidence" if recs['confidence_score'] > 90 else "Good Match")
                st.markdown(f"""<div class="metric-card"><h4>Quick Stats</h4>
                                <p><strong>Total Equipment:</strong> ${cost_breakdown['equipment']:,}</p>
                                <p><strong>Technology Grade:</strong> Enterprise</p></div>""", unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("### 🏗️ Photorealistic 3D Room Preview")
            with st.spinner("Rendering 3D model..."):
                fig_3d = create_photorealistic_3d_room(specs, recs)
                st.plotly_chart(fig_3d, use_container_width=True)

        with tabs[2]:
            st.markdown("### 💰 Comprehensive Cost Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investment", f"${cost_breakdown['total']:,}")
            col2.metric("Payback Period", f"{roi_analysis['payback_months']:.1f} months")
            col3.metric("3-Year ROI", f"{roi_analysis['roi_3_years']:.0f}%")
            st.plotly_chart(create_cost_breakdown_chart(cost_breakdown, roi_analysis), use_container_width=True)
            
        with tabs[3]:
            st.markdown("### 🔊 Environmental & Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Lighting Challenge:** {env_analysis['lighting']['challenge']}")
                st.markdown("**Recommendations:**")
                for rec_item in env_analysis['lighting']['recs']: st.write(f"• {rec_item}")
            with col2:
                st.info(f"**Estimated RT60 (Reverb):** {env_analysis['acoustics']['rt60']}")
                st.markdown("**Recommendations:**")
                for rec_item in env_analysis['acoustics']['recs']: st.write(f"• {rec_item}")
        
        with tabs[4]:
            st.markdown("### 📋 Executive Project Summary")
            st.success("This AI-generated solution is optimized for your room's dimensions and requirements, providing a reliable, enterprise-grade AV experience.")
            st.markdown(f"""#### 📊 Project Metrics
                        - **Total Investment:** ${cost_breakdown['total']:,}
                        - **Expected ROI (3-Years):** {roi_analysis['roi_3_years']:.0f}%
                        - **Payback Period:** {roi_analysis['payback_months']:.1f} months""")
            st.markdown("#### 🗓️ Implementation Timeline")
            timeline_df = pd.DataFrame({'Phase': ['Design', 'Procurement', 'Installation', 'Training'], 'Duration': ['2 weeks', '3-4 weeks', '1 week', '1 week']})
            st.dataframe(timeline_df, use_container_width=True)

    else:
        st.info("Get started by configuring your room specifications in the sidebar and clicking 'Generate AI Recommendations'.")

if __name__ == "__main__":
    main()
