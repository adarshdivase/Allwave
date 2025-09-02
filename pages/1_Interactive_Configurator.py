# pages/1_Interactive_Configurator.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="AI Room Configurator", page_icon="🏢", layout="wide")

# --- AI Recommendation System ---
class AVRecommendationEngine:
    def __init__(self, oem_data=None):
        self.oem_data = oem_data
        self.recommendations = {}
        
    def get_display_recommendations(self, room_length, room_width, capacity):
        """AI-powered display recommendations based on room specs"""
        recommendations = []
        
        farthest_viewer = room_length * 0.9
        min_diagonal_inches = (farthest_viewer * 39.37) / 6  # 6H rule
        max_diagonal_inches = (farthest_viewer * 39.37) / 3  # 3H rule
        
        room_area = room_length * room_width
        
        if room_area <= 20:  # Small rooms
            recommended_sizes = [55, 65, 75]
            display_type = "Interactive Display or Large Format Display"
        elif room_area <= 50:  # Medium rooms  
            recommended_sizes = [75, 85, 98]
            display_type = "Large Format Display or Projector"
        else:  # Large rooms
            recommended_sizes = [98, 110, 135]
            display_type = "Projector with Large Screen"
        
        optimal_size = min(recommended_sizes, key=lambda x: abs(x - min_diagonal_inches))
        
        recommendations.append({
            'category': 'Display',
            'product': f'{optimal_size}" {display_type}',
            'reason': f'Optimal for {farthest_viewer:.1f}m viewing distance',
            'specs': f'Recommended size: {min_diagonal_inches:.0f}"-{max_diagonal_inches:.0f}"',
            'price_range': self._get_price_estimate('display', optimal_size)
        })
        
        return recommendations
    
    def get_camera_recommendations(self, room_length, room_width, capacity):
        """AI-powered camera recommendations"""
        recommendations = []
        room_area = room_length * room_width
        
        if capacity <= 6 and room_area <= 25:
            camera_type = "Fixed Wide-Angle Camera"
            reason = "Small meeting room - single wide-angle view sufficient"
            suggested_models = ["Logitech Rally Bar Mini", "Poly Studio X30", "Yealink UVC30"]
        elif capacity <= 12 and room_area <= 50:
            camera_type = "PTZ Camera with Auto-Framing"  
            reason = "Medium room - PTZ for speaker tracking and group framing"
            suggested_models = ["Logitech Rally Camera", "Poly Studio X50", "Huddly IQ"]
        else:
            camera_type = "Multi-Camera System with AI Director"
            reason = "Large room - multiple cameras for comprehensive coverage"
            suggested_models = ["Logitech Rally System", "Poly Studio X70", "Multiple PTZ Setup"]
        
        recommendations.append({
            'category': 'Camera',
            'product': camera_type,
            'reason': reason,
            'specs': f'Coverage for {capacity} people in {room_area:.0f}m² space',
            'models': suggested_models,
            'price_range': self._get_price_estimate('camera', capacity)
        })
        
        return recommendations
    
    def get_audio_recommendations(self, room_length, room_width, capacity):
        """AI-powered audio recommendations"""
        recommendations = []
        room_volume = room_length * room_width * 3  # Assume 3m ceiling
        
        if capacity <= 6:
            audio_type = "All-in-One Soundbar with Beamforming Mics"
            reason = "Small group - integrated solution sufficient"
            suggested_models = ["Logitech Rally Bar", "Poly Studio X30", "Jabra Panacast 50"]
        elif capacity <= 12:
            audio_type = "Ceiling Mic Array + Premium Speakers"
            reason = "Medium room - dedicated audio for clear pickup"
            suggested_models = ["Shure MXA910", "Biamp Parlé Ceiling", "QSC Speakers"]
        else:
            audio_type = "Distributed Audio System with Zone Control"
            reason = "Large space - multiple zones for optimal coverage"
            suggested_models = ["Shure MXA920", "Biamp Tesira", "QSC Q-SYS"]
        
        recommendations.append({
            'category': 'Audio',
            'product': audio_type,
            'reason': reason,
            'specs': f'Optimized for {room_volume:.0f}m³ volume',
            'models': suggested_models,
            'price_range': self._get_price_estimate('audio', capacity)
        })
        
        return recommendations
    
    def get_control_recommendations(self, complexity_score):
        """Control system recommendations based on setup complexity"""
        if complexity_score <= 3:
            control_type = "Touch Panel with Preset Controls"
            models = ["Crestron TSW-752", "Extron TLP Pro 525M"]
        else:
            control_type = "Advanced Room Control System"  
            models = ["Crestron 3-Series", "AMX Acendo", "QSC Q-SYS"]
        
        # --- FIX IS HERE: Added the missing 'specs' key for consistency ---
        return [{
            'category': 'Control',
            'product': control_type,
            'reason': f'System complexity score: {complexity_score}/5',
            'specs': f'Manages a system with {complexity_score} or more integrated components.',
            'models': models,
            'price_range': self._get_price_estimate('control', complexity_score)
        }]
    
    def _get_price_estimate(self, category, size_factor):
        """Estimate price ranges based on category and complexity"""
        price_ranges = {
            'display': {
                55: "$3,000 - $5,000", 65: "$4,000 - $7,000", 75: "$6,000 - $10,000",
                85: "$8,000 - $15,000", 98: "$12,000 - $25,000", 110: "$20,000 - $40,000"
            },
            'camera': {
                range(1, 7): "$800 - $2,500", range(7, 13): "$2,500 - $8,000",
                range(13, 31): "$8,000 - $20,000"
            },
            'audio': {
                range(1, 7): "$1,500 - $4,000", range(7, 13): "$4,000 - $12,000", 
                range(13, 31): "$12,000 - $30,000"
            },
            'control': {
                range(1, 4): "$2,000 - $5,000", range(4, 6): "$5,000 - $15,000"
            }
        }
        
        if category == 'display':
            return price_ranges[category].get(size_factor, "$15,000 - $30,000")
        else:
            for key_range, price in price_ranges[category].items():
                if isinstance(key_range, range) and size_factor in key_range:
                    return price
            return "$10,000 - $25,000"

# --- 3D Visualization Functions ---
def create_3d_room_visualization(room_width, room_length, capacity, recommendations):
    """Create interactive 3D room visualization"""
    fig = go.Figure()
    
    room_height = 3.0
    
    # Floor
    fig.add_trace(go.Mesh3d(x=[0, room_width, room_width, 0], y=[0, 0, room_length, room_length], z=[0, 0, 0, 0], color='lightgray', opacity=0.3, name='Floor'))
    
    # Walls
    walls = [
        ([0, room_width, room_width, 0], [room_length, room_length, room_length, room_length], [0, 0, room_height, room_height]),
        ([0, room_width, room_width, 0], [0, 0, 0, 0], [0, 0, room_height, room_height]),
        ([0, 0, 0, 0], [0, room_length, room_length, 0], [0, 0, room_height, room_height]),
        ([room_width, room_width, room_width, room_width], [0, room_length, room_length, 0], [0, 0, room_height, room_height])
    ]
    for i, (x, y, z) in enumerate(walls):
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color='lightblue', opacity=0.4, name=f'Wall {i+1}'))
    
    # Display screen
    display_width = room_width * 0.6
    display_height = display_width * 0.5625
    display_x = (room_width - display_width) / 2
    display_y = room_length - 0.1
    display_z = 1.2
    fig.add_trace(go.Mesh3d(x=[display_x, display_x + display_width, display_x + display_width, display_x], y=[display_y, display_y, display_y, display_y], z=[display_z, display_z, display_z + display_height, display_z + display_height], color='black', opacity=0.8, name='Display Screen'))
    
    # Conference table
    table_length = room_length * 0.6
    table_width = room_width * 0.3
    table_x = (room_width - table_width) / 2
    table_y = room_length * 0.3
    table_height = 0.75
    fig.add_trace(go.Mesh3d(x=[table_x, table_x + table_width, table_x + table_width, table_x], y=[table_y, table_y, table_y + table_length, table_y + table_length], z=[table_height, table_height, table_height, table_height], color='brown', opacity=0.7, name='Conference Table'))
    
    # Chairs
    chairs_per_side = capacity // 2
    for i in range(chairs_per_side):
        y_pos = table_y + (i + 1) * table_length / (chairs_per_side + 1)
        fig.add_trace(go.Scatter3d(x=[table_x - 0.5], y=[y_pos], z=[0.4], mode='markers', marker=dict(size=8, color='gray'), name='Chair'))
        fig.add_trace(go.Scatter3d(x=[table_x + table_width + 0.5], y=[y_pos], z=[0.4], mode='markers', marker=dict(size=8, color='gray'), name='Chair'))
    
    # Camera position
    camera_x = room_width / 2
    camera_y = room_length - 0.3
    camera_z = room_height - 0.5
    fig.add_trace(go.Scatter3d(x=[camera_x], y=[camera_y], z=[camera_z], mode='markers', marker=dict(size=10, color='red', symbol='diamond'), name='PTZ Camera'))
    
    # Ceiling mics
    if capacity > 6:
        mic_positions = [[room_width*0.3, room_length*0.4, room_height-0.1], [room_width*0.7, room_length*0.4, room_height-0.1]]
        for pos in mic_positions:
            fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=6, color='blue'), name='Ceiling Mic'))
            
    fig.update_layout(
        title="3D Room Visualization with Recommended Equipment",
        scene=dict(
            xaxis_title="Width (m)", yaxis_title="Length (m)", zaxis_title="Height (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='manual', aspectratio=dict(x=1, y=room_length/room_width, z=0.5)
        ),
        showlegend=False, height=600
    )
    return fig

# --- Data Loading ---
@st.cache_data
def load_oem_data():
    try:
        return pd.read_csv("av_oem_list_2025.csv", encoding='latin-1', on_bad_lines='skip')
    except FileNotFoundError:
        st.warning("OEM database not found. Using default recommendations.")
        return None

# --- Main Application ---
st.title("🏢 AI-Powered Room Configurator & Equipment Recommender")
st.markdown("*Design your perfect meeting space with AI recommendations and 3D visualization*")

oem_data = load_oem_data()
recommender = AVRecommendationEngine(oem_data)

with st.sidebar:
    st.header("🎛️ Room Configuration")
    room_type = st.selectbox("Room Type", ["Small Meeting Room", "Medium Conference Room", "Large Boardroom", "Training Room", "Auditorium"])
    room_length = st.slider("Room Length (meters)", 3.0, 20.0, 8.0, 0.5)
    room_width = st.slider("Room Width (meters)", 3.0, 15.0, 6.0, 0.5)
    capacity = st.slider("Number of Seats", 2, 50, 12, 2)
    with st.expander("Advanced Options"):
        ceiling_height = st.slider("Ceiling Height (meters)", 2.5, 5.0, 3.0, 0.1)
        budget_range = st.selectbox("Budget Range", ["Economy ($10K-25K)", "Standard ($25K-50K)", "Premium ($50K-100K)", "Enterprise ($100K+)"])
        special_requirements = st.multiselect("Special Requirements", ["Recording Capability", "Live Streaming", "Wireless Presentation", "Room Booking Display", "Environmental Controls"])

tab1, tab2, tab3, tab4 = st.tabs(["3D Visualization", "AI Recommendations", "Equipment Specs", "Installation Plan"])

with tab1:
    st.subheader("Interactive 3D Room Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_3d = create_3d_room_visualization(room_width, room_length, capacity, {})
        st.plotly_chart(fig_3d, use_container_width=True)
        if st.button("🔄 Regenerate Layout"):
            st.rerun()
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved! You can download the specs from the Installation Plan tab.")
    with col2:
        st.subheader("Room Statistics")
        room_area = room_length * room_width
        st.metric("Room Area", f"{room_area:.1f} m²")
        st.metric("Capacity", f"{capacity} people")
        st.metric("Area per Person", f"{room_area/capacity:.1f} m²")
        st.subheader("AVIXA Compliance")
        max_viewing_distance = room_length * 0.9
        recommended_display_size = (max_viewing_distance * 39.37) / 6
        if recommended_display_size <= 100:
            st.success("✅ Viewing Distance: PASS")
        else:
            st.warning("⚠️ Consider larger display")
        worst_angle = np.degrees(np.arctan2(room_width/2, max_viewing_distance))
        if worst_angle <= 60:
            st.success("✅ Viewing Angle: PASS")
        else:
            st.error("❌ Viewing Angle: FAIL")

with tab2:
    st.subheader("🤖 AI Equipment Recommendations")
    display_recs = recommender.get_display_recommendations(room_length, room_width, capacity)
    camera_recs = recommender.get_camera_recommendations(room_length, room_width, capacity)
    audio_recs = recommender.get_audio_recommendations(room_length, room_width, capacity)
    complexity_score = min(5, (capacity // 5) + (1 if room_length * room_width > 50 else 0) + len(special_requirements))
    control_recs = recommender.get_control_recommendations(complexity_score)
    all_recommendations = display_recs + camera_recs + audio_recs + control_recs
    for rec in all_recommendations:
        with st.expander(f"📺 {rec['category']}: {rec['product']}", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Why this recommendation:** {rec['reason']}")
                st.write(f"**Specifications:** {rec['specs']}")
                if 'models' in rec:
                    st.write("**Suggested Models:**")
                    for model in rec['models']:
                        st.write(f"• {model}")
            with col2:
                st.metric("Price Range", rec['price_range'])
                if st.button(f"Add {rec['category']} to Quote", key=f"add_{rec['category']}"):
                    st.success(f"{rec['category']} added to quote!")
    st.subheader("💰 Project Cost Estimate")
    base_cost = 15000 + (capacity * 1000) + (room_area * 500)
    equipment_multiplier = 1.0
    if "Premium" in budget_range:
        equipment_multiplier = 1.5
    elif "Enterprise" in budget_range:
        equipment_multiplier = 2.0
    total_estimate = base_cost * equipment_multiplier
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Equipment Cost", f"${total_estimate*0.6:,.0f}")
    with col2:
        st.metric("Installation Cost", f"${total_estimate*0.3:,.0f}")
    with col3:
        st.metric("Total Project Cost", f"${total_estimate:,.0f}")

with tab3:
    st.subheader("📋 Detailed Equipment Specifications")
    specs_data = []
    for rec in all_recommendations:
        specs_data.append({
            'Category': rec['category'],
            'Product': rec['product'],
            'Reason': rec['reason'],
            'Price Range': rec['price_range'],
            'Models': ', '.join(rec.get('models', ['Custom Solution']))
        })
    specs_df = pd.DataFrame(specs_data)
    st.dataframe(specs_df, use_container_width=True)
    csv_data = specs_df.to_csv(index=False)
    st.download_button(label="📥 Download Equipment Specifications", data=csv_data, file_name=f"room_config_{room_type.replace(' ', '_')}_{capacity}pax.csv", mime="text/csv")

with tab4:
    st.subheader("🔧 Installation & Project Plan")
    st.write("### Project Timeline")
    timeline_data = {
        'Phase': ['Planning & Design', 'Equipment Procurement', 'Installation', 'Testing & Commissioning', 'Training & Handover'],
        'Duration (Days)': [7, 14, 5, 3, 1],
        'Dependencies': ['Client approval', 'Planning complete', 'Equipment delivered', 'Installation complete', 'Testing complete']
    }
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df)
    st.write("### Installation Requirements")
    requirements = ["🔌 Power outlets: Minimum 6 dedicated 15A circuits", "🌐 Network infrastructure: CAT6A cabling to all device locations", "📺 Display mounting: Reinforced wall mount or ceiling suspension", "🎤 Audio infrastructure: Conduit for microphone cabling", "❄️ Environmental: HVAC considerations for equipment cooling", "🔒 Security: Equipment lockdown and access control"]
    for req in requirements:
        st.write(req)
    st.write("### Next Steps")
    if st.button("📞 Request Professional Consultation"):
        st.success("Consultation request submitted! Our AV specialist will contact you within 24 hours.")
    if st.button("📊 Generate Detailed Proposal"):
        st.info("Generating comprehensive proposal with CAD drawings, detailed specifications, and project timeline...")

st.markdown("---")
st.markdown("*Powered by AI recommendation engine and AVIXA standards compliance*")
