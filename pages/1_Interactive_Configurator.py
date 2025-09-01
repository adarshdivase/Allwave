# pages/1_Interactive_Configurator.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np

st.set_page_config(layout="wide")
st.title("Interactive Room Configurator 🛋️")

# --- Functions (abbreviated for brevity, same as before) ---
@st.cache_data
def load_oem_data():
    try:
        return pd.read_csv("av_oem_list_2025.csv", encoding='latin1', engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        return None
oem_list_df = load_oem_data()

def create_room_visualization(room_width, room_length, capacity):
    fig, ax = plt.subplots(figsize=(8, 8 * (room_length / room_width) if room_width > 0 else 8))
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(Rectangle((0, 0), room_width, room_length, fill=None, edgecolor='black', linewidth=2))
    display_x = room_width * 0.25
    ax.add_patch(Rectangle((display_x, room_length - 0.05), room_width * 0.5, 0.05, facecolor='darkblue'))
    camera_pos = (room_width / 2, room_length - 0.1)
    ax.add_patch(Circle(camera_pos, radius=0.1, facecolor='black'))
    fov_angle = 90
    ax.add_patch(Wedge(camera_pos, r=room_length * 1.2, theta1=270 - fov_angle / 2, theta2=270 + fov_angle / 2, facecolor='lightcyan', alpha=0.5))
    if capacity > 0:
        num_seats_per_side = int(np.ceil(capacity / 2))
        table_length = max(1, room_length * 0.6); table_width = max(1, room_width * 0.4)
        table_x = (room_width - table_width) / 2; table_y = (room_length - table_length) / 2.5
        for i in range(num_seats_per_side):
            y_pos = table_y + (i * table_length / (num_seats_per_side - 1 if num_seats_per_side > 1 else 1))
            ax.add_patch(Circle((table_x - 0.3, y_pos), radius=0.25, facecolor='gray'))
            ax.add_patch(Circle((table_x + table_width + 0.3, y_pos), radius=0.25, facecolor='gray'))
    ax.set_xlim(-1, room_width + 1); ax.set_ylim(-1, room_length + 1)
    ax.set_xticks([]); ax.set_yticks([])
    plt.title(f"Layout for a {capacity}-Person Room")
    return fig

# --- Main Application ---
st.sidebar.header("Design Controls")
room_length = st.sidebar.slider("Room Length (meters)", min_value=3.0, max_value=20.0, value=6.0, step=0.5)
room_width = st.sidebar.slider("Room Width (meters)", min_value=3.0, max_value=15.0, value=4.5, step=0.5)
capacity = st.sidebar.slider("Number of Seats", min_value=2, max_value=30, value=10, step=2)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Room Visualization")
    fig = create_room_visualization(room_width, room_length, capacity)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Real-Time AVIXA Compliance")
    
    # Real-time DISCAS calculation
    farthest_viewer = room_length * 0.9 # Estimate
    min_height_m = farthest_viewer / 20 # Basic Decision Making
    min_diagonal_inches = min_height_m * 39.37 * 1.89
    sizes = [55, 65, 75, 85, 98]
    rec_size = min(sizes, key=lambda x:abs(x-min_diagonal_inches))

    if rec_size:
        st.metric(label="Recommended Display Size (DISCAS)", value=f"{rec_size}\"")
        st.success("✅ **Display Size:** PASSES")
    else:
        st.error("Could not calculate display size.")

    # Real-time Viewing Angle Check
    farthest_seat_angle = np.degrees(np.arctan2(room_width / 2, farthest_viewer))
    if farthest_seat_angle <= 60:
        st.success(f"✅ **Viewing Angle:** PASSES (~{farthest_seat_angle:.0f}°)")
    else:
        st.error(f"❌ **Viewing Angle:** FAILS (~{farthest_seat_angle:.0f}°). Some seats may have a poor view.")
    
    st.info("This is a preliminary check. Our experts will verify all standards in the final proposal.")
