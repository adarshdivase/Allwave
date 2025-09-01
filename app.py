import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np

# --- 1. App Setup and Data Loading ---
st.set_page_config(layout="wide")
st.title("Build Your Perfect Meeting Room ✨")
st.write("Select your room type, choose your options, and instantly see a recommended layout and equipment list.")

@st.cache_data
def load_oem_data():
    """Loads the OEM list for generating recommendations."""
    try:
        # Using a more generic name for the oem list file
        return pd.read_csv("av_oem_list_2025.csv", encoding='latin1', engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        st.error("OEM data file not found. Please ensure 'av_oem_list_2025.csv' is in the folder.")
        return None

oem_list_df = load_oem_data()

# --- 2. Visualization and Recommendation Engines ---
def create_room_visualization(room_width, room_length, capacity):
    """Generates a 2D top-down visualization that updates in real-time."""
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
        table_length = max(1, room_length * 0.6)
        table_width = max(1, room_width * 0.4)
        table_x = (room_width - table_width) / 2
        table_y = (room_length - table_length) / 2.5
        for i in range(num_seats_per_side):
            if num_seats_per_side > 1:
                y_pos = table_y + (i * table_length / (num_seats_per_side - 1))
            else:
                y_pos = table_y + table_length / 2
        
            ax.add_patch(Circle((table_x - 0.3, y_pos), radius=0.25, facecolor='gray'))
            ax.add_patch(Circle((table_x + table_width + 0.3, y_pos), radius=0.25, facecolor='gray'))
    
    ax.set_xlim(-1, room_width + 1); ax.set_ylim(-1, room_length + 1)
    ax.set_xticks([]); ax.set_yticks([])
    plt.title(f"Layout for a {capacity}-Person Room")
    return fig

def generate_recommendation(room_type, tier, add_wireless_pres, add_scheduler):
    """Generates a tailored BOQ based on customer selections."""
    if oem_list_df is None:
        return pd.DataFrame()

    playbooks = {
        "Huddle Room (2-4 People)": ["UC & Collaboration Devices", "Displays & Projectors"],
        "Conference Room (5-12 People)": ["UC & Collaboration Devices", "Displays & Projectors", "Audio: Microphones & Conferencing"],
        "Boardroom (12-20 People)": ["PTZ & Pro Video Cameras", "Displays & Projectors", "Audio: DSP & Mixing", "Control Systems"],
        "Town Hall / Large Space": ["PTZ & Pro Video Cameras", "Video Wall Technology", "Audio: Speakers & Amplifiers", "Audio: DSP & Mixing"]
    }
    brand_tiers = {"Standard": ["Logitech", "Poly", "Samsung"], "Premium": ["Cisco", "Crestron", "Shure", "Biamp"]}
    
    template = playbooks.get(room_type, [])
    boq_items = []

    for category in template:
        brands = oem_list_df[oem_list_df['Category'] == category]
        selected_brands = brands[brands['OEM / Brand'].str.contains('|'.join(brand_tiers[tier]), case=False)]
        brand = selected_brands['OEM / Brand'].iloc[0] if not selected_brands.empty else (brands['OEM / Brand'].iloc[0] if not brands.empty else "N/A")
        boq_items.append({"Component": category, "Recommendation": brand})

    if add_wireless_pres:
        boq_items.append({"Component": "Wireless Presentation", "Recommendation": "Barco ClickShare / Mersive Solstice"})
    if add_scheduler:
        boq_items.append({"Component": "Room Scheduling", "Recommendation": "Crestron / Logitech / Evoko"})

    return pd.DataFrame(boq_items)

# --- 3. Main Application ---
# Sidebar for customer inputs
st.sidebar.header("Design Your Room")
room_type = st.sidebar.selectbox(
    "1. Choose a Room Type",
    ["Huddle Room (2-4 People)", "Conference Room (5-12 People)", "Boardroom (12-20 People)", "Town Hall / Large Space"]
)

# Set room dimensions and capacity based on room type
if "Huddle" in room_type:
    capacity = 4; length = 4; width = 3
elif "Conference" in room_type:
    capacity = 10; length = 6; width = 4.5
elif "Boardroom" in room_type:
    capacity = 16; length = 9; width = 6
else: # Town Hall
    capacity = 30; length = 15; width = 10

tier = st.sidebar.selectbox("2. Choose a Quality Tier", ["Standard", "Premium"])

st.sidebar.header("Add-on Features")
add_wireless_pres = st.sidebar.checkbox("Wireless Presentation System")
add_scheduler = st.sidebar.checkbox("Room Scheduling Panel")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Room Visualization")
    fig = create_room_visualization(width, length, capacity)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Your Recommended Equipment")
    recommendation_df = generate_recommendation(room_type, tier, add_wireless_pres, add_scheduler)
    st.table(recommendation_df)
    st.info("This is a preliminary budget estimate. Our experts will provide a detailed, formal quote.")

if st.sidebar.button("Request a Formal Quote"):
    st.sidebar.success("Thank you! A design consultant will contact you shortly.")
