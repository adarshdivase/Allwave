# pages/1_Interactive_Configurator.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Interactive Configurator", layout="wide")

# --- SHARED FUNCTIONS AND CLASSES (Could be moved to a utils.py) ---

class ProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Samsung QMR 85"': {'price': 7999, 'image_url': 'https://image-us.samsung.com/SamsungUS/samsungbusiness/products/displays/4k-uhd/qm-r-series/85-qm85r/LH85QMRBGC-GO-GALLERY-600x600.jpg', 'specs': '4K UHD, 500 nits, 24/7', 'rating': 4.7},
                'LG LAEC 136" LED': {'price': 45999, 'image_url': 'https://www.lg.com/us/business/images/commercial-tvs/md_laec015-bu_136_all-in-one_dvled_display_e_d_v1/gallery/medium01.jpg', 'specs': 'All-in-One LED, 1.2mm pitch', 'rating': 4.9},
                'Sony Crystal LED': {'price': 89999, 'image_url': 'https://pro.sony/s3/2019/09/17154508/c-series-main-image-1.png?w=640', 'specs': 'MicroLED, 1800 nits, HDR', 'rating': 5.0}
            },
            'cameras': {
                'Logitech Rally Bar Mini': {'price': 3999, 'image_url': 'https://www.logitech.com/content/dam/logitech/en/products/video-conferencing/rally-bar-mini/gallery/rally-bar-mini-gallery-1-graphite.png', 'specs': '4K PTZ, AI Viewfinder', 'rating': 4.6},
                'Poly Studio X70': {'price': 8999, 'image_url': 'https://www.poly.com/content/dam/www/products/video/studio/studio-x70/poly-studio-x70-callouts-8-1-desktop.png.thumb.1280.1280.png', 'specs': 'Dual 4K cameras, NoiseBlockAI', 'rating': 4.8},
                'Cisco Room Kit Pro': {'price': 15999, 'image_url': 'https://www.cisco.com/c/en/us/products/collaboration-endpoints/webex-room-series/room-kit-pro/jcr:content/Grid/category_content/layout_0_1/layout-0-0/anchor_2/image.img.png/1632345582372.png', 'specs': 'Triple camera, 5K video', 'rating': 4.9}
            },
            'audio': {
                'Biamp Parlé': {'price': 4999, 'image_url': 'https://www.biamp.com/assets/2/15/Main-Images/Parle_VBC_2500_Front_1.png', 'specs': 'Beamforming mic bar, AEC', 'rating': 4.7},
                'Shure MXA920': {'price': 6999, 'image_url': 'https://www.shure.com/images/pr/press_release_mxa920_square_and_round_form_factors_white_grey/image_large_transparent', 'specs': 'Ceiling array, IntelliMix DSP', 'rating': 4.9},
                'QSC Q-SYS': {'price': 12999, 'image_url': 'https://www.qsc.com/resource-files/productimages/sys-cor-nv32h-f-m.png', 'specs': 'Full ecosystem, network audio', 'rating': 4.8}
            }
        }

class AIRecommender:
    def __init__(self):
        self.db = ProductDatabase()

    def get_recommendations(self, specs):
        rec = {}
        # Display logic
        view_dist = specs['length'] * 0.9
        if view_dist < 5: rec['display'] = self.db.products['displays']['Samsung QMR 85"']
        elif view_dist < 8: rec['display'] = self.db.products['displays']['LG LAEC 136" LED']
        else: rec['display'] = self.db.products['displays']['Sony Crystal LED']
        # Camera logic
        if specs['capacity'] <= 8: rec['camera'] = self.db.products['cameras']['Logitech Rally Bar Mini']
        elif specs['capacity'] <= 20: rec['camera'] = self.db.products['cameras']['Poly Studio X70']
        else: rec['camera'] = self.db.products['cameras']['Cisco Room Kit Pro']
        # Audio logic
        vol = specs['length'] * specs['width'] * specs['ceiling_height']
        if vol < 100: rec['audio'] = self.db.products['audio']['Biamp Parlé']
        elif vol < 250: rec['audio'] = self.db.products['audio']['Shure MXA920']
        else: rec['audio'] = self.db.products['audio']['QSC Q-SYS']
        
        rec['confidence_score'] = min(99, int(85 + (specs['length']/specs['width'])*2 + (20/specs['capacity'])*2 + random.uniform(-2, 2)))
        return rec

def create_cuboid(center, size, color='grey'):
    dx, dy, dz = size[0]/2, size[1]/2, size[2]/2
    x, y, z = center
    return go.Mesh3d(x=[x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx], y=[y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy], z=[z-dz, z-dz, z-dz, z-dz, z+dz, z+dz, z+dz, z+dz], i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color=color, opacity=1.0, flatshading=True, showlegend=False)

def create_3d_room(specs):
    fig = go.Figure()
    w, l, h = specs['width'], specs['length'], specs['ceiling_height']
    # Room Shell
    fig.add_trace(go.Mesh3d(x=[0, w, w, 0], y=[0, 0, l, l], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3], color='tan', opacity=0.7))
    fig.add_trace(go.Mesh3d(x=[0, w, w, 0], y=[l, l, l, l], z=[0, 0, h, h], i=[0, 0], j=[1, 2], k=[2, 3], color='lightgrey', opacity=0.7))
    fig.add_trace(go.Mesh3d(x=[0, 0, 0, 0], y=[0, l, l, 0], z=[0, 0, h, h], i=[0, 0], j=[1, 2], k=[2, 3], color='whitesmoke', opacity=0.7))
    
    # Audio Coverage Heatmap
    grid_x, grid_y = np.mgrid[0:w:30j, 0:l:30j]
    mic_pos = (w/2, l/3)
    distance = np.sqrt((grid_x - mic_pos[0])**2 + (grid_y - mic_pos[1])**2)
    heatmap_z = np.exp(-distance**2 / (2. * (l/3)**2)) # Gaussian falloff
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=np.full_like(heatmap_z, 0.01), surfacecolor=heatmap_z, colorscale='RdYlGn', cmin=0.1, cmax=1, showscale=False))

    # Furniture
    table_l, table_w, table_h = l*0.6, w*0.4, 0.75
    table_x, table_y = w/2, l*0.45
    fig.add_trace(create_cuboid((table_x, table_y, table_h-0.05), (table_w, table_l, 0.1), 'rgb(139, 69, 19)'))
    fig.add_trace(create_cuboid((table_x, table_y, (table_h-0.1)/2), (table_w*0.5, table_l*0.5, table_h-0.1), 'rgb(105,105,105)'))

    # Display & Camera
    display_w, display_h = w*0.7, w*0.7*(9/16)
    fig.add_trace(create_cuboid((w/2, l-0.05, 1.5), (display_w, 0.05, display_h), 'black'))
    fig.add_trace(create_cuboid((w/2, l-0.04, 1.5), (display_w*0.95, 0.05, display_h*0.9), '#313689'))
    fig.add_trace(create_cuboid((w/2, l-0.2, display_h+1.6), (0.2, 0.15, 0.15), 'darkgrey'))

    fig.update_layout(scene=dict(xaxis=dict(range=[0,w]), yaxis=dict(range=[0,l]), zaxis=dict(range=[0,h]), aspectmode='data', camera=dict(eye=dict(x=-1.5, y=-2.5, z=1.2))), height=600, margin=dict(l=0,r=0,b=0,t=0))
    return fig

def generate_pdf_report(specs, recs, costs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, 'AI Room Configurator Pro - Project Report', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, '1. Room Specifications', 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    for key, value in specs.items():
        pdf.cell(0, 8, f"  - {key.replace('_', ' ').title()}: {value}", 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, '2. AI Recommended Equipment', 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    for cat, details in recs.items():
        if isinstance(details, dict):
            product_name = next(iter(details))
            pdf.cell(0, 8, f"  - {cat.title()}: {product_name} (${details[product_name]['price']:,})", 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, '3. Financial Summary', 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    for key, value in costs.items():
        if isinstance(value, (int, float)):
             pdf.cell(0, 8, f"  - {key.replace('_', ' ').title()}: ${value:,.2f}", 0, 1)
    
    return pdf.output(dest='S').encode('latin1')

# --- INITIALIZE SESSION STATE ---
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'room_specs' not in st.session_state:
    st.session_state.room_specs = {
        'length': 8.0, 'width': 6.0, 'ceiling_height': 3.0, 'capacity': 12
    }

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://i.imgur.com/832fiQZ.png", width=150)
    st.title("Room Configurator")
    st.markdown("Define your room's parameters.")

    st.header("Project Presets")
    c1, c2 = st.columns(2)
    if c1.button("Huddle Room"):
        st.session_state.room_specs = {'length': 4.0, 'width': 3.0, 'ceiling_height': 2.8, 'capacity': 4}
    if c2.button("Exec Boardroom"):
        st.session_state.room_specs = {'length': 12.0, 'width': 7.0, 'ceiling_height': 3.5, 'capacity': 18}

    st.header("Room Specifications")
    specs = st.session_state.room_specs
    specs['length'] = st.slider("Room Length (m)", 3.0, 20.0, specs['length'])
    specs['width'] = st.slider("Room Width (m)", 3.0, 15.0, specs['width'])
    specs['ceiling_height'] = st.slider("Ceiling Height (m)", 2.5, 5.0, specs['ceiling_height'])
    specs['capacity'] = st.slider("Seating Capacity", 2, 50, specs['capacity'])

    if st.button("🚀 Generate AI Recommendations", type="primary", use_container_width=True):
        st.session_state.room_specs = specs
        recommender = AIRecommender()
        st.session_state.recommendations = recommender.get_recommendations(specs)
        st.success("Recommendations Generated!")

# --- MAIN PAGE LAYOUT ---
st.title("Interactive AV Configurator")

if st.session_state.recommendations is None:
    st.info("Adjust the settings in the sidebar and click 'Generate AI Recommendations' to begin.")
    st.image("https://i.imgur.com/5b2gCdv.png", caption="An example of a modern conference room.")
else:
    recs = st.session_state.recommendations
    specs = st.session_state.room_specs

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📐 3D Visualization", "💰 Cost Analysis", "🔊 Environmental", "📋 Project Summary"])

    with tab1:
        st.header("AI-Powered Equipment Recommendations")
        st.metric("AI Confidence Score", f"{recs['confidence_score']}%")
        for category, details in recs.items():
            if isinstance(details, dict):
                product_name = list(details.keys())[0]
                product_info = details[product_name]
                st.subheader(f"{category.title()}: {product_name}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(product_info['image_url'], caption=product_name)
                with col2:
                    st.markdown(f"**Price:** `${product_info['price']:,}`")
                    st.markdown(f"**Rating:** {'⭐' * int(product_info['rating'])} ({product_info['rating']}/5.0)")
                    st.markdown(f"**Key Specs:** {product_info['specs']}")
                st.divider()

    with tab2:
        st.header("Photorealistic 3D Room Preview")
        st.info(" heatmap on the floor represents the ideal audio coverage from the microphone system.", icon="💡")
        fig_3d = create_3d_room(specs)
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        st.header("Comprehensive Cost & ROI Analysis")
        equipment_cost = sum(d[list(d.keys())[0]]['price'] for d in recs.values() if isinstance(d, dict))
        install_cost = equipment_cost * 0.25
        total_investment = equipment_cost + install_cost
        annual_savings = (specs['capacity'] * 5 * 50) * 0.15 * 52 # 15% efficiency gain
        payback_months = (total_investment / annual_savings) * 12 if annual_savings > 0 else 0
        
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Equipment Cost", f"${equipment_cost:,.0f}")
        c2.metric("Install & Services", f"${install_cost:,.0f}")
        c3.metric("Total Investment", f"${total_investment:,.0f}")
        c4.metric("Payback Period", f"{payback_months:.1f} Months")

    with tab4:
        st.header("Environmental & Performance Analysis")
        # Simplified analysis
        rt60 = (specs['length']*specs['width']*specs['ceiling_height']) / (specs['length']*specs['width']*2 * 0.3)
        heat_load = specs['capacity']*100 + equipment_cost * 0.5
        st.warning(f"Estimated Reverb Time (RT60): **{rt60:.2f}s**. Consider acoustic treatment if > 0.6s.", icon="🔊")
        st.warning(f"Estimated Heat Load: **{heat_load:.0f}W**. Ensure adequate HVAC capacity.", icon="🔥")

    with tab5:
        st.header("Executive Project Summary")
        costs = {"equipment_cost": equipment_cost, "install_cost": install_cost, "total_investment": total_investment, "payback_months": payback_months}
        pdf_bytes = generate_pdf_report(specs, recs, costs)
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"AV_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
