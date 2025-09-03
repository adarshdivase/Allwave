import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from fpdf import FPDF
import graphviz

# --- Page Configuration ---
st.set_page_config(page_title="AI Room Configurator Ultra", page_icon="💎", layout="wide")

# --- Custom CSS for Ultra Styling ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 12px;
        border-radius: 12px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e3c72 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    /* Custom card styling */
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        color: black !important;
    }
    .review-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ffd700;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Product Database (Managed via CSV) ---
@st.cache_data
def load_products():
    """Loads product data from a CSV file."""
    try:
        df = pd.read_csv('product_database.csv')
        products = {}
        for category, group in df.groupby('category'):
            products[category] = group.set_index('model').to_dict(orient='index')
        return products
    except FileNotFoundError:
        st.error("Fatal Error: `product_database.csv` not found. Please create it.")
        return None

# --- Advanced AI Recommendation Engine ---
class AdvancedAVRecommender:
    def __init__(self, products_db):
        self.db = products_db

    def get_ai_recommendations(self, room_specs: Dict) -> Dict:
        """Generate recommendations based on a scoring system."""
        budget = room_specs['budget']
        packages = {}
        # Define budget tiers (e.g., Value: 0-30%, Balanced: 30-70%, Premium: 70%+)
        budget_tiers = {'Value': (0, budget * 0.4), 'Balanced': (budget * 0.4, budget * 0.8), 'Premium': (budget * 0.8, float('inf'))}

        for tier_name, (min_b, max_b) in budget_tiers.items():
            display = self._score_and_select('displays', room_specs, max_b * 0.4)
            camera = self._score_and_select('cameras', room_specs, max_b * 0.25)
            audio = self._score_and_select('audio', room_specs, max_b * 0.2)
            control = self._score_and_select('control', room_specs, max_b * 0.15)
            
            if all([display, camera, audio, control]):
                packages[tier_name] = {
                    'display': self.db['displays'][display],
                    'camera': self.db['cameras'][camera],
                    'audio': self.db['audio'][audio],
                    'control': self.db['control'][control],
                    'total_price': sum(p['price'] for p in [self.db['displays'][display], self.db['cameras'][camera], self.db['audio'][audio], self.db['control'][control]])
                }
        
        # Select best package based on budget
        selected_package_name = room_specs.get('package_preference', 'Balanced')
        if selected_package_name not in packages: # Fallback
            selected_package_name = min(packages.keys(), key=lambda k: abs(packages[k]['total_price'] - budget)) if packages else None

        if not selected_package_name:
             return {'error': "Could not find a suitable package for the given budget and room specs."}

        final_rec = packages[selected_package_name]
        final_rec['package_name'] = selected_package_name
        final_rec['all_packages'] = packages # for comparison
        final_rec['confidence_score'] = self._calculate_confidence(room_specs, final_rec)
        return final_rec

    def _score_and_select(self, category: str, specs: Dict, budget_limit: float) -> Optional[str]:
        """Scores products in a category and returns the best model name."""
        scores = {}
        for model, product in self.db[category].items():
            if product['price'] > budget_limit * 1.5: # Allow some budget flexibility
                continue
                
            score = 0
            # Score 1: Price conformity
            price_score = max(0, 100 - (abs(product['price'] - budget_limit) / budget_limit) * 100)
            score += price_score * 0.4

            # Score 2: Capacity match
            min_cap, max_cap = product['min_capacity'], product['max_capacity']
            if min_cap <= specs['capacity'] <= max_cap:
                score += 100 * 0.3
            elif specs['capacity'] > max_cap:
                score += (max_cap / specs['capacity']) * 100 * 0.3 # Penalize if too small

            # Score 3: Room size match
            room_area = specs['length'] * specs['width']
            min_area, max_area = product['min_room_area'], product['max_room_area']
            if min_area <= room_area <= max_area:
                score += 100 * 0.2
            
            # Score 4: User Rating
            score += (product['rating'] / 5.0) * 100 * 0.1

            scores[model] = score

        return max(scores, key=scores.get) if scores else None
    
    def _calculate_confidence(self, specs, rec):
        # A more nuanced confidence score
        base_score = 80
        # Budget fit
        price_diff = abs(rec['total_price'] - specs['budget']) / specs['budget']
        base_score -= price_diff * 20 # Penalize for being far from budget
        # Capacity fit
        area_per_person = (specs['length'] * specs['width']) / specs['capacity']
        if not (2.0 <= area_per_person <= 4.5):
            base_score -= 10 # Penalize for awkward density
        return min(99, int(base_score))

# --- PDF Report Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI Room Configurator - Project Summary', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
        
def generate_pdf_report(specs, recommendations, costs, roi):
    pdf = PDF()
    pdf.add_page()
    
    # Summary
    pdf.chapter_title('Executive Summary')
    summary_text = f"""
This report details the AI-generated AV solution for a {specs['room_type']} with a capacity of {specs['capacity']} people.
Total Estimated Investment: ${costs['total']:,.2f}
Expected Payback Period: {roi['payback_months']:.1f} months
3-Year ROI: {roi['roi_3_years']:.0f}%
"""
    pdf.chapter_body(summary_text)

    # Equipment
    pdf.chapter_title('Recommended Equipment')
    for category, item in recommendations.items():
        if isinstance(item, dict) and 'model' in item:
            item_text = f"**{category.title()}**: {item['model']} - ${item['price']:,}\nSpecs: {item['specs']}"
            pdf.chapter_body(item_text)

    # Costs
    pdf.chapter_title('Financial Analysis')
    cost_text = "\n".join([f"{k.replace('_', ' ').title()}: ${v:,.2f}" for k, v in costs.items()])
    pdf.chapter_body(cost_text)
    
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


# --- All other classes (ProductDatabase, CostCalculator, etc.) and functions from the original script should be placed here ---
# Make sure to adapt them to use the new recommendation structure (e.g., recommendations['display']['price'])

# --- Main Application Interface ---
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #1e3c72; font-size: 3em; margin-bottom: 0;">💎 AI Room Configurator Ultra</h1>
        <p style="color: #666; font-size: 1.2em; margin-top: 0;">Precision-Engineered AV Solutions with Photorealistic Previews</p>
    </div>
    """, unsafe_allow_html=True)
    
    products_db = load_products()
    if not products_db:
        return

    # Session State Initialization
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    # Sidebar
    with st.sidebar:
        st.markdown("### 📏 Room Specifications")
        length = st.slider("Room Length (m)", 3.0, 25.0, 10.0, 0.5)
        width = st.slider("Room Width (m)", 3.0, 20.0, 7.0, 0.5)
        ceiling_height = st.slider("Ceiling Height (m)", 2.5, 6.0, 3.0, 0.1)
        capacity = st.slider("Seating Capacity", 2, 100, 16)
        
        st.markdown("### 💰 Financials")
        budget = st.slider("Total Project Budget ($)", 5000, 250000, 50000, 1000)
        package_preference = st.radio("Package Preference", ('Value', 'Balanced', 'Premium'), index=1)

        st.markdown("### 🏗️ Room Characteristics")
        room_type = st.selectbox("Room Type", ['Boardroom', 'Conference Room', 'Training Room', 'Auditorium', 'Huddle Space'])
        wall_material = st.selectbox("Primary Wall Material", ['Drywall', 'Glass', 'Concrete', 'Wood Paneling'])
        windows = st.slider("Window Area (% of wall surface)", 0, 80, 25)
        
        if st.button("🚀 Generate AI Solution", type="primary"):
            room_specs = {
                'length': length, 'width': width, 'ceiling_height': ceiling_height,
                'capacity': capacity, 'budget': budget, 'package_preference': package_preference,
                'room_type': room_type, 'wall_material': wall_material, 'windows': windows
            }
            st.session_state.room_specs = room_specs
            recommender = AdvancedAVRecommender(products_db)
            st.session_state.recommendations = recommender.get_ai_recommendations(room_specs)
            st.success("✅ AI Solution Generated!")

    # Main Content
    if st.session_state.recommendations:
        recs = st.session_state.recommendations
        specs = st.session_state.room_specs
        
        if 'error' in recs:
            st.error(recs['error'])
            return

        tabs = st.tabs(["🎯 Solution Overview", "🏗️ 3D Interactive Room", "💰 Financial Analysis", "🔊 Environmental Analysis", "🔌 Wiring Diagram", "📋 Project Summary"])

        with tabs[0]: # Solution Overview
             # Display the selected package details... (similar to your original recommendation tab)
             st.header(f"AI Recommended Solution: **{recs['package_name']}** Package")
             
        with tabs[1]: # 3D Interactive Room
            st.header("🏗️ Interactive 3D Photorealistic Preview")
            st.info("Rotate (left-click & drag), Pan (right-click & drag), Zoom (scroll).")
            
            # This is where we will embed the custom 3D viewer
            # Create a dictionary of 3D models and their positions
            scene_data = {
                "room": {"width": specs['width'], "length": specs['length'], "height": specs['ceiling_height']},
                "assets": [
                    {"model": "conference_table.gltf", "position": [0, 0, 0], "scale": [specs['width']*0.3, 1, specs['length']*0.5]},
                    {"model": "wall_display.gltf", "position": [0, 1.5, -specs['length']/2 + 0.1], "scale": [2, 1, 1]},
                    # Add more assets based on recommendations...
                ]
            }

            # For now, we'll just show the data. You would pass this to a custom component.
            with st.expander("See 3D Scene Data"):
                st.json(scene_data)
            st.warning("3D Viewer component not implemented in this snippet. You would use `st.components.v1.html` to load a custom viewer with the data above.")


        with tabs[4]: # Wiring Diagram
            st.header("🔌 System Wiring & Connectivity")
            dot = graphviz.Digraph(comment='AV System Diagram')
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Nodes
            dot.node('display', f"Display\n({recs['display']['model']})")
            dot.node('camera', f"Camera\n({recs['camera']['model']})")
            dot.node('audio', f"Audio DSP/Mic\n({recs['audio']['model']})")
            dot.node('control', f"Control Panel\n({recs['control']['model']})")
            dot.node('codec', 'Video Codec / PC')
            
            # Edges
            dot.edge('codec', 'display', label='HDMI/DisplayPort')
            dot.edge('camera', 'codec', label='USB')
            dot.edge('audio', 'codec', label='USB/Dante')
            dot.edge('control', 'codec', label='Network/PoE')
            
            st.graphviz_chart(dot)

        with tabs[5]: # Project Summary
            # Use the functional PDF generator
            # calculator = CostCalculator() # Assuming this class exists and is updated
            # cost_breakdown = calculator.calculate_total_cost(recs, specs)
            # roi_analysis = calculator.calculate_roi(cost_breakdown, specs)
            # pdf_data = generate_pdf_report(specs, recs, cost_breakdown, roi_analysis)
            
            # st.download_button(
            #    label="📄 Download Full PDF Report",
            #    data=pdf_data,
            #    file_name=f"AV_Project_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
            #    mime="application/pdf"
            # )
            st.info("PDF generation logic is ready. Uncomment in the code once CostCalculator is integrated.")
            # ... rest of the summary tab
    else:
        # Welcome Screen
        st.info("Configure your room in the sidebar and click 'Generate AI Solution' to begin.")

if __name__ == "__main__":
    main()
