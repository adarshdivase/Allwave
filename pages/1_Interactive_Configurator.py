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
    }
    .review-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #ffd700;
    }
</style>
""", unsafe_allow_html=True)


class ProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Samsung QMR Series 98"': {
                    'price': 12999,
                    'image': '📺',
                    'specs': '4K UHD, 500 nits, 24/7 operation, MagicINFO',
                    'rating': 4.8,
                    'reviews': [
                        {'user': 'Tech Director, Fortune 500', 'rating': 5, 'text': 'Crystal clear image quality, perfect for our boardroom'},
                        {'user': 'AV Manager', 'rating': 4.5, 'text': 'Excellent display, easy integration with control systems'}
                    ]
                },
                'LG LAEC Series 136" LED': {
                    'price': 45999,
                    'image': '🖥️',
                    'specs': 'All-in-One LED, 1.2mm pixel pitch, HDR10',
                    'rating': 4.9,
                    'reviews': [
                        {'user': 'Corporate AV Lead', 'rating': 5, 'text': 'Stunning visuals, no bezels, worth every penny'},
                        {'user': 'Integration Specialist', 'rating': 4.8, 'text': 'Best LED wall solution we\'ve deployed'}
                    ]
                },
                'Sony Crystal LED': {
                    'price': 89999,
                    'image': '💎',
                    'specs': 'MicroLED, 1.0mm pitch, 1800 nits, HDR',
                    'rating': 5.0,
                    'reviews': [
                        {'user': 'Executive Board', 'rating': 5, 'text': 'Absolutely breathtaking quality'},
                        {'user': 'CTO', 'rating': 5, 'text': 'The future of display technology'}
                    ]
                }
            },
            'cameras': {
                'Logitech Rally Plus': {
                    'price': 5999,
                    'image': '📹',
                    'specs': '4K PTZ, 15x zoom, AI auto-framing, dual speakers',
                    'rating': 4.7,
                    'reviews': [
                        {'user': 'IT Director', 'rating': 5, 'text': 'Best video quality in hybrid meetings'},
                        {'user': 'Facility Manager', 'rating': 4.5, 'text': 'Easy setup, great tracking'}
                    ]
                },
                'Poly Studio X70': {
                    'price': 8999,
                    'image': '🎥',
                    'specs': 'Dual 4K cameras, 120° FOV, NoiseBlockAI',
                    'rating': 4.8,
                    'reviews': [
                        {'user': 'VP Engineering', 'rating': 5, 'text': 'Exceptional AI director mode'},
                        {'user': 'Meeting Room Admin', 'rating': 4.6, 'text': 'Crystal clear even in large rooms'}
                    ]
                },
                'Cisco Room Kit Pro': {
                    'price': 15999,
                    'image': '🎬',
                    'specs': 'Triple camera, 5K video, speaker tracking',
                    'rating': 4.9,
                    'reviews': [
                        {'user': 'Enterprise Architect', 'rating': 5, 'text': 'Enterprise-grade reliability'},
                        {'user': 'AV Consultant', 'rating': 4.8, 'text': 'Best-in-class for large spaces'}
                    ]
                }
            },
            'audio': {
                'Shure MXA920': {
                    'price': 6999,
                    'image': '🎤',
                    'specs': 'Ceiling array, steerable coverage, IntelliMix DSP',
                    'rating': 4.9,
                    'reviews': [
                        {'user': 'Audio Engineer', 'rating': 5, 'text': 'Invisible yet perfect audio capture'},
                        {'user': 'Consultant', 'rating': 4.8, 'text': 'Game-changer for ceiling installations'}
                    ]
                },
                'Biamp Parlé': {
                    'price': 4999,
                    'image': '🔊',
                    'specs': 'Beamforming mic bar, AEC, AGC, Dante',
                    'rating': 4.7,
                    'reviews': [
                        {'user': 'Systems Integrator', 'rating': 4.8, 'text': 'Excellent DSP, clean audio'},
                        {'user': 'Tech Lead', 'rating': 4.6, 'text': 'Great for medium rooms'}
                    ]
                },
                'QSC Q-SYS': {
                    'price': 12999,
                    'image': '🎵',
                    'specs': 'Full ecosystem, network audio, advanced DSP',
                    'rating': 4.8,
                    'reviews': [
                        {'user': 'AV Director', 'rating': 5, 'text': 'Most flexible platform available'},
                        {'user': 'Integrator', 'rating': 4.6, 'text': 'Powerful but requires expertise'}
                    ]
                }
            }
        }


class AdvancedAVRecommender:
    def __init__(self):
        self.db = ProductDatabase()

    def get_ai_recommendations(self, room_specs: Dict) -> Dict:
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
        if room_area < 25:
            product_name = 'Samsung QMR Series 98"'
        elif room_area < 60:
            product_name = 'LG LAEC Series 136" LED'
        else:
            product_name = 'Sony Crystal LED'
        product_info = self.db.products['displays'][product_name]
        return {'primary': product_name, **product_info}

    def _recommend_camera(self, specs):
        if specs['capacity'] <= 8:
            product_name = 'Logitech Rally Plus'
        elif specs['capacity'] <= 16:
            product_name = 'Poly Studio X70'
        else:
            product_name = 'Cisco Room Kit Pro'
        product_info = self.db.products['cameras'][product_name]
        return {'primary': product_name, **product_info}

    def _recommend_audio(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        if room_volume < 75:
            product_name = 'Biamp Parlé'
        elif room_volume < 150:
            product_name = 'Shure MXA920'
        else:
            product_name = 'QSC Q-SYS'
        product_info = self.db.products['audio'][product_name]
        return {'primary': product_name, **product_info}

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
            accessories.append({'item': 'Wireless presentation', 'model': 'Barco ClickShare CX-50', 'price': 1999})
        if 'Recording' in specs.get('special_requirements', []):
            accessories.append({'item': 'Recording appliance', 'model': 'Epiphan Pearl Nexus', 'price': 8999})
        accessories.append({'item': 'Cable management', 'model': 'FSR Floor boxes', 'price': 999})
        return accessories

    def _calculate_confidence(self, specs):
        base_score = 85
        ratio = specs['length'] / specs['width']
        if 1.2 <= ratio <= 1.8:
            base_score += 10
        area_per_person = (specs['length'] * specs['width']) / specs['capacity']
        if 2.5 <= area_per_person <= 4:
            base_score += 5
        return min(100, base_score)


class CostCalculator:
    def __init__(self):
        self.labor_rates = {'standard': 200, 'premium': 300}

    def calculate_total_cost(self, recommendations, specs):
        equipment_cost = sum(rec.get('price', 0) for rec in recommendations.values() if isinstance(rec, dict))
        equipment_cost += sum(item.get('price', 0) for item in recommendations.get('accessories', []))
        
        complexity = specs.get('complexity_score', 3)
        labor_hours = 8 + (complexity * 4)
        labor_rate = self.labor_rates['premium'] if equipment_cost > 50000 else self.labor_rates['standard']
        installation_cost = labor_hours * labor_rate
        
        infrastructure = self._calculate_infrastructure(specs)
        training = 2000 if specs['capacity'] > 12 else 1000
        support_year1 = equipment_cost * 0.1
        
        total = equipment_cost + installation_cost + infrastructure + training + support_year1
        return {
            'equipment': equipment_cost,
            'installation': installation_cost,
            'infrastructure': infrastructure,
            'training': training,
            'support_year1': support_year1,
            'total': total
        }

    def _calculate_infrastructure(self, specs):
        base_cost = 5000
        room_area = specs['length'] * specs['width']
        area_multiplier = max(1.0, room_area / 50)
        special_cost = 0
        if 'Recording' in specs.get('special_requirements', []):
            special_cost += 3000
        if 'Live Streaming' in specs.get('special_requirements', []):
            special_cost += 2000
        return int(base_cost * area_multiplier + special_cost)

    def calculate_roi(self, cost_breakdown, specs):
        total_investment = cost_breakdown['total']
        
        meeting_hours_per_week = specs['capacity'] * 4
        hourly_rate = 75
        annual_meeting_cost = meeting_hours_per_week * 52 * hourly_rate
        
        av_efficiency_gain = 0.15
        travel_reduction = 0.3
        
        annual_savings = annual_meeting_cost * av_efficiency_gain
        annual_travel_budget = specs['capacity'] * 5000
        travel_savings = annual_travel_budget * travel_reduction
        
        total_annual_savings = annual_savings + travel_savings
        
        payback_period = (total_investment / total_annual_savings) if total_annual_savings > 0 else float('inf')
        roi_3_years = ((total_annual_savings * 3 - total_investment) / total_investment) * 100 if total_investment > 0 else float('inf')
        
        return {
            'annual_savings': total_annual_savings,
            'payback_months': payback_period * 12,
            'roi_3_years': roi_3_years
        }


def create_photorealistic_3d_room(specs):
    fig = go.Figure()
    w, l, h = specs['width'], specs['length'], specs['ceiling_height']
    
    # Floor - FIX: Use a valid Plotly colorscale
    fig.add_trace(go.Surface(x=np.linspace(0, w, 2), y=np.linspace(0, l, 2), z=np.zeros((2, 2)), colorscale='ylorbr', showscale=False))
    
    # Walls
    wall_color = 'rgb(245, 245, 245)'
    fig.add_trace(go.Surface(x=[[0, w], [0, w]], y=[[l, l], [l, l]], z=[[0, 0], [h, h]], colorscale=[[0, wall_color], [1, wall_color]], showscale=False))
    
    # Conference Table
    table_l, table_w, table_h = l * 0.5, w * 0.35, 0.75
    table_x, table_y = w / 2, l * 0.4
    fig.add_trace(go.Mesh3d(x=[table_x - table_w/2, table_x + table_w/2, table_x + table_w/2, table_x - table_w/2], y=[table_y - table_l/2, table_y - table_l/2, table_y + table_l/2, table_y + table_l/2], z=[table_h, table_h, table_h, table_h], i=[0, 0], j=[1, 2], k=[2, 3], color='saddlebrown'))

    fig.update_layout(
        title={'text': "3D Room Configuration", 'x': 0.5},
        scene=dict(
            xaxis=dict(title="Width (m)", showbackground=False),
            yaxis=dict(title="Length (m)", showbackground=False),
            zaxis=dict(title="Height (m)", showbackground=False),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.5)),
            aspectmode='data'
        ),
        showlegend=False, height=600, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_cost_breakdown_chart(cost_data, roi_analysis):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cost Breakdown', 'Payback Period'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Cost Breakdown Pie Chart
    cost_labels = [k for k in cost_data if k != 'total']
    cost_values = [cost_data[k] for k in cost_labels]
    fig.add_trace(go.Pie(labels=cost_labels, values=cost_values, hole=0.4, marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']), row=1, col=1)

    # Payback Period Bar Chart
    payback_months = roi_analysis['payback_months']
    scenarios = ['Conservative', 'Realistic', 'Optimistic']
    payback_values = [payback_months * 1.2, payback_months, payback_months * 0.8]
    fig.add_trace(go.Bar(x=scenarios, y=payback_values, marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']), row=1, col=2)

    fig.update_yaxes(title_text="Months to Payback", row=1, col=2)
    fig.update_layout(height=400, showlegend=False, title_text="Financial Analysis")
    return fig


def main():
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #1e3c72; font-size: 3em; margin-bottom: 0;">🏢 AI Room Configurator Pro</h1>
            <p style="color: #666; font-size: 1.2em; margin-top: 0;">Enterprise-Grade AV Solutions Powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'room_specs' not in st.session_state:
        st.session_state.room_specs = None

    with st.sidebar:
        st.markdown("### 📏 Room Specifications")
        length = st.slider("Room Length (m)", 3.0, 20.0, 8.0, 0.5)
        width = st.slider("Room Width (m)", 3.0, 15.0, 6.0, 0.5)
        ceiling_height = st.slider("Ceiling Height (m)", 2.5, 5.0, 3.0, 0.1)
        capacity = st.slider("Seating Capacity", 4, 50, 12)
        
        st.markdown("### ⚙️ Special Requirements")
        special_req = st.multiselect("Additional Features", ['Recording', 'Live Streaming', 'Wireless Presentation', 'Room Scheduling'])
        
        if st.button("🚀 Generate AI Recommendations", type="primary"):
            st.session_state.room_specs = {
                'length': length, 'width': width, 'ceiling_height': ceiling_height, 'capacity': capacity,
                'special_requirements': special_req, 'complexity_score': len(special_req) + (1 if capacity > 16 else 0)
            }
            recommender = AdvancedAVRecommender()
            st.session_state.recommendations = recommender.get_ai_recommendations(st.session_state.room_specs)
            st.success("✅ Recommendations generated successfully!")

    if st.session_state.recommendations is not None:
        recommendations = st.session_state.recommendations
        room_specs = st.session_state.room_specs
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI Recommendations", "📐 3D Visualization", "💰 Cost Analysis", "📋 Project Summary"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                for category in ['display', 'camera', 'audio', 'control']:
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
                    </div>
                    """, unsafe_allow_html=True)
                    if 'reviews' in rec:
                        with st.expander("Show Customer Reviews"):
                            for review in rec['reviews']:
                                st.markdown(f"""<div class="review-card"><strong>{review['user']}</strong> - {'⭐' * int(review['rating'])}<br><em>"{review['text']}"</em></div>""", unsafe_allow_html=True)

            with col2:
                confidence = recommendations['confidence_score']
                st.metric("AI Confidence Score", f"{confidence}%", delta="High Confidence" if confidence > 90 else "Good Match")
                
                # Correctly calculate total equipment cost for the metric card
                total_equipment_cost = sum(rec.get('price', 0) for rec in recommendations.values() if isinstance(rec, dict))
                total_equipment_cost += sum(item.get('price', 0) for item in recommendations.get('accessories', []))

                st.markdown(f"""
                <div class="metric-card">
                    <h4>Quick Stats</h4>
                    <p><strong>Total Equipment:</strong> ${total_equipment_cost:,}</p>
                    <p><strong>Room Capacity:</strong> {room_specs['capacity']} people</p>
                    <p><strong>Technology Grade:</strong> Enterprise</p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### 🏗️ Photorealistic 3D Room Preview")
            with st.spinner("Rendering 3D model..."):
                try:
                    fig_3d = create_photorealistic_3d_room(room_specs)
                    st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.error(f"3D visualization failed: {e}")

        with tab3:
            st.markdown("### 💰 Comprehensive Cost Analysis")
            calculator = CostCalculator()
            cost_breakdown = calculator.calculate_total_cost(recommendations, room_specs)
            roi_analysis = calculator.calculate_roi(cost_breakdown, room_specs)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Investment", f"${cost_breakdown['total']:,.0f}")
            c2.metric("Payback Period", f"{roi_analysis['payback_months']:.1f} months")
            c3.metric("3-Year ROI", f"{roi_analysis['roi_3_years']:.0f}%")

            fig_cost = create_cost_breakdown_chart(cost_breakdown, roi_analysis)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 Executive Project Summary")
            st.markdown("This AI-generated solution is optimized for your room's dimensions and requirements, ensuring a high-performance, future-ready AV system.")
            
            # Implementation Timeline
            st.markdown("#### 🗓️ Implementation Timeline")
            timeline_data = {
                'Phase': ['Planning & Design', 'Procurement', 'Installation', 'Testing & Training', 'Go-Live'],
                'Duration': ['2 weeks', '3-4 weeks', '1 week', '1 week', '1 day']
            }
            st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)

    else:
        st.info("🎯 Use the sidebar to enter your room details and click 'Generate AI Recommendations'.")


if __name__ == "__main__":
    main()
