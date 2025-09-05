import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any

# --- Initialize Session State ---
if 'session_state' not in st.session_state:
    st.session_state.session_state = {
        'form_values': {},
        'camera_position': None,
        'active_features': set(['table_space', 'budget_tracking', 'camera_angles']),
        'selections': {},
        'last_recommendations': None,
        'table_config': None
    }

# --- Helper Functions ---
def persist_form_value(key, value):
    """Helper function to persist form values"""
    if 'form_values' not in st.session_state.session_state:
        st.session_state.session_state['form_values'] = {}
    st.session_state.session_state['form_values'][key] = value
    return value

def get_camera_position(room_specs):
    """Get persisted camera position or a default scaled to the room size"""
    # This function is kept dynamic to provide a good initial view.
    # User interactions will then be persisted by Plotly's uirevision feature.
    return {
        'eye': {'x': -1.5 * room_specs['length'], 'y': -1.5 * room_specs['width'], 'z': 1.2 * room_specs['ceiling_height']},
        'center': {'x': 0.5 * room_specs['length'], 'y': 0.5 * room_specs['width'], 'z': 0.3 * room_specs['ceiling_height']},
        'up': {'x': 0, 'y': 0, 'z': 1}
    }

def validate_form_inputs(room_specs, selections):
    """Validate all form inputs"""
    errors = []
    warnings = []
    
    # Room dimension validations (using a safe default of 1.5)
    min_area_per_person = 1.5
    if (room_specs['width'] * room_specs['length']) < (room_specs['capacity'] * min_area_per_person):
        errors.append(f"Room size is too small for the specified capacity. A minimum of {room_specs['capacity'] * min_area_per_person:.1f}m² is recommended.")

    # Aspect Ratio validation
    if room_specs['width'] > 0:
        aspect_ratio = room_specs['length'] / room_specs['width']
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            warnings.append("Room's aspect ratio may be challenging for AV equipment placement and viewing angles.")
            
    # Budget validations
    if selections.get('budget_tier') == 'Budget' and \
       len(selections.get('special_features', [])) > 2:
        warnings.append("Multiple special features selected for a 'Budget' tier may compromise core component quality.")
        
    return errors, warnings

def create_feature_controls():
    """Create and manage feature controls"""
    features = {
        'table_space': {
            'label': 'Table Space Calculator',
            'description': 'Calculate and display optimal table dimensions in the analysis tab.'
        },
        'budget_tracking': {
            'label': 'Budget Tracking',
            'description': 'Enable real-time budget validation against selected tier limits.'
        },
        'camera_angles': {
            'label': 'Camera Angle Lock',
            'description': 'Lock the 3D view camera position between interactions.'
        }
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Feature Controls")
    
    active_features = st.session_state.session_state.get('active_features', set())
    
    for feature_id, feature in features.items():
        is_active = st.sidebar.checkbox(
            f"{feature['label']}",
            value=feature_id in active_features,
            help=feature['description'],
            key=f"feature_{feature_id}"
        )
        
        if is_active:
            active_features.add(feature_id)
        else:
            active_features.discard(feature_id)
    
    st.session_state.session_state['active_features'] = active_features


# --- Data and Logic Classes (Keep as they are) ---
class EnhancedProductDatabase:
    def __init__(self):
        self.products = {
            'displays': {
                'Budget': {
                    'BenQ IRP55': {'price': 1299, 'specs': '55" 4K Interactive Display, 20-point touch', 'rating': 4.2, 'brand': 'BenQ'},
                    'LG 75UP8000PUA': {'price': 1899, 'specs': '75" 4K LED, webOS, ThinQ AI', 'rating': 4.4, 'brand': 'LG'},
                    'Samsung QB65R': {'price': 2499, 'specs': '65" 4K QLED Business Display', 'rating': 4.5, 'brand': 'Samsung'}
                },
                'Professional': {
                    'Sharp/NEC 100" 4K Display': {'price': 15999, 'specs': '100" 4K UHD, 500 nits, 24/7 Operation', 'rating': 4.7, 'brand': 'Sharp/NEC'},
                    'Sony BRAVIA FW-85BZ40H': {'price': 8999, 'specs': '85" 4K Pro Display, Android TV', 'rating': 4.6, 'brand': 'Sony'},
                    'Planar UltraRes X Series': {'price': 12999, 'specs': '86" 4K Multi-touch Display', 'rating': 4.5, 'brand': 'Planar'}
                },
                'Premium': {
                    'LG MAGNIT 136"': {'price': 75000, 'specs': 'MicroLED, 4K, AI-powered processing, Cable-less', 'rating': 4.9, 'brand': 'LG'},
                    'Samsung "The Wall" 146"': {'price': 99999, 'specs': 'MicroLED, 4K, 0.8mm Pixel Pitch, AI Upscaling', 'rating': 5.0, 'brand': 'Samsung'},
                    'Sony Crystal LED 220"': {'price': 150000, 'specs': 'Crystal LED, 4K+, Seamless Modular Design', 'rating': 5.0, 'brand': 'Sony'}
                }
            },
            'cameras': {
                'Budget': {
                    'Logitech MeetUp': {'price': 899, 'specs': '4K Ultra HD, 120° FOV, Built-in Speakers', 'rating': 4.3, 'brand': 'Logitech'},
                    'Poly Studio P5': {'price': 699, 'specs': 'HD Webcam, Automatic Group Framing', 'rating': 4.2, 'brand': 'Poly'},
                    'Jabra PanaCast': {'price': 1199, 'specs': '4K Panoramic Camera, 180° FOV', 'rating': 4.4, 'brand': 'Jabra'}
                },
                'Professional': {
                    'Logitech Rally Bar': {'price': 3999, 'specs': '4K PTZ, AI Viewfinder, RightSight Auto-Framing', 'rating': 4.8, 'brand': 'Logitech'},
                    'Poly Studio E70': {'price': 4200, 'specs': 'Dual 4K sensors, Poly DirectorAI, Speaker Tracking', 'rating': 4.9, 'brand': 'Poly'},
                    'Aver CAM520 Pro3': {'price': 2899, 'specs': '4K PTZ, 18x Optical Zoom, AI Auto-Framing', 'rating': 4.6, 'brand': 'Aver'}
                },
                'Premium': {
                    'Cisco Room Kit EQ': {'price': 19999, 'specs': 'AI-powered Quad Camera, Speaker Tracking, Codec', 'rating': 5.0, 'brand': 'Cisco'},
                    'Crestron Flex UC-MM30-Z': {'price': 15999, 'specs': 'Advanced AI Camera, 4K PTZ, Zoom Integration', 'rating': 4.9, 'brand': 'Crestron'},
                    'Polycom Studio X70': {'price': 12999, 'specs': 'Dual 4K cameras, AI-powered Director', 'rating': 4.8, 'brand': 'Polycom'}
                }
            },
            'audio': {
                'Budget': {
                    'Yamaha YVC-1000': {'price': 1299, 'specs': 'USB/Bluetooth Speakerphone, Adaptive Echo Canceller', 'rating': 4.3, 'brand': 'Yamaha'},
                    'ClearOne CHAT 50': {'price': 599, 'specs': 'USB Speakerphone, Duplex Audio', 'rating': 4.1, 'brand': 'ClearOne'},
                    'Jabra Speak 750': {'price': 399, 'specs': 'UC Speakerphone, 360° Microphone', 'rating': 4.4, 'brand': 'Jabra'}
                },
                'Professional': {
                    'QSC Core Nano': {'price': 2500, 'specs': 'Network Audio I/O, Q-SYS Ecosystem, Software DSP', 'rating': 4.7, 'brand': 'QSC'},
                    'Biamp TesiraFORTE X 400': {'price': 4500, 'specs': 'AEC, Dante/AVB, USB Audio, Launch Config', 'rating': 4.8, 'brand': 'Biamp'},
                    'ClearOne BMA 360': {'price': 3299, 'specs': 'Beamforming Mic Array, 360° Coverage', 'rating': 4.6, 'brand': 'ClearOne'}
                },
                'Premium': {
                    'Shure MXA920 Ceiling Array': {'price': 6999, 'specs': 'Automatic Coverage, Steerable Coverage, IntelliMix DSP', 'rating': 5.0, 'brand': 'Shure'},
                    'Sennheiser TeamConnect Ceiling 2': {'price': 5999, 'specs': 'AI-Enhanced Audio, Beam Steering Technology', 'rating': 4.9, 'brand': 'Sennheiser'},
                    'Audio-Technica ATUC-50CU': {'price': 4999, 'specs': 'Ceiling Array, AI Noise Reduction', 'rating': 4.8, 'brand': 'Audio-Technica'}
                }
            },
            'control_systems': {
                'Budget': {
                    'Extron TouchLink Pro 725T': {'price': 1999, 'specs': '7" Touchpanel, PoE+, Web Interface', 'rating': 4.3, 'brand': 'Extron'},
                    'AMX Modero X Series NXD-700Vi': {'price': 2299, 'specs': '7" Touch Panel, Built-in Video', 'rating': 4.4, 'brand': 'AMX'},
                    'Crestron TSW-570': {'price': 1799, 'specs': '5" Touch Screen, Wi-Fi, PoE', 'rating': 4.5, 'brand': 'Crestron'}
                },
                'Professional': {
                    'Crestron Flex UC': {'price': 3999, 'specs': 'Tabletop Touchpanel, UC Integration', 'rating': 4.6, 'brand': 'Crestron'},
                    'AMX Enova DGX': {'price': 5999, 'specs': 'Digital Matrix Switching, Control System', 'rating': 4.7, 'brand': 'AMX'},
                    'Extron DTP3 CrossPoint': {'price': 7999, 'specs': '4K60 Matrix Switching, Advanced Control', 'rating': 4.8, 'brand': 'Extron'}
                },
                'Premium': {
                    'Crestron NVX System': {'price': 15999, 'specs': 'Enterprise Control Platform, AV over IP', 'rating': 4.9, 'brand': 'Crestron'},
                    'Q-SYS Core 8 Flex': {'price': 12999, 'specs': 'Unified AV/IT Platform, Software-based', 'rating': 5.0, 'brand': 'QSC'},
                    'Biamp Vocia MS-1': {'price': 18999, 'specs': 'Networked Paging System, Enterprise Grade', 'rating': 4.9, 'brand': 'Biamp'}
                }
            },
            'lighting': {
                'Budget': {
                    'Philips Hue Pro': {'price': 899, 'specs': 'Smart LED System, App Control', 'rating': 4.2, 'brand': 'Philips'},
                    'Lutron Caseta Pro': {'price': 1299, 'specs': 'Wireless Dimming System', 'rating': 4.4, 'brand': 'Lutron'},
                    'Leviton Decora Smart': {'price': 799, 'specs': 'Wi-Fi Enabled Switches and Dimmers', 'rating': 4.1, 'brand': 'Leviton'}
                },
                'Professional': {
                    'Crestron DIN-2MC2': {'price': 2999, 'specs': '2-Channel Dimmer, 0-10V Control', 'rating': 4.6, 'brand': 'Crestron'},
                    'Lutron Quantum': {'price': 4999, 'specs': 'Total Light Management System', 'rating': 4.8, 'brand': 'Lutron'},
                    'ETC ColorSource': {'price': 3999, 'specs': 'LED Architectural Lighting', 'rating': 4.7, 'brand': 'ETC'}
                },
                'Premium': {
                    'Ketra N4 Hub': {'price': 8999, 'specs': 'Natural Light Technology, Circadian Rhythm', 'rating': 5.0, 'brand': 'Lutron/Ketra'},
                    'USAI BeveLED': {'price': 12999, 'specs': 'Architectural LED Lighting System', 'rating': 4.9, 'brand': 'USAI'},
                    'Signify Interact Pro': {'price': 15999, 'specs': 'IoT-connected Lighting Management', 'rating': 4.8, 'brand': 'Signify'}
                }
            }
        }
        
        self.room_templates = {
            'Huddle Room (2-6 people)': {
                'typical_size': (3, 4), 'capacity_range': (2, 6),
                'recommended_tier': 'Budget', 'typical_usage': 'Quick meetings, brainstorming'
            },
            'Small Conference (6-12 people)': {
                'typical_size': (4, 6), 'capacity_range': (6, 12),
                'recommended_tier': 'Professional', 'typical_usage': 'Team meetings, presentations'
            },
            'Large Conference (12-20 people)': {
                'typical_size': (6, 10), 'capacity_range': (12, 20),
                'recommended_tier': 'Professional', 'typical_usage': 'Department meetings, training'
            },
            'Boardroom (8-16 people)': {
                'typical_size': (5, 8), 'capacity_range': (8, 16),
                'recommended_tier': 'Premium', 'typical_usage': 'Executive meetings, board meetings'
            },
            'Training Room (20-50 people)': {
                'typical_size': (8, 12), 'capacity_range': (20, 50),
                'recommended_tier': 'Professional', 'typical_usage': 'Training, workshops, seminars'
            },
            'Auditorium (50+ people)': {
                'typical_size': (12, 20), 'capacity_range': (50, 200),
                'recommended_tier': 'Premium', 'typical_usage': 'Large presentations, events'
            }
        }

class BudgetManager:
    def __init__(self, budget_tier):
        self.tier_limits = {
            'Budget': {'min': 5000, 'max': 25000},
            'Professional': {'min': 25000, 'max': 75000},
            'Premium': {'min': 75000, 'max': 500000}
        }
        self.current_tier = budget_tier
        self.running_total = 0
    
    def validate_item(self, item_cost):
        new_total = self.running_total + item_cost
        tier_limit = self.tier_limits[self.current_tier]['max']
        
        if 'budget_tracking' in st.session_state.session_state['active_features'] and new_total > tier_limit:
            return False, f"Budget exceeded for {self.current_tier} tier (${tier_limit:,})"
        
        self.running_total = new_total
        return True, None

class MaximizedAVRecommender:
    def __init__(self):
        self.db = EnhancedProductDatabase()
    
    def get_comprehensive_recommendations(self, room_specs: Dict, user_preferences: Dict, budget_manager: BudgetManager) -> Dict:
        budget_tier = user_preferences.get('budget_tier', 'Professional')
        brand_preference = user_preferences.get('preferred_brands', [])
        special_features = user_preferences.get('special_features', [])
        
        recommendations = {}
        
        # Sequentially recommend and validate budget
        recommendations['display'] = self._recommend_display_advanced(room_specs, budget_tier, brand_preference)
        budget_manager.validate_item(recommendations['display']['price'])
        
        recommendations['camera'] = self._recommend_camera_advanced(room_specs, budget_tier, brand_preference)
        budget_manager.validate_item(recommendations['camera']['price'])
        
        recommendations['audio'] = self._recommend_audio_advanced(room_specs, budget_tier, special_features)
        budget_manager.validate_item(recommendations['audio']['price'])
        
        recommendations['control'] = self._recommend_control_advanced(room_specs, budget_tier)
        budget_manager.validate_item(recommendations['control']['price'])
        
        recommendations['lighting'] = self._recommend_lighting_advanced(room_specs, budget_tier, user_preferences, special_features)
        budget_manager.validate_item(recommendations['lighting']['price'])
        
        recommendations['accessories'] = self._recommend_accessories_advanced(room_specs, special_features)
        for acc in recommendations['accessories']:
             budget_manager.validate_item(acc['price'])
        
        # Add other analysis parts
        recommendations['alternatives'] = self._generate_alternatives(room_specs, budget_tier)
        recommendations['confidence_score'] = self._calculate_advanced_confidence(room_specs, user_preferences)
        recommendations['room_analysis'] = self._analyze_room_characteristics(room_specs)
        recommendations['upgrade_path'] = self._suggest_upgrade_path(room_specs, budget_tier)
        
        return recommendations
    
    def _recommend_display_advanced(self, specs, tier, brands):
        room_area = specs['length'] * specs['width']
        viewing_distance = max(specs['length'], specs['width']) * 0.6
        
        if room_area < 20 and specs['capacity'] <= 8:
            size_category = 'small'
        elif room_area < 60 and specs['capacity'] <= 20:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        products = self.db.products['displays'][tier]
        
        if brands:
            products = {k: v for k, v in products.items() if v['brand'] in brands}
        
        if products:
            selected = list(products.items())[0]
            for name, product in products.items():
                if size_category == 'large' and ('100"' in name or 'Wall' in name):
                    selected = (name, product)
                    break
                elif size_category == 'medium' and any(size in name for size in ['75"', '85"', '86"']):
                    selected = (name, product)
                    break
        else:
            products = self.db.products['displays'][tier]
            selected = list(products.items())[0]
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['viewing_distance_optimal'] = f"{viewing_distance:.1f}m"
        result['brightness_needed'] = self._calculate_brightness_needs(specs)
        return result
    
    def _recommend_camera_advanced(self, specs, tier, brands):
        products = self.db.products['cameras'][tier]
        
        if brands:
            products = {k: v for k, v in products.items() if v['brand'] in brands}
        
        if not products:
            products = self.db.products['cameras'][tier]
        
        room_depth = max(specs['length'], specs['width'])
        
        if specs['capacity'] <= 6 and room_depth <= 5:
            camera_type = 'fixed'
        elif specs['capacity'] <= 16:
            camera_type = 'ptz'
        else:
            camera_type = 'multi_camera'
        
        selected = list(products.items())[0]
        for name, product in products.items():
            if camera_type == 'multi_camera' and ('EQ' in name or 'Studio X' in name):
                selected = (name, product)
                break
            elif camera_type == 'ptz' and ('Rally' in name or 'E70' in name):
                selected = (name, product)
                break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['recommended_mounting'] = self._suggest_camera_mounting(specs)
        result['coverage_analysis'] = self._analyze_camera_coverage(specs, camera_type)
        return result
    
    def _recommend_audio_advanced(self, specs, tier, features):
        products = self.db.products['audio'][tier]
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        acoustic_features = specs.get('environment', {}).get('acoustic_features', [])
        
        config_type = 'table_system' # Default
        if 'Sound Absorption Needed' in acoustic_features or 'Echo Control Required' in acoustic_features:
            config_type = 'premium_processing'
        elif room_volume > 200 or specs.get('environment', {}).get('ceiling_type') == "Open Plenum":
            config_type = 'distributed'
        elif specs['ceiling_height'] > 3.5:
            config_type = 'ceiling_array'
        
        selected = list(products.items())[0]
        for name, product in products.items():
            if config_type == 'ceiling_array' and ('MXA920' in name or 'Ceiling' in name):
                selected = (name, product)
                break
            elif config_type == 'premium_processing' and ('TesiraFORTE' in name or 'Core' in name):
                selected = (name, product)
                break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['configuration'] = self._design_audio_config(specs, config_type)
        result['acoustic_analysis'] = self._analyze_acoustics(specs)
        return result
    
    def _recommend_control_advanced(self, specs, tier):
        products = self.db.products['control_systems'][tier]
        complexity_score = len(specs.get('special_requirements', [])) + (specs['capacity'] // 10)
        complexity_score += len(specs.get('environment', {}).get('env_controls', []))

        selected = list(products.items())[0]
        if complexity_score > 4:
            for name, product in products.items():
                if 'NVX' in name or 'DGX' in name or 'Core' in name:
                    selected = (name, product)
                    break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['integration_options'] = self._suggest_integrations(specs)
        return result
    
    def _recommend_lighting_advanced(self, specs, tier, user_prefs, features):
        products = self.db.products['lighting'][tier]
        env_config = specs.get('environment', {})
        
        needs_daylight_sync = 'Circadian Lighting' in features or env_config.get('natural_light') in ['High', 'Very High']
        
        selected = list(products.items())[0]
        if needs_daylight_sync and tier == 'Premium':
            for name, product in products.items():
                if 'Ketra' in name or 'Natural Light' in product['specs']:
                    selected = (name, product)
                    break
        
        result = selected[1].copy()
        result['model'] = selected[0]
        result['lighting_analysis'] = self._analyze_lighting_needs(specs, features)
        return result
    
    def _recommend_accessories_advanced(self, specs, features):
        accessories = []
        accessibility_needs = specs.get('environment', {}).get('accessibility', [])
        
        accessories.append({'category': 'Cable Management', 'item': 'Under-table Cable Tray System', 'model': 'FSR FL-500P Series', 'price': 1299, 'necessity': 'Essential'})
        if 'Wireless Presentation' in features:
            accessories.append({'category': 'Wireless Presentation', 'item': 'Professional Wireless System', 'model': 'Barco ClickShare Conference CX-50', 'price': 2999, 'necessity': 'Required'})
        if 'Room Scheduling' in features:
            accessories.append({'category': 'Room Booking', 'item': 'Smart Room Panel', 'model': 'Crestron TSS-1070-B-S', 'price': 1899, 'necessity': 'Required'})
        if 'Hearing Loop System' in accessibility_needs:
            accessories.append({'category': 'Accessibility', 'item': 'Inductive Loop System', 'model': 'Williams AV PLR BP1', 'price': 1500, 'necessity': 'Required'})

        if specs['capacity'] > 16:
            accessories.append({'category': 'Power Management', 'item': 'Intelligent Power Distribution', 'model': 'Middle Atlantic UPS-2200R', 'price': 1599, 'necessity': 'Recommended'})
        
        return accessories
    
    def _generate_alternatives(self, specs, tier):
        alternatives = {}
        all_tiers = ['Budget', 'Professional', 'Premium']
        
        for alt_tier in all_tiers:
            if alt_tier != tier:
                alt_recs = {}
                for category in ['displays', 'cameras', 'audio']:
                    products = self.db.products[category][alt_tier]
                    alt_recs[category] = list(products.items())[0]
                alternatives[alt_tier] = alt_recs
        
        return alternatives
    
    def _analyze_room_characteristics(self, specs):
        room_area = specs['length'] * specs['width']
        aspect_ratio = max(specs['length'], specs['width']) / min(specs['length'], specs['width']) if min(specs['length'], specs['width']) > 0 else 1
        
        return {
            'size_category': self._categorize_room_size(room_area),
            'shape_analysis': self._analyze_room_shape(aspect_ratio),
            'acoustic_properties': self._estimate_acoustic_properties(specs),
            'lighting_challenges': self._identify_lighting_challenges(specs),
            'capacity_efficiency': self._analyze_capacity_efficiency(specs)
        }
    
    def _suggest_upgrade_path(self, specs, current_tier):
        tiers = ['Budget', 'Professional', 'Premium']
        current_index = tiers.index(current_tier)
        upgrade_path = []
        
        if current_index < len(tiers) - 1:
            next_tier = tiers[current_index + 1]
            upgrade_path.append({
                'phase': 'Short-term (6-12 months)', 'tier': next_tier, 'focus': 'Core AV upgrade',
                'estimated_cost': self._estimate_tier_cost(specs, next_tier) - self._estimate_tier_cost(specs, current_tier)
            })
        
        if current_index < len(tiers) - 2:
            ultimate_tier = tiers[-1]
            upgrade_path.append({
                'phase': 'Long-term (2-3 years)', 'tier': ultimate_tier, 'focus': 'Premium features & AI integration',
                'estimated_cost': self._estimate_tier_cost(specs, ultimate_tier) - self._estimate_tier_cost(specs, current_tier)
            })
        
        return upgrade_path

    def _generate_smart_upgrade_plan(self, specs, current_tier, estimated_cost):
        """Generate a detailed phased upgrade plan"""
        phases = {
            'Immediate (0-3 months)': {
                'priorities': [
                    'Essential software upgrades and licensing',
                    'Control system programming optimization',
                    'Staff training on current systems'
                ], 'cost_percentage': 0.15, 'focus': 'Maximizing current infrastructure'
            },
            'Phase 1 (3-6 months)': {
                'priorities': ['Display system upgrade','Camera system enhancement','Basic audio improvements'],
                'cost_percentage': 0.35, 'focus': 'Core AV capabilities'
            },
            'Phase 2 (6-9 months)': {
                'priorities': ['Advanced audio processing implementation','Lighting control system upgrade','Room automation integration'],
                'cost_percentage': 0.30, 'focus': 'Enhanced functionality'
            },
            'Final Phase (9-12 months)': {
                'priorities': ['Premium features activation','AI analytics integration','Complete system optimization'],
                'cost_percentage': 0.20, 'focus': 'Premium capabilities'
            }
        }
        for phase in phases.values():
            phase['budget'] = estimated_cost * phase['cost_percentage']
        roi_metrics = {
            'Productivity Gain': '15-20%', 'Energy Savings': '10-15%',
            'Maintenance Cost Reduction': '25-30%', 'System Downtime Reduction': '40-50%'
        }
        return {'phases': phases, 'roi_metrics': roi_metrics, 'total_investment': estimated_cost, 'monthly_investment': estimated_cost / 12 if estimated_cost > 0 else 0}
    
    def _calculate_brightness_needs(self, specs):
        env = specs.get('environment', {})
        light_levels = {"Very Low": -50, "Low": -25, "Moderate": 0, "High": 50, "Very High": 100}
        natural_light_adjust = light_levels.get(env.get('natural_light', 'Moderate'), 0)
        return int(350 + natural_light_adjust)
    
    def _suggest_camera_mounting(self, specs):
        return "Ceiling mount recommended for optimal coverage" if specs['ceiling_height'] > 3.5 else "Wall mount at display location"
    
    def _analyze_camera_coverage(self, specs, camera_type):
        room_area = specs['length'] * specs['width']
        coverage_factor = 80 if camera_type == 'multi_camera' else 50
        coverage = min(100, (room_area / coverage_factor) * 100)
        return f"{coverage:.0f}% optimal coverage"

    def _design_audio_config(self, specs, config_type):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        return {
            'type': config_type, 'coverage': f"{min(100, room_volume / 2):.0f}%",
            'microphone_count': max(2, specs['capacity'] // 4), 'speaker_zones': max(1, specs['capacity'] // 8),
            'processing_power': 'High' if room_volume > 150 else 'Standard'
        }
    
    def _analyze_acoustics(self, specs):
        env = specs.get('environment', {})
        wall_absorb = {"Drywall": 0.1, "Glass": 0.03, "Concrete": 0.02, "Wood Panels": 0.15, "Acoustic Panels": 0.8}
        absorption_coeff = wall_absorb.get(env.get('wall_material', 'Drywall'), 0.1)
        
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        surface_area = 2 * (specs['length'] * specs['width'] + specs['length'] * specs['ceiling_height'] + specs['width'] * specs['ceiling_height'])
        rt60_estimate = 0.161 * room_volume / (absorption_coeff * surface_area) if absorption_coeff > 0 else float('inf')
        return {
            'rt60_estimate': f"{rt60_estimate:.2f} seconds", 'acoustic_treatment_needed': rt60_estimate > 0.8,
            'sound_masking_recommended': 'Speech Privacy Important' in env.get('acoustic_features', []),
            'echo_risk': 'High' if env.get('wall_material') in ['Glass', 'Concrete'] else 'Low'
        }
    
    def _suggest_integrations(self, specs):
        integrations = ['Microsoft Teams', 'Zoom', 'Google Meet']
        if specs['capacity'] > 16:
            integrations.extend(['Cisco Webex', 'BlueJeans'])
        if 'VTC' in specs.get('special_requirements', []):
            integrations.append('Polycom RealPresence')
        return integrations
    
    def _analyze_lighting_needs(self, specs, features):
        env = specs.get('environment', {})
        natural_light = env.get('natural_light', 'Moderate')
        return {
            'natural_light_factor': f"{natural_light}",
            'artificial_light_zones': max(1, (specs['length'] * specs['width']) // 20),
            'dimming_required': True,
            'color_temperature_control': 'Circadian Lighting' in features or env.get('color_scheme') != 'Neutral',
            'daylight_harvesting': 'Daylight Harvesting' in env.get('env_controls', [])
        }
    
    def _calculate_advanced_confidence(self, specs, preferences):
        base_confidence = 0.85
        if len(specs.get('special_requirements', [])) > 3: base_confidence -= 0.1
        if preferences.get('preferred_brands'): base_confidence += 0.05
        if preferences.get('budget_tier') == 'Premium': base_confidence += 0.1
        return min(0.99, max(0.70, base_confidence))
    
    def _categorize_room_size(self, area):
        if area < 20: return "Small (Huddle)"
        elif area < 50: return "Medium (Conference)"
        elif area < 100: return "Large (Training)"
        else: return "Extra Large (Auditorium)"
    
    def _analyze_room_shape(self, aspect_ratio):
        if aspect_ratio < 1.3: return "Square - Good for collaboration"
        elif aspect_ratio < 2.0: return "Rectangular - Versatile layout"
        else: return "Long/Narrow - Challenging for AV"
    
    def _estimate_acoustic_properties(self, specs):
        room_volume = specs['length'] * specs['width'] * specs['ceiling_height']
        return {
            'reverb_category': 'High' if room_volume > 200 else 'Moderate' if room_volume > 80 else 'Low',
            'treatment_needed': room_volume > 150, 'echo_potential': max(specs['length'], specs['width']) > 10
        }
    
    def _identify_lighting_challenges(self, specs):
        challenges = []
        if specs.get('environment', {}).get('natural_light') in ['High', 'Very High']: challenges.append("High natural light - glare control needed")
        if specs['ceiling_height'] > 4: challenges.append("High ceiling - requires powerful fixtures")
        if 'Presentations' in specs.get('environment', {}).get('room_purpose', []): challenges.append("Presentation mode requires zoned lighting")
        return challenges if challenges else ["Standard lighting requirements"]
    
    def _analyze_capacity_efficiency(self, specs):
        efficiency = specs['capacity'] / (specs['length'] * specs['width']) if (specs['length'] * specs['width']) > 0 else 0
        if efficiency > 0.5: return "High density - space optimization good"
        elif efficiency > 0.3: return "Moderate density - balanced layout"
        else: return "Low density - spacious environment"
    
    def _estimate_tier_cost(self, specs, tier):
        base_costs = {'Budget': 15000, 'Professional': 45000, 'Premium': 120000}
        return int(base_costs[tier] * (1 + (specs['capacity'] / 50)))

class EnhancedVisualizationEngine:
    def calculate_table_requirements(self, room_specs):
        capacity = room_specs['capacity']
        space_per_person_perimeter = 0.75 
        min_table_length = max(capacity * 0.6, 2.0)
        min_table_width = max(capacity * 0.3, 1.0)
        max_length = room_specs['length'] * 0.7
        max_width = room_specs['width'] * 0.4
        table_length = min(min_table_length, max_length)
        table_width = min(min_table_width, max_width)
        perimeter_seats = int((table_length * 2 + table_width * 2) / space_per_person_perimeter)
        return {
            'length': table_length,
            'width': table_width,
            'area': table_length * table_width,
            'seats': min(capacity, perimeter_seats)
        }

    def create_3d_room_visualization(self, room_specs, recommendations, viz_config):
        fig = go.Figure()

        length, width, height = room_specs['length'], room_specs['width'], room_specs['ceiling_height']

        # Floor, walls, and ceiling rendering
        fig.add_trace(go.Surface(
            x=[[0, length], [0, length]], y=[[0, 0], [width, width]], z=[[0, 0], [0, 0]],
            colorscale=[[0, 'rgb(245, 245, 240)'], [1, 'rgb(245, 245, 240)']],
            showscale=False, name='Floor', hoverinfo='skip'
        ))
        fig.add_trace(go.Surface(
            x=[[0, 0], [0, 0]], y=[[0, width], [0, width]], z=[[0, 0], [height, height]],
            colorscale=[[0, 'rgb(210, 220, 215)'], [1, 'rgb(210, 220, 215)']],
            showscale=False, name='Back Wall', hoverinfo='skip'
        ))
        fig.add_trace(go.Surface(
            x=[[0, length], [0, length]], y=[[0, 0], [0, 0]], z=[[0, 0], [height, height]],
            colorscale=[[0, 'rgb(225, 230, 225)'], [1, 'rgb(225, 230, 225)']],
            showscale=False, name='Left Wall', opacity=0.8, hoverinfo='skip'
        ))
        fig.add_trace(go.Surface(
            x=[[0, length], [0, length]], y=[[width, width], [width, width]], z=[[0, 0], [height, height]],
            colorscale=[[0, 'rgb(225, 230, 225)'], [1, 'rgb(225, 230, 225)']],
            showscale=False, name='Right Wall', opacity=0.8, hoverinfo='skip'
        ))
        fig.add_trace(go.Surface(
            x=[[0, length], [0, length]], y=[[0, 0], [width, width]], z=[[height, height], [height, height]],
            colorscale=[[0, 'rgb(230, 230, 240)'], [1, 'rgb(230, 230, 240)']],
            showscale=False, name='Ceiling', opacity=0.9, hoverinfo='skip'
        ))
        
        # Display Screen and Frame
        screen_width = min(3, width * 0.6)
        screen_height = screen_width * 9 / 16
        screen_y_start = (width - screen_width) / 2
        screen_z_start = height * 0.3
        display_customdata_str = f"{screen_width:.1f}m × {screen_height:.1f}m"
        fig.add_trace(go.Surface(
            x=[[0.05, 0.05], [0.05, 0.05]],
            y=[[screen_y_start, screen_y_start + screen_width], [screen_y_start, screen_y_start + screen_width]],
            z=[[screen_z_start, screen_z_start], [screen_z_start + screen_height, screen_z_start + screen_height]],
            colorscale=[[0, 'rgb(25, 25, 25)'], [1, 'rgb(25, 25, 25)']], showscale=False, name='Display Screen',
            hovertemplate='<b>Display System</b><br>Size: %{customdata}<extra></extra>',
            customdata=np.full((2, 2), display_customdata_str)
        ))
        bezel_thickness = 0.02
        fig.add_trace(go.Surface(
            x=[[0.04, 0.04], [0.04, 0.04]],
            y=[[screen_y_start - bezel_thickness, screen_y_start + screen_width + bezel_thickness],
               [screen_y_start - bezel_thickness, screen_y_start + screen_width + bezel_thickness]],
            z=[[screen_z_start - bezel_thickness, screen_z_start - bezel_thickness],
               [screen_z_start + screen_height + bezel_thickness, screen_z_start + screen_height + bezel_thickness]],
            colorscale=[[0, 'rgb(50, 50, 50)'], [1, 'rgb(50, 50, 50)']], showscale=False, name='Display Frame', hoverinfo='skip'
        ))
        
        # Conference Table
        table_config = st.session_state.session_state.get('table_config')
        if table_config:
            table_length = table_config['length']
            table_width = table_config['width']
        else:
            table_length = min(length * 0.7, 4)
            table_width = min(width * 0.4, 1.5)

        table_height = 0.75
        table_x_center = length * 0.6
        table_y_center = width * 0.5
        table_customdata_str = f"{table_length:.1f}m × {table_width:.1f}m"

        fig.add_trace(go.Surface(
            x=[[table_x_center - table_length / 2, table_x_center + table_length / 2],
               [table_x_center - table_length / 2, table_x_center + table_length / 2]],
            y=[[table_y_center - table_width / 2, table_y_center - table_width / 2],
               [table_y_center + table_width / 2, table_y_center + table_width / 2]],
            z=[[table_height, table_height], [table_height, table_height]],
            colorscale=[[0, 'rgb(139, 115, 85)'], [1, 'rgb(160, 130, 95)']],
            showscale=False, name='Conference Table',
            hovertemplate='<b>Conference Table</b><br>Size: %{customdata}<extra></extra>',
            customdata=np.full((2, 2), table_customdata_str)
        ))

        # Other elements (Camera, Chairs, Speakers, etc.)
        camera_width = screen_width * 0.8; camera_y_center = width * 0.5; camera_z = screen_z_start + screen_height + 0.15
        camera_y_coords = np.linspace(camera_y_center - camera_width / 2, camera_y_center + camera_width / 2, 20)
        camera_x = np.full_like(camera_y_coords, 0.1); camera_z_coords = np.full_like(camera_y_coords, camera_z)
        camera_model = recommendations.get('camera', {}).get('model', 'Professional Camera')
        fig.add_trace(go.Scatter3d(
            x=camera_x, y=camera_y_coords, z=camera_z_coords, mode='markers',
            marker=dict(size=4, color='rgb(60, 60, 60)', symbol='square'), name='Camera System',
            hovertemplate='<b>Camera System</b><br>Model: %{customdata}<extra></extra>', customdata=[camera_model] * len(camera_x)
        ))
        capacity = min(room_specs['capacity'], 12); chairs_per_side = min(6, capacity // 2)
        chair_x_positions = []; chair_y_positions = []; chair_z_positions = []
        if chairs_per_side > 0:
            for i in range(chairs_per_side):
                chair_x = table_x_center - table_length / 2 + (i + 1) * table_length / (chairs_per_side + 1)
                chair_x_positions.append(chair_x); chair_y_positions.append(table_y_center - table_width / 2 - 0.4); chair_z_positions.append(0.85)
                if len(chair_x_positions) < capacity:
                    chair_x_positions.append(chair_x); chair_y_positions.append(table_y_center + table_width / 2 + 0.4); chair_z_positions.append(0.85)
        fig.add_trace(go.Scatter3d(
            x=chair_x_positions, y=chair_y_positions, z=chair_z_positions, mode='markers',
            marker=dict(size=8, color='rgb(70, 130, 180)', symbol='square', opacity=0.8),
            name=f'Seating ({len(chair_x_positions)} chairs)', hovertemplate='<b>Chair</b><br>Position: %{x:.1f}, %{y:.1f}<extra></extra>'
        ))
        speaker_positions = [(length * 0.25, width * 0.25), (length * 0.75, width * 0.25), (length * 0.25, width * 0.75), (length * 0.75, width * 0.75)]
        speaker_x = [p[0] for p in speaker_positions]; speaker_y = [p[1] for p in speaker_positions]; speaker_z = [height - 0.1] * 4
        fig.add_trace(go.Scatter3d(
            x=speaker_x, y=speaker_y, z=speaker_z, mode='markers',
            marker=dict(size=6, color='rgb(220, 220, 220)', symbol='circle', line=dict(color='rgb(100, 100, 100)', width=1)),
            name='Ceiling Speakers', hovertemplate='<b>Ceiling Speaker</b><br>Zone Coverage<extra></extra>'
        ))
        light_rows = 2; light_cols = 3; light_x = []; light_y = []; light_z = []
        for i in range(light_rows):
            for j in range(light_cols):
                light_x.append(length * (i + 1) / (light_rows + 1)); light_y.append(width * (j + 1) / (light_cols + 1)); light_z.append(height - 0.05)
        fig.add_trace(go.Scatter3d(
            x=light_x, y=light_y, z=light_z, mode='markers',
            marker=dict(size=5, color='rgb(255, 255, 200)', symbol='circle', opacity=0.8),
            name='LED Lighting', hovertemplate='<b>LED Light</b><br>Recessed Ceiling Mount<extra></extra>'
        ))
        fig.add_trace(go.Scatter3d(
            x=[length - 0.1], y=[width * 0.85], z=[height * 0.35], mode='markers',
            marker=dict(size=10, color='rgb(240, 240, 240)', symbol='square', line=dict(color='rgb(150, 150, 150)', width=2)),
            name='Touch Control Panel', hovertemplate='<b>Control Panel</b><br>Wall-mounted Touch Interface<extra></extra>'
        ))

        camera_position = get_camera_position(room_specs)
        
        fig.update_layout(
            title=dict(text="Professional Conference Room - 3D Layout", x=0.5, font=dict(size=16, color='#2c3e50')),
            scene=dict(
                xaxis=dict(title=dict(text=f"Length ({length:.1f}m)", font=dict(size=12)), showgrid=False, showbackground=False, showline=False, showticklabels=True, range=[0, length + 0.5]),
                yaxis=dict(title=dict(text=f"Width ({width:.1f}m)", font=dict(size=12)), showgrid=False, showbackground=False, showline=False, showticklabels=True, range=[0, width + 0.5]),
                zaxis=dict(title=dict(text=f"Height ({height:.1f}m)", font=dict(size=12)), showgrid=False, showbackground=False, showline=False, showticklabels=True, range=[0, height + 0.2]),
                bgcolor='rgb(248, 249, 250)',
                aspectmode='data'
            ),
            height=600, showlegend=True,
            legend=dict(
                x=1.02, y=0.8,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1,
                font=dict(size=11, color='black')
            ),
            margin=dict(l=0, r=120, t=50, b=0),
            paper_bgcolor='rgb(255, 255, 255)', plot_bgcolor='rgb(255, 255, 255)',
            scene_camera=camera_position,
            uirevision='camera_lock' if 'camera_angles' in st.session_state.session_state['active_features'] else None
        )

        return fig

    @staticmethod
    def create_equipment_layout_2d(room_specs, recommendations):
        fig = go.Figure()
        
        length, width = room_specs['length'], room_specs['width']
        
        # Room Outline and Background
        fig.add_shape(type="rect", x0=-0.2, y0=-0.2, x1=length+0.2, y1=width+0.2, line=dict(color="rgba(200,200,200,0.5)", width=2), fillcolor="rgba(240,240,240,0.3)", layer='below')
        fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color="rgb(70,70,70)", width=3), fillcolor="rgba(250,250,250,1)")
        
        # Windows
        if room_specs.get('environment', {}).get('windows', 0) > 0:
            window_sections = int(room_specs.get('environment', {}).get('windows', 0) / 20)
            if window_sections > 0:
                window_width_section = width / (window_sections * 2) 
                for i in range(window_sections):
                    y_start = (width / 2) - (window_sections * window_width_section / 2) + (i * window_width_section * 2)
                    fig.add_shape(type="rect", x0=length-0.1, y0=y_start, x1=length, y1=y_start + window_width_section, line=dict(color="rgb(150,200,255)", width=2), fillcolor="rgba(200,230,255,0.7)")

        # Equipment Shapes
        screen_width_2d = min(width * 0.6, 3.5)
        screen_start = (width - screen_width_2d) / 2
        fig.add_shape(type="rect", x0=0, y0=screen_start, x1=0.15, y1=screen_start + screen_width_2d, line=dict(color="rgb(50,50,50)", width=2), fillcolor="rgb(80,80,80)")
        
        table_length, table_width = min(length * 0.7, 4.5), min(width * 0.4, 1.5)
        table_x, table_y = length * 0.6, width * 0.5
        fig.add_shape(type="rect", x0=table_x - table_length/2, y0=table_y - table_width/2, x1=table_x + table_length/2, y1=table_y + table_width/2, line=dict(color="rgb(120,85,60)", width=2), fillcolor="rgb(139,115,85)")
        
        # Chairs
        capacity = min(room_specs['capacity'], 12)
        chairs_per_side = min(6, capacity // 2)
        chair_positions = []
        if chairs_per_side > 0:
            for i in range(chairs_per_side):
                x_pos = table_x - table_length/2 + ((i + 1) * table_length/(chairs_per_side + 1))
                chair_positions.extend([(x_pos, table_y - table_width/2 - 0.4), (x_pos, table_y + table_width/2 + 0.4)])
        
        for x, y in chair_positions[:capacity]:
            fig.add_shape(type="circle", x0=x-0.25, y0=y-0.25, x1=x+0.25, y1=y+0.25, line=dict(color="rgb(70,130,180)"), fillcolor="rgb(100,149,237)")
        
        # Coverage Zones
        camera_points = [[0.1, width*0.45], [0.1, width*0.55], [length*0.8, width*0.2], [length*0.8, width*0.8]]
        fig.add_shape(type="path", path=f"M {camera_points[0][0]},{camera_points[0][1]} L {camera_points[1][0]},{camera_points[1][1]} L {camera_points[3][0]},{camera_points[3][1]} L {camera_points[2][0]},{camera_points[2][1]} Z", line=dict(color="rgba(100,200,100,0.3)", width=1), fillcolor="rgba(100,200,100,0.1)")
        
        speaker_positions = [(length*0.25, width*0.25), (length*0.75, width*0.25), (length*0.25, width*0.75), (length*0.75, width*0.75)]
        for x, y in speaker_positions:
            fig.add_shape(type="circle", x0=x-1.5, y0=y-1.5, x1=x+1.5, y1=y+1.5, line=dict(color="rgba(100,100,255,0.1)"), fillcolor="rgba(100,100,255,0.05)")
        
        # Annotations
        annotations = [
            dict(x=0.1, y=width*0.5, text="Display", showarrow=True, arrowhead=2, ax=40, ay=-30),
            dict(x=length*0.5, y=width*0.1, text="Camera Coverage", showarrow=False, font=dict(color="green", size=10)),
            dict(x=length*0.8, y=width*0.5, text="Audio Coverage", showarrow=False, font=dict(color="blue", size=10))
        ]
        
        fig.update_layout(
            title=dict(text="Enhanced Floor Plan with Equipment Layout", y=0.95, x=0.5, xanchor='center'),
            xaxis=dict(title="Length (m)", range=[-1, length+1], scaleanchor="y", scaleratio=1, showspikes=False),
            yaxis=dict(title="Width (m)", range=[-1, width+1], showspikes=False),
            height=750,
            showlegend=False,
            annotations=annotations,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=80, b=50, l=50, r=50),
            dragmode='pan'
        )
        
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

# --- Main Application Function ---
def main():
    # Keep the CSS markdown here
    st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # Keep your full CSS here

    st.title("🏢 AI Room Configurator Pro Max")
    st.markdown("### Transform Your Space with Intelligent AV Design")

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="premium-card" style="padding: 1.5rem; margin-top: -50px;"><h2>🎛️ Room Configuration</h2></div>', unsafe_allow_html=True)
        
        form = st.session_state.session_state.get('form_values', {})

        template = persist_form_value('template', 
            st.selectbox("Room Template", 
                         list(EnhancedProductDatabase().room_templates.keys()), 
                         index=list(EnhancedProductDatabase().room_templates.keys()).index(form.get('template', 'Small Conference (6-12 people)')),
                         help="Choose a template to start.")
        )
        template_info = EnhancedProductDatabase().room_templates[template]

        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        length = persist_form_value('length',
            col1.slider("Length (m)", 2.0, 20.0, form.get('length', float(template_info['typical_size'][0])), 0.5)
        )
        width = persist_form_value('width',
            col2.slider("Width (m)", 2.0, 20.0, form.get('width', float(template_info['typical_size'][1])), 0.5)
        )
        ceiling_height = persist_form_value('ceiling_height',
            col1.slider("Ceiling Height (m)", 2.4, 6.0, form.get('ceiling_height', 3.0), 0.1)
        )
        capacity = persist_form_value('capacity',
            col2.slider("Capacity", 2, 100, form.get('capacity', template_info['capacity_range'][1]))
        )

        st.markdown("---")
        st.subheader("🌟 Environment & Atmosphere")
        env_col1, env_col2 = st.columns(2)
        with env_col1:
            windows = persist_form_value('windows', 
                st.slider("Windows (%)", 0, 80, form.get('windows', 20), 5)
            )
            natural_light = persist_form_value('natural_light', 
                st.select_slider("Natural Light Level", options=["Very Low", "Low", "Moderate", "High", "Very High"], value=form.get('natural_light', "Moderate"))
            )
        with env_col2:
            ceiling_type = persist_form_value('ceiling_type',
                st.selectbox("Ceiling Type", ["Standard", "Drop Ceiling", "Open Plenum", "Acoustic Tiles"], index=["Standard", "Drop Ceiling", "Open Plenum", "Acoustic Tiles"].index(form.get('ceiling_type', 'Standard')))
            )
            wall_material = persist_form_value('wall_material',
                st.selectbox("Wall Material", ["Drywall", "Glass", "Concrete", "Wood Panels", "Acoustic Panels"], index=["Drywall", "Glass", "Concrete", "Wood Panels", "Acoustic Panels"].index(form.get('wall_material', 'Drywall')))
            )
        
        room_purpose = persist_form_value('room_purpose',
            st.multiselect("Primary Activities", ["Video Conferencing", "Presentations", "Training", "Board Meetings", "Collaborative Work", "Hybrid Meetings"], default=form.get('room_purpose', ["Video Conferencing", "Presentations"]))
        )
        acoustic_features = persist_form_value('acoustic_features',
            st.multiselect("Acoustic Considerations", ["Sound Absorption Needed", "Echo Control Required", "External Noise Issues", "Speech Privacy Important"], default=form.get('acoustic_features', []))
        )
        env_controls = persist_form_value('env_controls',
            st.multiselect("Control Systems", ["Automated Lighting", "Motorized Shades", "Climate Control", "Occupancy Sensors", "Daylight Harvesting"], default=form.get('env_controls', []))
        )
        color_scheme_temp = persist_form_value('color_scheme',
            st.select_slider("Color Temperature", options=["Warm", "Neutral", "Cool"], value=form.get('color_scheme', "Neutral"))
        )
        design_style = persist_form_value('design_style',
            st.selectbox("Interior Design Style", ["Modern Corporate", "Executive", "Creative/Tech", "Minimalist"], index=["Modern Corporate", "Executive", "Creative/Tech", "Minimalist"].index(form.get('design_style', 'Modern Corporate')))
        )
        accessibility = persist_form_value('accessibility',
            st.multiselect("Accessibility Requirements", ["Wheelchair Access", "Hearing Loop System", "High Contrast Displays", "Voice Control"], default=form.get('accessibility', []))
        )

        st.markdown("---")
        st.subheader("💰 Budget & Brands")
        budget_tier = persist_form_value('budget_tier',
            st.selectbox("Budget Tier", ['Budget', 'Professional', 'Premium'], index=['Budget', 'Professional', 'Premium'].index(form.get('budget_tier', 'Professional')))
        )
        preferred_brands = persist_form_value('preferred_brands',
            st.multiselect("Preferred Brands", ['Samsung', 'LG', 'Sony', 'Crestron', 'Cisco', 'Logitech', 'QSC', 'Shure'], default=form.get('preferred_brands', []))
        )
        special_features = persist_form_value('special_features',
            st.multiselect("Required Features", ['Wireless Presentation', 'Digital Whiteboard', 'Room Scheduling', 'Noise Reduction', 'AI Analytics'], default=form.get('special_features', []))
        )
        
        create_feature_controls()

    if st.button("🚀 Generate AI Recommendation"):
        form_values = st.session_state.session_state['form_values']
        environment_config = {
            'windows': form_values.get('windows'),
            'natural_light': form_values.get('natural_light'),
            'ceiling_type': form_values.get('ceiling_type'),
            'wall_material': form_values.get('wall_material'),
            'room_purpose': form_values.get('room_purpose'),
            'acoustic_features': form_values.get('acoustic_features'),
            'env_controls': form_values.get('env_controls'),
            'color_scheme': form_values.get('color_scheme'),
            'design_style': form_values.get('design_style'),
            'accessibility': form_values.get('accessibility')
        }
        room_specs = {
            'template': form_values.get('template'),
            'length': form_values.get('length'),
            'width': form_values.get('width'),
            'ceiling_height': form_values.get('ceiling_height'),
            'capacity': form_values.get('capacity'),
            'environment': environment_config,
            'special_requirements': []
        }
        user_preferences = {
            'budget_tier': form_values.get('budget_tier'),
            'preferred_brands': form_values.get('preferred_brands'),
            'special_features': form_values.get('special_features')
        }

        errors, warnings = validate_form_inputs(room_specs, user_preferences)

        if errors:
            st.error("🚨 Please correct the following errors:\n\n* " + "\n* ".join(errors))
        else:
            if warnings:
                st.warning("⚠️ Consider these warnings:\n\n* " + "\n* ".join(warnings))
            
            try:
                recommender = MaximizedAVRecommender()
                viz_engine = EnhancedVisualizationEngine()
                
                table_config = viz_engine.calculate_table_requirements(room_specs)
                st.session_state.session_state['table_config'] = table_config
                
                budget_manager = BudgetManager(user_preferences['budget_tier'])
                recommendations = recommender.get_comprehensive_recommendations(
                    room_specs, user_preferences, budget_manager
                )
                
                st.session_state.session_state['last_recommendations'] = recommendations
                st.session_state.session_state['room_specs'] = room_specs
                st.session_state.session_state['budget_tier'] = user_preferences['budget_tier']
                
                st.success("✅ AI Analysis Complete!")
                
            except Exception as e:
                st.error(f"An error occurred during recommendation generation: {str(e)}")

    if st.session_state.session_state.get('last_recommendations'):
        recommendations = st.session_state.session_state['last_recommendations']
        room_specs = st.session_state.session_state['room_specs']
        budget_tier = st.session_state.session_state['budget_tier']
        recommender = MaximizedAVRecommender()

        total_cost = sum(rec['price'] for rec in recommendations.values() if isinstance(rec, dict) and 'price' in rec)
        for acc in recommendations.get('accessories', []):
            total_cost += acc['price']

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Investment", f"${total_cost:,.0f}")
        with col2: st.metric("AI Confidence", f"{recommendations['confidence_score']:.0%}")
        with col3: st.metric("Room Size", f"{room_specs['length']}m × {room_specs['width']}m")
        with col4: st.metric("Capacity", f"{room_specs['capacity']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recommendations", "📊 Analysis", "🎨 Visualization", "🔄 Alternatives", "📋 Report"])
        
        with tab1:
            # ... (Tab 1 display code remains the same)
            pass
        with tab2:
            # ... (Tab 2 display code remains the same, including the TypeError fix)
            pass
        with tab3:
            # ... (Tab 3 display code remains the same)
            pass
        with tab4:
            # ... (Tab 4 display code remains the same)
            pass
        with tab5:
            # ... (Tab 5 display code remains the same)
            pass

if __name__ == "__main__":
    main()
