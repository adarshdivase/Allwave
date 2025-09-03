# pages/2_AI_Support_Chatbot_Enhanced.py
import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Tuple
import PyPDF2
import fitz  # PyMuPDF - better PDF handling
from io import BytesIO
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

# Import maintenance system components
import sys
sys.path.append('.')

warnings.filterwarnings("ignore")

# --- Enhanced RAG Configuration ---
class RAGConfig:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 75
        self.top_k_retrieval = 6
        self.similarity_threshold = 0.15
        self.max_context_length = 2500

config = RAGConfig()

# --- Predictive Maintenance Integration ---
@dataclass
class MaintenanceConfig:
    """Configuration for predictive maintenance system"""
    risk_threshold_high: float = 0.7
    risk_threshold_medium: float = 0.4
    prediction_horizon_days: int = 30
    min_confidence: float = 0.6

maintenance_config = MaintenanceConfig()

class MaintenancePipeline:
    """Pipeline connecting chatbot with predictive maintenance system"""
    
    def __init__(self):
        self.equipment_profiles = {
            'HVAC': {
                'expected_life': 15 * 365,
                'maintenance_interval': 90,
                'failure_indicators': ['temperature_variance', 'energy_consumption', 'vibration'],
                'seasonal_factors': True
            },
            'IT_EQUIPMENT': {
                'expected_life': 5 * 365,
                'maintenance_interval': 180,
                'failure_indicators': ['cpu_temperature', 'disk_usage', 'memory_errors'],
                'seasonal_factors': False
            },
            'ELECTRICAL': {
                'expected_life': 20 * 365,
                'maintenance_interval': 365,
                'failure_indicators': ['voltage_fluctuation', 'current_load', 'temperature'],
                'seasonal_factors': False
            },
            'FIRE_SAFETY': {
                'expected_life': 10 * 365,
                'maintenance_interval': 180,
                'failure_indicators': ['sensor_sensitivity', 'battery_level', 'response_time'],
                'seasonal_factors': False
            }
        }
        
        # Load or generate maintenance data
        self.maintenance_data = self._load_maintenance_data()
    
    def _load_maintenance_data(self) -> Dict:
        """Load or generate maintenance data for equipment"""
        # In a real system, this would connect to actual maintenance databases
        # For demo, we'll generate realistic maintenance data
        
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'AV_EQUIPMENT']
        
        for i in range(50):  # Generate 50 pieces of equipment
            equipment_id = f"{random.choice(equipment_types)}_{str(i+1).zfill(3)}"
            equipment_type = equipment_id.split('_')[0]
            
            # Generate realistic metrics
            age_days = random.randint(30, 2555)
            usage_intensity = random.uniform(0.2, 0.95)
            days_since_maintenance = random.randint(5, 300)
            environmental_score = random.uniform(0.1, 0.9)
            
            # Calculate failure probability
            if equipment_type in self.equipment_profiles:
                profile = self.equipment_profiles[equipment_type]
                age_factor = min(age_days / profile['expected_life'], 1.0)
                usage_factor = usage_intensity
                maintenance_factor = min(days_since_maintenance / profile['maintenance_interval'], 1.0)
                
                failure_probability = 0.1 + (age_factor * 0.3) + (usage_factor * 0.2) + \
                                      (maintenance_factor * 0.3) + (environmental_score * 0.2)
                failure_probability = min(max(failure_probability, 0.0), 1.0)
            else:
                failure_probability = random.uniform(0.05, 0.8)
            
            # Determine risk level
            if failure_probability >= maintenance_config.risk_threshold_high:
                risk_level = 'HIGH'
            elif failure_probability >= maintenance_config.risk_threshold_medium:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            equipment_data[equipment_id] = {
                'type': equipment_type,
                'location': random.choice(['Building A', 'Building B', 'Server Room', 'Main Hall', 'Conference Room']),
                'age_days': age_days,
                'usage_intensity': usage_intensity,
                'days_since_maintenance': days_since_maintenance,
                'environmental_score': environmental_score,
                'failure_probability': failure_probability,
                'risk_level': risk_level,
                'last_maintenance': (datetime.now() - timedelta(days=days_since_maintenance)).strftime('%Y-%m-%d'),
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 5000),
                'criticality': random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
            }
        
        return equipment_data
    
    def get_equipment_by_type(self, equipment_type: str) -> List[Dict]:
        """Get all equipment of a specific type"""
        equipment_type = equipment_type.upper()
        return [
            {'id': eq_id, **eq_data} 
            for eq_id, eq_data in self.maintenance_data.items() 
            if eq_data['type'] == equipment_type
        ]
    
    def get_equipment_by_risk(self, risk_level: str) -> List[Dict]:
        """Get equipment by risk level"""
        risk_level = risk_level.upper()
        return [
            {'id': eq_id, **eq_data}
            for eq_id, eq_data in self.maintenance_data.items()
            if eq_data['risk_level'] == risk_level
        ]
    
    def get_maintenance_schedule(self, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming maintenance schedule"""
        target_date = datetime.now() + timedelta(days=days_ahead)
        
        scheduled_items = []
        for eq_id, eq_data in self.maintenance_data.items():
            next_maintenance = datetime.strptime(eq_data['next_maintenance'], '%Y-%m-%d')
            if next_maintenance <= target_date:
                scheduled_items.append({
                    'id': eq_id,
                    'date': eq_data['next_maintenance'],
                    'days_until': (next_maintenance - datetime.now()).days,
                    **eq_data
                })
        
        return sorted(scheduled_items, key=lambda x: x['days_until'])
    
    def get_equipment_details(self, equipment_id: str) -> Dict:
        """Get detailed information about specific equipment"""
        equipment_id = equipment_id.upper()
        if equipment_id in self.maintenance_data:
            return {'id': equipment_id, **self.maintenance_data[equipment_id]}
        return None
    
    def generate_maintenance_recommendations(self, equipment_data: Dict) -> List[str]:
        """Generate maintenance recommendations for equipment"""
        recommendations = []
        
        risk_level = equipment_data.get('risk_level', 'LOW')
        equipment_type = equipment_data.get('type', '')
        failure_prob = equipment_data.get('failure_probability', 0)
        days_since_maintenance = equipment_data.get('days_since_maintenance', 0)
        
        if risk_level == 'HIGH':
            recommendations.extend([
                "🚨 URGENT: Schedule immediate inspection",
                "📋 Create detailed maintenance plan",
                "💰 Budget for potential replacement",
                "👥 Assign experienced technician"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "⚠️ Schedule preventive maintenance within 2 weeks",
                "📊 Increase monitoring frequency",
                "🔍 Inspect related systems",
                "📅 Plan maintenance window"
            ])
        else:
            recommendations.extend([
                "✅ Equipment is in good condition",
                "📅 Continue regular maintenance schedule",
                "📊 Monitor performance trends"
            ])
        
        # Type-specific recommendations
        if equipment_type == 'HVAC':
            if days_since_maintenance > 90:
                recommendations.append("🌡️ Replace air filters")
            recommendations.append("🔧 Check thermostat calibration")
            
        elif equipment_type == 'IT_EQUIPMENT':
            if failure_prob > 0.5:
                recommendations.append("💾 Backup critical data")
            recommendations.append("❄️ Check cooling systems")
            recommendations.append("🧹 Clean dust from components")
            
        elif equipment_type == 'ELECTRICAL':
            recommendations.extend([
                "⚡ Test electrical connections",
                "🔌 Inspect circuit breakers",
                "📏 Check voltage stability"
            ])
        
        return recommendations

# --- Enhanced Text Processing with Maintenance Integration ---
def detect_query_intent(query: str) -> Dict[str, Any]:
    """Enhanced query analysis with maintenance-specific intent detection"""
    query_lower = query.lower()
    
    # Maintenance-specific keywords
    maintenance_keywords = {
        'maintenance': ['maintenance', 'service', 'repair', 'fix', 'schedule'],
        'risk': ['risk', 'failure', 'predict', 'probability', 'alert', 'warning'],
        'status': ['status', 'health', 'condition', 'state', 'check'],
        'schedule': ['schedule', 'calendar', 'upcoming', 'planned', 'when'],
        'cost': ['cost', 'budget', 'expense', 'price', 'money'],
        'recommendation': ['recommend', 'suggest', 'advice', 'should', 'what to do']
    }
    
    # Equipment categories
    equipment_categories = {
        'camera': ['camera', 'video', 'recording', 'webcam', 'camcorder', 'lens', 'canon', 'nikon', 'sony'],
        'audio': ['audio', 'microphone', 'speaker', 'sound', 'mic', 'headphone', 'amplifier'],
        'computer': ['laptop', 'computer', 'pc', 'workstation', 'desktop', 'server', 'tablet'],
        'display': ['projector', 'display', 'monitor', 'screen', 'tv', 'television'],
        'network': ['network', 'wifi', 'router', 'switch', 'cable', 'internet'],
        'hvac': ['hvac', 'air', 'conditioning', 'heating', 'ventilation', 'climate'],
        'electrical': ['electrical', 'power', 'voltage', 'circuit', 'panel'],
        'fire_safety': ['fire', 'smoke', 'alarm', 'sprinkler', 'safety', 'exit']
    }
    
    # Count matches for maintenance intent
    maintenance_matches = {}
    for category, keywords in maintenance_keywords.items():
        maintenance_matches[category] = sum(1 for keyword in keywords if keyword in query_lower)
    
    # Count matches for equipment categories
    equipment_matches = {}
    for category, keywords in equipment_categories.items():
        equipment_matches[category] = sum(1 for keyword in keywords if keyword in query_lower)
    
    # Determine primary intent
    best_maintenance_intent = max(maintenance_matches, key=maintenance_matches.get) if max(maintenance_matches.values()) > 0 else None
    best_equipment_category = max(equipment_matches, key=equipment_matches.get) if max(equipment_matches.values()) > 0 else 'general'
    
    # Detect if maintenance-related query
    is_maintenance_query = max(maintenance_matches.values()) > 0
    
    return {
        'category': best_equipment_category,
        'maintenance_intent': best_maintenance_intent,
        'is_maintenance_query': is_maintenance_query,
        'equipment_confidence': max(equipment_matches.values()),
        'maintenance_confidence': max(maintenance_matches.values()),
        'is_comprehensive': any(keyword in query_lower for keyword in ['all', 'everything', 'complete', 'full', 'entire', 'list', 'show me']),
        'is_question': any(word in query_lower for word in ['what', 'how', 'where', 'when', 'why', 'which'])
    }

def generate_maintenance_response(query: str, query_info: Dict, maintenance_pipeline: MaintenancePipeline) -> str:
    """Generate maintenance-specific responses"""
    
    maintenance_intent = query_info.get('maintenance_intent')
    equipment_category = query_info.get('category', 'general')
    
    response = ""
    
    if maintenance_intent == 'risk':
        # Risk assessment queries
        high_risk_equipment = maintenance_pipeline.get_equipment_by_risk('HIGH')
        medium_risk_equipment = maintenance_pipeline.get_equipment_by_risk('MEDIUM')
        
        response += f"🚨 **Risk Assessment Summary**\n\n"
        response += f"🔴 **High Risk Equipment:** {len(high_risk_equipment)} items\n"
        response += f"🟡 **Medium Risk Equipment:** {len(medium_risk_equipment)} items\n\n"
        
        if high_risk_equipment:
            response += "**🚨 Critical Equipment Requiring Immediate Attention:**\n\n"
            for eq in high_risk_equipment[:5]:  # Show top 5
                response += f"• **{eq['id']}** ({eq['type']}) - {eq['failure_probability']:.1%} failure risk\n"
                response += f"  📍 Location: {eq['location']} | Last service: {eq['last_maintenance']}\n\n"
    
    elif maintenance_intent == 'schedule':
        # Maintenance schedule queries
        upcoming_maintenance = maintenance_pipeline.get_maintenance_schedule(30)
        
        response += f"📅 **Upcoming Maintenance Schedule (Next 30 Days)**\n\n"
        
        if upcoming_maintenance:
            for item in upcoming_maintenance[:8]:  # Show next 8 items
                days_text = f"in {item['days_until']} days" if item['days_until'] > 0 else "overdue"
                response += f"• **{item['id']}** - {days_text} ({item['date']})\n"
                response += f"  🏢 {item['location']} | 💰 Est. cost: ${item['maintenance_cost']}\n\n"
        else:
            response += "✅ No maintenance scheduled in the next 30 days.\n\n"
    
    elif maintenance_intent == 'status':
        # Equipment status queries
        if equipment_category != 'general':
            # Convert category to equipment type
            category_mapping = {
                'hvac': 'HVAC',
                'computer': 'IT_EQUIPMENT',
                'electrical': 'ELECTRICAL',
                'fire_safety': 'FIRE_SAFETY'
            }
            equipment_type = category_mapping.get(equipment_category)
            
            if equipment_type:
                equipment_list = maintenance_pipeline.get_equipment_by_type(equipment_type)
                
                response += f"📊 **{equipment_type.replace('_', ' ').title()} Equipment Status**\n\n"
                
                # Summary statistics
                total_equipment = len(equipment_list)
                high_risk = len([eq for eq in equipment_list if eq['risk_level'] == 'HIGH'])
                medium_risk = len([eq for eq in equipment_list if eq['risk_level'] == 'MEDIUM'])
                low_risk = len([eq for eq in equipment_list if eq['risk_level'] == 'LOW'])
                
                response += f"📈 **Summary:** {total_equipment} total units\n"
                response += f"🔴 High Risk: {high_risk} | 🟡 Medium Risk: {medium_risk} | 🟢 Low Risk: {low_risk}\n\n"
                
                # Show details for each piece of equipment
                for eq in equipment_list[:6]:  # Show first 6
                    risk_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(eq['risk_level'], '⚪')
                    response += f"{risk_emoji} **{eq['id']}**\n"
                    response += f"   📍 {eq['location']} | Age: {eq['age_days']} days | Risk: {eq['failure_probability']:.1%}\n"
                    response += f"   🔧 Last maintenance: {eq['last_maintenance']} | Next: {eq['next_maintenance']}\n\n"
    
    elif maintenance_intent == 'recommendation':
        # Recommendation queries
        high_risk_equipment = maintenance_pipeline.get_equipment_by_risk('HIGH')
        
        response += f"💡 **Maintenance Recommendations**\n\n"
        
        if high_risk_equipment:
            response += "**🚨 Priority Actions:**\n\n"
            for eq in high_risk_equipment[:3]:  # Top 3 priority items
                recommendations = maintenance_pipeline.generate_maintenance_recommendations(eq)
                response += f"**{eq['id']}** ({eq['type']}):\n"
                for rec in recommendations[:4]:  # Show top 4 recommendations
                    response += f"  {rec}\n"
                response += "\n"
        else:
            response += "✅ All equipment is in good condition. Continue regular maintenance schedules.\n\n"
    
    elif maintenance_intent == 'cost':
        # Cost analysis queries
        upcoming_maintenance = maintenance_pipeline.get_maintenance_schedule(90)
        total_cost = sum(item['maintenance_cost'] for item in upcoming_maintenance)
        
        response += f"💰 **Maintenance Cost Analysis (Next 90 Days)**\n\n"
        response += f"💵 **Total Estimated Cost:** ${total_cost:,}\n"
        response += f"📊 **Items Requiring Maintenance:** {len(upcoming_maintenance)}\n"
        response += f"📈 **Average Cost per Item:** ${total_cost // len(upcoming_maintenance) if upcoming_maintenance else 0:,}\n\n"
        
        if upcoming_maintenance:
            # Show highest cost items
            expensive_items = sorted(upcoming_maintenance, key=lambda x: x['maintenance_cost'], reverse=True)[:5]
            response += "**💸 Highest Cost Items:**\n\n"
            for item in expensive_items:
                response += f"• **{item['id']}** - ${item['maintenance_cost']:,}\n"
                response += f"  📍 {item['location']} | Risk: {item['risk_level']}\n\n"
    
    return response

# --- Enhanced Response Generation ---
def generate_enhanced_response(query: str, search_results: List[Dict], maintenance_pipeline: MaintenancePipeline = None) -> str:
    """Generate enhanced response combining document search and maintenance data"""
    query_info = detect_query_intent(query)
    
    # If it's a maintenance query and we have the maintenance pipeline
    if query_info['is_maintenance_query'] and maintenance_pipeline:
        maintenance_response = generate_maintenance_response(query, query_info, maintenance_pipeline)
        if maintenance_response:
            # Combine maintenance response with document search if available
            if search_results:
                response = maintenance_response + "\n\n---\n\n📚 **Related Documentation:**\n\n"
                for result in search_results[:2]:  # Show top 2 document results
                    preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                    response += f"📄 **{os.path.basename(result['source'])}**\n{preview}\n\n"
                return response
            else:
                return maintenance_response
    
    # Original document search response logic
    if not search_results:
        help_text = f"""🤔 **I couldn't find specific information about "{query}" in your documents.**

💡 **I can help you with:**
• 📷 Equipment searches (cameras, audio, computers)
• 📊 Inventory lookups  
• 🔧 **Maintenance predictions and schedules** *NEW!*
• 📈 **Risk assessments and recommendations** *NEW!*
• 📄 Document content searches

🔍 **Try asking:**
• "Show me all cameras"
• "What's the maintenance schedule?"
• "Which equipment is at high risk?"
• "What are the maintenance recommendations?"
• "Show equipment status for HVAC systems"
"""
        
        # Add maintenance data summary if available
        if maintenance_pipeline:
            high_risk_count = len(maintenance_pipeline.get_equipment_by_risk('HIGH'))
            upcoming_maintenance = len(maintenance_pipeline.get_maintenance_schedule(7))
            help_text += f"\n📊 **Current System Status:**\n"
            help_text += f"• 🚨 High risk equipment: {high_risk_count}\n"
            help_text += f"• 📅 Maintenance due this week: {upcoming_maintenance}\n"
        
        return help_text

    # Standard document search response
    response = f"🎯 **Found {len(search_results)} relevant results for: \"{query}\"**\n\n"
    
    # Add category insight if detected
    if query_info['category'] != 'general':
        category_emojis = {
            'camera': '📷',
            'audio': '🎵', 
            'computer': '💻',
            'display': '🖥️',
            'network': '🌐',
            'hvac': '🌡️',
            'electrical': '⚡'
        }
        emoji = category_emojis.get(query_info['category'], '🔍')
        response += f"{emoji} **Category:** {query_info['category'].title()} Equipment\n\n"
    
    response += "---\n\n"
    
    # Group results by source
    results_by_source = {}
    for result in search_results:
        source_name = os.path.basename(result['source'])
        if source_name not in results_by_source:
            results_by_source[source_name] = []
        results_by_source[source_name].append(result)
    
    # Display results
    for source_name, source_results in results_by_source.items():
        if len(source_results) == 1:
            result = source_results[0]
            # Assuming format_search_result exists, but it's missing in the provided code.
            # Creating a placeholder for it to avoid an error.
            def format_search_result(content, metadata, query_info):
                 return f"📄 **{os.path.basename(metadata.get('path', 'Unknown'))}**\n{content[:300]}...\n\n"
            response += format_search_result(result['content'], result['metadata'], query_info)
        else:
            response += f"📁 **Multiple matches in {source_name}:**\n\n"
            for i, result in enumerate(source_results[:3], 1):
                preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                response += f"**Match {i}:** {preview}\n\n"
        
        response += "---\n\n"
    
    # Add maintenance insights if relevant
    if maintenance_pipeline and query_info.get('equipment_confidence', 0) > 0:
        response += "🔧 **Maintenance Insights:**\n\n"
        response += "💡 *Try asking: 'What's the maintenance status for this equipment?' or 'Show risk assessment'*\n\n"
    
    return response

# Keep all the original functions for document processing
def clean_text(text: str) -> str:
    """Enhanced text cleaning with better formatting preservation"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\-.,;:()[\]{}!?@#$%^&*+=<>/"\'`~|\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def smart_chunking(text: str, chunk_size: int = 500) -> List[str]:
    """Smarter chunking that preserves sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
    return chunks

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz) with fallback to PyPDF2"""
    text_content = ""
    
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.get_text()
        doc.close()
        return text_content
    except Exception as e:
        st.warning(f"PyMuPDF failed for {file_path}, trying PyPDF2...")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page.extract_text()
            return text_content
        except Exception as e2:
            st.error(f"Both PDF extraction methods failed for {file_path}: {str(e2)}")
            return f"Error extracting PDF content: {str(e2)}"

@st.cache_resource
def load_documents_enhanced():
    """Load documents with enhanced processing and better user feedback"""
    st.info("🔍 **Loading and processing your documents...**")
    
    documents = []
    file_paths = []
    file_metadata = []
    
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf"]
    
    total_files_found = 0
    successful_loads = 0
    
    for pattern in file_patterns:
        files = glob.glob(pattern, recursive=True)
        total_files_found += len(files)
        
        for file_path in files:
            try:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                st.info(f"📂 Processing: **{file_name}** ({file_size:,} bytes)")
                
                content = ""
                file_type = ""
                
                if file_path.endswith('.pdf'):
                    content = extract_text_from_pdf(file_path)
                    file_type = 'pdf'
                    
                elif file_path.endswith('.csv'):
                    try:
                        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                        df = None
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding)
                                break
                            except:
                                continue
                        
                        if df is not None:
                            content = f"📊 **Equipment Database: {file_name}**\n\n"
                            content += f"📈 **Summary:** {len(df)} total items | {len(df.columns)} data fields\n\n"
                            content += f"📋 **Available Fields:** {', '.join(df.columns)}\n\n"
                            
                            content += "🔍 **Equipment Details:**\n\n"
                            for idx, row in df.head(100).iterrows():
                                item_details = []
                                for col, val in row.items():
                                    if pd.notna(val) and str(val).strip():
                                        item_details.append(f"{col}: {val}")
                                
                                if item_details:
                                    content += f"**Item {idx + 1}:** {' | '.join(item_details)}\n"
                            
                            file_type = 'csv'
                        else:
                            raise Exception("Could not read CSV with any encoding")
                        
                    except Exception as e:
                        st.warning(f"⚠️ CSV parsing failed, reading as text: {str(e)}")
                        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            content = f"📄 **Data from {file_name}:**\n\n" + f.read()
                        file_type = 'text'
                        
                else:
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except:
                            continue
                    
                    file_type = 'text'
                
                if content and len(content.strip()) > 50:
                    clean_content = clean_text(content)
                    documents.append(clean_content)
                    file_paths.append(file_path)
                    file_metadata.append({
                        'name': file_name,
                        'path': file_path,
                        'type': file_type,
                        'size': len(clean_content),
                        'original_size': file_size
                    })
                    successful_loads += 1
                    st.success(f"✅ **Successfully loaded:** {file_name}")
                else:
                    st.warning(f"⚠️ **Skipped:** {file_name} (insufficient content)")
                
            except Exception as e:
                st.error(f"❌ **Failed to load** {file_path}: {str(e)}")
    
    if successful_loads > 0:
        st.success(f"🎉 **Successfully loaded {successful_loads}/{total_files_found} documents!**")
    else:
        st.error("❌ **No documents were loaded successfully**")
    
    return documents, file_paths, file_metadata

# Note: The function 'create_enhanced_search_index' was incomplete in the provided code.
# I am providing a plausible completion for it to make the script functional.
@st.cache_resource
def create_enhanced_search_index(_documents, _file_paths, _metadata):
    """Create enhanced search index using SentenceTransformers and FAISS"""
    st.info("🧠 **Creating search index... This might take a moment.**")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = []
    chunk_metadata = []

    for i, doc in enumerate(_documents):
        chunks = smart_chunking(doc, chunk_size=config.chunk_size)
        all_chunks.extend(chunks)
        for chunk in chunks:
            chunk_metadata.append({
                'source': _file_paths[i],
                'metadata': _metadata[i]
            })

    if not all_chunks:
        st.warning("No text chunks were generated. The search index will be empty.")
        return None, [], []

    st.info(f"📊 **Generated {len(all_chunks)} text chunks for indexing.**")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    st.success(f"✅ **Search index created successfully!**")
    
    return index, all_chunks, chunk_metadata

# Main application logic would typically follow here, e.g.:
# if __name__ == "__main__":
#     st.title("AI Support Chatbot Enhanced")
#     docs, paths, meta = load_documents_enhanced()
#     if docs:
#         index, chunks, chunk_meta = create_enhanced_search_index(docs, paths, meta)
#         maintenance_pipeline = MaintenancePipeline()
#         # ... rest of the Streamlit UI and interaction logic
