import streamlit as st
import pandas as pd
import tempfile
import os
from typing import Optional, List, Dict, Any
import nest_asyncio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Model imports
try:
    from llama_index.llms.huggingface import HuggingFaceLLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from llama_index.readers.docling import DoclingReader
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

nest_asyncio.apply()

class OptimizedDataAnalyzer:
    """Optimized analyzer for your 9-column data structure"""
    
    # Define the expected column structure
    EXPECTED_COLUMNS = {
        'Number': 'categorical',      # Unique identifier
        'Task type': 'categorical',   # Category
        'Assignment group': 'categorical',  # Category  
        'Assigned to': 'categorical', # Category
        'State': 'categorical',       # Category
        'Short description': 'text',  # Long text
        'Priority': 'priority',       # Special categorical with ordering
        'Created': 'datetime',        # Date/time
        'Resolve notes': 'text'       # Long text
    }
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.setup_data_types()
        self.categorize_columns()
    
    def setup_data_types(self):
        """Setup proper data types based on known column structure"""
        # Handle datetime column
        if 'Created' in self.df.columns:
            self.df['Created'] = pd.to_datetime(self.df['Created'], errors='coerce')
            self.df['Month'] = self.df['Created'].dt.to_period('M')
            self.df['Week'] = self.df['Created'].dt.to_period('W')
            self.df['Hour'] = self.df['Created'].dt.hour
            self.df['Day_of_Week'] = self.df['Created'].dt.day_name()
        
        # Handle priority column (detect any ordering pattern)
        if 'Priority' in self.df.columns:
            self._setup_priority_ordering()
    
    def _setup_priority_ordering(self):
        """Auto-detect priority ordering from data"""
        priorities = self.df['Priority'].dropna().unique()
        
        # Try to detect common priority patterns
        priority_map = {}
        for i, priority in enumerate(sorted(priorities)):
            priority_str = str(priority).lower()
            if 'critical' in priority_str or '1' in priority_str:
                priority_map[priority] = 1
            elif 'high' in priority_str or '2' in priority_str:
                priority_map[priority] = 2
            elif 'medium' in priority_str or 'moderate' in priority_str or '3' in priority_str:
                priority_map[priority] = 3
            elif 'low' in priority_str or '4' in priority_str:
                priority_map[priority] = 4
            else:
                priority_map[priority] = i + 1
        
        self.df['Priority_Order'] = self.df['Priority'].map(priority_map)
    
    def categorize_columns(self):
        """Categorize available columns by type"""
        self.categorical_columns = []
        self.text_columns = []
        self.datetime_columns = []
        self.identifier_columns = []
        
        for col in self.df.columns:
            if col in ['Month', 'Week', 'Hour', 'Day_of_Week', 'Priority_Order']:
                continue  # Skip derived columns
                
            if col == 'Created':
                self.datetime_columns.append(col)
            elif col in ['Number']:
                self.identifier_columns.append(col)
            elif col in ['Short description', 'Resolve notes']:
                self.text_columns.append(col)
            elif col in ['Task type', 'Assignment group', 'Assigned to', 'State', 'Priority']:
                self.categorical_columns.append(col)
    
    def _apply_dark_theme(self, fig):
        """Apply dark theme to plotly figures"""
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            title_font_color='#ffffff',
            legend=dict(
                bgcolor='rgba(45, 55, 72, 0.8)',
                bordercolor='rgba(255,255,255,0.2)',
                font_color='#e2e8f0'
            ),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)',
                tickcolor='rgba(255,255,255,0.2)',
                title_font_color='#a0aec0'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)',
                tickcolor='rgba(255,255,255,0.2)',
                title_font_color='#a0aec0'
            )
        )
        return fig
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        summary = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'date_range': 'N/A',
            'missing_data_pct': round((self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100, 1)
        }
        
        # Add column-specific summaries
        for col in self.categorical_columns:
            if col in self.df.columns:
                summary[f'{col}_unique'] = self.df[col].nunique()
                summary[f'{col}_most_common'] = self.df[col].mode().iloc[0] if not self.df[col].empty else 'N/A'
        
        if 'Created' in self.df.columns:
            summary['date_range'] = f"{self.df['Created'].min().strftime('%Y-%m-%d')} to {self.df['Created'].max().strftime('%Y-%m-%d')}"
            summary['records_per_month'] = round(len(self.df) / max(1, self.df['Month'].nunique()), 1)
        
        return summary
    
    def create_smart_dashboard(self) -> List[Dict[str, Any]]:
        """Create intelligent dashboard based on available data"""
        charts = []
        
        # 1. Overview metrics
        charts.append(self._create_overview_chart())
        
        # 2. Categorical distributions
        for col in self.categorical_columns[:4]:  # Top 4 categorical columns
            if col in self.df.columns:
                chart = self._create_categorical_chart(col)
                if chart:
                    charts.append(chart)
        
        # 3. Time-based analysis
        if 'Created' in self.df.columns:
            charts.extend(self._create_time_charts())
        
        # 4. Incident summary and issue categories
        charts.extend(self._create_summary_and_categories_charts())
        
        return charts
    
    def _create_overview_chart(self) -> Dict[str, Any]:
        """Create overview metrics chart"""
        summary = self.get_data_summary()
        
        metrics = ['Total Records']
        values = [summary['total_records']]
        colors = ['#667eea']
        
        # Add categorical summaries
        for col in self.categorical_columns[:3]:
            if f'{col}_unique' in summary:
                metrics.append(f'Unique {col}')
                values.append(summary[f'{col}_unique'])
                colors.append('#764ba2' if len(colors) == 1 else '#38b2ac' if len(colors) == 2 else '#ed8936')
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=colors, 
                   text=[f"{val:,}" for val in values], textposition='outside',
                   textfont=dict(color='#ffffff', size=12))
        ])
        fig.update_layout(
            title="📊 Data Overview", 
            showlegend=False, 
            height=400,
            title_font_size=18,
            title_x=0.5
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig)
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': 'Data Overview',
            'description': 'Key metrics from your dataset'
        }
    
    def _create_categorical_chart(self, column: str) -> Optional[Dict[str, Any]]:
        """Create chart for categorical column"""
        if column not in self.df.columns:
            return None
        
        value_counts = self.df[column].value_counts().head(15)  # Top 15 to avoid overcrowding
        
        if len(value_counts) <= 1:
            return None
        
        # Dark theme color palettes
        pie_colors = ['#667eea', '#764ba2', '#38b2ac', '#ed8936', '#f56565', '#9f7aea', '#38a169', '#3182ce']
        
        # Choose chart type based on number of categories
        if len(value_counts) <= 8:
            # Pie chart for fewer categories
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {column}',
                color_discrete_sequence=pie_colors
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_color='white',
                textfont_size=11
            )
        else:
            # Bar chart for more categories
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f'Top Values in {column}',
                color=value_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            height=450
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig)
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{column} Distribution',
            'description': f'Distribution of values in {column}'
        }
    
    def _create_time_charts(self) -> List[Dict[str, Any]]:
        """Create time-based analysis charts"""
        charts = []
        
        if 'Created' not in self.df.columns:
            return charts
        
        # Monthly trend
        if 'Month' in self.df.columns:
            monthly_data = self.df.groupby('Month').size().reset_index(name='Count')
            monthly_data['Month_str'] = monthly_data['Month'].astype(str)
            
            fig = px.line(
                monthly_data, x='Month_str', y='Count',
                title='📈 Monthly Trends', markers=True,
                line_shape='spline'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                title_font_size=16,
                title_x=0.5,
                height=400
            )
            fig.update_traces(
                line_color='#667eea',
                marker_color='#764ba2',
                marker_size=8,
                line_width=3
            )
            
            # Apply dark theme
            fig = self._apply_dark_theme(fig)
            
            charts.append({
                'type': 'plotly',
                'figure': fig,
                'title': 'Monthly Trends',
                'description': 'Records volume over time'
            })
        
        # Day of week pattern
        if 'Day_of_Week' in self.df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = self.df['Day_of_Week'].value_counts().reindex(day_order, fill_value=0)
            
            fig = px.bar(
                x=day_counts.index, y=day_counts.values,
                title='📅 Pattern by Day of Week',
                color=day_counts.values, 
                color_continuous_scale=['#4a5568', '#667eea', '#764ba2']
            )
            fig.update_layout(
                title_font_size=16,
                title_x=0.5,
                height=400
            )
            
            # Apply dark theme
            fig = self._apply_dark_theme(fig)
            
            charts.append({
                'type': 'plotly',
                'figure': fig,
                'title': 'Weekly Patterns',
                'description': 'Activity pattern by day of week'
            })
        
        return charts
    
    def _create_summary_and_categories_charts(self) -> List[Dict[str, Any]]:
        """Create incident summary and issue categories analysis"""
        charts = []
        
        # 1. Incident Summary Dashboard
        summary = self.get_data_summary()
        
        # Create comprehensive summary metrics
        metrics = []
        values = []
        colors = []
        
        # Basic metrics
        metrics.extend(['Total Records'])
        values.append(summary['total_records'])
        
        # Calculate status metrics if State column exists
        if 'State' in self.df.columns:
            state_counts = self.df['State'].value_counts()
            # Add top 3 states to summary
            for i, (state, count) in enumerate(state_counts.head(3).items()):
                metrics.append(f'{state}')
                values.append(count)
        
        # Priority metrics if Priority column exists
        if 'Priority' in self.df.columns:
            priority_counts = self.df['Priority'].value_counts()
            # Add top priority to summary
            if len(priority_counts) > 0:
                top_priority, top_count = priority_counts.iloc[0], priority_counts.iloc[0]
                metrics.append(f'Top Priority: {priority_counts.index[0]}')
                values.append(top_count)
        
        # Generate colors for all metrics
        colors = ['#667eea', '#764ba2', '#38b2ac', '#ed8936', '#9f7aea', '#8c564b', '#e377c2'][:len(metrics)]
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=colors,
                   text=[f"{val:,}" for val in values], textposition='outside')
        ])
        fig.update_layout(
            title="📋 Incident Summary Dashboard", 
            showlegend=False, 
            height=450,
            yaxis_title="Count",
            xaxis_tickangle=-45,
            title_font_size=18,
            title_x=0.5
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig)
        
        charts.append({
            'type': 'plotly',
            'figure': fig,
            'title': 'Incident Summary',
            'description': 'Complete overview of incident status and priorities'
        })
        
        # 2. Issue Categories/Types Analysis - Use Task type column
        if 'Task type' in self.df.columns:
            task_counts = self.df['Task type'].value_counts().head(10)
            
            if len(task_counts) > 0:
                fig = px.pie(
                    values=task_counts.values,
                    names=task_counts.index,
                    title='🏷️ Issue Categories by Task Type',
                    color_discrete_sequence=['#667eea', '#764ba2', '#38b2ac', '#ed8936', '#f56565', '#9f7aea', '#38a169', '#3182ce', '#805ad5', '#d69e2e']
                )
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_color='white',
                    textfont_size=11
                )
                fig.update_layout(
                    height=500,
                    title_font_size=16,
                    title_x=0.5
                )
                
                # Apply dark theme
                fig = self._apply_dark_theme(fig)
                
                charts.append({
                    'type': 'plotly',
                    'figure': fig,
                    'title': 'Issue Categories',
                    'description': 'Distribution of different types of issues being handled'
                })
        
        # 3. Assignment Analysis
        if 'Assignment group' in self.df.columns:
            assignment_counts = self.df['Assignment group'].value_counts().head(8)
            
            if len(assignment_counts) > 0:
                fig = px.bar(
                    x=assignment_counts.values,
                    y=assignment_counts.index,
                    orientation='h',
                    title='👥 Assignment Groups Distribution',
                    color=assignment_counts.values,
                    color_continuous_scale=['#4a5568', '#667eea', '#764ba2', '#38b2ac']
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=450,
                    xaxis_title="Number of Records",
                    yaxis_title="Assignment Group",
                    title_font_size=16,
                    title_x=0.5
                )
                
                # Apply dark theme
                fig = self._apply_dark_theme(fig)
                
                charts.append({
                    'type': 'plotly',
                    'figure': fig,
                    'title': 'Assignment Groups',
                    'description': 'Distribution of records across different assignment groups'
                })
        
        return charts
    
    def get_intelligent_insights(self) -> str:
        """Generate data-driven insights"""
        insights = []
        summary = self.get_data_summary()
        
        # Data quality insight
        if summary['missing_data_pct'] > 0:
            insights.append(f"📊 Data completeness: {100 - summary['missing_data_pct']:.1f}% ({summary['missing_data_pct']:.1f}% missing)")
        else:
            insights.append("✅ Data is 100% complete with no missing values")
        
        # Categorical distribution insights
        for col in self.categorical_columns[:3]:
            if col in self.df.columns and f'{col}_unique' in summary:
                unique_count = summary[f'{col}_unique']
                most_common = summary[f'{col}_most_common']
                total_records = summary['total_records']
                
                if unique_count > 1:
                    concentration = (self.df[col] == most_common).sum() / total_records * 100
                    insights.append(f"📋 {col}: {unique_count} unique values, most common is '{most_common}' ({concentration:.1f}%)")
        
        # Time-based insights
        if 'Created' in self.df.columns and summary['date_range'] != 'N/A':
            insights.append(f"📅 Time range: {summary['date_range']} (~{summary.get('records_per_month', 0):.1f} records/month)")
            
            if 'Month' in self.df.columns:
                busiest_month = self.df['Month'].mode().iloc[0] if not self.df['Month'].empty else None
                if busiest_month:
                    month_count = self.df[self.df['Month'] == busiest_month].shape[0]
                    insights.append(f"📈 Busiest period: {busiest_month} with {month_count} records")
        
        # Priority insights (if available)
        if 'Priority' in self.df.columns:
            priority_dist = self.df['Priority'].value_counts()
            top_priority = priority_dist.index[0]
            top_pct = (priority_dist.iloc[0] / len(self.df)) * 100
            insights.append(f"🚨 Most common priority: {top_priority} ({top_pct:.1f}% of records)")
        
        return '\n'.join([f"• {insight}" for insight in insights])
    
    def create_custom_chart(self, question: str) -> List[Dict[str, Any]]:
        """Create charts based on user questions"""
        question_lower = question.lower()
        charts = []
        
        # Detect what user is asking for
        for col in self.categorical_columns:
            if col.lower().replace(' ', '') in question_lower.replace(' ', ''):
                chart = self._create_categorical_chart(col)
                if chart:
                    charts.append(chart)
        
        # Time-related requests
        if any(word in question_lower for word in ['time', 'trend', 'month', 'day', 'hour', 'when']):
            charts.extend(self._create_time_charts())
        
        # Summary and categories requests
        if any(word in question_lower for word in ['summary', 'categories', 'types', 'issues']):
            charts.extend(self._create_summary_and_categories_charts())
        
        return charts

class OptimizedRAGChatbot:
    """Optimized RAG Chatbot for your specific 9-column structure"""
    
    def __init__(self, model_type="huggingface"):
        self.model_type = model_type
        self.index = None
        self.chat_engine = None
        self.processed_files = []
        self.dataframe = None
        self.analyzer = None
        
        self.setup_models()
        if DOCLING_AVAILABLE:
            self.setup_docling_reader()
    
    def setup_models(self):
        """Setup models optimized for your data structure"""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
            
            self.setup_huggingface_model()
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 256
            Settings.chunk_overlap = 100
            
            st.success(f"✅ {self.model_type.title()} models optimized for your data structure!")
            
        except Exception as e:
            st.error(f"❌ Error setting up {self.model_type} models: {e}")
    
    def setup_huggingface_model(self):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face dependencies not available")
        
        model_name = "Gensyn/Qwen2.5-1.5B-Instruct"
        
        with st.spinner(f"🔄 Loading {model_name}..."):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if device == "cuda":
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
                )
                
                self.llm = HuggingFaceLLM(
                    context_window=6144, max_new_tokens=1536, model_name=model_name,
                    tokenizer_name=model_name, device_map="auto",
                    tokenizer_kwargs={"trust_remote_code": True},
                    model_kwargs={"torch_dtype": torch.float16, "trust_remote_code": True},
                    generate_kwargs={"temperature": 0.1, "do_sample": True, "top_p": 0.9, "repetition_penalty": 1.2}
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float32, trust_remote_code=True
                )
                
                self.llm = HuggingFaceLLM(
                    context_window=4096, max_new_tokens=1024, model_name=model_name,
                    tokenizer_name=model_name, tokenizer_kwargs={"trust_remote_code": True},
                    model_kwargs={"torch_dtype": torch.float32, "trust_remote_code": True},
                    generate_kwargs={"temperature": 0.05, "do_sample": True, "top_p": 0.85}
                )
    
    def setup_docling_reader(self):
        if DOCLING_AVAILABLE:
            try:
                self.docling_reader = DoclingReader()
                self.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=100, separator=" ")
                st.success("✅ Document reader optimized for your data!")
            except Exception as e:
                st.error(f"❌ Error setting up document reader: {e}")
                self.docling_reader = None
    
    def create_system_prompt(self, data_summary: Dict) -> str:
        """Create optimized system prompt for your data structure"""
        
        columns_info = f"""
DATASET STRUCTURE:
- Total Records: {data_summary.get('total_records', 0):,}
- Columns: Number, Task type, Assignment group, Assigned to, State, Short description, Priority, Created, Resolve notes
- Date Range: {data_summary.get('date_range', 'N/A')}
"""
        
        system_prompt = f"""You are an expert data analyst specializing in operational data analysis.

{columns_info}

ANALYSIS CAPABILITIES:
- Number: Unique identifiers and tracking
- Task type: Categorization and type analysis  
- Assignment group: Team and organizational analysis
- Assigned to: Individual workload and performance analysis
- State: Status tracking and workflow analysis
- Short description: Content and pattern analysis
- Priority: Urgency and importance analysis
- Created: Time-based trends and patterns
- Resolve notes: Resolution analysis and insights

INSTRUCTIONS:
1. **For visualization requests**: Respond briefly, let the chart system handle it
2. **For analysis requests**: Provide detailed insights from the actual data
3. **Be Data-Driven**: Use exact numbers, percentages, and patterns from the data
4. **Be Analytical**: Focus on trends, distributions, correlations, and business insights
5. **Be Accurate**: Only reference data you can see in the provided context

RESPONSE STYLE:
- Start with direct answers
- Support with specific data points
- Include actionable insights
- Use professional but accessible language

Remember: Accuracy is paramount. Base all insights on the actual data provided."""

        return system_prompt
    
    def process_files(self, uploaded_files: List) -> bool:
        """Process files optimized for your 9-column structure"""
        try:
            documents = []
            dataframes = []
            
            with st.spinner("📊 Processing your data files..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Load DataFrame
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(io.BytesIO(uploaded_file.getbuffer()))
                        else:
                            df = pd.read_excel(io.BytesIO(uploaded_file.getbuffer()))
                        
                        df['source_file'] = uploaded_file.name
                        dataframes.append(df)
                        
                        # Process with docling if available
                        if self.docling_reader:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                                temp_file.write(uploaded_file.getbuffer())
                                temp_path = temp_file.name
                            
                            try:
                                file_documents = self.docling_reader.load_data(file_path=temp_path)
                                
                                for doc in file_documents:
                                    doc.metadata.update({
                                        "filename": uploaded_file.name,
                                        "file_type": "structured_data",
                                        "total_rows": len(df),
                                        "columns": list(df.columns)
                                    })
                                
                                documents.extend(file_documents)
                                
                                # Add structured summary
                                summary_doc = self._create_data_summary_document(df, uploaded_file.name)
                                if summary_doc:
                                    documents.append(summary_doc)
                                
                            finally:
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                        
                        self.processed_files.append(uploaded_file.name)
                        st.success(f"✅ Processed: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            if not dataframes:
                st.error("No files were successfully processed.")
                return False
            
            # Combine data and create analyzer
            self.dataframe = pd.concat(dataframes, ignore_index=True)
            self.analyzer = OptimizedDataAnalyzer(self.dataframe)
            
            # Create RAG index
            if documents and self.docling_reader:
                with st.spinner("🔍 Creating optimized search system..."):
                    nodes = self.node_parser.get_nodes_from_documents(documents)
                    self.index = VectorStoreIndex(nodes)
                    
                    data_summary = self.analyzer.get_data_summary()
                    system_prompt = self.create_system_prompt(data_summary)
                    
                    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
                    
                    self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                        self.index.as_retriever(similarity_top_k=8, similarity_cutoff=0.6),
                        memory=memory, system_prompt=system_prompt, verbose=True
                    )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            return False
    
    def _create_data_summary_document(self, df: pd.DataFrame, filename: str):
        """Create structured summary for better retrieval"""
        try:
            summary_parts = [f"FILE: {filename}", f"TOTAL RECORDS: {len(df):,}"]
            
            # Column summaries
            for col in OptimizedDataAnalyzer.EXPECTED_COLUMNS.keys():
                if col in df.columns:
                    if col == 'Created':
                        date_range = f"{df[col].min()} to {df[col].max()}"
                        summary_parts.append(f"{col}: {date_range}")
                    elif col in ['Short description', 'Resolve notes']:
                        avg_length = df[col].str.len().mean() if not df[col].empty else 0
                        summary_parts.append(f"{col}: Average length {avg_length:.0f} characters")
                    else:
                        unique_count = df[col].nunique()
                        most_common = df[col].mode().iloc[0] if not df[col].empty else 'N/A'
                        summary_parts.append(f"{col}: {unique_count} unique values, most common: {most_common}")
            
            summary_text = "\n\n".join(summary_parts)
            
            return Document(
                text=summary_text,
                metadata={"type": "data_summary", "filename": filename, "content_type": "structured_summary"}
            )
        
        except Exception as e:
            st.error(f"Error creating summary document: {e}")
            return None
    
    def chat(self, question: str) -> str:
        """Optimized chat for your data structure"""
        question_lower = question.lower()
        if any(word in question_lower for word in ['chart', 'plot', 'graph', 'visualization', 'dashboard']):
            return ""
        
        if self.chat_engine:
            try:
                with st.spinner(f"🔍 Analyzing your data with {self.model_type.title()}..."):
                    response = self.chat_engine.chat(question)
                    response_text = str(response)
                    return self._clean_response(response_text)
            except Exception as e:
                return f"Error generating response: {e}"
        else:
            return self._basic_analysis(question)
    
    def _clean_response(self, response: str) -> str:
        """Clean response for better readability"""
        if self.model_type == "huggingface":
            import re
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = response.strip()
        
        return response
    
    def _basic_analysis(self, question: str) -> str:
        """Basic analysis when RAG is not available"""
        if not self.analyzer:
            return "Please upload and process your data files first."
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['summary', 'overview', 'describe']):
            summary = self.analyzer.get_data_summary()
            insights = self.analyzer.get_intelligent_insights()
            
            return f"""
**📊 Data Summary:**
- Total Records: {summary['total_records']:,}
- Date Range: {summary.get('date_range', 'N/A')}
- Data Completeness: {100 - summary['missing_data_pct']:.1f}%

**🔍 Key Insights:**
{insights}

**📋 Available Analysis:**
Ask about specific columns like Task type, Assignment group, Priority, time trends, or request visualizations.
"""
        
        elif any(word in question_lower for word in ['pattern', 'trend', 'insight']):
            return self.analyzer.get_intelligent_insights()
        
        else:
            return ("I can analyze your data in detail. Try asking:\n"
                   "- 'Give me a summary' for overview\n"
                   "- 'What patterns do you see?' for insights\n" 
                   "- 'Show me charts' for visualizations\n"
                   "- Questions about specific columns or trends")
    
    def create_visualizations(self, question: str) -> List[Dict[str, Any]]:
        """Create visualizations optimized for your data"""
        if not self.analyzer:
            return []
        
        question_lower = question.lower()
        
        # Dashboard request
        if any(word in question_lower for word in ['dashboard', 'multiple', 'all charts', 'overview']):
            return self.analyzer.create_smart_dashboard()
        
        # Specific chart requests
        return self.analyzer.create_custom_chart(question)
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get processing information"""
        model_info = "Qwen2.5-1.5B-Instruct (HuggingFace)"
        
        return {
            "processed_files": self.processed_files,
            "total_chunks": len(self.index.docstore.docs) if self.index else 0,
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
            "llm_model": model_info,
            "model_type": self.model_type,
            "has_rag": self.chat_engine is not None,
            "has_analyzer": self.analyzer is not None,
            "total_records": len(self.dataframe) if self.dataframe is not None else 0
        }

def create_streamlit_app():
    """Create optimized Streamlit interface with ChatGPT-like chat history"""
    st.set_page_config(
        page_title="Incident Triage using GenAI",
        page_icon="🚀", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme and modern styling
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #1e1e2e, #2d3748);
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 2px solid #4a5568;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-style: italic;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #2d3748, #4a5568);
        color: #e2e8f0;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Quick action buttons */
    .quick-action-btn {
        background: linear-gradient(135deg, #38b2ac, #319795);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 20px;
        margin: 5px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(56, 178, 172, 0.3);
        width: 100%;
    }
    
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 178, 172, 0.4);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #2d3748, #4a5568);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed #667eea;
    }
    
    /* Progress bars and success messages */
    .stSuccess {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #4299e1, #3182ce);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1);
        color: white;
        border: 2px solid #4a5568;
        border-radius: 25px;
        padding: 12px 20px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar headers */
    .sidebar-header {
        color: #667eea;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Getting started styling */
    .getting-started {
        background: linear-gradient(135deg, #2d3748, #4a5568);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "hf_chatbot" not in st.session_state:
        if HF_AVAILABLE:
            st.session_state.hf_chatbot = OptimizedRAGChatbot(model_type="huggingface")
        else:
            st.error("❌ Hugging Face dependencies not available")
            st.session_state.hf_chatbot = None
    
    st.markdown('<h1 class="main-title"> Incident Triage using GenAI </h1>', unsafe_allow_html=True)
    # st.markdown('<p class="subtitle">AI-powered analysis optimized for your 9-column data structure</p>', unsafe_allow_html=True)
    
    # Main layout with sidebar for file upload and main area for chat
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">📁 Upload Data Files</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose your data files",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            key="hf_uploader",
            help="Upload Excel or CSV files with your incident data"
        )
        
        if uploaded_files and st.session_state.hf_chatbot:
            if st.button("🚀 Process Files", type="primary", key="hf_process"):
                success = st.session_state.hf_chatbot.process_files(uploaded_files)
                if success:
                    st.balloons()
                    st.success("🎉 Ready for analysis!")
                    st.rerun()
        
        # File info with better styling
        if st.session_state.hf_chatbot:
            file_info = st.session_state.hf_chatbot.get_file_info()
            if file_info:
                st.markdown('<h3 class="sidebar-header">📊 Data Overview</h3>', unsafe_allow_html=True)
                
                # Create metrics with custom styling
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Files", len(file_info['processed_files']))
                    st.metric("🔢 Records", f"{file_info.get('total_records', 0):,}")
                with col2:
                    if file_info['total_chunks'] > 0:
                        st.metric("🧩 Chunks", file_info['total_chunks'])
                    else:
                        st.metric("🧩 Chunks", "0")
                
                # Status indicators with better styling
                st.markdown("---")
                if file_info.get('has_rag'):
                    st.markdown("✅ **Advanced Analysis Ready**")
                if file_info.get('has_analyzer'):
                    st.markdown("✅ **Visualizations Ready**")
        
        # Clear chat history button with better styling
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", key="clear_history", help="Clear all chat messages"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.hf_chatbot and st.session_state.hf_chatbot.analyzer:
        # Chat history display area with scrolling
        # st.markdown('<h2 class="section-header">💬 Chat History</h2>', unsafe_allow_html=True)
        
        # Create a container for chat messages with custom CSS for scrolling
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            # Display chat history in reverse order (newest at bottom)
            for i, (question, response, charts) in enumerate(st.session_state.chat_history):
                # User message with new styling
                st.markdown(f'''
                <div class="user-message">
                    <strong>👤 You:</strong><br>{question}
                </div>
                ''', unsafe_allow_html=True)
                
                # AI response with new styling
                if response:
                    st.markdown(f'''
                    <div class="ai-message">
                        <strong>🤖 AI Assistant:</strong><br>{response}
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Charts with container styling
                if charts:
                    for j, chart in enumerate(charts):
                        if chart['type'] == 'plotly':
                            with st.container():
                                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                st.plotly_chart(chart['figure'], use_container_width=True, key=f"history_chart_{i}_{j}")
                                st.caption(f"📊 {chart['description']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
        else:
            st.markdown('''
            <div style="text-align: center; padding: 40px; color: #a0aec0;">
                <h3> Welcome to your AI Data Analyst!</h3>
                <p>Start a conversation by asking about your data or use the quick actions below.</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick action buttons with better styling
        st.markdown('<h3 class="section-header"> Quick Actions</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Data Overview", key="overview_btn", help="Get comprehensive data summary"):
                question = "Give me a comprehensive data overview"
                process_question(question, st.session_state.hf_chatbot)
        
        with col2:
            if st.button("📈 Smart Dashboard", key="dashboard_btn", help="Create interactive dashboard"):
                question = "Create a smart dashboard"
                process_question(question, st.session_state.hf_chatbot)
        
        with col3:
            if st.button("🔍 Find Insights", key="insights_btn", help="Discover data patterns"):
                question = "What key insights and patterns do you see?"
                process_question(question, st.session_state.hf_chatbot)
        
        with col4:
            if st.button("📅 Time Analysis", key="time_btn", help="Analyze time-based trends"):
                question = "Show me time-based trends and patterns"
                process_question(question, st.session_state.hf_chatbot)
        
        # Chat input at the bottom with better styling
        # st.markdown('<h3 class="section-header">💭 Ask a Question</h3>', unsafe_allow_html=True)
        
        # Use a form to handle the input and submission
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input(
                "Ask about your data:",
                placeholder="e.g., 'Show priority distribution' or 'Compare assignment groups'",
                key="chat_input",
                help="Type your question about the data and press Enter or click Send"
            )
            submit_button = st.form_submit_button("Send 💬", type="primary")
            
            if submit_button and question:
                process_question(question, st.session_state.hf_chatbot)
    
    else:
        # Getting started interface with better styling
        # st.markdown('<h2 class="section-header">🚀 Get Started</h2>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="getting-started">
            <h3>👈 Upload your data files in the sidebar to begin analysis!</h3>
        </div>
        ''', unsafe_allow_html=True)

def process_question(question: str, chatbot):
    """Process a question and update chat history"""
    if not question.strip():
        return
    
    # Get AI response
    response = chatbot.chat(question)
    
    # Get visualizations
    charts = chatbot.create_visualizations(question)
    
    # Add to chat history
    st.session_state.chat_history.append((question, response, charts))
    
    # Rerun to update the interface
    st.rerun()

if __name__ == "__main__":
    create_streamlit_app()
