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
from sentence_transformers import SentenceTransformer

# LlamaIndex imports (keeping RAG functionality)
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Model imports
try:
    from llama_index.llms.langchain import LangChainLLM
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from llama_index.llms.huggingface import HuggingFaceLLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Try Docling import
try:
    from llama_index.readers.docling import DoclingReader
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Apply nest_asyncio for Jupyter/Streamlit compatibility
nest_asyncio.apply()

class ExcelAnalyzer:
    """Smart Excel data analyzer and visualizer for any Excel data"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = self.df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'datetime_columns': len(self.datetime_columns),
            'missing_values': self.df.isnull().sum().sum(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        }
        return summary
    
    def create_distribution_charts(self) -> List[Dict[str, Any]]:
        """Create distribution charts - ONLY USEFUL ONES for ANY dataset"""
        charts = []
        
        # Numeric distributions (usually always useful)
        for col in self.numeric_columns[:5]:
            fig = px.histogram(
                self.df, 
                x=col, 
                title=f'Distribution of {col}',
                marginal="box",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            charts.append({
                'type': 'plotly',
                'figure': fig,
                'title': f'Distribution: {col}',
                'description': f'Histogram and box plot showing distribution of {col}'
            })
        
        # SMART FILTER: Only useful categorical columns
        useful_categorical_columns = []
        
        for col in self.categorical_columns:
            if self._is_useful_for_visualization(col):
                useful_categorical_columns.append(col)
        
        # Create pie charts only for useful columns
        for col in useful_categorical_columns[:3]:
            value_counts = self.df[col].value_counts().head(10)
            if len(value_counts) > 1:
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f'Distribution of {col}',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                charts.append({
                    'type': 'plotly',
                    'figure': fig,
                    'title': f'Distribution: {col}',
                    'description': f'Pie chart showing distribution of {col} categories'
                })
        
        return charts
    
    def create_correlation_heatmap(self) -> Optional[Dict[str, Any]]:
        """Create correlation heatmap for numeric columns"""
        if len(self.numeric_columns) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_columns].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': 'Correlation Analysis',
            'description': 'Correlation heatmap showing relationships between numeric variables'
        }
    
    def create_comparison_charts(self, group_by: str = None) -> List[Dict[str, Any]]:
        """Create comparison charts"""
        charts = []
        
        if group_by and group_by in self.categorical_columns:
            # Group by categorical column and analyze numeric columns
            for num_col in self.numeric_columns[:3]:
                grouped_data = self.df.groupby(group_by)[num_col].agg(['mean', 'count']).reset_index()
                
                # Bar chart of means
                fig = px.bar(
                    grouped_data,
                    x=group_by,
                    y='mean',
                    title=f'Average {num_col} by {group_by}',
                    text='count',
                    color='mean',
                    color_continuous_scale='viridis'
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(showlegend=False)
                
                charts.append({
                    'type': 'plotly',
                    'figure': fig,
                    'title': f'{num_col} by {group_by}',
                    'description': f'Average {num_col} grouped by {group_by} with count labels'
                })
        
        return charts
    
    def create_time_series_charts(self) -> List[Dict[str, Any]]:
        """Create time series charts if datetime columns exist"""
        charts = []
        
        if not self.datetime_columns:
            return charts
        
        date_col = self.datetime_columns[0]
        
        for num_col in self.numeric_columns[:2]:
            # Sort by date
            df_sorted = self.df.sort_values(date_col)
            
            fig = px.line(
                df_sorted,
                x=date_col,
                y=num_col,
                title=f'{num_col} Over Time',
                markers=True
            )
            
            charts.append({
                'type': 'plotly',
                'figure': fig,
                'title': f'Time Series: {num_col}',
                'description': f'Time series plot of {num_col} over {date_col}'
            })
        
        return charts
    
    def create_smart_dashboard(self) -> List[Dict[str, Any]]:
        """Create a smart dashboard based on data characteristics"""
        charts = []
        
        # Add data summary chart
        summary = self.get_data_summary()
        
        # Data overview
        fig = go.Figure(data=[
            go.Bar(
                x=['Rows', 'Columns', 'Numeric Cols', 'Text Cols', 'Missing Values'],
                y=[summary['total_rows'], summary['total_columns'], 
                   summary['numeric_columns'], summary['categorical_columns'], 
                   summary['missing_values']],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        fig.update_layout(title="Data Overview", showlegend=False)
        
        charts.append({
            'type': 'plotly',
            'figure': fig,
            'title': 'Data Overview',
            'description': 'Quick overview of your dataset structure'
        })
        
        # Add distribution charts
        charts.extend(self.create_distribution_charts())
        
        # Add correlation heatmap
        corr_chart = self.create_correlation_heatmap()
        if corr_chart:
            charts.append(corr_chart)
        
        # Add time series if available
        charts.extend(self.create_time_series_charts())
        
        return charts
    
    def _is_useful_for_visualization(self, col: str) -> bool:
        """Universal logic to determine if ANY column is worth visualizing"""
        
        # Rule 1: Skip if too many unique values (probably IDs/names)
        unique_count = self.df[col].nunique()
        total_rows = len(self.df)
        
        if unique_count > total_rows * 0.8:
            return False
        
        if unique_count > 50:
            return False
        
        # Rule 2: Skip single-value columns
        if unique_count <= 1:
            return False
        
        # Rule 3: Skip if values are too long
        avg_length = self.df[col].astype(str).str.len().mean()
        if avg_length > 30:
            return False
        
        # Rule 4: Skip if looks like IDs
        sample_values = self.df[col].dropna().head(20).astype(str)
        
        id_like_count = 0
        for val in sample_values:
            if (any(char.isdigit() for char in val) and len(val) > 6) or \
            (val.replace('-', '').replace('_', '').isalnum() and len(val) > 8):
                id_like_count += 1
        
        if len(sample_values) > 0 and id_like_count > len(sample_values) * 0.7:
            return False
        
        # Rule 5: KEEP if reasonable number of categories
        if 2 <= unique_count <= 20:
            return True
        
        if 20 < unique_count <= 50 and unique_count < total_rows * 0.5:
            return True
        
        return False

class ExcelRAGChatbot:
    """Enhanced Excel RAG Chatbot for general Excel data analysis"""
    
    def __init__(self, model_type="ollama"):
        self.model_type = model_type
        self.index = None
        self.chat_engine = None
        self.processed_files = []
        self.dataframe_for_viz = None
        self.analyzer = None
        
        # Initialize components
        self.setup_models()
        if DOCLING_AVAILABLE:
            self.setup_docling_reader()
        else:
            st.warning("⚠️ Docling not available - RAG functionality limited")
    
    def setup_models(self):
        """Setup LLM and embedding models based on model type"""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
            
            if self.model_type == "ollama":
                self.setup_ollama_model()
            elif self.model_type == "huggingface":
                self.setup_huggingface_model()
            
            # Configure LlamaIndex settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 256      # IMPROVED: Smaller chunks for better retrieval
            Settings.chunk_overlap = 100   # IMPROVED: More overlap for context continuity
            
            st.success(f"✅ {self.model_type.title()} models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error setting up {self.model_type} models: {e}")
    
    def setup_ollama_model(self):
        """Setup Ollama model"""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama dependencies not available")
        
        langchain_llm = ChatOllama(
            model="gemma3:27b",
            base_url="http://localhost:11434",
            temperature=0.1,    # IMPROVED: Lower temperature for more consistent responses
        )
        self.llm = LangChainLLM(llm=langchain_llm)
    
    def setup_huggingface_model(self):
        """Setup Hugging Face model with enhanced accuracy settings"""
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face dependencies not available")
        
        model_name = "Gensyn/Qwen2.5-1.5B-Instruct"
        
        with st.spinner(f"🔄 Loading {model_name} with enhanced accuracy..."):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if device == "cuda":
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # IMPROVED: Enhanced accuracy settings for GPU
                self.llm = HuggingFaceLLM(
                    context_window=6144,        # IMPROVED: Increased context window
                    max_new_tokens=1536,        # IMPROVED: Longer, more detailed responses
                    model_name=model_name,
                    tokenizer_name=model_name,
                    device_map="auto",
                    tokenizer_kwargs={"trust_remote_code": True},
                    
                    model_kwargs={
                        "torch_dtype": torch.float16,
                        "trust_remote_code": True
                    },
                    # IMPROVED: Better generation settings
                    generate_kwargs={
                        "temperature": 0.1,      # Lower temperature = more consistent
                        "do_sample": True,
                        "top_p": 0.9,           # More focused sampling
                        "repetition_penalty": 1.2
                    }
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                
                # IMPROVED: Enhanced accuracy settings for CPU
                self.llm = HuggingFaceLLM(
                    context_window=4096,
                    max_new_tokens=1024,        # IMPROVED: Increased for CPU too
                    model_name=model_name,
                    tokenizer_name=model_name,
                    tokenizer_kwargs={"trust_remote_code": True},
                    model_kwargs={
                        "torch_dtype": torch.float32,
                        "trust_remote_code": True
                    },
                    generate_kwargs={
                        "temperature": 0.05,     # Very low for accuracy
                        "do_sample": True,
                        "top_p": 0.85
                    }
                )
        
        st.info(f"🎯 Using {device} with enhanced accuracy settings")
    
    def setup_docling_reader(self):
        """Setup Dockling reader with better chunking"""
        if DOCLING_AVAILABLE:
            try:
                self.docling_reader = DoclingReader()
                
                # IMPROVED: Better chunking strategy
                self.node_parser = SentenceSplitter(
                    chunk_size=256,      # Smaller chunks = more precise retrieval
                    chunk_overlap=100,   # More overlap = better context continuity
                    separator=" "        # Split on sentences for better coherence
                )
                st.success("✅ Enhanced Dockling reader initialized!")
            except Exception as e:
                st.error(f"❌ Error setting up Dockling: {e}")
                self.docling_reader = None
    
    def create_enhanced_system_prompt(self, data_summary: Dict) -> str:
        """Create data-aware system prompt for better accuracy"""
        
        columns_info = ""
        if self.analyzer:
            numeric_cols = ", ".join(self.analyzer.numeric_columns[:5])
            categorical_cols = ", ".join(self.analyzer.categorical_columns[:5])
            columns_info = f"""
        DATASET CONTEXT:
        - Total Records: {len(self.dataframe_for_viz):,}
        - Numeric Columns: {numeric_cols}
        - Categorical Columns: {categorical_cols}
        """
        
        enhanced_prompt = f"""You are an expert data analyst with deep expertise in Excel data analysis.

        IMPORTANT: For visualization requests (charts, plots, graphs, dashboard), give ONLY a brief response like "Creating charts from your data" - do NOT provide coding examples or explanations.


        {columns_info}

        CORE INSTRUCTIONS:
        1. **For visualization requests**: Respond briefly, let the chart system handle it
        2. **For analysis requests**: Provide detailed insights from the actual data
        3. **Be Precise**: Base ALL answers on the actual data provided in context
        4. **Be Specific**: Use exact numbers, percentages, and column names from the data
        5. **Be Analytical**: Provide insights relevant to data analysis and decision-making
        6. **Be Accurate**: If you don't see specific data in context, say "I don't see that information in the provided data"
        7. **Be Comprehensive**: When analyzing, look for patterns, trends, outliers, and correlations

        RESPONSE FORMAT:
        - Start with direct answer to the question
        - Support with specific data points and numbers
        - End with analytical insights or observations
        - Use clear, professional language

        ACCURACY REQUIREMENTS:
        - Never make up numbers or statistics not in the provided context
        - Always cite specific data when making claims
        - Distinguish between what you observe vs. what you infer
        - Acknowledge limitations when data is insufficient

        Remember: Your credibility depends on accuracy. Be precise and evidence-based."""

        return enhanced_prompt
    
    def _create_data_summary_document(self, df: pd.DataFrame, filename: str):
        """Create structured summary document for better retrieval"""
        try:
            # Create comprehensive data summary
            summary_parts = []
            
            # Basic stats
            summary_parts.append(f"FILE: {filename}")
            summary_parts.append(f"TOTAL RECORDS: {len(df):,}")
            summary_parts.append(f"TOTAL COLUMNS: {len(df.columns)}")
            
            # Column information
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols:
                summary_parts.append(f"NUMERIC COLUMNS: {', '.join(numeric_cols)}")
                # Add statistical summary
                stats_summary = df[numeric_cols].describe()
                summary_parts.append(f"NUMERIC STATISTICS:\n{stats_summary.to_string()}")
            
            if categorical_cols:
                summary_parts.append(f"CATEGORICAL COLUMNS: {', '.join(categorical_cols)}")
                # Add value counts for key categorical columns
                for col in categorical_cols[:3]:  # First 3 categorical columns
                    value_counts = df[col].value_counts().head(10)
                    summary_parts.append(f"{col} DISTRIBUTION: {value_counts.to_dict()}")
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                missing_info = missing_data[missing_data > 0].to_dict()
                summary_parts.append(f"MISSING DATA: {missing_info}")
            
            summary_text = "\n\n".join(summary_parts)
            
            return Document(
                text=summary_text,
                metadata={
                    "type": "data_summary",
                    "filename": filename,
                    "content_type": "structured_summary"
                }
            )
        
        except Exception as e:
            st.error(f"Error creating summary document: {e}")
            return None
    
    def process_excel_files(self, uploaded_files: List) -> bool:
        """Enhanced file processing with better retrieval"""
        try:
            documents = []
            dataframes = []
            
            with st.spinner("📊 Processing Excel files with enhanced accuracy..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Load as DataFrame
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(io.BytesIO(uploaded_file.getbuffer()))
                        else:
                            df = pd.read_excel(io.BytesIO(uploaded_file.getbuffer()))
                        
                        df['source_file'] = uploaded_file.name
                        dataframes.append(df)
                        
                        # IMPROVED: Enhanced document processing
                        if self.docling_reader:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                                temp_file.write(uploaded_file.getbuffer())
                                temp_path = temp_file.name
                            
                            try:
                                file_documents = self.docling_reader.load_data(file_path=temp_path)
                                
                                # ENHANCED: Add rich metadata
                                for doc in file_documents:
                                    doc.metadata.update({
                                        "filename": uploaded_file.name,
                                        "file_type": "excel",
                                        "total_rows": len(df),
                                        "columns": list(df.columns),
                                        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                                        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
                                    })
                                
                                documents.extend(file_documents)
                                
                                # IMPROVED: Add structured data summaries as documents
                                summary_doc = self._create_data_summary_document(df, uploaded_file.name)
                                if summary_doc:
                                    documents.append(summary_doc)
                                
                            finally:
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                        
                        self.processed_files.append(uploaded_file.name)
                        st.success(f"✅ Enhanced processing: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            if not dataframes:
                st.error("No files were successfully processed.")
                return False
            
            # Combine dataframes and create analyzer
            self.dataframe_for_viz = pd.concat(dataframes, ignore_index=True)
            self.analyzer = ExcelAnalyzer(self.dataframe_for_viz)
            
            # IMPROVED: Create RAG index with better configuration
            if documents and self.docling_reader:
                with st.spinner("🔍 Creating enhanced retrieval system..."):
                    nodes = self.node_parser.get_nodes_from_documents(documents)
                    self.index = VectorStoreIndex(nodes)
                    
                    # Get data summary for enhanced prompt
                    data_summary = self.analyzer.get_data_summary()
                    enhanced_system_prompt = self.create_enhanced_system_prompt(data_summary)
                    
                    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)  # IMPROVED: Increased memory
                    
                    # IMPROVED: Better retrieval configuration
                    self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                        self.index.as_retriever(
                            similarity_top_k=8,          # IMPROVED: More context chunks
                            similarity_cutoff=0.6        # IMPROVED: Filter low-quality matches
                        ),
                        memory=memory,
                        system_prompt=enhanced_system_prompt,
                        verbose=True,
                        context_prompt_template="Context information is below:\n{context_str}\n\nBased on this context and your expertise, please answer the question: {query_str}"
                    )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error during enhanced processing: {e}")
            return False
    
    def chat(self, question: str) -> str:
        """Enhanced chat with response validation"""
        question_lower = question.lower()
        if any(word in question_lower for word in ['chart', 'plot', 'graph', 'visualization', 'dashboard']):
            return ""
        
        if self.chat_engine:
            try:
                with st.spinner(f"🔍 Analyzing data with enhanced accuracy using {self.model_type.title()}..."):
                    response = self.chat_engine.chat(question)
                    response_text = str(response)
                    
                    # IMPROVED: Validate and enhance response
                    validated_response = self._validate_and_enhance_response(response_text, question)
                    return validated_response
            except Exception as e:
                return f"Error generating response: {e}"
        else:
            # Enhanced basic data analysis
            if self.analyzer:
                return self._enhanced_basic_data_analysis(question)
            else:
                return "Please upload and process Excel files first."
    
    def _validate_and_enhance_response(self, response: str, question: str) -> str:
        """Validate response accuracy and add data-driven enhancements"""
        
        # Clean any reasoning artifacts (for HuggingFace models)
        if self.model_type == "huggingface":
            import re
            # Remove thinking tags and content
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = response.strip()
        
        # Add data validation footer if response seems generic
        if len(response) < 100 or "I don't have" in response:
            if self.analyzer:
                summary = self.analyzer.get_data_summary()
                response += f"\n\n**📊 Quick Data Context:**\nYour dataset has {summary['total_rows']:,} rows and {summary['total_columns']} columns. Ask more specific questions for detailed analysis."
        
        return response
    
    def _enhanced_basic_data_analysis(self, question: str) -> str:
        """Enhanced basic data analysis with more accuracy"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['summary', 'overview', 'describe']):
            summary = self.analyzer.get_data_summary()
            
            # Add more detailed analysis
            insights = []
            
            # Data quality insights
            completeness = (1 - summary['missing_values'] / (summary['total_rows'] * summary['total_columns'])) * 100
            insights.append(f"Data Completeness: {completeness:.1f}%")
            
            # Column analysis
            if self.analyzer.categorical_columns:
                for col in self.analyzer.categorical_columns[:2]:
                    unique_vals = self.dataframe_for_viz[col].nunique()
                    most_common = self.dataframe_for_viz[col].mode().iloc[0] if not self.dataframe_for_viz[col].empty else "N/A"
                    insights.append(f"{col}: {unique_vals} unique values, most common: {most_common}")
            
            if self.analyzer.numeric_columns:
                for col in self.analyzer.numeric_columns[:2]:
                    col_mean = self.dataframe_for_viz[col].mean()
                    col_std = self.dataframe_for_viz[col].std()
                    insights.append(f"{col}: Mean={col_mean:.2f}, Std={col_std:.2f}")
            
            return f"""
            **📊 Data Summary:**
            - Total Rows: {summary['total_rows']:,}
            - Total Columns: {summary['total_columns']}
            - Numeric Columns: {summary['numeric_columns']}
            - Text Columns: {summary['categorical_columns']}
            - Missing Values: {summary['missing_values']:,}
            - Memory Usage: {summary['memory_usage']}
            
            **🔍 Key Insights:**
            {chr(10).join([f"- {insight}" for insight in insights])}
            
            **📋 Available Columns:**
            - Numeric: {', '.join(self.analyzer.numeric_columns[:5])}
            - Categorical: {', '.join(self.analyzer.categorical_columns[:5])}
            """
        
        elif any(word in question_lower for word in ['pattern', 'trend', 'insight']):
            insights = []
            
            # Enhanced correlation analysis
            if len(self.analyzer.numeric_columns) >= 2:
                corr_matrix = self.dataframe_for_viz[self.analyzer.numeric_columns].corr()
                # Find top 3 correlations
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:  # Only meaningful correlations
                            correlations.append({
                                'cols': (corr_matrix.columns[i], corr_matrix.columns[j]),
                                'value': corr_val
                            })
                
                correlations.sort(key=lambda x: abs(x['value']), reverse=True)
                for corr in correlations[:3]:
                    relationship = "strong positive" if corr['value'] > 0.7 else "moderate positive" if corr['value'] > 0.3 else "moderate negative" if corr['value'] < -0.3 else "strong negative"
                    insights.append(f"{relationship} correlation between {corr['cols'][0]} and {corr['cols'][1]} (r={corr['value']:.3f})")
            
            # Enhanced categorical analysis
            for col in self.analyzer.categorical_columns[:2]:
                value_counts = self.dataframe_for_viz[col].value_counts()
                if len(value_counts) > 1:
                    top_category = value_counts.index[0]
                    top_percentage = (value_counts.iloc[0] / len(self.dataframe_for_viz)) * 100
                    insights.append(f"{col}: '{top_category}' dominates with {top_percentage:.1f}% of records")
            
            # Data quality patterns
            missing_data = self.dataframe_for_viz.isnull().sum()
            if missing_data.sum() > 0:
                missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)
                worst_col = missing_cols.index[0]
                missing_pct = (missing_cols.iloc[0] / len(self.dataframe_for_viz)) * 100
                insights.append(f"Data quality note: {worst_col} missing {missing_pct:.1f}% of values")
            
            return "**🔍 Key Data Patterns:**\n" + '\n'.join([f"- {insight}" for insight in insights])
        
        else:
            return ("I can provide detailed analysis of your data. Try asking for:\n"
                   "- 'Give me a summary' for data overview\n" 
                   "- 'What patterns do you see?' for insights\n"
                   "- Specific visualizations like 'bar chart of [column name]'")

    def create_visualizations(self, question: str) -> List[Dict[str, Any]]:
        """Create ANY chart for ANY question based on actual data"""
        if not self.analyzer:
            return []
        
        question_lower = question.lower()
        
        # ADD THIS DASHBOARD DETECTION:
        if any(word in question_lower for word in ['dashboard', 'multiple visualizations', 'visualizations', 'charts']):
            # Return the smart dashboard directly
            return self.analyzer.create_smart_dashboard()
        
        # Parse chart type from question (existing code)
        chart_type = self._detect_chart_type(question_lower)
        if not chart_type:
            return []
        
        # Rest of your existing code...
        target_columns = self._detect_target_columns(question_lower)
        chart = self._create_dynamic_chart(chart_type, target_columns, question_lower)
        return [chart] if chart else []

    def _detect_chart_type(self, question_lower: str) -> str:
        """Detect what type of chart user wants"""
        chart_keywords = {
            'bar': ['bar chart', 'bar graph', 'bar plot', 'bars'],
            'pie': ['pie chart', 'pie graph', 'pie plot'],
            'line': ['line chart', 'line graph', 'line plot', 'trend'],
            'scatter': ['scatter plot', 'scatter chart', 'scatter graph'],
            'histogram': ['histogram', 'hist', 'distribution'],
            'heatmap': ['heatmap', 'heat map', 'correlation'],
            'box': ['box plot', 'boxplot', 'box chart'],
            'area': ['area chart', 'area plot', 'area graph']
        }
        
        for chart_type, keywords in chart_keywords.items():

            if any(keyword in question_lower for keyword in keywords):
                return chart_type
        
        return None
    
    def _is_useful_for_visualization(self, col: str) -> bool:
        """Universal logic to determine if ANY column is worth visualizing"""
        
        # Rule 1: Skip if too many unique values (probably IDs/names)
        unique_count = self.df[col].nunique()
        total_rows = len(self.df)
        
        if unique_count > total_rows * 0.8:
            return False
        
        if unique_count > 50:
            return False
        
        # Rule 2: Skip single-value columns
        if unique_count <= 1:
            return False
        
        # Rule 3: Skip if values are too long
        avg_length = self.df[col].astype(str).str.len().mean()
        if avg_length > 30:
            return False
        
        # Rule 4: Skip if looks like IDs
        sample_values = self.df[col].dropna().head(20).astype(str)
        
        id_like_count = 0
        for val in sample_values:
            if (any(char.isdigit() for char in val) and len(val) > 6) or \
            (val.replace('-', '').replace('_', '').isalnum() and len(val) > 8):
                id_like_count += 1
        
        if len(sample_values) > 0 and id_like_count > len(sample_values) * 0.7:
            return False
        
        # Rule 5: KEEP if reasonable number of categories
        if 2 <= unique_count <= 20:
            return True
        
        if 20 < unique_count <= 50 and unique_count < total_rows * 0.5:
            return True
        
        return False
    
    

    def _detect_target_columns(self, question_lower: str) -> Dict[str, List[str]]:
        """Detect which columns user wants to visualize"""
        target_cols = {
            'categorical': [],
            'numeric': [],
            'datetime': []
        }
        
        # Check for column names mentioned in question
        all_columns = (self.analyzer.categorical_columns + 
                    self.analyzer.numeric_columns + 
                    self.analyzer.datetime_columns)
        
        for col in all_columns:
            if col.lower() in question_lower:
                if col in self.analyzer.categorical_columns:
                    target_cols['categorical'].append(col)
                elif col in self.analyzer.numeric_columns:
                    target_cols['numeric'].append(col)
                elif col in self.analyzer.datetime_columns:
                    target_cols['datetime'].append(col)
        
        return target_cols

    def _create_dynamic_chart(self, chart_type: str, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create the specific chart requested with actual data"""
        try:
            if chart_type == 'bar':
                return self._create_bar_chart(target_columns, question_lower)
            elif chart_type == 'pie':
                return self._create_pie_chart(target_columns, question_lower)
            elif chart_type == 'line':
                return self._create_line_chart(target_columns, question_lower)
            elif chart_type == 'scatter':
                return self._create_scatter_chart(target_columns, question_lower)
            elif chart_type == 'histogram':
                return self._create_histogram_chart(target_columns, question_lower)
            elif chart_type == 'heatmap':
                return self._create_heatmap_chart(target_columns, question_lower)
            elif chart_type == 'box':
                return self._create_box_chart(target_columns, question_lower)
            elif chart_type == 'area':
                return self._create_area_chart(target_columns, question_lower)
            
            return None
        
        except Exception as e:
            return None

    def _create_bar_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create bar chart from actual data"""
        # For bar charts, we need categorical data
        if not target_columns['categorical']:
            # Use first available categorical column
            if not self.analyzer.categorical_columns:
                return None
            cat_col = self.analyzer.categorical_columns[0]
        else:
            cat_col = target_columns['categorical'][0]
        
        # Check if user wants percentage
        if 'percentage' in question_lower or 'percent' in question_lower:
            value_counts = self.dataframe_for_viz[cat_col].value_counts()
            percentages = (value_counts / len(self.dataframe_for_viz) * 100).round(1)
            
            fig = px.bar(
                x=percentages.index,
                y=percentages.values,
                title=f'Percentage Distribution of {cat_col}',
                labels={'x': cat_col, 'y': 'Percentage (%)'}
            )
            fig.update_traces(text=[f'{val}%' for val in percentages.values], textposition='outside')
            
        else:
            # Regular count chart
            value_counts = self.dataframe_for_viz[cat_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Count Distribution of {cat_col}',
                labels={'x': cat_col, 'y': 'Count'}
            )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{cat_col} Distribution',
            'description': f'Bar chart of {cat_col} distribution'
        }

    def _create_pie_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create pie chart from actual data"""
        if not target_columns['categorical']:
            if not self.analyzer.categorical_columns:
                return None
            cat_col = self.analyzer.categorical_columns[0]
        else:
            cat_col = target_columns['categorical'][0]
        
        value_counts = self.dataframe_for_viz[cat_col].value_counts()
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Distribution of {cat_col}'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{cat_col} Distribution',
            'description': f'Pie chart of {cat_col} distribution'
        }

    def _create_line_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create line chart from actual data"""
        if not target_columns['numeric'] or not target_columns['datetime']:
            # Need both numeric and time data for line chart
            if not self.analyzer.numeric_columns or not self.analyzer.datetime_columns:
                return None
            num_col = self.analyzer.numeric_columns[0]
            date_col = self.analyzer.datetime_columns[0]
        else:
            num_col = target_columns['numeric'][0]
            date_col = target_columns['datetime'][0] if target_columns['datetime'] else self.analyzer.datetime_columns[0]
        
        df_sorted = self.dataframe_for_viz.sort_values(date_col)
        
        fig = px.line(
            df_sorted,
            x=date_col,
            y=num_col,
            title=f'{num_col} Over Time',
            markers=True
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{num_col} Trend',
            'description': f'Line chart showing {num_col} over {date_col}'
        }

    def _create_scatter_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create scatter chart from actual data"""
        if len(target_columns['numeric']) < 2:
            if len(self.analyzer.numeric_columns) < 2:
                return None
            x_col = self.analyzer.numeric_columns[0]
            y_col = self.analyzer.numeric_columns[1]
        else:
            x_col = target_columns['numeric'][0]
            y_col = target_columns['numeric'][1]
        
        fig = px.scatter(
            self.dataframe_for_viz,
            x=x_col,
            y=y_col,
            title=f'{y_col} vs {x_col}'
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{y_col} vs {x_col}',
            'description': f'Scatter plot showing relationship between {x_col} and {y_col}'
        }

    def _create_histogram_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create histogram from actual data"""
        if not target_columns['numeric']:
            if not self.analyzer.numeric_columns:
                return None
            num_col = self.analyzer.numeric_columns[0]
        else:
            num_col = target_columns['numeric'][0]
        
        fig = px.histogram(
            self.dataframe_for_viz,
            x=num_col,
            title=f'Distribution of {num_col}',
            marginal="box"
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{num_col} Distribution',
            'description': f'Histogram showing distribution of {num_col}'
        }

    def _create_heatmap_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create heatmap from actual data"""
        if len(self.analyzer.numeric_columns) < 2:
            return None
        
        corr_matrix = self.dataframe_for_viz[self.analyzer.numeric_columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r'
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': 'Correlation Heatmap',
            'description': 'Heatmap showing correlations between numeric variables'
        }

    def _create_box_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create box plot from actual data"""
        if not target_columns['numeric']:
            if not self.analyzer.numeric_columns:
                return None
            num_col = self.analyzer.numeric_columns[0]
        else:
            num_col = target_columns['numeric'][0]
        
        fig = px.box(
            self.dataframe_for_viz,
            y=num_col,
            title=f'Box Plot of {num_col}'
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{num_col} Box Plot',
            'description': f'Box plot showing distribution and outliers in {num_col}'
        }

    def _create_area_chart(self, target_columns: Dict, question_lower: str) -> Optional[Dict[str, Any]]:
        """Create area chart from actual data"""
        if not target_columns['numeric'] or not target_columns['datetime']:
            if not self.analyzer.numeric_columns or not self.analyzer.datetime_columns:
                return None
            num_col = self.analyzer.numeric_columns[0]
            date_col = self.analyzer.datetime_columns[0]
        else:
            num_col = target_columns['numeric'][0]
            date_col = target_columns['datetime'][0] if target_columns['datetime'] else self.analyzer.datetime_columns[0]
        
        df_sorted = self.dataframe_for_viz.sort_values(date_col)
        
        fig = px.area(
            df_sorted,
            x=date_col,
            y=num_col,
            title=f'{num_col} Over Time (Area)'
        )
        
        return {
            'type': 'plotly',
            'figure': fig,
            'title': f'{num_col} Area Chart',
            'description': f'Area chart showing {num_col} over {date_col}'
        }
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about processed files"""
        model_info = "gemma3:27b (Ollama)" if self.model_type == "ollama" else "Qwen2.5-1.5B-Instruct (HuggingFace)"
        
        return {
            "processed_files": self.processed_files,
            "total_chunks": len(self.index.docstore.docs) if self.index else 0,
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
            "llm_model": model_info,
            "model_type": self.model_type,
            "has_rag": self.chat_engine is not None,
            "has_analyzer": self.analyzer is not None
        }

def create_streamlit_app():
    """Create the Streamlit interface"""
    st.set_page_config(
        page_title="🦙 Excel RAG Chatbot",
        page_icon="🦙", 
        layout="wide"
    )
    
    st.title("🦙 Excel RAG Chatbot")
    st.markdown("*Enhanced AI-powered Excel analysis for any data type*")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🌐 Ollama Models", "🤗 Hugging Face Models"])
    
    with tab1:
        st.markdown("### 🌐 Ollama Local Models")
        st.info("💡 Fast, local processing with Gemma 3 27B via Ollama")
        
        if 'ollama_chatbot' not in st.session_state:
            if OLLAMA_AVAILABLE:
                st.session_state.ollama_chatbot = ExcelRAGChatbot(model_type="ollama")
            else:
                st.error("❌ Ollama dependencies not available")
                st.session_state.ollama_chatbot = None
        
        if st.session_state.ollama_chatbot:
            create_chat_interface(st.session_state.ollama_chatbot, "ollama")
    
    with tab2:
        st.markdown("### 🤗 Hugging Face Direct Models")
        st.info("🎨 Enhanced Qwen2.5-1.5B model with improved accuracy")
        
        if 'hf_chatbot' not in st.session_state:
            if HF_AVAILABLE:
                st.session_state.hf_chatbot = ExcelRAGChatbot(model_type="huggingface")
            else:
                st.error("❌ Hugging Face dependencies not available")
                st.session_state.hf_chatbot = None
        
        if st.session_state.hf_chatbot:
            create_chat_interface(st.session_state.hf_chatbot, "huggingface")

def create_chat_interface(chatbot, model_type):
    """Create the chat interface for a specific model type"""
    
    with st.sidebar:
        st.header(f"📁 Upload Excel Files ({model_type.title()})")
        
        uploaded_files = st.file_uploader(
            f"Choose Excel files for {model_type.title()}",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            key=f"{model_type}_uploader"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary", key=f"{model_type}_process"):
                success = chatbot.process_excel_files(uploaded_files)
                if success:
                    st.balloons()
                    st.success("🎉 Ready to chat!")
                    st.rerun()
        
        file_info = chatbot.get_file_info()
        if file_info:
            st.markdown("### 📊 Processed Data")
            st.metric("Files Processed", len(file_info['processed_files']))
            if file_info['total_chunks'] > 0:
                st.metric("Data Chunks", file_info['total_chunks'])
            st.metric("Model", file_info['llm_model'])
            
            if file_info.get('has_rag'):
                st.success("✅ RAG Chat Enabled")
            else:
                st.warning("⚠️ Basic Analysis Only")
            
            if file_info.get('has_analyzer'):
                st.success("✅ Visualizations Ready")
    
    if chatbot.analyzer:
        st.markdown(f"### 💬 Chat with Your Excel Data ({model_type.title()})")
        
        # Sample questions
        st.markdown("**💡 Try asking:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Data summary", key=f"{model_type}_summary"):
                st.session_state[f"{model_type}_sample_question"] = "Give me a comprehensive summary of the data"
        
        with col2:
            if st.button("📈 Create dashboard", key=f"{model_type}_dashboard"):
                st.session_state[f"{model_type}_sample_question"] = "Show me a dashboard with multiple visualizations"
        
        with col3:
            if st.button("🔍 Find patterns", key=f"{model_type}_patterns"):
                st.session_state[f"{model_type}_sample_question"] = "What patterns and trends do you see in the data?"
        
        # with col4:
        #     if st.button("📊 Distribution charts", key=f"{model_type}_distribution"):
        #         st.session_state[f"{model_type}_sample_question"] = "Show me distribution charts for my data"
        
        # Chat input
        question = st.text_input(
            f"Ask a question about your data ({model_type.title()}):",
            value=getattr(st.session_state, f"{model_type}_sample_question", ''),
            placeholder="e.g., 'Create a bar chart of [column name]' or 'Show correlation heatmap'",
            key=f"{model_type}_question"
        )
        
        if question:
            if hasattr(st.session_state, f"{model_type}_sample_question"):
                del st.session_state[f"{model_type}_sample_question"]
            
            # Get response
            response = chatbot.chat(question)
            
            st.markdown(f"#### 🤖 AI Response ({model_type.title()})")
            st.markdown(response)
            
            # ADD THIS MISSING CODE:
            charts = chatbot.create_visualizations(question)
            if charts:
                st.markdown("#### 📊 Data Visualizations")
                
                for i, chart in enumerate(charts):
                    if chart['type'] == 'plotly':
                        st.plotly_chart(chart['figure'], use_container_width=True, key=f"{model_type}_chart_{i}")
                        st.caption(f"📈 {chart['description']}")
                        
                        if i > 0 and i % 2 == 1:  # Add spacing every 2 charts
                            st.markdown("---")
    
    else:
        st.markdown(f"### 🚀 Get Started with {model_type.title()}")
        st.info("👈 Upload your Excel files in the sidebar to begin!")
        
        col1, col2 = st.columns(2)
        
    

if __name__ == "__main__":
    create_streamlit_app()
