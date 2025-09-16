import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path
from io import BytesIO
import traceback

import streamlit as st
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.provider import get_chat_model, get_embedding_model, require_env_for, choose_provider
from tools.search_adapter import require_tavily_api_key
from tools.embeddings import embed_texts
from graph.builder import build_graph
from graph.state import GraphState
from utils.io import read_text_auto, save_text


# Load environment variables
load_dotenv()


def validate_api_keys(provider_choice="auto"):
    """Validate required API keys and return validation status."""
    validation_results = {
        "llm_valid": False,
        "tavily_valid": False,
        "llm_error": None,
        "tavily_error": None,
        "provider_warnings": []
    }
    
    # Map UI choices to provider names
    provider_map = {"Auto": "auto", "Gemini": "gemini", "Mistral": "mistral"}
    provider = provider_map.get(provider_choice, provider_choice.lower())
    
    # Check LLM API keys based on provider choice
    try:
        if provider == "gemini":
            require_env_for("gemini")
        elif provider == "mistral":
            require_env_for("mistral")
        elif provider == "auto":
            # For auto mode, check available keys
            has_gemini = bool(os.getenv("GEMINI_API_KEY"))
            has_mistral = bool(os.getenv("MISTRAL_API_KEY"))
            
            if not has_gemini and not has_mistral:
                raise RuntimeError("At least one API key required for auto mode: GEMINI_API_KEY or MISTRAL_API_KEY")
            elif not has_gemini:
                validation_results["provider_warnings"].append("⚠️ GEMINI_API_KEY missing - will use Mistral only")
            elif not has_mistral:
                validation_results["provider_warnings"].append("⚠️ MISTRAL_API_KEY missing - will use Gemini only")
        
        validation_results["llm_valid"] = True
    except RuntimeError as e:
        validation_results["llm_error"] = str(e)
    
    # Check Tavily API key
    try:
        require_tavily_api_key()
        validation_results["tavily_valid"] = True
    except RuntimeError as e:
        validation_results["tavily_error"] = str(e)
    
    return validation_results


def extract_sources_from_state(state):
    """Extract source URLs from analysis state for display."""
    sources = []
    
    # Extract from market summary if available
    if state.market_summary:
        market_data = convert_to_serializable(state.market_summary)
        if 'sources_sample' in market_data:
            sources.extend(market_data['sources_sample'])
    
    # Extract from logs for any mentioned URLs
    if state.logs:
        for log in state.logs:
            if 'http' in log:
                # Extract URLs from log entries
                words = log.split()
                for word in words:
                    if word.startswith('http'):
                        sources.append(word.rstrip('.,;!?'))
    
    return list(set(sources))  # Remove duplicates


def check_pdf_support():
    """Check if PDF processing dependencies are available."""
    try:
        import pymupdf4llm
        return True, "pymupdf4llm available"
    except ImportError:
        return False, "pymupdf4llm not installed. Install with: pip install pymupdf4llm"


def convert_to_serializable(obj):
    """Convert any Pydantic models to dicts recursively."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path."""
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None


def run_analysis_pipeline(file_path: str, provider: str, role: str, region: str, lang: str, allow_offline_embeddings: bool = False):
    """Run the complete analysis pipeline and return results."""
    try:
        # Set embedding configuration
        if allow_offline_embeddings:
            os.environ['ALLOW_OFFLINE_EMBEDDINGS'] = 'true'
        else:
            os.environ['ALLOW_OFFLINE_EMBEDDINGS'] = 'false'
        
        # Map UI provider choice to provider system
        provider_map = {"Auto": "auto", "Gemini": "gemini", "Mistral": "mistral"}
        provider_name = provider_map.get(provider, provider.lower())
        
        # Enhanced provider tracking for better feedback
        provider_info = {
            "requested": provider,
            "resolved": None,
            "model_name": None,
            "fallback_used": False,
            "warnings": []
        }
        
        # Get models using new provider system with enhanced error tracking
        try:
            chat_model = get_chat_model(provider_name, None)
            embedding_model = get_embedding_model(provider_name, None)
            
            # Extract actual provider and model information
            if hasattr(chat_model, 'model'):
                model_name = chat_model.model
                provider_info["model_name"] = model_name
                
                if 'gemini' in model_name.lower():
                    provider_info["resolved"] = "Gemini"
                    if '2.0' in model_name:
                        provider_info["model_display"] = "Gemini 2.0 Flash"
                    elif '1.5' in model_name:
                        provider_info["model_display"] = "Gemini 1.5 Pro"
                    else:
                        provider_info["model_display"] = "Gemini"
                elif 'mistral' in model_name.lower():
                    provider_info["resolved"] = "Mistral"
                    if 'large' in model_name.lower():
                        provider_info["model_display"] = "Mistral Large"
                    else:
                        provider_info["model_display"] = "Mistral"
                else:
                    provider_info["resolved"] = "Unknown"
                    provider_info["model_display"] = model_name
            
            # For Auto mode, check if we had to fall back
            if provider_name == "auto":
                if provider_info["resolved"] == "Mistral":
                    # Check if Gemini was available but failed
                    gemini_available = bool(os.getenv("GEMINI_API_KEY"))
                    if gemini_available:
                        provider_info["fallback_used"] = True
                        provider_info["warnings"].append("Gemini unavailable, falling back to Mistral")
                        
        except Exception as e:
            # Enhanced error handling for provider failures
            error_msg = str(e)
            if provider_name == "auto":
                provider_info["warnings"].append(f"Auto mode failed: {error_msg}")
            
            # Provide specific guidance based on error type
            if "api" in error_msg.lower() or "key" in error_msg.lower():
                raise Exception(f"API key error: {error_msg}. Please check your API keys in the sidebar.")
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                raise Exception(f"API quota exceeded: {error_msg}. Please check your usage limits or try a different provider.")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise Exception(f"Network error: {error_msg}. Please check your internet connection.")
            else:
                raise Exception(f"Provider initialization failed: {error_msg}")
        
        # Build the analysis graph with provider choice
        graph = build_graph(provider_choice=provider_name)
        
        # Create initial state
        # Extract model names for compatibility
        chat_model_name = provider_info.get("model_display", "Unknown")
        embed_model_name = "Unknown"
        
        try:
            if hasattr(chat_model, 'model'):
                chat_model_name = chat_model.model
            elif hasattr(chat_model, 'model_name'):
                chat_model_name = chat_model.model_name
        except:
            pass
            
        try:
            if hasattr(embedding_model, 'model'):
                embed_model_name = embedding_model.model
            elif hasattr(embedding_model, 'model_name'):
                embed_model_name = embedding_model.model_name
        except:
            pass
        
        initial_state = GraphState(
            file_path=file_path,
            target_role=role,
            market_region=region,
            lang=lang,
            logs=[],
            provider=provider_name,
            chat_model_name=chat_model_name,
            embed_model_name=embed_model_name
        )
        
        
        # Execute the graph
        result = graph.invoke(initial_state.model_dump())
        final_state = GraphState(**result)
        
        return {
            "success": True,
            "state": final_state,
            "error": None,
            "provider_info": provider_info
        }
        
    except Exception as e:
        return {
            "success": False,
            "state": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "provider_info": {
                "requested": provider,
                "resolved": None,
                "model_display": None,
                "fallback_used": False,
                "warnings": [f"Analysis failed: {str(e)}"]
            }
        }


# Streamlit UI
st.set_page_config(
    page_title="Skill Gap Analyst",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Skill Gap Analyst")
st.markdown("### Upload your CV and get a comprehensive skill gap analysis")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Provider selector (move to top to validate based on selection)
provider = st.sidebar.selectbox(
    "🧠 LLM Provider",
    options=["Auto", "Gemini", "Mistral"],
    index=0,
    help="Choose the LLM provider for analysis"
)

# Validate API keys based on selected provider
api_validation = validate_api_keys(provider)
all_keys_valid = api_validation["llm_valid"] and api_validation["tavily_valid"]

# Live Mode Status Badge
if all_keys_valid:
    st.sidebar.success("🚀 **LIVE MODE** - Real APIs Active")
else:
    st.sidebar.error("🔒 **BLOCKED** - Missing API Keys")

# Show provider warnings if any
for warning in api_validation.get("provider_warnings", []):
    st.sidebar.warning(warning)

# API Key Status
with st.sidebar.expander("🔑 API Key Status", expanded=not all_keys_valid):
    if api_validation["llm_valid"]:
        st.success("✅ LLM API Key: Valid")
    else:
        st.error(f"❌ LLM API Key: {api_validation['llm_error']}")
    
    if api_validation["tavily_valid"]:
        st.success("✅ Tavily API Key: Valid")
    else:
        st.error(f"❌ Tavily API Key: {api_validation['tavily_error']}")
    
    if not all_keys_valid:
        st.markdown("**Required API Keys:**")
        if provider == "Auto":
            st.markdown("- `GEMINI_API_KEY` and/or `MISTRAL_API_KEY` (at least one required)")
        elif provider == "Gemini":
            st.markdown("- `GEMINI_API_KEY` for Gemini provider")
        elif provider == "Mistral":
            st.markdown("- `MISTRAL_API_KEY` for Mistral provider")
        st.markdown("- `TAVILY_API_KEY` for web search")
        st.markdown("Set these in your `.env` file or environment variables.")

st.sidebar.markdown("---")

# Role input
role = st.sidebar.text_input(
    "🎯 Target Role",
    value="Senior AI Engineer",
    help="The role you want to analyze skills for"
)

# Region selector
region = st.sidebar.selectbox(
    "🌍 Market Region",
    options=["Global", "US", "Europe", "Asia-Pacific", "North America"],
    index=0,
    help="Market region for skill demand analysis"
)

# Language selector
language = st.sidebar.selectbox(
    "🗣️ Report Language",
    options=["en", "id"],
    format_func=lambda x: {"en": "English", "id": "Indonesian"}[x],
    index=0,
    help="Language for the generated report"
)

# Offline embeddings checkbox
allow_offline_embeddings = st.sidebar.checkbox(
    "Allow offline embeddings",
    value=False,
    help="When unchecked, requires live embeddings (live mode)",
    disabled=not all_keys_valid
)

if not allow_offline_embeddings:
    st.sidebar.info("🔒 Live embeddings required (live-only mode)")
else:
    st.sidebar.warning("⚠️ Offline fallback enabled")

# Show provider/model info if valid
if all_keys_valid:
    try:
        provider_map = {"Auto": "auto", "Gemini": "gemini", "Mistral": "mistral"}
        provider_name = provider_map.get(provider, provider.lower())
        
        chat_model = get_chat_model(provider_name, None)
        embedding_model = get_embedding_model(provider_name, None)
        
        st.sidebar.markdown("**🤖 Active LLM:**")
        
        # Enhanced provider detection with display names
        chat_model_name = "Unknown"
        embed_model_name = "Unknown"
        model_display = "Unknown"
        
        try:
            if hasattr(chat_model, 'model'):
                chat_model_name = chat_model.model
                
                # Create display-friendly names
                if 'gemini' in chat_model_name.lower():
                    if '2.0' in chat_model_name:
                        model_display = "Gemini 2.0 Flash"
                    elif '1.5' in chat_model_name:
                        model_display = "Gemini 1.5 Pro"
                    else:
                        model_display = "Gemini"
                elif 'mistral' in chat_model_name.lower():
                    if 'large' in chat_model_name.lower():
                        model_display = "Mistral Large"
                    else:
                        model_display = "Mistral"
                else:
                    model_display = chat_model_name
            elif hasattr(chat_model, 'model_name'):
                chat_model_name = chat_model.model_name
                model_display = chat_model_name
        except:
            pass
            
        try:
            if hasattr(embedding_model, 'model'):
                embed_model_name = embedding_model.model
            elif hasattr(embedding_model, 'model_name'):
                embed_model_name = embedding_model.model_name
        except:
            pass
        
        # Show active LLM with colored badge
        if "Gemini" in model_display:
            st.sidebar.success(f"✅ {model_display}")
        elif "Mistral" in model_display:
            st.sidebar.info(f"✅ {model_display}")
        else:
            st.sidebar.info(f"✅ {model_display}")
        
        # Show technical details
        st.sidebar.markdown(f"*Chat: `{chat_model_name}`*")
        st.sidebar.markdown(f"*Embeddings: `{embed_model_name}`*")
        
        # Show Auto mode status with health indication
        if provider == "Auto":
            if 'gemini' in chat_model_name.lower():
                st.sidebar.success("🎯 Auto: Gemini Active")
            elif 'mistral' in chat_model_name.lower():
                # Check if this was a fallback (Gemini key exists but failed)
                gemini_available = bool(os.getenv("GEMINI_API_KEY"))
                if gemini_available:
                    st.sidebar.warning("🔄 Auto: Fell back to Mistral")
                else:
                    st.sidebar.info("🎯 Auto: Mistral Active")
                st.sidebar.info("� Auto mode: Using Mistral")
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Provider init failed: {str(e)[:50]}...")

st.sidebar.markdown("---")

# Display current settings
st.sidebar.markdown("**Current Settings:**")
st.sidebar.markdown(f"- Provider: `{provider}`")
st.sidebar.markdown(f"- Role: `{role}`")
st.sidebar.markdown(f"- Region: `{region}`")
st.sidebar.markdown(f"- Language: `{language}`")

# Main Content Area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📄 Upload CV")
    
    uploaded_file = st.file_uploader(
        "Choose a CV file",
        type=['txt', 'pdf'],
        help="Upload your CV in .txt or .pdf format"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📁 File size: {len(uploaded_file.getvalue()):,} bytes")
        
        # Check PDF support if PDF file is uploaded
        if uploaded_file.name.endswith('.pdf'):
            pdf_supported, pdf_message = check_pdf_support()
            if pdf_supported:
                st.success("🔧 PDF processing available")
            else:
                st.error(f"⚠️ PDF Support Missing: {pdf_message}")
                st.info("💡 To enable PDF support, run: `pip install pymupdf4llm`")
        
        # Show file preview for text files
        if uploaded_file.name.endswith('.txt'):
            with st.expander("📖 File Preview"):
                content = uploaded_file.getvalue().decode('utf-8')
                st.text_area("Content", content[:500] + "..." if len(content) > 500 else content, height=150)

with col2:
    st.header("🚀 Analysis")
    
    if uploaded_file is not None:
        # Show blocking message if API keys are missing
        if not all_keys_valid:
            st.error("🔒 **Analysis Blocked**: Missing required API keys")
            st.info("👆 Configure your API keys in the sidebar to enable analysis")
            
        analyze_button_disabled = not all_keys_valid
        
        if st.button(
            "🔍 Analyze Skills", 
            type="primary", 
            use_container_width=True,
            disabled=analyze_button_disabled,
            help="Analyze your CV for skill gaps" if all_keys_valid else "Configure API keys first"
        ):
            # Save uploaded file temporarily
            temp_file_path = save_uploaded_file(uploaded_file)
            
            try:
                with st.spinner("🔄 Running skill gap analysis..."):
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🧠 Initializing LLM provider...")
                    progress_bar.progress(10)
                    
                    status_text.text("🔗 Building analysis pipeline...")
                    progress_bar.progress(20)
                    
                    status_text.text("⚡ Executing analysis...")
                    progress_bar.progress(30)
                    
                    # Run the analysis
                    result = run_analysis_pipeline(
                        file_path=temp_file_path,
                        provider=provider,
                        role=role,
                        region=region,
                        lang=language,
                        allow_offline_embeddings=allow_offline_embeddings
                    )
                    
                    progress_bar.progress(100)
                    
                    # Show provider information if analysis succeeded
                    if result["success"] and result.get("provider_info"):
                        provider_info = result["provider_info"]
                        if provider_info.get("model_display"):
                            status_text.text(f"✅ Analysis complete using {provider_info['model_display']}!")
                        else:
                            status_text.text("✅ Analysis complete!")
                    else:
                        status_text.text("✅ Analysis complete!")
                    
                    # Store results in session state
                    st.session_state.analysis_result = result
                    st.session_state.analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    else:
        st.info("👆 Please upload a CV file to start the analysis")

# Display Results
if 'analysis_result' in st.session_state:
    result = st.session_state.analysis_result
    timestamp = st.session_state.analysis_timestamp
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    if result["success"]:
        state = result["state"]
        provider_info = result.get("provider_info", {})
        
        # Enhanced provider information display
        if provider_info:
            model_display = provider_info.get("model_display")
            fallback_used = provider_info.get("fallback_used", False)
            warnings = provider_info.get("warnings", [])
            
            # Show active LLM with colored badge
            if model_display:
                if "Gemini" in model_display:
                    st.success(f"🤖 **Active LLM:** {model_display}")
                elif "Mistral" in model_display:
                    st.info(f"🤖 **Active LLM:** {model_display}")
                else:
                    st.info(f"🤖 **Active LLM:** {model_display}")
            
            # Show fallback warning for Auto mode
            if fallback_used:
                st.warning("⚠️ **Auto Mode:** Gemini unavailable, fell back to Mistral")
            
            # Show any warnings
            for warning in warnings:
                if "fallback" in warning.lower():
                    st.warning(f"⚠️ {warning}")
                else:
                    st.info(f"ℹ️ {warning}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📄 CV Processing",
                "✅ Success" if state.cv_text else "❌ Failed"
            )
        
        with col2:
            if state.skill_profile:
                skill_data = convert_to_serializable(state.skill_profile)
                total_skills = len(skill_data.get('explicit', [])) + len(skill_data.get('implicit', []))
                st.metric("🎯 Skills Found", total_skills)
            else:
                st.metric("🎯 Skills Found", "N/A")
        
        with col3:
            if state.market_summary:
                market_data = convert_to_serializable(state.market_summary)
                market_skills = len(market_data.get('in_demand_skills', []))
                st.metric("📈 Market Skills", market_skills)
            else:
                st.metric("📈 Market Skills", "N/A")
        
        with col4:
            report_size = len(state.report_md) if state.report_md else 0
            st.metric("📋 Report Size", f"{report_size:,} chars")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Final Report", "� Sources Used", "�📄 CV Structure", "🎯 Skill Profile", "📊 Market Summary", "� Download"])
        
        with tab1:
            st.subheader("📋 Comprehensive Skill Gap Report")
            if state.report_md:
                st.markdown(state.report_md)
            else:
                st.warning("⚠️ Report generation failed. Check the analysis logs.")
        
        with tab2:
            st.subheader("🔗 Sources Used in Analysis")
            sources = extract_sources_from_state(state)
            
            if sources:
                st.success(f"✅ Found {len(sources)} source(s) from live data gathering")
                
                for i, source in enumerate(sources, 1):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**Source {i}:**")
                    with col2:
                        st.markdown(f"[{source}]({source})")
                
                # Show sources breakdown
                with st.expander("📊 Sources Analysis", expanded=False):
                    st.markdown("**Source Types:**")
                    domains = []
                    for source in sources:
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(source).netloc
                            domains.append(domain)
                        except:
                            domains.append("Unknown")
                    
                    unique_domains = list(set(domains))
                    for domain in unique_domains:
                        count = domains.count(domain)
                        st.markdown(f"- {domain}: {count} source(s)")
            else:
                st.warning("⚠️ No live sources found. This may indicate:")
                st.markdown("- Analysis used cached/offline data")
                st.markdown("- Market intelligence gathering failed")
                st.markdown("- API quota limits reached")
                
                if not allow_offline_embeddings:
                    st.info("🔒 Live-only mode was enabled - check logs for API issues")
        
        with tab3:
            st.subheader("📄 Parsed CV Structure")
            if state.cv_struct:
                cv_data = convert_to_serializable(state.cv_struct)
                with st.expander("🔍 View CV Structure JSON", expanded=True):
                    st.json(cv_data)
            else:
                st.warning("⚠️ CV parsing failed.")
        
        with tab4:
            st.subheader("🎯 Skill Profile Analysis")
            if state.skill_profile:
                skill_data = convert_to_serializable(state.skill_profile)
                
                # Skill categories breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Explicit Skills", len(skill_data.get('explicit', [])))
                with col2:
                    st.metric("Implicit Skills", len(skill_data.get('implicit', [])))
                with col3:
                    st.metric("Transferable Skills", len(skill_data.get('transferable', [])))
                
                with st.expander("🔍 View Skill Profile JSON", expanded=False):
                    st.json(skill_data)
            else:
                st.warning("⚠️ Skill analysis failed.")
        
        with tab5:
            st.subheader("📊 Market Intelligence Summary")
            if state.market_summary:
                market_data = convert_to_serializable(state.market_summary)
                
                # Market skills breakdown
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("In-Demand Skills", len(market_data.get('in_demand_skills', [])))
                with col2:
                    st.metric("Common Tools", len(market_data.get('common_tools', [])))
                with col3:
                    st.metric("Frameworks", len(market_data.get('frameworks', [])))
                with col4:
                    st.metric("Nice-to-Have", len(market_data.get('nice_to_have', [])))
                
                with st.expander("🔍 View Market Summary JSON", expanded=False):
                    st.json(market_data)
            else:
                st.warning("⚠️ Market analysis failed.")
        
        with tab6:
            st.subheader("� Download Artifacts")
            
            # Prepare download data
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.markdown("**📋 Reports**")
                
                # Download final report
                if state.report_md:
                    st.download_button(
                        label="📋 Download Markdown Report",
                        data=state.report_md,
                        file_name=f"skill_gap_report_{timestamp}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                # Download execution logs
                logs_data = {
                    "timestamp": timestamp,
                    "logs": state.logs,
                    "settings": {
                        "provider": provider,
                        "role": role,
                        "region": region,
                        "language": language
                    }
                }
                st.download_button(
                    label="📜 Download Execution Logs",
                    data=json.dumps(logs_data, indent=2),
                    file_name=f"execution_logs_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with download_col2:
                st.markdown("**📊 Analysis Data**")
                
                # Download CV structure
                if state.cv_struct:
                    cv_data = convert_to_serializable(state.cv_struct)
                    st.download_button(
                        label="📄 Download CV Structure",
                        data=json.dumps(cv_data, indent=2),
                        file_name=f"cv_structure_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Download skill profile
                if state.skill_profile:
                    skill_data = convert_to_serializable(state.skill_profile)
                    st.download_button(
                        label="🎯 Download Skill Profile",
                        data=json.dumps(skill_data, indent=2),
                        file_name=f"skill_profile_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Download market summary
                if state.market_summary:
                    market_data = convert_to_serializable(state.market_summary)
                    st.download_button(
                        label="📊 Download Market Summary",
                        data=json.dumps(market_data, indent=2),
                        file_name=f"market_summary_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        # Show analysis logs
        if state.logs:
            with st.expander("📜 Analysis Logs", expanded=False):
                for log in state.logs:
                    st.text(log)
    
    else:
        st.error("❌ Analysis Failed")
        
        # Enhanced error information from provider_info
        provider_info = result.get("provider_info", {})
        if provider_info:
            warnings = provider_info.get("warnings", [])
            requested_provider = provider_info.get("requested", "unknown")
            
            # Show detailed error with context
            st.error(f"**Error:** {result['error']}")
            
            # Show provider context
            if requested_provider != "unknown":
                st.info(f"**Provider:** {requested_provider}")
            
            # Show any warnings that might explain the failure
            for warning in warnings:
                st.warning(f"⚠️ {warning}")
        else:
            st.error(f"Error: {result['error']}")
        
        # Provide guidance based on error type
        error_msg = result['error'].lower()
        if "api key" in error_msg:
            st.info("💡 **Solution:** Check your API keys in the sidebar")
        elif "quota" in error_msg or "limit" in error_msg:
            st.info("💡 **Solution:** Check your API usage limits or try a different provider")
        elif "network" in error_msg or "connection" in error_msg:
            st.info("💡 **Solution:** Check your internet connection and try again")
        elif "provider" in error_msg:
            st.info("💡 **Solution:** Try switching to a different provider or check API keys")
        
        if result.get('traceback'):
            with st.expander("🔍 Technical Details", expanded=False):
                st.code(result['traceback'], language='python')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🎯 Skill Gap Analyst - Powered by LangGraph & Streamlit<br>
        Built for comprehensive CV analysis and career development insights
    </div>
    """, 
    unsafe_allow_html=True
)
