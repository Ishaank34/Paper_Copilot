"""
PaperCopilot - Streamlit Frontend

A clean, user-friendly interface for analyzing research papers using
semantic search and local LLM inference with Ollama.

This application provides single-user desktop access to comprehensive
paper analysis capabilities including student summaries, research mentor
feedback, contributions extraction, and question answering.

Architecture:
- Single "Analyze Paper" button generates ALL standard analyses upfront
- Results cached in session_state for instant tab switching
- Dynamic question tabs have their own interaction buttons
- No analysis reruns when switching tabs or reloading

Requirements:
    - Ollama service running on localhost:11434
    - qwen3:8b model downloaded (ollama pull qwen3:8b)
    - Streamlit installed

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import logging
from typing import Optional, Dict

# Add src directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from paper_analyzer import PaperAnalyzer
import llm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Streamlit Configuration
# ============================================================================

st.set_page_config(
    page_title="PaperCopilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize session state variables for caching."""
    if "analyzer" not in st.session_state:
        st.session_state.analyzer: Optional[PaperAnalyzer] = None
    
    if "paper_loaded" not in st.session_state:
        st.session_state.paper_loaded = False
    
    if "paper_info" not in st.session_state:
        st.session_state.paper_info: Dict = {}
    
    if "temp_file_path" not in st.session_state:
        st.session_state.temp_file_path: Optional[str] = None
    
    if "analyses_generated" not in st.session_state:
        st.session_state.analyses_generated = False
    
    # Cache for ALL analysis results (comprehensive cache)
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache: Dict[str, Optional[str]] = {}


# ============================================================================
# Utility Functions
# ============================================================================

def load_paper_from_upload(uploaded_file) -> bool:
    """
    Load a paper from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        True if paper loaded successfully, False otherwise
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            dir=Path.cwd()
        ) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        st.session_state.temp_file_path = temp_path
        
        # Initialize PaperAnalyzer
        with st.spinner("🔄 Initializing analyzer..."):
            analyzer = PaperAnalyzer(provider="ollama", model_name="qwen3:8b")
        
        # Load paper
        with st.spinner("📖 Loading and parsing paper..."):
            analyzer.load_paper(temp_path)
        
        # Get paper info
        paper_info = analyzer.get_paper_info()
        
        # Cache in session state
        st.session_state.analyzer = analyzer
        st.session_state.paper_info = paper_info
        st.session_state.paper_loaded = True
        st.session_state.analyses_generated = False  # Reset analyses flag
        st.session_state.analysis_cache = {}  # Clear cache on new paper
        
        logger.info(f"Paper loaded: {paper_info['title']}")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to load paper: {str(e)}")
        logger.error(f"Paper loading error: {e}", exc_info=True)
        return False


def generate_all_analyses() -> bool:
    """
    Generate all standard analyses and cache them.
    
    This function runs all 9 standard analysis methods and caches results
    in session_state. This is called once when user clicks "Analyze Paper".
    
    Returns:
        True if all analyses succeeded, False if any failed
    """
    if not st.session_state.paper_loaded or st.session_state.analyzer is None:
        st.error("❌ Paper not loaded. Please load a paper first.")
        return False
    
    analyzer = st.session_state.analyzer
    all_succeeded = True
    
    # Define analyses to run
    analyses = {
        "student_summary": ("generate_student_summary", []),
        "cse_explanation": ("generate_research_mentor_analysis", []),
        "research_mentor": ("generate_research_mentor_analysis", []),
        "contributions": ("extract_contributions", []),
        "limitations": ("extract_limitations", []),
        "future_work": ("extract_future_work", []),
        "datasets_metrics": ("extract_datasets_and_metrics", []),
        "methodology": ("explain_methodology", []),
        "roadmap": ("generate_reading_roadmap", []),
    }
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_analyses = len(analyses)
    completed = 0
    
    # Run each analysis
    for cache_key, (method_name, args) in analyses.items():
        try:
            status_text.text(f"🔄 Generating {cache_key}...")
            
            # Get method from analyzer
            method = getattr(analyzer, method_name)
            
            # Run analysis
            result = method(*args)
            
            # Cache result
            st.session_state.analysis_cache[cache_key] = result
            logger.info(f"Analysis completed: {cache_key}")
            
            completed += 1
            progress_bar.progress(completed / total_analyses)
            
        except Exception as e:
            error_msg = f"❌ Failed to generate {cache_key}: {str(e)}"
            st.error(error_msg)
            st.session_state.analysis_cache[cache_key] = None
            logger.error(f"Analysis error ({cache_key}): {e}", exc_info=True)
            all_succeeded = False
            completed += 1
            progress_bar.progress(completed / total_analyses)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if all_succeeded:
        st.success("✅ All analyses completed successfully!")
        st.session_state.analyses_generated = True
    else:
        st.warning("⚠️ Some analyses failed. Check error messages above.")
        st.session_state.analyses_generated = True  # Mark as attempted
    
    return all_succeeded


def run_dynamic_analysis(
    analysis_key: str,
    method_name: str,
    *args,
    **kwargs
) -> Optional[str]:
    """
    Run a dynamic analysis (e.g., custom question) with caching.
    
    This is for question-answering analyses that take user input.
    Results are cached so re-asking the same question returns instantly.
    
    Args:
        analysis_key: Unique key for caching
        method_name: Name of PaperAnalyzer method to call
        *args: Positional arguments for the method
        **kwargs: Keyword arguments for the method
        
    Returns:
        Analysis result or None if error occurred
    """
    # Check if result is cached
    if analysis_key in st.session_state.analysis_cache:
        logger.debug(f"Using cached result for {analysis_key}")
        return st.session_state.analysis_cache[analysis_key]
    
    # Paper must be loaded
    if not st.session_state.paper_loaded or st.session_state.analyzer is None:
        st.warning("⚠️ Please load a paper first.")
        return None
    
    try:
        # Get the method from analyzer
        analyzer = st.session_state.analyzer
        method = getattr(analyzer, method_name)
        
        # Run analysis with spinner
        with st.spinner(f"🔄 Analyzing paper..."):
            result = method(*args, **kwargs)
        
        # Cache result
        st.session_state.analysis_cache[analysis_key] = result
        logger.info(f"Dynamic analysis completed: {analysis_key}")
        
        return result
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Dynamic analysis error ({analysis_key}): {e}", exc_info=True)
        return None


def get_cached_analysis(cache_key: str) -> Optional[str]:
    """
    Get a cached analysis result.
    
    Args:
        cache_key: Key in analysis_cache
        
    Returns:
        Cached result or None if not available
    """
    return st.session_state.analysis_cache.get(cache_key)


def display_paper_info():
    """Display paper metadata in the main interface."""
    if not st.session_state.paper_loaded:
        return
    
    info = st.session_state.paper_info
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Title", info.get("title", "Unknown")[:30])
    with col2:
        st.metric("Pages", info.get("num_pages", "N/A"))
    with col3:
        st.metric("Chunks", info.get("num_chunks", "N/A"))
    with col4:
        st.metric("Indexed Vectors", info.get("indexed_vectors", "N/A"))


# ============================================================================
# Sidebar Configuration
# ============================================================================

def build_sidebar():
    """Build the sidebar with title, description, and PDF uploader."""
    with st.sidebar:
        # Title and description
        st.markdown("# 📚 PaperCopilot")
        st.markdown("""
        An intelligent research paper analysis tool powered by:
        - **Semantic Search**: FAISS vector database
        - **Local LLM**: Ollama + Qwen3:8b
        - **RAG**: Retrieval-augmented generation
        
        Upload any research paper PDF and explore its contents through
        multiple analytical lenses.
        """)
        
        st.divider()
        
        # PDF Upload Section
        st.markdown("### 📤 Upload Paper")
        uploaded_file = st.file_uploader(
            "Choose a PDF paper",
            type="pdf",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.markdown(f"**Selected**: {uploaded_file.name}")
            
            if st.button("✅ Load Paper", use_container_width=True, type="primary"):
                if load_paper_from_upload(uploaded_file):
                    st.success("✅ Paper loaded successfully!")
                else:
                    st.error("❌ Failed to load paper. Check error messages above.")
        
        st.divider()
        
        # Analysis Generation Section
        if st.session_state.paper_loaded:
            st.markdown("### 🚀 Generate Analyses")
            
            if not st.session_state.analyses_generated:
                if st.button(
                    "📊 Generate All Analyses",
                    use_container_width=True,
                    type="primary"
                ):
                    generate_all_analyses()
            else:
                st.success("✅ Analyses generated!")
                if st.button(
                    "🔄 Regenerate Analyses",
                    use_container_width=True,
                    help="Regenerate all analyses (slower)"
                ):
                    generate_all_analyses()
            
            st.divider()
            
            # Paper Info
            st.markdown("### 📋 Paper Info")
            info = st.session_state.paper_info
            st.info(f"""
            **Title**: {info.get('title', 'Unknown')}
            
            **Pages**: {info.get('num_pages', 'N/A')}
            
            **Chunks**: {info.get('num_chunks', 'N/A')}
            
            **Cached Results**: {len([k for k, v in st.session_state.analysis_cache.items() if v is not None])} / 9
            """)
        
        # System Info
        st.divider()
        st.markdown("### ⚙️ System")
        st.caption("Provider: Ollama (Local)")
        st.caption("Model: Qwen3:8b")
        st.caption("Backend: RAG with FAISS")


# ============================================================================
# Main Interface - Tab Definitions (Display-Only)
# ============================================================================

def tab_student_summary():
    """Tab: Student-friendly summary of the paper (display-only)."""
    st.header("📄 Student Summary")
    st.markdown("""
    An undergraduate-friendly explanation of the paper translated into
    accessible language without requiring deep domain expertise.
    """)
    
    result = get_cached_analysis("student_summary")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_cse_explanation():
    """Tab: CSE-focused explanation of the paper (display-only)."""
    st.header("💻 Explain Like I'm a CSE Student")
    st.markdown("""
    Technical explanation tailored for computer science students,
    focusing on algorithms, data structures, and computational aspects.
    """)
    
    result = get_cached_analysis("cse_explanation")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_research_mentor():
    """Tab: Deep research-level analysis (display-only)."""
    st.header("👨‍🏫 Research Mentor Analysis")
    st.markdown("""
    Deep, research-level analysis as if from a senior PhD mentor,
    providing critical evaluation, assumptions, strengths, and weaknesses.
    """)
    
    result = get_cached_analysis("research_mentor")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_contributions():
    """Tab: Key contributions extraction (display-only)."""
    st.header("⭐ Key Contributions")
    st.markdown("""
    Clearly identified novelty claims and main research contributions
    presented in a structured, enumerated format.
    """)
    
    result = get_cached_analysis("contributions")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_limitations():
    """Tab: Limitations and constraints (display-only)."""
    st.header("⚠️ Limitations")
    st.markdown("""
    Comprehensive analysis of both explicit and implicit limitations,
    edge cases, scalability concerns, and generalization constraints.
    """)
    
    result = get_cached_analysis("limitations")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_future_work():
    """Tab: Future research directions (display-only)."""
    st.header("🔮 Future Research Directions")
    st.markdown("""
    Natural extensions, open research questions, and new opportunities
    that build on this work.
    """)
    
    result = get_cached_analysis("future_work")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_datasets_metrics():
    """Tab: Datasets and evaluation metrics (display-only)."""
    st.header("📊 Datasets & Metrics")
    st.markdown("""
    Systematic documentation of experimental configuration including
    datasets, metrics, baselines, and evaluation setup for reproducibility.
    """)
    
    result = get_cached_analysis("datasets_metrics")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_methodology():
    """Tab: Technical methodology explanation (display-only)."""
    st.header("⚙️ Methodology")
    st.markdown("""
    Detailed explanation of the technical methodology, from training pipeline
    to deployment, with clear architectural and algorithmic details.
    """)
    
    result = get_cached_analysis("methodology")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_reading_roadmap():
    """Tab: Structured reading roadmap (display-only)."""
    st.header("🛣️ Reading Roadmap")
    st.markdown("""
    Personalized reading guide through the paper with difficulty levels,
    section recommendations, and time estimates for efficient understanding.
    """)
    
    result = get_cached_analysis("roadmap")
    if result:
        st.markdown(result)
    elif st.session_state.paper_loaded:
        st.info("ℹ️ Click 'Generate All Analyses' in the sidebar to generate this analysis.")
    else:
        st.warning("⚠️ Load a paper first.")


def tab_ask_questions():
    """Tab: General question answering (with dynamic button)."""
    st.header("❓ Ask Questions")
    st.markdown("""
    Ask any question about the paper and get context-grounded answers
    based on the paper's content using semantic search and retrieval.
    """)
    
    if not st.session_state.paper_loaded:
        st.warning("⚠️ Load a paper first.")
        return
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What datasets were used? How does the model perform?",
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Ask", use_container_width=True):
        if not question or not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            # Use unique cache key based on question hash
            cache_key = f"qa_{hash(question) % 10**8}"
            result = run_dynamic_analysis(
                cache_key,
                "answer_question",
                question
            )
            if result:
                st.markdown(result)


def tab_deep_questions():
    """Tab: Deep analytical questions (with dynamic button)."""
    st.header("🔬 Deep Questions")
    st.markdown("""
    Ask deeper analytical questions and get comprehensive analysis
    exploring "why" and "how" with nuanced insights grounded in the paper.
    """)
    
    if not st.session_state.paper_loaded:
        st.warning("⚠️ Load a paper first.")
        return
    
    question = st.text_area(
        "Your analytical question:",
        placeholder="e.g., Why does this approach outperform baselines? What assumptions underlie this method?",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("🧠 Analyze", use_container_width=True):
        if not question or not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            # Use unique cache key based on question hash
            cache_key = f"deep_qa_{hash(question) % 10**8}"
            result = run_dynamic_analysis(
                cache_key,
                "deep_answer",
                question
            )
            if result:
                st.markdown(result)


# ============================================================================
# Main Application Logic
# ============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Build sidebar
    build_sidebar()
    
    # Main content area
    if not st.session_state.paper_loaded:
        # Landing page
        st.markdown("""
        # Welcome to PaperCopilot 📚
        
        **PaperCopilot** is an intelligent research paper analysis system that uses:
        - 🔍 **Semantic Search**: FAISS vector database for intelligent retrieval
        - 🤖 **Local LLM**: Ollama with Qwen3:8b for privacy-first analysis
        - 🎯 **RAG Architecture**: Retrieval-Augmented Generation for accurate insights
        
        ## Getting Started
        
        1. **Upload a Paper**: Use the sidebar to upload a PDF research paper
        2. **Load**: Click "Load Paper" to parse and index the paper
        3. **Generate**: Click "Generate All Analyses" to create all analyses at once
        4. **Explore**: Use the tabs below to explore different analytical perspectives
        
        ## Analysis Perspectives
        
        Once you generate analyses, explore these perspectives:
        
        - 📄 **Student Summary**: Undergraduate-friendly overview
        - 💻 **CSE Explanation**: Technical depth for computer scientists
        - 👨‍🏫 **Research Mentor**: Deep critical analysis
        - ⭐ **Contributions**: Key novelties and innovations
        - ⚠️ **Limitations**: Constraints and edge cases
        - 🔮 **Future Work**: Research directions and extensions
        - 📊 **Datasets & Metrics**: Experimental configuration
        - ⚙️ **Methodology**: Technical approach and implementation
        - 🛣️ **Reading Roadmap**: Personalized reading guide
        - ❓ **Ask Questions**: General Q&A with paper content
        - 🔬 **Deep Questions**: Analytical exploration
        
        ## Architecture Highlights
        
        - **Single-Click Analysis**: Generate all 9 standard analyses at once
        - **Instant Tab Switching**: All results cached, no reruns
        - **Dynamic Q&A**: Ask custom questions with caching
        - **Local Processing**: 100% offline after model download
        - **Privacy First**: All data stays on your machine
        
        ## Requirements
        
        - Ollama service running on localhost:11434
        - Qwen3:8b model downloaded: `ollama pull qwen3:8b`
        - Python dependencies installed
        
        **Start by uploading a paper in the sidebar →**
        """)
        
    else:
        # Paper loaded - show analysis interface
        st.markdown("## Paper Analysis Dashboard")
        display_paper_info()
        st.divider()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "📄 Student Summary",
            "💻 CSE Student",
            "👨‍🏫 Research Mentor",
            "⭐ Contributions",
            "⚠️ Limitations",
            "🔮 Future Work",
            "📊 Datasets & Metrics",
            "⚙️ Methodology",
            "🛣️ Roadmap",
            "❓ Questions",
            "🔬 Deep Questions"
        ])
        
        with tab1:
            tab_student_summary()
        
        with tab2:
            tab_cse_explanation()
        
        with tab3:
            tab_research_mentor()
        
        with tab4:
            tab_contributions()
        
        with tab5:
            tab_limitations()
        
        with tab6:
            tab_future_work()
        
        with tab7:
            tab_datasets_metrics()
        
        with tab8:
            tab_methodology()
        
        with tab9:
            tab_reading_roadmap()
        
        with tab10:
            tab_ask_questions()
        
        with tab11:
            tab_deep_questions()


if __name__ == "__main__":
    main()

