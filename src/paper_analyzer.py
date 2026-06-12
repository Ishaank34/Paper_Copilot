"""
Paper Analyzer Module - Orchestrates the PaperCopilot Backend

This module provides the PaperAnalyzer class, which orchestrates all backend
components (parser, embeddings, vectorstore, prompts, llm_client) to provide
comprehensive paper analysis capabilities.

The PaperAnalyzer acts as the central coordinator, handling:
- Paper parsing and chunk extraction
- Embedding generation and storage in vector database
- Semantic retrieval of relevant paper sections
- LLM-powered analysis using retrieved context (RAG)
- Support for multiple papers (future enhancement)

Example:
    >>> analyzer = PaperAnalyzer()
    >>> analyzer.load_paper("paper.pdf")
    >>> summary = analyzer.generate_student_summary()
    >>> mentor_analysis = analyzer.generate_research_mentor_analysis()
    >>> answer = analyzer.answer_question("What datasets were used?")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from parser import PDFParser
from embeddings import embed_text, embed_chunks
from vectorstore import VectorStore
import prompts
import llm_client

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
DEFAULT_TOP_K = 5  # Number of chunks to retrieve for context
CHUNK_SIZE = 1000  # Approximate characters per chunk
MIN_CHUNK_SIZE = 100  # Minimum characters for a chunk to be meaningful
SYSTEM_PROMPT_BASE = (
    "You are an expert academic research assistant. "
    "Your role is to help students and researchers understand research papers deeply. "
    "Be accurate, insightful, and grounded in the provided paper content. "
    "When unsure, acknowledge the limitation explicitly."
)


class PaperAnalyzer:
    """
    Orchestrator for comprehensive paper analysis using RAG (Retrieval-Augmented Generation).
    
    This class manages the complete pipeline:
    1. Parse PDF and extract structured content
    2. Generate embeddings for all paper sections/chunks
    3. Store embeddings in FAISS vector database
    4. Retrieve top-k relevant chunks for any query
    5. Use LLM with retrieved context to generate insights
    
    Attributes:
        paper_path: Path to the loaded PDF paper
        paper_name: Human-readable name of the paper
        parser: PDFParser instance
        vectorstore: VectorStore instance for semantic search
        paper_metadata: Metadata about the loaded paper
        paper_sections: Extracted sections from the paper
        paper_chunks: Chunked text for embedding
        _papers: Dictionary storing data for multiple papers (future use)
    """
    
    def __init__(self, provider: str = "gemini", model_name: Optional[str] = None):
        """
        Initialize the PaperAnalyzer with empty state and specified LLM provider.
        
        Args:
            provider: LLM provider to use (default: "gemini")
                     Supported: "gemini", "ollama", "claude", "openai"
            model_name: Specific model to use. If None, uses provider's default:
                       - Gemini: "gemini-2.5-flash"
                       - Ollama: "qwen3:8b"
                       - Claude: "claude-3-5-sonnet-20241022"
                       - OpenAI: "gpt-4o"
        """
        self.paper_path: Optional[Path] = None
        self.paper_name: Optional[str] = None
        self.parser: Optional[PDFParser] = None
        self.vectorstore = VectorStore()
        self.vectorstore.create_index()
        
        self.paper_metadata: Dict[str, Any] = {}
        self.paper_sections: Dict[str, str] = {}
        self.paper_chunks: List[str] = []
        
        # Future multi-paper support: store data for each paper
        self._papers: Dict[str, Dict[str, Any]] = {}
        
        # LLM provider configuration
        self.provider = provider.lower()
        self.model_name = model_name
        
        logger.info(f"PaperAnalyzer initialized with provider: {self.provider} (model: {model_name or 'default'})")
    
    # ========================================================================
    # Paper Loading & Preparation
    # ========================================================================
    
    def load_paper(self, pdf_path: str) -> None:
        """
        Load and prepare a research paper for analysis.
        
        This method:
        1. Parses the PDF to extract text and metadata
        2. Extracts and organizes paper sections
        3. Creates chunks for embedding
        4. Generates embeddings for all chunks
        5. Indexes embeddings in the vector store
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If parsing or embedding fails
            
        Example:
            >>> analyzer = PaperAnalyzer()
            >>> analyzer.load_paper("path/to/paper.pdf")
        """
        try:
            self.paper_path = Path(pdf_path)
            if not self.paper_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            self.paper_name = self.paper_path.stem
            logger.info(f"Loading paper: {self.paper_name}")
            
            # Parse PDF
            self.parser = PDFParser(pdf_path)
            self.paper_metadata = self.parser.extract_metadata()
            logger.info(
                f"Paper: {self.paper_metadata.get('title', 'Unknown')} "
                f"({self.paper_metadata.get('num_pages', 0)} pages)"
            )
            
            # Extract text and sections
            full_text = self.parser.extract_text()
            self.paper_sections = self._organize_sections(full_text)
            
            # Create chunks for embedding
            self.paper_chunks = self._create_chunks(full_text)
            logger.info(f"Created {len(self.paper_chunks)} chunks for embedding")
            
            # Generate embeddings and index
            self._index_paper_chunks()
            
            # Store paper in multi-paper registry
            self._papers[self.paper_name] = {
                'path': str(self.paper_path),
                'metadata': self.paper_metadata,
                'sections': self.paper_sections,
                'chunks': self.paper_chunks,
            }
            
            logger.info(f"Paper '{self.paper_name}' loaded and indexed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load paper: {e}")
            raise
    
    def _organize_sections(self, full_text: str) -> Dict[str, str]:
        """
        Organize paper text into logical sections.
        
        Args:
            full_text: Complete paper text
            
        Returns:
            Dictionary mapping section names to their content
        """
        # Simple section organization (can be enhanced with better parsing)
        sections = {'full_text': full_text}
        logger.debug("Organized paper into sections")
        return sections
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split paper text into meaningful chunks for embedding.
        
        Uses a simple chunking strategy: split by approximate character count
        while respecting sentence boundaries to maintain semantic coherence.
        
        Args:
            text: Paper text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < CHUNK_SIZE:
                current_chunk += sentence + ". "
            else:
                if len(current_chunk) > MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if len(current_chunk) > MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())
        
        logger.debug(f"Created {len(chunks)} chunks from paper text")
        return chunks
    
    def _index_paper_chunks(self) -> None:
        """
        Generate embeddings for paper chunks and index them.
        
        Raises:
            RuntimeError: If embedding or indexing fails
        """
        try:
            if not self.paper_chunks:
                logger.warning("No chunks to index")
                return
            
            logger.info(f"Embedding {len(self.paper_chunks)} chunks...")
            embeddings = embed_chunks(self.paper_chunks)
            
            # Create metadata for each chunk
            metadata_list = [
                {
                    'paper_name': self.paper_name,
                    'chunk_index': i,
                    'chunk_text': chunk,
                    'section': 'unknown',
                }
                for i, chunk in enumerate(self.paper_chunks)
            ]
            
            # Add to vector store
            self.vectorstore.add_embeddings(embeddings, metadata_list)
            logger.info(f"Indexed {len(self.paper_chunks)} chunks in vector store")
            
        except Exception as e:
            logger.error(f"Failed to index paper chunks: {e}")
            raise
    
    # ========================================================================
    # Retrieval & Querying
    # ========================================================================
    
    def _retrieve_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve top-k most relevant chunks for a query using semantic search.
        
        Args:
            query: Query or question about the paper
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple of (concatenated_context, result_list) where:
            - concatenated_context: Combined text for LLM (RAG)
            - result_list: Individual retrieval results with metadata
            
        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            # Embed the query
            query_embedding = embed_text(query)
            
            # Search vector store
            results = self.vectorstore.search(query_embedding, k=top_k)
            logger.debug(f"Retrieved {len(results)} chunks for query")
            
            # Concatenate retrieved chunks as context
            context_parts = []
            for result in results:
                chunk_text = result['metadata'].get('chunk_text', '')
                score = result['score']
                context_parts.append(f"[Relevance: {score:.2f}] {chunk_text}")
            
            context = "\n\n".join(context_parts)
            return context, results
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise
    
    # ========================================================================
    # Analysis Methods
    # ========================================================================
    
    def generate_student_summary(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Generate an undergraduate-friendly explanation of the paper.
        
        Purpose:
            Translate technical content into accessible language for students
            without deep expertise in the domain.
        
        Returns:
            Student-friendly summary of the paper
            
        Raises:
            RuntimeError: If paper not loaded or analysis fails
            
        Example:
            >>> summary = analyzer.generate_student_summary()
            >>> print(summary)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Generating student summary...")
            
            # Retrieve broad overview chunks
            context, _ = self._retrieve_context(
                "main contribution problem solved approach results",
                top_k=top_k
            )
            
            # Get student-focused prompt
            system_prompt = SYSTEM_PROMPT_BASE + " Explain to an undergraduate student."
            user_prompt = prompts.get_student_summary_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Student summary generated")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate student summary: {e}")
            raise
    
    def generate_research_mentor_analysis(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Generate deep, research-level analysis of the paper.
        
        Purpose:
            Act as a senior PhD mentor providing critical analysis, identifying
            assumptions, strengths, weaknesses, and research implications.
        
        Returns:
            Deep mentor-level analysis
            
        Raises:
            RuntimeError: If paper not loaded or analysis fails
            
        Example:
            >>> analysis = analyzer.generate_research_mentor_analysis()
            >>> print(analysis)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Generating research mentor analysis...")
            
            # Retrieve comprehensive context
            context, _ = self._retrieve_context(
                "methodology technical approach assumptions limitations strengths weaknesses",
                top_k=top_k
            )
            
            # Get mentor-focused prompt
            system_prompt = SYSTEM_PROMPT_BASE + " Provide deep research-level analysis."
            user_prompt = prompts.get_research_mentor_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Research mentor analysis generated")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate mentor analysis: {e}")
            raise
    
    def review_paper(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Generate a peer review of the paper.
        
        Purpose:
            Analyze the paper as a reviewer for a top-tier conference (NeurIPS,
            ICLR, ICML), evaluating novelty, rigor, and impact.
        
        Returns:
            Comprehensive peer review
            
        Raises:
            RuntimeError: If paper not loaded or review fails
            
        Example:
            >>> review = analyzer.review_paper()
            >>> print(review)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Generating peer review...")
            
            # Retrieve evaluation-focused context
            context, _ = self._retrieve_context(
                "novelty experiments evaluation baselines results contributions",
                top_k=top_k
            )
            
            # Get reviewer prompt
            system_prompt = SYSTEM_PROMPT_BASE + " Provide constructive peer review feedback."
            user_prompt = prompts.get_reviewer_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Peer review generated")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate review: {e}")
            raise
    
    def extract_contributions(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Extract and articulate the paper's main research contributions.
        
        Purpose:
            Clearly identify novelty claims and main contributions in a
            structured, enumerated format.
        
        Returns:
            Structured list of contributions
            
        Raises:
            RuntimeError: If paper not loaded or extraction fails
            
        Example:
            >>> contributions = analyzer.extract_contributions()
            >>> print(contributions)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Extracting contributions...")
            
            # Retrieve contribution-focused context
            context, _ = self._retrieve_context(
                "contribution novel method innovation improvement results",
                top_k=top_k
            )
            
            # Get contributions prompt
            system_prompt = SYSTEM_PROMPT_BASE
            user_prompt = prompts.get_contributions_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Contributions extracted")
            return response
            
        except Exception as e:
            logger.error(f"Failed to extract contributions: {e}")
            raise
    
    def extract_limitations(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Identify and articulate limitations of the paper.
        
        Purpose:
            Uncover both explicit and implicit limitations, edge cases,
            and constraints of the proposed approach.
        
        Returns:
            Comprehensive list of limitations
            
        Raises:
            RuntimeError: If paper not loaded or extraction fails
            
        Example:
            >>> limitations = analyzer.extract_limitations()
            >>> print(limitations)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Extracting limitations...")
            
            # Retrieve limitations-focused context
            context, _ = self._retrieve_context(
                "limitations constraints edge cases assumptions scalability generalization",
                top_k=top_k
            )
            
            # Get limitations prompt
            system_prompt = SYSTEM_PROMPT_BASE
            user_prompt = prompts.get_limitations_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Limitations extracted")
            return response
            
        except Exception as e:
            logger.error(f"Failed to extract limitations: {e}")
            raise
    
    def extract_future_work(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Suggest future research directions inspired by this paper.
        
        Purpose:
            Identify natural extensions, open questions, and new research
            opportunities that build on this work.
        
        Returns:
            Suggested future work and research directions
            
        Raises:
            RuntimeError: If paper not loaded or extraction fails
            
        Example:
            >>> future = analyzer.extract_future_work()
            >>> print(future)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Extracting future work...")
            
            # Retrieve context for future directions
            context, _ = self._retrieve_context(
                "future work open questions extensions limitations applications",
                top_k=top_k
            )
            
            # Get future work prompt
            system_prompt = SYSTEM_PROMPT_BASE
            user_prompt = prompts.get_future_work_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Future work extracted")
            return response
            
        except Exception as e:
            logger.error(f"Failed to extract future work: {e}")
            raise
    
    def extract_datasets_and_metrics(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Extract evaluation details: datasets, metrics, baselines, and setup.
        
        Purpose:
            Systematically document experimental configuration for reproducibility
            and understanding of evaluation scope.
        
        Returns:
            Structured evaluation details
            
        Raises:
            RuntimeError: If paper not loaded or extraction fails
            
        Example:
            >>> eval_details = analyzer.extract_datasets_and_metrics()
            >>> print(eval_details)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Extracting datasets and metrics...")
            
            # Retrieve evaluation-focused context
            context, _ = self._retrieve_context(
                "dataset metrics evaluation experimental setup baselines results",
                top_k=top_k
            )
            
            # Get dataset/metrics prompt
            system_prompt = SYSTEM_PROMPT_BASE
            user_prompt = prompts.get_dataset_metrics_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Datasets and metrics extracted")
            return response
            
        except Exception as e:
            logger.error(f"Failed to extract datasets and metrics: {e}")
            raise
    
    def explain_methodology(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Provide detailed explanation of the technical methodology.
        
        Purpose:
            Explain how the method works, from training pipeline to deployment,
            with clear architectural and algorithmic details.
        
        Returns:
            Comprehensive methodology explanation
            
        Raises:
            RuntimeError: If paper not loaded or explanation fails
            
        Example:
            >>> methodology = analyzer.explain_methodology()
            >>> print(methodology)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Explaining methodology...")
            
            # Retrieve methodology-focused context
            context, _ = self._retrieve_context(
                "methodology approach algorithm training inference architecture implementation",
                top_k=top_k
            )
            
            # Get methodology prompt
            system_prompt = SYSTEM_PROMPT_BASE
            user_prompt = prompts.get_methodology_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Methodology explained")
            return response
            
        except Exception as e:
            logger.error(f"Failed to explain methodology: {e}")
            raise
    
    def generate_reading_roadmap(self, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Generate a structured reading roadmap for the paper.
        
        Purpose:
            Guide readers through the paper with a personalized reading plan
            that balances understanding with time efficiency.
        
        Returns:
            Structured reading roadmap with difficulty, sections, and time estimates
            
        Raises:
            RuntimeError: If paper not loaded or generation fails
            
        Example:
            >>> roadmap = analyzer.generate_reading_roadmap()
            >>> print(roadmap)
        """
        self._check_paper_loaded()
        
        try:
            logger.info("Generating reading roadmap...")
            
            # Retrieve comprehensive context
            context, _ = self._retrieve_context(
                "abstract introduction methodology experiments conclusion key figures",
                top_k=top_k
            )
            
            # Get roadmap prompt
            system_prompt = SYSTEM_PROMPT_BASE
            user_prompt = prompts.get_roadmap_prompt()
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Reading roadmap generated")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate reading roadmap: {e}")
            raise
    
    def answer_question(self, question: str, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Answer a general question about the paper.
        
        Purpose:
            Provide direct, context-grounded answers to user questions using
            retrieved paper sections. Prioritizes accuracy over hallucination.
        
        Args:
            question: User's question about the paper
            top_k: Number of context chunks to retrieve
            
        Returns:
            Answer based on paper content
            
        Raises:
            RuntimeError: If paper not loaded or answering fails
            ValueError: If question is empty
            
        Example:
            >>> answer = analyzer.answer_question("What is the main contribution?")
            >>> print(answer)
        """
        self._check_paper_loaded()
        
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        try:
            logger.info(f"Answering question: {question[:50]}...")
            
            # Retrieve context relevant to the question
            context, _ = self._retrieve_context(question, top_k=top_k)
            
            # Get QA prompt
            system_prompt = SYSTEM_PROMPT_BASE + " Answer only based on provided context."
            user_prompt = prompts.get_qa_prompt(question)
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Question answered")
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise
    
    def deep_answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Provide a deep, analytical answer to a question about the paper.
        
        Purpose:
            Go beyond surface-level answers to explore "why" and "how" deeply,
            providing comprehensive analysis grounded in the paper.
        
        Args:
            question: Analytical question about the paper
            top_k: Number of context chunks to retrieve
            
        Returns:
            Deep, analytical answer with reasoning and nuance
            
        Raises:
            RuntimeError: If paper not loaded or answering fails
            ValueError: If question is empty
            
        Example:
            >>> answer = analyzer.deep_answer(
            ...     "Why does this method outperform baselines?"
            ... )
            >>> print(answer)
        """
        self._check_paper_loaded()
        
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        try:
            logger.info(f"Providing deep answer: {question[:50]}...")
            
            # Retrieve comprehensive context
            context, _ = self._retrieve_context(question, top_k=top_k)
            
            # Get deep QA prompt
            system_prompt = SYSTEM_PROMPT_BASE + " Provide deep analytical insights."
            user_prompt = prompts.get_deep_qa_prompt(question)
            
            # Generate response
            response = llm_client.answer_with_context(
                context=context,
                question=user_prompt,
                system_prompt=system_prompt,
                provider=self.provider,
                model_name=self.model_name
            )
            
            logger.info("Deep answer provided")
            return response
            
        except Exception as e:
            logger.error(f"Failed to provide deep answer: {e}")
            raise
    
    # ========================================================================
    # Utility & Support Methods
    # ========================================================================
    
    def _check_paper_loaded(self) -> None:
        """
        Verify that a paper has been loaded.
        
        Raises:
            RuntimeError: If no paper is currently loaded
        """
        if self.paper_name is None or not self.paper_chunks:
            raise RuntimeError(
                "No paper loaded. Call load_paper() first to load a research paper."
            )
    
    def get_paper_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded paper.
        
        Returns:
            Dictionary with paper metadata, section count, chunk count, etc.
            
        Raises:
            RuntimeError: If no paper is loaded
            
        Example:
            >>> info = analyzer.get_paper_info()
            >>> print(f"Paper: {info['title']}, {info['num_pages']} pages")
        """
        self._check_paper_loaded()
        
        return {
            'name': self.paper_name,
            'path': str(self.paper_path),
            'title': self.paper_metadata.get('title', 'Unknown'),
            'num_pages': self.paper_metadata.get('num_pages', 0),
            'num_chunks': len(self.paper_chunks),
            'num_sections': len(self.paper_sections),
            'indexed_vectors': self.vectorstore.index.ntotal if self.vectorstore.index else 0,
        }
    
    def get_available_papers(self) -> List[str]:
        """
        Get list of currently available papers (loaded or stored).
        
        Returns:
            List of paper names
        """
        return list(self._papers.keys())
    
    def reset(self) -> None:
        """
        Reset the analyzer to initial state.
        
        Useful for loading a new paper or clearing memory.
        
        Example:
            >>> analyzer.reset()
            >>> analyzer.load_paper("another_paper.pdf")
        """
        self.paper_path = None
        self.paper_name = None
        self.parser = None
        self.vectorstore = VectorStore()
        self.vectorstore.create_index()
        self.paper_metadata = {}
        self.paper_sections = {}
        self.paper_chunks = []
        llm_client.reset_client()
        logger.info("PaperAnalyzer reset to initial state")
    
    # ========================================================================
    # Future Multi-Paper Support (Stubs)
    # ========================================================================
    
    def compare_papers(self, paper1: str, paper2: str, aspect: str = "methodology") -> str:
        """
        Compare two papers on a specific aspect (future implementation).
        
        This method is a placeholder for future multi-paper comparison functionality.
        When implemented, it will enable:
        - Comparing methodologies across papers
        - Identifying differences in experimental approaches
        - Finding papers with similar contributions
        - Tracking evolution of ideas across multiple papers
        
        Args:
            paper1: Name of first paper
            paper2: Name of second paper
            aspect: Aspect to compare (methodology, results, datasets, etc.)
            
        Returns:
            Comparison analysis
            
        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            "Multi-paper comparison not yet implemented. "
            "Load a single paper and use individual analysis methods."
        )
    
    def extract_cross_paper_themes(self, papers: Optional[List[str]] = None) -> str:
        """
        Extract common themes across multiple papers (future implementation).
        
        This is a placeholder for identifying patterns and connections across
        the loaded papers in the system.
        
        Args:
            papers: List of paper names to analyze (all if None)
            
        Returns:
            Analysis of cross-paper themes
            
        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            "Cross-paper theme extraction not yet implemented. "
            "Use single-paper analysis methods currently."
        )


# ============================================================================
# Demonstration & Testing
# ============================================================================

if __name__ == "__main__":
    """Example usage of PaperAnalyzer."""
    logger.info("PaperAnalyzer Module - Example Usage")
    
    try:
        # Initialize analyzer
        analyzer = PaperAnalyzer()
        
        # Example: Load a paper
        paper_path = "papers/example_paper.pdf"
        logger.info(f"To use the analyzer, load a paper:")
        logger.info(f"  analyzer.load_paper('{paper_path}')")
        
        # Show available analysis methods
        logger.info("\nAvailable analysis methods:")
        methods = [
            "generate_student_summary()",
            "generate_research_mentor_analysis()",
            "review_paper()",
            "extract_contributions()",
            "extract_limitations()",
            "extract_future_work()",
            "extract_datasets_and_metrics()",
            "explain_methodology()",
            "generate_reading_roadmap()",
            "answer_question(question)",
            "deep_answer(question)",
            "get_paper_info()",
        ]
        for method in methods:
            logger.info(f"  - {method}")
        
    except Exception as e:
        logger.error(f"Example error: {e}")
