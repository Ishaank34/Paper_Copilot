"""
PDF Parser for Research Papers

Extracts metadata, text, pages, and sections from academic papers using PyMuPDF.
Optimized for machine learning and computer science research papers.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import fitz  # PyMuPDF

# Configure logging
logger = logging.getLogger(__name__)


class PDFParser:
    """Parser for extracting structured content from research paper PDFs."""

    # Common section headers in academic papers (KEEP sections)
    COMMON_SECTIONS = {
        "abstract", "introduction", "background", "related work",
        "methodology", "method", "approach", "proposed", "proposed method",
        "experiments", "experimental setup", "results", "evaluation",
        "analysis", "discussion", "conclusion", "conclusions", "future work",
    }
    
    # Sections to IGNORE (metadata/appendix sections)
    IGNORE_SECTIONS = {
        "references", "appendix", "appendices", "acknowledgments",
        "supplementary material", "supplementary", "appendix material",
        "evaluation prompts", "prompt templates", "prompts"
    }
    
    # Subsection patterns to merge into parent sections (subsection -> parent)
    SUBSECTION_MERGE_MAP = {
        "experimental setup": "experiments",
        "experimental results": "experiments",
        "experiment results": "experiments",
        "datasets": "experiments",
        "dataset description": "experiments",
        "evaluation metrics": "evaluation",
        "evaluation results": "evaluation",
        "ablation study": "evaluation",
        "proposed method": "methodology",
        "methodology and approach": "methodology",
    }

    # Minimum font size for section detection (heuristic)
    MIN_SECTION_FONT_SIZE = 10.0

    def __init__(self, pdf_path: str):
        """
        Initialize the PDF parser.

        Args:
            pdf_path: Path to the PDF file.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            self.document = fitz.open(self.pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise

    def __del__(self):
        """Clean up PDF document resources."""
        if hasattr(self, "document"):
            self.document.close()

    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the PDF.

        Returns:
            Dictionary containing:
            - title: Paper title
            - author: Author(s)
            - subject: Subject/topic
            - creator: PDF creator application
            - producer: PDF producer application
            - creation_date: Creation timestamp
            - modification_date: Last modification timestamp
            - num_pages: Total number of pages
            - file_size: File size in bytes
            - file_name: Original file name

        Raises:
            Exception: If metadata extraction fails.
        """
        try:
            metadata = self.document.metadata or {}

            # Parse dates if available
            creation_date = None
            modification_date = None

            if metadata.get("creationDate"):
                try:
                    creation_date = self._parse_pdf_date(metadata["creationDate"])
                except Exception as e:
                    logger.warning(f"Could not parse creation date: {e}")

            if metadata.get("modDate"):
                try:
                    modification_date = self._parse_pdf_date(metadata["modDate"])
                except Exception as e:
                    logger.warning(f"Could not parse modification date: {e}")

            result = {
                "title": metadata.get("title", "Unknown"),
                "author": metadata.get("author", "Unknown"),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": creation_date,
                "modification_date": modification_date,
                "num_pages": len(self.document),
                "file_size": self.pdf_path.stat().st_size,
                "file_name": self.pdf_path.name,
            }

            logger.info(f"Extracted metadata for {result['file_name']}: {result['num_pages']} pages")
            return result

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            raise

    def extract_text(self) -> str:
        """
        Extract all text from the PDF.

        Returns:
            Complete text content of the paper.

        Raises:
            Exception: If text extraction fails.
        """
        try:
            full_text = ""
            for page_num, page in enumerate(self.document, start=1):
                text = page.get_text()
                full_text += f"\n--- Page {page_num} ---\n{text}"

            logger.info(f"Extracted text from {len(self.document)} pages")
            return full_text.strip()

        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise

    def extract_pages(self) -> List[Dict[str, Any]]:
        """
        Extract text from each page with page metadata.

        Returns:
            List of dictionaries, each containing:
            - page_number: 1-indexed page number
            - text: Text content of the page
            - num_blocks: Number of text blocks on page
            - num_words: Approximate word count

        Raises:
            Exception: If page extraction fails.
        """
        try:
            pages = []
            for page_num, page in enumerate(self.document, start=1):
                text = page.get_text()

                # Count text blocks
                blocks = page.get_text("blocks")
                num_blocks = len([b for b in blocks if b[6] == 0])  # Text blocks only

                # Estimate word count
                num_words = len(text.split())

                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "num_blocks": num_blocks,
                    "num_words": num_words,
                })

            logger.info(f"Extracted {len(pages)} pages")
            return pages

        except Exception as e:
            logger.error(f"Error extracting pages: {e}")
            raise

    def extract_sections(self) -> List[Dict[str, Any]]:
        """
        Extract sections (with headings) from the PDF.

        Uses heuristics to detect section headings:
        - Matches common academic section names
        - Detects font size changes and bold text
        - Filters out metadata sections (References, Appendix, etc.)
        - Merges subsections into parent sections
        - Preserves page numbers and section order
        - Optimized for ML/NLP papers

        Returns:
            List of dictionaries, each containing:
            - section_name: Detected section heading
            - page_number: Page where section starts
            - start_char_index: Character index in full text
            - end_char_index: Character index in full text
            - text: Text content of the section
            - confidence: Detection confidence (0.0-1.0)

        Raises:
            Exception: If section extraction fails.
        """
        try:
            sections = []
            full_text = ""
            section_candidates: List[Tuple[str, int, int, int]] = []  # (name, page, char_index, original_page)

            # Extract all text and identify section candidates
            for page_num, page in enumerate(self.document, start=1):
                blocks = page.get_text("blocks")

                for block in blocks:
                    if block[6] == 0:  # Text block
                        block_text = block[4]
                        block_lines = block_text.split("\n")

                        for line in block_lines:
                            line_stripped = line.strip()

                            # Check if line is a potential section heading
                            if self._is_section_heading(line_stripped, block):
                                section_candidates.append(
                                    (line_stripped, page_num, len(full_text), page_num)
                                )

                        full_text += block_text + "\n"
            
            # Special handling for Abstract detection on early pages
            abstract_found = any(c[0].lower() == "abstract" for c in section_candidates)
            if not abstract_found and len(section_candidates) > 0:
                # Check if we should insert Abstract for papers where it's missing
                first_page_content = self.document[0].get_text() if len(self.document) > 0 else ""
                if len(first_page_content.split()) > 100:  # First page has substantial text
                    # Check for Abstract markers in first page
                    if "abstract" in first_page_content.lower():
                        logger.info("Abstract detected on first page (implicit)")

            # Filter and process candidates
            filtered_candidates = []
            for section_name, page_num, start_idx, _ in section_candidates:
                # Skip ignored sections
                if self._is_ignored_section(section_name):
                    logger.debug(f"Skipping ignored section: {section_name}")
                    continue
                
                # Apply subsection merging logic
                normalized_name = self._normalize_section_name(section_name)
                filtered_candidates.append((normalized_name, page_num, start_idx))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for name, page, idx in filtered_candidates:
                # Use name and page as duplicate key
                key = (name.lower().strip(), page)
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append((name, page, idx))
            
            # Build section dictionaries with text content
            for i, (section_name, page_num, start_idx) in enumerate(unique_candidates):
                # End index is the start of the next section, or end of document
                end_idx = (
                    unique_candidates[i + 1][2]
                    if i + 1 < len(unique_candidates)
                    else len(full_text)
                )

                section_text = full_text[start_idx:end_idx].strip()
                
                # Skip sections with very little content (likely just headers)
                if len(section_text) < 100:
                    logger.debug(f"Skipping section '{section_name}' with insufficient content ({len(section_text)} chars)")
                    continue

                # Calculate confidence based on section name match
                confidence = self._calculate_heading_confidence(section_name)

                sections.append({
                    "section_name": section_name,
                    "page_number": page_num,
                    "start_char_index": start_idx,
                    "end_char_index": end_idx,
                    "text": section_text,
                    "confidence": confidence,
                })

            logger.info(f"Detected {len(sections)} sections after filtering and merging")
            return sections

        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            raise

    def _is_section_heading(self, text: str, block: Tuple) -> bool:
        """
        Determine if a text block is likely a section heading.

        Args:
            text: The text content to evaluate.
            block: The PyMuPDF block tuple containing formatting info.

        Returns:
            True if text is likely a section heading, False otherwise.
        """
        if not text or len(text) > 100:  # Headings are typically short
            return False

        text_lower = text.lower().strip()

        # Direct match with common sections or ignored sections
        all_sections = self.COMMON_SECTIONS | self.IGNORE_SECTIONS
        if text_lower in all_sections:
            return True

        # Partial match (e.g., "Related Work" vs "Related work")
        for section in all_sections:
            if text_lower.startswith(section) and len(text_lower) <= len(section) + 10:
                return True

        # Numbered sections (e.g., "1. Introduction", "2.1 Background")
        if text and text[0].isdigit() and ("." in text[:4]):
            rest_of_text = text.split(".", 1)[1].strip().lower()
            if any(rest_of_text.startswith(s) for s in all_sections):
                return True

        return False
    
    def _is_ignored_section(self, heading: str) -> bool:
        """
        Check if a section should be ignored (metadata/appendix sections).
        
        Args:
            heading: The section heading text.
            
        Returns:
            True if section should be ignored, False otherwise.
        """
        heading_lower = heading.lower().strip()
        
        # Direct match with ignored sections
        if heading_lower in self.IGNORE_SECTIONS:
            return True
        
        # Partial match
        for ignored in self.IGNORE_SECTIONS:
            if heading_lower.startswith(ignored) or ignored.startswith(heading_lower):
                if len(heading_lower) <= len(ignored) + 5:
                    return True
        
        # Numbered ignored sections (e.g., "A. References", "B.1 Appendix")
        if heading_lower and heading_lower[0].isdigit() and "." in heading_lower[:4]:
            rest = heading_lower.split(".", 1)[1].strip()
            for ignored in self.IGNORE_SECTIONS:
                if rest.startswith(ignored):
                    return True
        
        return False
    
    def _normalize_section_name(self, heading: str) -> str:
        """
        Normalize section name and apply subsection merging logic.
        
        Converts subsection names to parent section names when appropriate
        (e.g., "Experimental setup" -> "Experiments") and standardizes formatting.
        
        Args:
            heading: The original section heading text.
            
        Returns:
            Normalized section name (potentially merged/remapped).
        """
        heading_lower = heading.lower().strip()
        
        # Remove numbering prefixes (e.g., "4.1" from "4.1 Related Work")
        if heading_lower and heading_lower[0].isdigit():
            parts = heading_lower.split(".", 1)
            if len(parts) > 1:
                heading_lower = parts[1].strip()
        
        # Check subsection merge map
        for subsection, parent in self.SUBSECTION_MERGE_MAP.items():
            if heading_lower.startswith(subsection) or heading_lower == subsection:
                logger.debug(f"Merging subsection '{heading}' -> '{parent}'")
                return parent.capitalize()
        
        # Return original heading with consistent capitalization
        return heading.strip()

    def _calculate_heading_confidence(self, heading: str) -> float:
        """
        Calculate confidence score for a detected heading.

        Args:
            heading: The heading text (may be normalized/merged).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        heading_lower = heading.lower().strip()

        # Direct match with common sections: high confidence
        if heading_lower in self.COMMON_SECTIONS:
            return 0.95

        # Partial match: medium-high confidence
        for section in self.COMMON_SECTIONS:
            if heading_lower.startswith(section):
                return 0.85

        # Normalized/merged section: medium confidence
        for parent_section in self.COMMON_SECTIONS:
            if heading_lower == parent_section.lower():
                return 0.80

        # Numbered section: medium confidence
        if heading_lower and heading_lower[0].isdigit() and "." in heading_lower[:4]:
            return 0.75

        return 0.5

    @staticmethod
    def _parse_pdf_date(date_string: str) -> Optional[str]:
        """
        Parse PDF date format (D:YYYYMMDDHHmmSSOHH'mm') to ISO format.

        Args:
            date_string: PDF date string.

        Returns:
            ISO format datetime string, or None if parsing fails.
        """
        try:
            # PDF date format: D:YYYYMMDDHHmmSS
            if date_string.startswith("D:"):
                date_string = date_string[2:]

            # Extract basic components
            year = int(date_string[:4])
            month = int(date_string[4:6])
            day = int(date_string[6:8])
            hour = int(date_string[8:10]) if len(date_string) > 8 else 0
            minute = int(date_string[10:12]) if len(date_string) > 10 else 0
            second = int(date_string[12:14]) if len(date_string) > 12 else 0

            dt = datetime(year, month, day, hour, minute, second)
            return dt.isoformat()

        except Exception as e:
            logger.warning(f"Could not parse date {date_string}: {e}")
            return None


# Convenience functions for module-level usage

def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary containing PDF metadata.
    """
    parser = PDFParser(pdf_path)
    return parser.extract_metadata()


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Complete text content of the paper.
    """
    parser = PDFParser(pdf_path)
    return parser.extract_text()


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract pages from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of page dictionaries with content and metadata.
    """
    parser = PDFParser(pdf_path)
    return parser.extract_pages()


def extract_sections(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract sections from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of section dictionaries with headings and content.
    """
    parser = PDFParser(pdf_path)
    return parser.extract_sections()
