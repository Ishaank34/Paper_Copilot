"""
Test script for parser.py

Loads all PDFs from the papers/ folder and tests extraction functions.
Prints structured output and warnings for any issues.
"""

import logging
from pathlib import Path
from typing import List

from src.parser import extract_metadata, extract_text, extract_pages, extract_sections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def test_pdf(pdf_path: Path) -> None:
    """
    Test all extraction functions on a single PDF.

    Args:
        pdf_path: Path to the PDF file.
    """
    print("\n" + "=" * 80)
    print(f"Testing: {pdf_path.name}")
    print("=" * 80)

    try:
        # Extract metadata
        metadata = extract_metadata(str(pdf_path))

        # Extract text
        full_text = extract_text(str(pdf_path))

        # Extract pages
        pages = extract_pages(str(pdf_path))

        # Extract sections
        sections = extract_sections(str(pdf_path))

        # Print basic information
        print(f"\nPaper Name: {metadata['file_name']}")
        print(f"Title: {metadata['title']}")
        print(f"Authors: {metadata['author']}")
        print(f"Page Count: {metadata['num_pages']}")

        # Warnings
        warnings = []

        if metadata['title'] == "Unknown" or not metadata['title']:
            warnings.append("⚠ WARNING: Title is missing")

        if len(sections) < 3:
            warnings.append(f"⚠ WARNING: Fewer than 3 sections detected ({len(sections)} found)")

        if not full_text or len(full_text.strip()) == 0:
            warnings.append("⚠ WARNING: Extraction returned empty text")

        if warnings:
            print("\n" + "-" * 80)
            for warning in warnings:
                print(warning)

        # Print detected sections
        print("\n" + "-" * 80)
        print(f"Detected Sections ({len(sections)} total):\n")

        if sections:
            for i, section in enumerate(sections, 1):
                section_name = section["section_name"]
                start_idx = section["start_char_index"]
                end_idx = section["end_char_index"]
                text = section["text"]
                confidence = section["confidence"]

                char_length = end_idx - start_idx
                preview = text[:200].replace("\n", " ")

                print(f"{i}. {section_name}")
                print(f"   Page: {section['page_number']}")
                print(f"   Character Length: {char_length}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Preview: {preview}...")
                print()
        else:
            print("No sections detected.")

        print("-" * 80)
        print(f"Total extracted text length: {len(full_text)} characters")
        print(f"Total pages: {len(pages)}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error testing {pdf_path.name}: {e}", exc_info=True)


def find_pdf_files(papers_folder: Path) -> List[Path]:
    """
    Find all PDF files in the papers folder.

    Args:
        papers_folder: Path to the papers directory.

    Returns:
        List of PDF file paths.
    """
    if not papers_folder.exists():
        logger.warning(f"Papers folder does not exist: {papers_folder}")
        return []

    pdfs = list(papers_folder.glob("*.pdf"))

    if not pdfs:
        logger.warning(f"No PDF files found in {papers_folder}")
        return []

    return sorted(pdfs)


def main():
    """Main test runner."""
    print("\n" + "=" * 80)
    print("PaperCopilot Parser Test Suite")
    print("=" * 80)

    papers_folder = Path("papers")
    pdf_files = find_pdf_files(papers_folder)

    if not pdf_files:
        print(f"\nNo PDF files found in '{papers_folder}' folder.")
        print("Please add PDF files to the papers/ directory and run this test again.")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s) to test:\n")
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_path.name}")

    for pdf_path in pdf_files:
        test_pdf(pdf_path)

    print("\n" + "=" * 80)
    print("Test suite completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
