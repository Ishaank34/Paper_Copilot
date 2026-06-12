"""
Integration Test Suite for PaperAnalyzer

Tests the complete PaperCopilot backend pipeline:
- Paper loading and parsing
- Chunk creation and embedding
- Vector store indexing
- Semantic retrieval
- LLM-powered analysis

Test Paper: CLaRa.pdf (from papers folder)
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from paper_analyzer import PaperAnalyzer
import logging

# Configure logging to reduce noise during tests
logging.basicConfig(level=logging.WARNING)

# Test tracking
tests_passed = 0
tests_failed = 0
test_results = []


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test_header(test_name: str) -> None:
    """Print a formatted test header."""
    print(f"\n[TEST] {test_name}")
    print("-" * 80)


def print_result(test_name: str, success: bool, error: str = None) -> None:
    """Print test result with status."""
    global tests_passed, tests_failed
    
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {test_name}")
    
    if error:
        print(f"  Error: {error}")
    
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    test_results.append((test_name, success))


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max length for display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n  ... (truncated, total length: {len(text)} chars)"


def print_output(output: str, label: str = "Output") -> None:
    """Print analysis output in a readable format."""
    print(f"\n{label}:")
    print("-" * 80)
    print(truncate_text(output))
    print("-" * 80)


# ============================================================================
# Main Test Suite
# ============================================================================

def main():
    """Run all tests."""
    
    print_section("PAPERCOPILOT BACKEND INTEGRATION TESTS")
    
    # Initialize analyzer with Ollama provider
    print("\nInitializing PaperAnalyzer with Ollama provider...")
    analyzer = PaperAnalyzer(provider="ollama", model_name="qwen3:8b")
    print("✓ PaperAnalyzer initialized successfully (Provider: Ollama, Model: qwen3:8b)")
    
    # Test 0: Load Paper
    print_test_header("Test 0: load_paper()")
    paper_path = Path("papers") / "CLaRa.pdf"
    
    if not paper_path.exists():
        print_result(
            "load_paper()",
            False,
            f"Paper not found at {paper_path}. Please ensure CLaRa.pdf exists in the papers folder."
        )
        print("\n" + "=" * 80)
        print("UNABLE TO CONTINUE: Paper file not found")
        print(f"Expected location: {paper_path.absolute()}")
        print("=" * 80)
        return
    
    try:
        analyzer.load_paper(str(paper_path))
        paper_info = analyzer.get_paper_info()
        print_result("load_paper()", True)
        print(f"  Paper: {paper_info['title']}")
        print(f"  Pages: {paper_info['num_pages']}")
        print(f"  Chunks: {paper_info['num_chunks']}")
        print(f"  Indexed vectors: {paper_info['indexed_vectors']}")
    except Exception as e:
        print_result("load_paper()", False, str(e))
        print("\n" + "=" * 80)
        print("UNABLE TO CONTINUE: Failed to load paper")
        print("=" * 80)
        return
    
    # Test 1: generate_student_summary()
    print_test_header("Test 1: generate_student_summary()")
    try:
        result = analyzer.generate_student_summary()
        print_result("generate_student_summary()", True)
        print_output(result, "Student Summary")
    except Exception as e:
        print_result("generate_student_summary()", False, str(e))
    
    # Test 2: generate_research_mentor_analysis()
    print_test_header("Test 2: generate_research_mentor_analysis()")
    try:
        result = analyzer.generate_research_mentor_analysis()
        print_result("generate_research_mentor_analysis()", True)
        print_output(result, "Research Mentor Analysis")
    except Exception as e:
        print_result("generate_research_mentor_analysis()", False, str(e))
    
    # Test 3: extract_contributions()
    print_test_header("Test 3: extract_contributions()")
    try:
        result = analyzer.extract_contributions()
        print_result("extract_contributions()", True)
        print_output(result, "Contributions")
    except Exception as e:
        print_result("extract_contributions()", False, str(e))
    
    # Test 4: extract_limitations()
    print_test_header("Test 4: extract_limitations()")
    try:
        result = analyzer.extract_limitations()
        print_result("extract_limitations()", True)
        print_output(result, "Limitations")
    except Exception as e:
        print_result("extract_limitations()", False, str(e))
    
    # Test 5: extract_future_work()
    print_test_header("Test 5: extract_future_work()")
    try:
        result = analyzer.extract_future_work()
        print_result("extract_future_work()", True)
        print_output(result, "Future Work")
    except Exception as e:
        print_result("extract_future_work()", False, str(e))
    
    # Test 6: extract_datasets_and_metrics()
    print_test_header("Test 6: extract_datasets_and_metrics()")
    try:
        result = analyzer.extract_datasets_and_metrics()
        print_result("extract_datasets_and_metrics()", True)
        print_output(result, "Datasets and Metrics")
    except Exception as e:
        print_result("extract_datasets_and_metrics()", False, str(e))
    
    # Test 7: explain_methodology()
    print_test_header("Test 7: explain_methodology()")
    try:
        result = analyzer.explain_methodology()
        print_result("explain_methodology()", True)
        print_output(result, "Methodology Explanation")
    except Exception as e:
        print_result("explain_methodology()", False, str(e))
    
    # Test 8: generate_reading_roadmap()
    print_test_header("Test 8: generate_reading_roadmap()")
    try:
        result = analyzer.generate_reading_roadmap()
        print_result("generate_reading_roadmap()", True)
        print_output(result, "Reading Roadmap")
    except Exception as e:
        print_result("generate_reading_roadmap()", False, str(e))
    
    # Test 9: answer_question() - Question 1
    print_test_header("Test 9: answer_question() - Question 1")
    question_1 = "Why does CLaRa use latent reasoning?"
    try:
        result = analyzer.answer_question(question_1)
        print_result("answer_question(q1)", True)
        print(f"\nQuestion: {question_1}")
        print_output(result, "Answer")
    except Exception as e:
        print_result("answer_question(q1)", False, str(e))
    
    # Test 10: deep_answer() - Question 2
    print_test_header("Test 10: deep_answer() - Question 2")
    question_2 = "What assumptions does CLaRa make?"
    try:
        result = analyzer.deep_answer(question_2)
        print_result("deep_answer(q2)", True)
        print(f"\nQuestion: {question_2}")
        print_output(result, "Deep Answer")
    except Exception as e:
        print_result("deep_answer(q2)", False, str(e))
    
    # Print final summary
    print_section("TEST SUMMARY")
    print(f"\nTotal Tests Run: {tests_passed + tests_failed}")
    print(f"Tests Passed:   {tests_passed}")
    print(f"Tests Failed:   {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The PaperCopilot backend is working correctly.")
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed.")
        print("See details above for error messages.")
    
    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
