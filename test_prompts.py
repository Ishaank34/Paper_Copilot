"""Test prompts.py module"""

import sys
from src.prompts import (
    get_student_summary_prompt,
    get_research_mentor_prompt,
    get_reviewer_prompt,
    get_contributions_prompt,
    get_limitations_prompt,
    get_future_work_prompt,
    get_dataset_metrics_prompt,
    get_equation_explanation_prompt,
    get_prerequisites_prompt,
    get_roadmap_prompt,
    get_methodology_prompt,
    get_qa_prompt,
    get_deep_qa_prompt,
    get_compare_papers_prompt,
    get_all_prompts,
    list_available_prompts,
    get_prompt_by_name,
)

print("Testing prompts.py module...\n")

# Test 1: Import all functions
try:
    print("✓ All prompt functions imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check that all prompts return strings
test_functions = [
    ("student_summary", get_student_summary_prompt),
    ("research_mentor", get_research_mentor_prompt),
    ("reviewer", get_reviewer_prompt),
    ("contributions", get_contributions_prompt),
    ("limitations", get_limitations_prompt),
    ("future_work", get_future_work_prompt),
    ("dataset_metrics", get_dataset_metrics_prompt),
    ("equation_explanation", get_equation_explanation_prompt),
    ("prerequisites", get_prerequisites_prompt),
    ("roadmap", get_roadmap_prompt),
    ("methodology", get_methodology_prompt),
]

for name, func in test_functions:
    try:
        prompt = func()
        assert isinstance(prompt, str), f"Expected string, got {type(prompt)}"
        assert len(prompt) > 100, f"Prompt too short: {len(prompt)} chars"
        print(f"✓ {name}: {len(prompt)} characters")
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        sys.exit(1)

# Test 3: QA prompts with questions
try:
    qa_prompt = get_qa_prompt("What problem does this paper solve?")
    assert isinstance(qa_prompt, str)
    assert "What problem does this paper solve?" in qa_prompt
    print(f"✓ get_qa_prompt: {len(qa_prompt)} characters")
except Exception as e:
    print(f"✗ get_qa_prompt failed: {e}")
    sys.exit(1)

try:
    deep_qa_prompt = get_deep_qa_prompt("Why is this approach novel?")
    assert isinstance(deep_qa_prompt, str)
    assert "Why is this approach novel?" in deep_qa_prompt
    print(f"✓ get_deep_qa_prompt: {len(deep_qa_prompt)} characters")
except Exception as e:
    print(f"✗ get_deep_qa_prompt failed: {e}")
    sys.exit(1)

# Test 4: Compare papers prompt
try:
    compare_prompt = get_compare_papers_prompt()
    assert isinstance(compare_prompt, str)
    assert len(compare_prompt) > 100
    print(f"✓ get_compare_papers_prompt: {len(compare_prompt)} characters")
except Exception as e:
    print(f"✗ get_compare_papers_prompt failed: {e}")
    sys.exit(1)

# Test 5: Utility functions
try:
    all_prompts = get_all_prompts()
    assert isinstance(all_prompts, dict)
    assert len(all_prompts) >= 11
    print(f"✓ get_all_prompts: {len(all_prompts)} prompts available")
except Exception as e:
    print(f"✗ get_all_prompts failed: {e}")
    sys.exit(1)

try:
    available = list_available_prompts()
    assert isinstance(available, list)
    assert "student_summary" in available
    assert "research_mentor" in available
    assert len(available) >= 11
    print(f"✓ list_available_prompts: {len(available)} prompts")
except Exception as e:
    print(f"✗ list_available_prompts failed: {e}")
    sys.exit(1)

try:
    prompt = get_prompt_by_name("student_summary")
    assert isinstance(prompt, str)
    assert len(prompt) > 100
    print(f"✓ get_prompt_by_name('student_summary'): success")
    
    none_result = get_prompt_by_name("nonexistent")
    assert none_result is None
    print(f"✓ get_prompt_by_name('nonexistent'): correctly returns None")
except Exception as e:
    print(f"✗ get_prompt_by_name failed: {e}")
    sys.exit(1)

# Test 6: Check prompt quality
try:
    # Check that prompts include key instructions
    student = get_student_summary_prompt()
    assert "IMPORTANT" in student
    assert "simple" in student.lower() or "intuition" in student.lower()
    
    mentor = get_research_mentor_prompt()
    assert "IMPORTANT" in mentor
    assert "assumptions" in mentor.lower()
    
    reviewer = get_reviewer_prompt()
    assert "novelty" in reviewer.lower()
    assert "experiments" in reviewer.lower()
    
    print("✓ Prompts include appropriate instructions and keywords")
except Exception as e:
    print(f"✗ Prompt quality check failed: {e}")
    sys.exit(1)

# Test 7: Verify prompts are distinct
try:
    prompts_text = [func() for _, func in test_functions]
    lengths = [len(p) for p in prompts_text]
    
    # Check that prompts have varied lengths (not duplicated)
    unique_lengths = len(set(lengths))
    assert unique_lengths > len(test_functions) * 0.7
    print(f"✓ Prompts are distinct (lengths vary appropriately)")
except Exception as e:
    print(f"✗ Distinctness check failed: {e}")
    sys.exit(1)

# Test 8: Verify docstrings are present
try:
    functions_with_docs = [
        get_student_summary_prompt,
        get_research_mentor_prompt,
        get_reviewer_prompt,
        get_contributions_prompt,
        get_limitations_prompt,
        get_future_work_prompt,
        get_dataset_metrics_prompt,
        get_equation_explanation_prompt,
        get_prerequisites_prompt,
        get_roadmap_prompt,
        get_methodology_prompt,
        get_qa_prompt,
        get_deep_qa_prompt,
        get_compare_papers_prompt,
    ]
    
    for func in functions_with_docs:
        assert func.__doc__ is not None and len(func.__doc__) > 50
    
    print(f"✓ All {len(functions_with_docs)} functions have comprehensive docstrings")
except Exception as e:
    print(f"✗ Docstring check failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)

print("\n📚 Available prompts:")
for i, name in enumerate(list_available_prompts(), 1):
    print(f"  {i:2d}. {name}")

print("\n✨ Plus: get_qa_prompt(question) and get_deep_qa_prompt(question)")
print("✨ Plus: get_compare_papers_prompt()")
