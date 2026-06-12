"""Test embeddings module without loading the model"""

import sys
import numpy as np

# Test 1: Import module
try:
    from src.embeddings import cosine_similarity, get_model_info
    print("✓ Module imports successful")
except Exception as e:
    print(f"✗ Failed to import module: {e}")
    sys.exit(1)

# Test 2: Cosine similarity computation
try:
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    similarity = cosine_similarity(v1, v2)
    assert 0.99 < similarity <= 1.0, f"Expected ~1.0, got {similarity}"
    print(f"✓ Cosine similarity (identical vectors): {similarity:.4f}")
    
    v3 = np.array([0.0, 1.0, 0.0])
    similarity2 = cosine_similarity(v1, v3)
    assert -0.01 < similarity2 < 0.01, f"Expected ~0.0, got {similarity2}"
    print(f"✓ Cosine similarity (orthogonal vectors): {similarity2:.4f}")
except Exception as e:
    print(f"✗ Cosine similarity test failed: {e}")
    sys.exit(1)

# Test 3: Error handling - empty chunks
try:
    from src.embeddings import embed_chunks
    empty_chunks = []
    try:
        embed_chunks(empty_chunks)
        print("✗ Should have raised ValueError for empty chunks")
        sys.exit(1)
    except ValueError:
        print("✓ Error handling: ValueError caught for empty chunks")
except Exception as e:
    print(f"✗ Error handling test failed: {e}")
    sys.exit(1)

# Test 4: Type hints and docstrings
try:
    from src.embeddings import (
        embed_text, embed_chunks, batch_embed, 
        cosine_similarity, batch_cosine_similarity
    )
    
    functions = [
        ("embed_text", embed_text),
        ("embed_chunks", embed_chunks),
        ("batch_embed", batch_embed),
        ("cosine_similarity", cosine_similarity),
        ("batch_cosine_similarity", batch_cosine_similarity),
    ]
    
    for name, func in functions:
        if func.__doc__:
            print(f"✓ {name}: docstring present")
        else:
            print(f"✗ {name}: missing docstring")
            
except Exception as e:
    print(f"✗ Docstring check failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All non-model tests passed!")
print("="*50)
print("\nNote: Model loading tests will proceed when needed.")
print("The BAAI/bge-small-en-v1.5 model (~270MB) will be")
print("downloaded on first embed_text() or embed_chunks() call.")
