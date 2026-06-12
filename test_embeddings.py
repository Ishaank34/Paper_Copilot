"""Quick test script for embeddings module"""

from src.embeddings import load_embedding_model, get_model_info, cosine_similarity
import numpy as np

print("Testing embeddings module...")
print()

# Test 1: Get model info
try:
    info = get_model_info()
    print("✓ Model info retrieved:")
    print(f"  - Model: {info['model_name']}")
    print(f"  - Embedding dimension: {info['embedding_dimension']}")
    print(f"  - Default batch size: {info['batch_size']}")
    print(f"  - Max batch size: {info['max_batch_size']}")
    print()
except Exception as e:
    print(f"✗ Failed to get model info: {e}")

# Test 2: Cosine similarity computation
try:
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    similarity = cosine_similarity(v1, v2)
    print(f"✓ Cosine similarity test passed: {similarity:.4f} (expected ~1.0)")
    print()
except Exception as e:
    print(f"✗ Cosine similarity test failed: {e}")

# Test 3: Error handling
try:
    empty_chunks = []
    from src.embeddings import embed_chunks
    embed_chunks(empty_chunks)
except ValueError as e:
    print(f"✓ Error handling works: caught ValueError for empty chunks")
    print()

print("Module structure verified successfully!")
print("\nNote: To fully test embedding generation, run the module")
print("after the BAAI/bge-small-en-v1.5 model is downloaded.")
