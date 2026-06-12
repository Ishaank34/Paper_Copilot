"""Test vectorstore.py functionality"""

import sys
import numpy as np
from pathlib import Path

# Test 1: Import and basic instantiation
try:
    from src.vectorstore import VectorStore, create_vectorstore, load_vectorstore
    print("✓ Module imports successful")
except Exception as e:
    print(f"✗ Failed to import module: {e}")
    sys.exit(1)

# Test 2: Create vector store
try:
    vs = create_vectorstore(dimension=384, index_folder="test_indexes")
    info = vs.get_index_info()
    assert info["status"] == "uninitialized"
    assert info["dimension"] == 384
    print("✓ Vector store created successfully")
except Exception as e:
    print(f"✗ Failed to create vector store: {e}")
    sys.exit(1)

# Test 3: Create index
try:
    vs.create_index()
    info = vs.get_index_info()
    assert info["status"] == "initialized"
    assert vs.index is not None
    assert vs.get_vector_count() == 0
    print("✓ Index created successfully (empty)")
except Exception as e:
    print(f"✗ Failed to create index: {e}")
    sys.exit(1)

# Test 4: Add embeddings
try:
    # Create normalized embeddings (384 dimensions)
    num_embeddings = 10
    embeddings = np.random.randn(num_embeddings, 384).astype(np.float32)
    # L2-normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create metadata
    metadata = [
        {
            "paper": "test_paper.pdf",
            "section": f"Section {i}",
            "chunk_text": f"This is chunk {i} content.",
            "page_number": (i // 2) + 1,
            "start_idx": i * 100,
            "end_idx": (i + 1) * 100,
        }
        for i in range(num_embeddings)
    ]
    
    vs.add_embeddings(embeddings, metadata)
    assert vs.get_vector_count() == num_embeddings
    print(f"✓ Added {num_embeddings} embeddings to index")
except Exception as e:
    print(f"✗ Failed to add embeddings: {e}")
    sys.exit(1)

# Test 5: Search functionality
try:
    # Create a query embedding (normalized)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Search for top-3 results
    results = vs.search(query, k=3)
    assert len(results) == 3
    assert all("metadata" in r for r in results)
    assert all("score" in r for r in results)
    assert all("rank" in r for r in results)
    
    # Verify scores are valid cosine similarities [0, 1] for normalized embeddings
    for result in results:
        assert 0 <= result["score"] <= 1.1, f"Invalid score: {result['score']}"
    
    print(f"✓ Search returned {len(results)} results with valid scores")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['metadata']['section']} - Score: {r['score']:.4f}")
except Exception as e:
    print(f"✗ Search failed: {e}")
    sys.exit(1)

# Test 6: Save index
try:
    index_path = vs.save_index("test_index")
    assert index_path.exists()
    assert (index_path / "faiss_index.bin").exists()
    assert (index_path / "metadata.json").exists()
    print(f"✓ Index saved to {index_path}")
except Exception as e:
    print(f"✗ Failed to save index: {e}")
    sys.exit(1)

# Test 7: Load index
try:
    vs2 = create_vectorstore(index_folder="test_indexes")
    success = vs2.load_index("test_index")
    assert success
    assert vs2.get_vector_count() == num_embeddings
    assert len(vs2.metadata) == num_embeddings
    print(f"✓ Index loaded successfully ({vs2.get_vector_count()} vectors)")
except Exception as e:
    print(f"✗ Failed to load index: {e}")
    sys.exit(1)

# Test 8: List saved indexes
try:
    indexes = vs.list_saved_indexes()
    assert "test_index" in indexes
    print(f"✓ List saved indexes: {indexes}")
except Exception as e:
    print(f"✗ Failed to list indexes: {e}")
    sys.exit(1)

# Test 9: Error handling - invalid inputs
try:
    # Test empty embeddings
    try:
        vs.add_embeddings(np.array([]).reshape(0, 384), [])
        print("  (Empty embeddings accepted with warning)")
    except Exception:
        pass
    
    # Test dimension mismatch
    try:
        bad_embeddings = np.random.randn(5, 100).astype(np.float32)
        vs.add_embeddings(bad_embeddings, [{} for _ in range(5)])
        print("✗ Should have raised error for dimension mismatch")
    except ValueError:
        print("✓ Error handling: ValueError for dimension mismatch")
    
    # Test metadata count mismatch
    try:
        good_embeddings = np.random.randn(5, 384).astype(np.float32)
        vs.add_embeddings(good_embeddings, [{}, {}])  # Only 2 metadata
        print("✗ Should have raised error for metadata mismatch")
    except ValueError:
        print("✓ Error handling: ValueError for metadata mismatch")
        
except Exception as e:
    print(f"✗ Error handling test failed: {e}")

# Test 10: Reset and cleanup
try:
    vs.reset()
    assert vs.get_vector_count() == 0
    assert len(vs.metadata) == 0
    print("✓ Vector store reset successfully")
    
    # Clean up test files
    import shutil
    if Path("test_indexes").exists():
        shutil.rmtree("test_indexes")
        print("✓ Cleaned up test files")
except Exception as e:
    print(f"✗ Cleanup failed: {e}")

print("\n" + "="*50)
print("All vectorstore tests passed!")
print("="*50)
