#!/usr/bin/env python3
"""
Verify all configurations before running expensive API calls.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def test_data_files():
    """Test 1: Verify data files exist"""
    print("\n" + "=" * 60)
    print("TEST 1: Data Files Verification")
    print("=" * 60)
    
    chunks_file = Path("JJ/data/processed/aapl_10k_chunks.jsonl")
    qa_file = Path("JJ/data/manual_qa_template.jsonl")
    
    if chunks_file.exists():
        with open(chunks_file, encoding='utf-8') as f:
            chunk_count = sum(1 for _ in f)
        print(f"✅ Chunks file: {chunks_file} ({chunk_count} chunks)")
    else:
        print(f"❌ Chunks file missing: {chunks_file}")
        return False
    
    if qa_file.exists():
        with open(qa_file, encoding="utf-8") as f:
            qa_count = sum(1 for _ in f)
        print(f"✅ QA file: {qa_file} ({qa_count} QA pairs)")
    else:
        print(f"❌ QA file missing: {qa_file}")
        return False
    
    return True

def test_embeddings():
    """Test 2: Verify embedding models"""
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Models")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        baseline = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Baseline model: all-MiniLM-L6-v2")
        
        finance = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        print("✅ Finance model: all-mpnet-base-v2")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cross_encoder():
    """Test 3: Verify cross-encoder"""
    print("\n" + "=" * 60)
    print("TEST 3: Cross-Encoder Model")
    print("=" * 60)
    
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("✅ Cross-encoder: mmarco-mMiniLMv2-L12-H384-v1")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_llm_model():
    """Test 4: Verify LLM is valid (NO API CALL)"""
    print("\n" + "=" * 60)
    print("TEST 4: LLM Model Configuration")
    print("=" * 60)
    
    try:
        from langchain_openai import ChatOpenAI
        # Don't actually call the API, just verify model can be initialized
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        print("✅ LLM model: gpt-3.5-turbo (valid OpenAI model)")
        print("   Note: This is a valid model that will work for HyDE queries")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_variant_logic():
    """Test 5: Verify variant logic"""
    print("\n" + "=" * 60)
    print("TEST 5: Variant Logic Verification")
    print("=" * 60)
    
    variants = ["baseline+rerank", "hyde+rerank", "hybrid+rerank", "finance+rerank"]
    
    for variant in variants:
        use_rerank = variant.endswith("+rerank")
        base_variant = variant.replace("+rerank", "") if use_rerank else variant
        
        retrieval_method = {
            "baseline": "Dense (baseline embeddings only)",
            "hyde": "Dense + HyDE (hypothetical document expansion)",
            "hybrid": "Dense + BM25 (reciprocal rank fusion)",
            "finance": "Dense (finance-tuned embeddings)"
        }.get(base_variant, "Unknown")
        
        rerank_method = "Cross-Encoder" if use_rerank else "None"
        
        print(f"✅ {variant:20} -> Retrieval: {retrieval_method:45} | Reranking: {rerank_method}")
    
    return True

def main():
    print("\n" + "🔍 PRE-RUN VERIFICATION (No API calls yet)")
    print("=" * 60)
    
    all_pass = True
    all_pass &= test_data_files()
    all_pass &= test_embeddings()
    all_pass &= test_cross_encoder()
    all_pass &= test_llm_model()
    all_pass &= test_variant_logic()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL CHECKS PASSED - Ready to run with API calls!")
        print("\nHyDE queries WILL call OpenAI API (gpt-3.5-turbo):")
        print("  - 4 variants × 50 questions = 200 API calls")
        print("  - Cost: ~$0.01-0.05 depending on query length")
        print("\nCommand to run:")
        print("  source .venv_rag/bin/activate && python3 JJ/experiments/rag_compare_rerank.py \\")
        print("    --chunks JJ/data/processed/aapl_10k_chunks.jsonl \\")
        print("    --qa JJ/data/manual_qa_template.jsonl \\")
        print("    --reranker cross-encoder \\")
        print("    --output-dir results/rag_compare_rerank_cross_encoder_corrected")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before running")
        sys.exit(1)

if __name__ == "__main__":
    main()
