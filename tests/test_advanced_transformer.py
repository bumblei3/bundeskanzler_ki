#!/usr/bin/env python3
"""
Test script for the AdvancedTransformerModel
Tests basic functionality of GPT-2, BERT, and RoBERTa models
"""

import sys
import os
sys.path.append('/home/tobber/bkki_venv')

from advanced_transformer_model import AdvancedTransformerModel
import torch

def test_gpt2_model():
    """Test GPT-2 model loading and text generation"""
    print("Testing GPT-2 model...")

    try:
        model = AdvancedTransformerModel(model_type='gpt2')
        print("✓ GPT-2 model loaded successfully")

        # Test text generation
        prompt = "Der Bundeskanzler sagte:"
        response = model.generate_response(prompt, max_length=50)
        print(f"✓ GPT-2 response: {response}")

        # Test embedding extraction
        embedding = model.get_embedding(prompt)
        print(f"✓ GPT-2 embedding shape: {embedding.shape}")

        return True
    except Exception as e:
        print(f"✗ GPT-2 test failed: {e}")
        return False

def test_bert_model():
    """Test BERT model loading and functionality"""
    print("\nTesting BERT model...")

    try:
        model = AdvancedTransformerModel(model_type='bert')
        print("✓ BERT model loaded successfully")

        # Test embedding extraction
        text = "Die KI-Entwicklung ist wichtig für Deutschland."
        embedding = model.get_embedding(text)
        print(f"✓ BERT embedding shape: {embedding.shape}")

        return True
    except Exception as e:
        print(f"✗ BERT test failed: {e}")
        return False

def test_roberta_model():
    """Test RoBERTa model loading and functionality"""
    print("\nTesting RoBERTa model...")

    try:
        model = AdvancedTransformerModel(model_type='roberta')
        print("✓ RoBERTa model loaded successfully")

        # Test embedding extraction
        text = "Künstliche Intelligenz verändert unsere Welt."
        embedding = model.get_embedding(text)
        print(f"✓ RoBERTa embedding shape: {embedding.shape}")

        return True
    except Exception as e:
        print(f"✗ RoBERTa test failed: {e}")
        return False

def test_hybrid_model():
    """Test hybrid model creation"""
    print("\nTesting hybrid model creation...")

    try:
        model = AdvancedTransformerModel(model_type='hybrid')
        print("✓ Hybrid model created successfully")

        # Test with sample text
        text = "Bundeskanzler Scholz besucht Berlin."
        embedding = model.get_embedding(text)
        print(f"✓ Hybrid embedding shape: {embedding.shape}")

        return True
    except Exception as e:
        print(f"✗ Hybrid model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Advanced Transformer Model Tests")
    print("=" * 50)

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    tests = [
        test_gpt2_model,
        test_bert_model,
        test_roberta_model,
        test_hybrid_model
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Advanced Transformer Model is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())