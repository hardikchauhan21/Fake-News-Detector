"""
Test script to verify preprocessing functionality
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import preprocess_pipeline, clean_text, remove_stopwords, stem_text

def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("Testing Fake News Detector preprocessing...")
    print("=" * 50)

    # Test text
    test_text = "Breaking: Scientists DISCOVER miracle CURE!!! 100% guaranteed results!!"
    print(f"Original text: {test_text}")

    # Test individual functions
    cleaned = clean_text(test_text)
    print(f"\nAfter cleaning: {cleaned}")

    no_stopwords = remove_stopwords(cleaned)
    print(f"After stopword removal: {no_stopwords}")

    stemmed = stem_text(no_stopwords)
    print(f"After stemming: {stemmed}")

    # Test full pipeline
    pipeline_result = preprocess_pipeline(test_text)
    print(f"\nFull pipeline result: {pipeline_result}")

    # Additional test cases
    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "BREAKING: GOVERNMENT ANNOUNCES NEW POLICY TODAY!!",
        "According to sources, the event will take place tomorrow.",
        "SHOCKING: CELEBRITY SEEN DOING SOMETHING!!!"
    ]

    print("\n" + "=" * 50)
    print("Additional test cases:")
    for i, text in enumerate(test_cases, 1):
        result = preprocess_pipeline(text)
        print(f"{i}. Input:  {text}")
        print(f"   Output: {result}")
        print()

    print("✅ Preprocessing tests completed successfully!")

if __name__ == "__main__":
    test_preprocessing()