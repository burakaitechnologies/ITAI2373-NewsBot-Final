"""
Quick test script to verify all dependencies are installed correctly
"""
print("Testing setup...\n")

try:
    import pandas as pd
    print("[OK] pandas imported")
except ImportError as e:
    print(f"[ERROR] pandas failed: {e}")

try:
    import numpy as np
    print("[OK] numpy imported")
except ImportError as e:
    print(f"[ERROR] numpy failed: {e}")

try:
    import sklearn
    print("[OK] scikit-learn imported")
except ImportError as e:
    print(f"[ERROR] scikit-learn failed: {e}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("[OK] spaCy imported and model loaded")
except Exception as e:
    print(f"[ERROR] spaCy failed: {e}")

try:
    import nltk
    print("[OK] NLTK imported")
except ImportError as e:
    print(f"[ERROR] NLTK failed: {e}")

try:
    import matplotlib
    print("[OK] matplotlib imported")
except ImportError as e:
    print(f"[ERROR] matplotlib failed: {e}")

try:
    from textblob import TextBlob
    print("[OK] TextBlob imported")
except ImportError as e:
    print(f"[ERROR] TextBlob failed: {e}")

print("\n[OK] Setup test complete!")
