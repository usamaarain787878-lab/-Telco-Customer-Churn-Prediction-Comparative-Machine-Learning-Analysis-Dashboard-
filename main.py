import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Aapka purana import ab iske niche aayega
from my_preprocessing_framework import preprocess_data

# File Name: main.py
# --- Sabse upar framework se function import kiya ---
from my_preprocessing_framework import preprocess_data

# Fake data pipeline chalane ke liye
raw_data = "  HELLO WORLD!  "

print("Pipeline execution shuru ho rahi hai...")

# Ab function call karenge to error nahi aayega
final_result = preprocess_data(raw_data)

print("Pipeline ka result:", final_result)