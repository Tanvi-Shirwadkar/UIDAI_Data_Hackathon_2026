# src/config.py
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access variables
DATA_DIR = os.getenv("DATA_PATH", "data/")
OUTPUT_DIR = os.getenv("OUTPUT_PATH", "output/")

# Verify they exist
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found at: {DATA_DIR}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR) # Create if it doesn't exist