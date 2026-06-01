from pathlib import Path
import os

def get_base_dir() -> Path:
    # Return the project root directory (parent of the recursive_lm package).
    return Path(__file__).resolve().parent.parent