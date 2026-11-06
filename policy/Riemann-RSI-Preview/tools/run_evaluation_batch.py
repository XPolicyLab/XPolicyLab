"""Dispatch evaluation batches through the unmodified official SDK."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.evaluation_batch import main

if __name__ == '__main__':
    main()
