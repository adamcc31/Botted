"""
Enables `python -m model_training` as a shortcut for the training pipeline.

Usage:
    python -m model_training --data data/signals.csv --output models/v1/
    python -m model_training --data data/signals.db --format sqlite --log-level DEBUG
"""

from .pipeline import main

if __name__ == "__main__":
    main()
