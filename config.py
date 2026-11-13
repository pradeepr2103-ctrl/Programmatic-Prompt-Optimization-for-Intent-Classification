"""
Global Configuration and Hyperparameters.
"""

import os

# Dataset Settings
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "banking77_subsets", "banking77_10intents.json")
INTENTS_METADATA_PATH = os.path.join(os.path.dirname(__file__), "data", "banking77_subsets", "intents_metadata.json")

# Evaluation Settings
RANDOM_SEED = 42
DEFAULT_TEMPERATURE = 0.0

# Optimization Hyperparameters
DSPY_BOOTSTRAP_MAX_DEMOS = 5
DSPY_COPRO_CANDIDATES = 4

TEXTGRAD_EPOCHS = 2
TEXTGRAD_BATCH_SIZE = 8
TEXTGRAD_MAX_GRADIENTS_PER_STEP = 3

ADALFLOW_DEMOS = 5
ADALFLOW_DIAGNOSIS_SAMPLES = 25

# Output Directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
