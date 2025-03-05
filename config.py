# config.py
import os

class Config:
    ENABLE_LLM_VALIDATION = True  # Master switch
    LLM_VALIDATION_THRESHOLD = 0.7  # Validation threshold

# Global access method
config = Config()

MCTS_CONFIG = {
    'max_depth': 15,
    'max_simulations': 2000,
    'relation_weights': {
        'CAUSES': {'explicit': 1.0, 'implicit': 0.7},
        'TIME_SEQUENCE': {'main': 0.6, 'branch': 0.5},
        'OTHER': 0.4
    }
}
RELATION_PENALTIES = {
    'INVALID_RELATION': 0.7,      # Penalty for incorrect relation type
    'LOW_WEIGHT_CAUSATION': 0.5    # Penalty for unconfigured weight in causal relationship
}
LLM_CONFIG = {
    # It is recommended to load the API Key through environment variables or configuration methods
    'api_key': os.environ.get("OPENAI_API_KEY", "key"),
    'max_retries': 50,
    'timeout': 30.0
}

SAFETY_FILTERS = {
    'violence_keywords': ['murder', 'kill', 'attack', 'wound', 'blood', 'death', 'fight'],
    'suspense_threshold': 0.75  # Safety threshold for suspense intensity
}
RELATION_DEFAULTS = {
    'Hierarchy': 0.7,      # Default weight for hierarchical relationship
    'Causation': 0.8,      # Default weight for causal relationship
    'Time': 0.6            # Default weight for time relationship
}

RELATION_VALIDATION = {
    "expected": {"TIME", "INFERENCE", "HIERARCHY", "CAUSATION", "CONCURRENT", "ASSOCIATION", "SPECIAL_CAUSATION"},
    "default_weight": 0.5,
    "required_for": {
        "culprit_path": ["CAUSATION", "INFERENCE"]  # The culprit path must contain the inference relationship
    }
}

MCTS_CONFIG = {
    "relation_penalty_rates": {  # New relation type penalty factor
        "INVALID_RELATION": 0.7,
        "LOW_WEIGHT_CAUSATION": 0.5
    },
    "max_depth": 15,
    "min_culprit_clues": 3  # The final scene requires at least 3 culprit clues
}

DIRECT_GENERATION_PARAMS = {
    'temperature': 0.7,          # 0.4 higher than the validation mode
    'top_p': 0.95,               # Allows for a larger creative space
    'max_tokens': 5000           # Support for long texts
}