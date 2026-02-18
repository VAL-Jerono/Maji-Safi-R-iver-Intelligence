"""
Configuration for EY Water Quality Challenge
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Snowflake Configuration
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "HYB96890")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SNOWFLAKE_LEARNING_DB")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")

# Connection parameters for Snowflake
CONNECTION_PARAMS = {
    "account": SNOWFLAKE_ACCOUNT,
    "user": SNOWFLAKE_USER,
    "password": SNOWFLAKE_PASSWORD,
    "warehouse": SNOWFLAKE_WAREHOUSE,
    "database": SNOWFLAKE_DATABASE,
    "schema": SNOWFLAKE_SCHEMA,
}

# Data Configuration
TARGET_VARIABLES = [
    'total_alkalinity',
    'electrical_conductance',
    'dissolved_reactive_phosphorus'
]

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Paths
DATA_DIR = "data/"
RESULTS_DIR = "results/"
MODELS_DIR = "models/"

# Feature Engineering
POLYNOMIAL_FEATURES = True
POLYNOMIAL_DEGREE = 2