"""Upload CSV files to Snowflake"""

import pandas as pd
from snowflake.snowpark import Session
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONNECTION_PARAMS = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

def upload_csv_to_snowflake(csv_path, table_name):
    """Upload CSV file to Snowflake table"""
    
    session = Session.builder.configs(CONNECTION_PARAMS).create()
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Write to Snowflake - Fixed: removed 'mode' parameter
        session.write_pandas(df, table_name, auto_create_table=True)
        
        print(f"✅ {table_name}: {len(df)} rows uploaded")
        
    except Exception as e:
        print(f"❌ {table_name}: {e}")
    finally:
        session.close()

def main():
    """Upload all data files"""
    
    print("\n" + "="*70)
    print("UPLOADING DATA TO SNOWFLAKE")
    print("="*70)
    
    # Verify credentials are loaded
    if not CONNECTION_PARAMS["user"]:
        print("❌ ERROR: Credentials not loaded from .env file")
        print("Make sure .env file exists in the project root")
        return
    
    print(f"\nConnecting as: {CONNECTION_PARAMS['user']}")
    print(f"Account: {CONNECTION_PARAMS['account']}")
    print(f"Warehouse: {CONNECTION_PARAMS['warehouse']}\n")
    
    # Map CSV files to table names
    uploads = {
        "data/water_quality_training_dataset.csv": "water_quality_training_dataset",
        "data/landsat_features_training.csv": "landsat_features_training",
        "data/landsat_features_validation.csv": "landsat_features_validation",
        "data/terraclimate_features_training.csv": "terraclimate_features_training",
        "data/terraclimate_features_validation.csv": "terraclimate_features_validation",
        "data/submission_template.csv": "submission_template",
    }
    
    for csv_file, table_name in uploads.items():
        if os.path.exists(csv_file):
            print(f"Uploading {csv_file}...")
            upload_csv_to_snowflake(csv_file, table_name)
        else:
            print(f"⚠️  File not found: {csv_file}")
    
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()