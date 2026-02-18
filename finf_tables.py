"""Verify all tables with correct Snowflake syntax"""

import os
from dotenv import load_dotenv
from snowflake.snowpark import Session

load_dotenv()

CONNECTION_PARAMS = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

print("\n" + "="*70)
print("COMPLETE DATA VERIFICATION")
print("="*70)

session = Session.builder.configs(CONNECTION_PARAMS).create()

tables = [
    "water_quality_training_dataset",
    "landsat_features_training",
    "landsat_features_validation",
    "terraclimate_features_training",
    "terraclimate_features_validation",
    "submission_template"
]

print("\nTable Row Counts:\n")

for table in tables:
    try:
        # Use quoted names to preserve case
        count = session.sql(f'SELECT COUNT(*) as cnt FROM "{table}"').to_pandas()
        row_count = count.iloc[0]['CNT']
        print(f"✅ {table:45s} {row_count:>6,} rows")
    except Exception as e:
        print(f"❌ {table}: {e}")

# Show column details for main table
print("\n" + "-"*70)
print("WATER QUALITY TRAINING DATASET - COLUMNS:")
print("-"*70)

try:
    schema = session.sql('DESCRIBE TABLE "water_quality_training_dataset"').to_pandas()
    print(schema[['name', 'type']].to_string(index=False))
except Exception as e:
    print(f"Error: {e}")

# Show sample data
print("\n" + "-"*70)
print("SAMPLE DATA (First 5 rows):")
print("-"*70)

try:
    sample = session.sql('SELECT * FROM "water_quality_training_dataset" LIMIT 5').to_pandas()
    print(sample)
except Exception as e:
    print(f"Error: {e}")

session.close()

print("\n" + "="*70)
print("✅ VERIFICATION COMPLETE")
print("="*70)
print("\nYou're ready to start the challenge!")
print("Next: Run the benchmark model notebook")