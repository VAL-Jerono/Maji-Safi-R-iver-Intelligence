"""
Test Snowflake connection and data access
"""

import sys
sys.path.insert(0, '..')

from src.snowflake_utils import SnowflakeManager

def test_connection():
    """Test basic connection"""
    print("\n" + "="*60)
    print("TEST 1: Snowflake Connection")
    print("="*60)
    
    try:
        session = SnowflakeManager.connect()
        result = session.sql("SELECT CURRENT_USER() as user, CURRENT_WAREHOUSE() as warehouse").to_pandas()
        print(f"✅ Connected as: {result.iloc[0]['USER']}")
        print(f"✅ Using warehouse: {result.iloc[0]['WAREHOUSE']}")
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\n" + "="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    
    try:
        # Load main dataset
        quality = SnowflakeManager.load_table('water_quality_training_dataset')
        print(f"✅ Quality data shape: {quality.shape}")
        print(f"   Columns: {quality.columns.tolist()}")
        
        # Load feature datasets
        landsat = SnowflakeManager.load_table('landsat_features_training')
        print(f"✅ Landsat features shape: {landsat.shape}")
        
        terraclimate = SnowflakeManager.load_table('terraclimate_features_training')
        print(f"✅ TerraClimate features shape: {terraclimate.shape}")
        
        # Load submission template
        submission = SnowflakeManager.load_table('submission_template')
        print(f"✅ Submission template shape: {submission.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_data_quality():
    """Test data quality"""
    print("\n" + "="*60)
    print("TEST 3: Data Quality")
    print("="*60)
    
    try:
        quality = SnowflakeManager.load_table('water_quality_training_dataset')
        
        print(f"\nData Info:")
        print(f"  Shape: {quality.shape}")
        print(f"  Missing values:\n{quality.isnull().sum()}")
        print(f"  Data types:\n{quality.dtypes}")
        
        return True
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("EY WATER QUALITY CHALLENGE - CONNECTION TESTS")
    print("="*60)
    
    results = {
        "Connection": test_connection(),
        "Data Loading": test_data_loading(),
        "Data Quality": test_data_quality()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    # Disconnect
    SnowflakeManager.disconnect()
    
    if all(results.values()):
        print("\n🎉 All tests passed! Ready to start development.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()