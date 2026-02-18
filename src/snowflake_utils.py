"""
Snowflake utility functions
"""

from snowflake.snowpark import Session
from config import CONNECTION_PARAMS
import pandas as pd
from typing import Optional

class SnowflakeManager:
    """Manage Snowflake connections and operations"""
    
    _session: Optional[Session] = None
    
    @classmethod
    def connect(cls) -> Session:
        """Create and return Snowflake session"""
        if cls._session is None:
            try:
                cls._session = Session.builder.configs(CONNECTION_PARAMS).create()
                print("✅ Connected to Snowflake")
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                raise
        return cls._session
    
    @classmethod
    def load_table(cls, table_name: str) -> pd.DataFrame:
        """Load table from Snowflake as pandas DataFrame"""
        session = cls.connect()
        try:
            df = session.table(table_name).to_pandas()
            print(f"✅ Loaded {table_name}: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Failed to load {table_name}: {e}")
            raise
    
    @classmethod
    def execute_sql(cls, query: str) -> pd.DataFrame:
        """Execute SQL query and return as DataFrame"""
        session = cls.connect()
        try:
            df = session.sql(query).to_pandas()
            print(f"✅ Query executed: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise
    
    @classmethod
    def save_table(cls, df: pd.DataFrame, table_name: str, mode: str = "overwrite"):
        """Save DataFrame to Snowflake table"""
        session = cls.connect()
        try:
            session.write_pandas(df, table_name, mode=mode)
            print(f"✅ Saved to {table_name}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
            raise
    
    @classmethod
    def disconnect(cls):
        """Close Snowflake session"""
        if cls._session:
            cls._session.close()
            cls._session = None
            print("✅ Disconnected from Snowflake")