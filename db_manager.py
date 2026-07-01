import os
import sqlite3
import pandas as pd
from datetime import datetime
from config import BASE_DIR
from utils import log_info, log_error, log_warning

DB_PATH = os.path.join(BASE_DIR, "trend_alfa.db")

def get_connection():
    """Returns a SQLite connection to the primary database."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database if required. SQLite creates the file automatically."""
    log_info(f"Initialized SQLite database at {DB_PATH}")

def save_pipeline_stage(df, table_name, date_str=None):
    """
    Saves a DataFrame to the specified SQLite table.
    - Appends a 'Date' column to the DataFrame to support historical time-series queries.
    - Drops & recreates table on schema mismatch to prevent silent failures.
    """
    if df is None or df.empty:
        log_warning(f"Attempted to save empty DataFrame to {table_name}. Skipping.")
        return

    # Use today's date if not provided
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Create a copy so we don't mutate the original dataframe in memory
    df_save = df.copy()
    
    # Insert Date as the first column for cleaner organization
    if "Date" in df_save.columns:
        df_save.drop(columns=["Date"], inplace=True)
    df_save.insert(0, "Date", date_str)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table exists and compare schema
            cursor.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone()[0] == 1
            
            if table_exists:
                # Get existing columns from the table
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                existing_cols = {row[1] for row in cursor.fetchall()}
                new_cols = set(df_save.columns)
                
                if existing_cols == new_cols:
                    # Schema matches — just delete old rows for this date and append
                    cursor.execute(f"DELETE FROM {table_name} WHERE Date = ?", (date_str,))
                    conn.commit()
                else:
                    # Schema mismatch — drop and recreate
                    log_warning(f"Schema mismatch for '{table_name}'. Dropping and recreating table.")
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    conn.commit()
                    table_exists = False
            
            # Save the new data
            df_save.to_sql(table_name, conn, if_exists="append", index=False)
            log_info(f"Successfully saved {len(df_save)} records to table '{table_name}' for Date: {date_str}")
            
    except Exception as e:
        log_error(f"Failed to save data to SQLite table '{table_name}': {e}")

def load_pipeline_stage(table_name, date_str=None):
    """
    Loads a DataFrame from the specified SQLite table.
    - If date_str is provided, loads data for that specific date.
    - If date_str is None, loads the MOST RECENT date available in the table.
    Returns None if table doesn't exist or is empty.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if cursor.fetchone()[0] == 0:
                return None
                
            if date_str:
                query = f"SELECT * FROM {table_name} WHERE Date = ?"
                df = pd.read_sql_query(query, conn, params=(date_str,))
            else:
                # Find the most recent date
                cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
                max_date = cursor.fetchone()[0]
                if not max_date:
                    return None
                query = f"SELECT * FROM {table_name} WHERE Date = ?"
                df = pd.read_sql_query(query, conn, params=(max_date,))
                
            if df.empty:
                return None
                
            # Drop the Date column when returning to keep compatibility with existing pipeline logic
            if "Date" in df.columns:
                df.drop(columns=["Date"], inplace=True)
                
            return df
    except Exception as e:
        log_error(f"Failed to load data from SQLite table '{table_name}': {e}")
        return None
