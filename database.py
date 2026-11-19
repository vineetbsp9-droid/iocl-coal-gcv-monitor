import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "iocl_plant_logs.db"

def init_db():
    """Initialize the database and create the prediction logs table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Prediction history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            moisture REAL,
            ash REAL,
            volatile_matter REAL,
            fixed_carbon REAL,
            sulfur REAL,
            predicted_gcv REAL,
            quality_grade TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(moisture, ash, vm, fc, sulfur, gcv, grade):
    """Save each individual prediction to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO prediction_logs (timestamp, moisture, ash, volatile_matter, fixed_carbon, sulfur, predicted_gcv, quality_grade)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, moisture, ash, vm, fc, sulfur, gcv, grade))
    
    conn.commit()
    conn.close()

def get_logs():
    """Fetch all historical records from the database."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC", conn)
    conn.close()
    return df