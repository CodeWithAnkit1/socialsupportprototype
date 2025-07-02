import sqlite3
from utils.logger import get_logger

logger = get_logger("database")

# Initialize SQLite DB
def init_db():
    conn = sqlite3.connect("social_support.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emirates_id TEXT,
            name TEXT,
            phone TEXT,
            address TEXT,
            dependents INTEGER,
            submitted_income REAL,
            submitted_loans REAL,
            extracted_income REAL,
            extracted_loans REAL
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database and table initialized.")

def insert_application(emirates_id, name, phone, address, dependents, submitted_income, submitted_loans, extracted_income, extracted_loans):
    conn = sqlite3.connect("social_support.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO applications (
            emirates_id, name, phone, address, dependents,
            submitted_income, submitted_loans, extracted_income, extracted_loans
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (emirates_id, name, phone, address, dependents, submitted_income, submitted_loans, extracted_income, extracted_loans))
    conn.commit()
    conn.close()
    logger.info(f"Inserted application for {name} (Emirates ID: {emirates_id})")
