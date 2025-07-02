import sys
import os
import sqlite3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.logger import get_logger
logger = get_logger("sample_query")

def get_all_applications():
    conn = sqlite3.connect("social_support.db")
    c = conn.cursor()
    c.execute("SELECT * FROM applications")
    rows = c.fetchall()
    conn.close()
    logger.info(f"Fetched {len(rows)} applications from database.")
    return rows

# Example usage:
if __name__ == "__main__":
    apps = get_all_applications()
    for app in apps:
        print(app)