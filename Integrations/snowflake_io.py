#try
import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

def connect_snowflake():
    """Establish connection using environment variables"""
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )
    return conn

def setup_tables():
    """Initialise required tables if they don’t exist"""
    ddl_tickets = """
    CREATE TABLE IF NOT EXISTS SUPPORT_TICKETS (
        TICKET_ID INTEGER AUTOINCREMENT,
        USER_TEXT STRING,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (TICKET_ID)
    );
    """
    ddl_results = """
    CREATE TABLE IF NOT EXISTS TICKET_RESULTS (
        RESULT_ID INTEGER AUTOINCREMENT,
        TICKET_ID INTEGER,
        PREDICTED_CATEGORY STRING,
        SENTIMENT STRING,
        SENTIMENT_CONFIDENCE FLOAT,
        ESCALATION_DECISION STRING,
        REASON STRING,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (RESULT_ID)
    );
    """

    conn = connect_snowflake()
    cur = conn.cursor()
    try:
        cur.execute(ddl_tickets)
        cur.execute(ddl_results)
        conn.commit()
        print("✅ Tables created or already exist.")
    finally:
        cur.close()
        conn.close()


def insert_ticket(user_text: str) -> int:
    """Insert user message and return ticket_id"""
    conn = connect_snowflake()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO SUPPORT_TICKETS (USER_TEXT) VALUES (%s)", (user_text,))
        cur.execute("SELECT LASTVAL()")  # works as LAST_INSERT_ID alternative
        ticket_id = cur.fetchone()[0]
        conn.commit()
        return ticket_id
    finally:
        cur.close()
        conn.close()


def insert_results(ticket_id: int, category: str, sentiment: str, conf: float, decision: str, reason: str):
    """Store the processed results"""
    conn = connect_snowflake()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO TICKET_RESULTS
            (TICKET_ID, PREDICTED_CATEGORY, SENTIMENT, SENTIMENT_CONFIDENCE, ESCALATION_DECISION, REASON)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (ticket_id, category, sentiment, conf, decision, reason))
        conn.commit()
        print(f"✅ Inserted results for Ticket {ticket_id}")
    finally:
        cur.close()
        conn.close()
