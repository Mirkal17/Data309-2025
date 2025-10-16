import snowflake.connector

def connect_snowflake():
    """Connect to Snowflake and return the connection object."""
    try:
        conn = snowflake.connector.connect(
            user='AMIRROJALI17',
            password='Mirkal383917!!',  
            account='xl21198.australia-east.azure',
            warehouse='DATA309PROJECT',      
            database='SUPPORT_AI_DB',
            schema='AI_PIPELINE'
        )
        print("✅ Connection successful!")
        return conn
    except Exception as e:
        print("❌ Connection failed:", e)
        return None

if __name__ == "__main__":
    connect_snowflake()
