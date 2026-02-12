import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir) # Go up one level to backend/
dotenv_path = os.path.join(backend_dir, '.env')

# Load environment variables
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment configuration from {dotenv_path}")
else:
    load_dotenv()
    print("Loaded environment configuration from current directory")

def get_mysql_uri():
    required_vars = ['MYSQLUSER', 'MYSQLPASSWORD', 'MYSQLHOST', 'MYSQLPORT', 'MYSQLDATABASE']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"Error: Missing environment variables: {missing}")
        print("Make sure you have a .env file with MySQL credentials.")
        return None
    
    return (
        f"mysql+pymysql://{os.getenv('MYSQLUSER')}:"
        f"{os.getenv('MYSQLPASSWORD')}@"
        f"{os.getenv('MYSQLHOST')}:"
        f"{os.getenv('MYSQLPORT')}/"
        f"{os.getenv('MYSQLDATABASE')}"
    )

def main():
    uri = get_mysql_uri()
    if not uri:
        sys.exit(1)

    print(f"Target Database Host: {os.getenv('MYSQLHOST')}")
    print(f"Target Database Name: {os.getenv('MYSQLDATABASE')}")
    print("Preparing to drop tables: 'log', 'standards_document'...")

    try:
        engine = create_engine(uri)
        with engine.connect() as conn:
            # Drop log table
            print("Dropping table 'log'...", end=" ")
            conn.execute(text("DROP TABLE IF EXISTS log"))
            print("Done.")

            # Drop standards_document table
            print("Dropping table 'standards_document'...", end=" ")
            conn.execute(text("DROP TABLE IF EXISTS standards_document"))
            print("Done.")
            
            conn.commit()
            print("\nSUCCESS: Unused tables removed from the database.")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
