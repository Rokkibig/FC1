import sqlite3
import os

DB_FILE = "forex_data.db"
SCHEMA_FILE = "database/schema.sqlite.sql"

def main():
    """
    Sets up the SQLite database by creating the .db file and executing the schema.
    """
    print("🚀 Setting up SQLite database...")

    # Check if the database file already exists
    if os.path.exists(DB_FILE):
        print(f"✅ Database file '{DB_FILE}' already exists. Setup is complete.")
        return

    try:
        # Read the schema file
        print(f"📖 Reading schema from '{SCHEMA_FILE}'...")
        with open(SCHEMA_FILE, 'r') as f:
            schema_sql = f.read()

        # Connect to the database (this will create the file)
        print(f"✍️ Creating database file '{DB_FILE}'...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Execute the schema script
        print("🏗️ Executing schema to create tables and indexes...")
        cursor.executescript(schema_sql)

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        print(f"🎉 Successfully created and initialized database '{DB_FILE}'.")

    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        # Clean up a potentially partially created file
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

if __name__ == "__main__":
    main()
