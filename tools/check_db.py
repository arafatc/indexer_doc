#!/usr/bin/env python3
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cursor = conn.cursor()

print("Checking available tables in database...")
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
tables = cursor.fetchall()
print(f"Found {len(tables)} tables:")
for table in tables:
    print(f"  - {table[0]}")

print("\nChecking for document-related tables...")
doc_tables = [t[0] for t in tables if 'doc' in t[0].lower()]
print(f"Document tables: {doc_tables}")

if doc_tables:
    for table in doc_tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} rows")

cursor.close()
conn.close()
