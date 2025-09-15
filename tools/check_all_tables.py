#!/usr/bin/env python3

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cursor = conn.cursor()

print('Checking all tables:')
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
tables = cursor.fetchall()
for table in tables:
    table_name = table[0]
    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
    count = cursor.fetchone()[0]
    print(f'  - {table_name}: {count} rows')

cursor.close()
conn.close()
