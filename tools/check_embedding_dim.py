#!/usr/bin/env python3
"""
Check embedding dimensions in the database
"""

import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

def check_embedding_dimensions():
    """Check the actual embedding dimensions stored in the database"""
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv('PG_HOST', 'localhost'),
            port=int(os.getenv('PG_PORT', '5432')),
            user=os.getenv('PG_USER', 'postgres'),
            password=os.getenv('PG_PASSWORD', 'postgres'),
            database=os.getenv('PG_DATABASE', 'rag_db')
        )
        
        cur = conn.cursor()
        
        # Check semantic table
        print("CHECKING: Checking semantic strategy table...")
        cur.execute("SELECT COUNT(*) FROM data_llamaindex_enhanced_semantic")
        count = cur.fetchone()[0]
        print(f"   Rows: {count}")
        
        if count > 0:
            cur.execute("SELECT embedding FROM data_llamaindex_enhanced_semantic LIMIT 1")
            row = cur.fetchone()
            if row and row[0]:
                # Parse the embedding array to get its length
                embedding_str = row[0]
                # Remove brackets and split by comma
                embedding_list = embedding_str.strip('[]').split(',')
                embedding_dim = len(embedding_list)
                print(f"   Actual embedding dimension: {embedding_dim}")
            else:
                print("   No embedding found")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR: Error: {e}")

if __name__ == "__main__":
    check_embedding_dimensions()
