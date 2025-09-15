#!/usr/bin/env python3
"""
Simple Table Deletion Script - Delete multiple tables by name

Usage:
  python delete_tables_simple.py table1 table2 table3
  python delete_tables_simple.py --force table1 table2 table3
  python delete_tables_simple.py --list  # Show all tables first
"""

import os
import sys
import argparse
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_db():
    """Connect to the database"""
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        return conn
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)

def list_tables(conn):
    """List all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename;
    """)
    
    tables = cursor.fetchall()
    print(f"\nFound {len(tables)} tables:")
    print(f"{'Table Name':<40} {'Size':<10}")
    print("-" * 52)
    for table_name, size in tables:
        print(f"{table_name:<40} {size:<10}")
    
    cursor.close()
    return [table[0] for table in tables]

def table_exists(conn, table_name):
    """Check if a table exists"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = %s
        );
    """, (table_name,))
    
    exists = cursor.fetchone()[0]
    cursor.close()
    return exists

def get_table_info(conn, table_name):
    """Get basic table information"""
    cursor = conn.cursor()
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
        row_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size(%s)) as size;
        """, (table_name,))
        table_size = cursor.fetchone()[0]
        
        cursor.close()
        return row_count, table_size
    except Exception as e:
        cursor.close()
        return None, None

def delete_table(conn, table_name, force=False):
    """Delete a single table"""
    if not table_exists(conn, table_name):
        print(f"ERROR: Table '{table_name}' does not exist")
        return False
    
    # Get table info
    row_count, table_size = get_table_info(conn, table_name)
    if row_count is not None:
        print(f" Table '{table_name}': {row_count:,} rows, {table_size}")
    
    # Confirmation unless forced
    if not force:
        response = input(f"     Delete table '{table_name}'? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print(f"     Skipped '{table_name}'")
            return False
    
    # Delete the table
    cursor = conn.cursor()
    try:
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
        conn.commit()
        print(f"   SUCCESS: Deleted '{table_name}'")
        cursor.close()
        return True
    except Exception as e:
        conn.rollback()
        print(f"   ERROR: Error deleting '{table_name}': {e}")
        cursor.close()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Simple tool to delete multiple PostgreSQL tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all tables first
  python delete_tables_simple.py --list
  
  # Delete specific tables (with confirmation)
  python delete_tables_simple.py table1 table2 table3
  
  # Delete without confirmation
  python delete_tables_simple.py --force table1 table2 table3
  
  # Delete all tables matching a pattern (requires existing table names)
  python delete_tables_simple.py llamaindex_enhanced_* vector_store_*
        """
    )
    
    parser.add_argument('tables', nargs='*', help='Table names to delete')
    parser.add_argument('--list', action='store_true', help='List all tables')
    parser.add_argument('--force', action='store_true', help='Delete without confirmation')
    
    args = parser.parse_args()
    
    # Connect to database
    conn = connect_db()
    
    try:
        if args.list:
            list_tables(conn)
        
        elif args.tables:
            print(f"\n  Preparing to delete {len(args.tables)} table(s)...")
            
            success_count = 0
            for table_name in args.tables:
                if delete_table(conn, table_name, force=args.force):
                    success_count += 1
            
            print(f"\nINFO: Summary: {success_count}/{len(args.tables)} tables deleted successfully")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n  Operation cancelled by user")
        conn.rollback()
    
    except Exception as e:
        print(f"ERROR: Error: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()