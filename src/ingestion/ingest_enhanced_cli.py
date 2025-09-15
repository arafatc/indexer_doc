#!/usr/bin/env python3
"""
Enhanced CLI interface for LlamaIndex document ingestion with advanced chunking strategies.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ingest_llamaindex_enhanced import (
    ingest_documents_enhanced, 
    CHUNKING_STRATEGY, 
    ENABLE_TABLE_EXTRACTION,
    ENABLE_IMAGE_EXTRACTION, 
    ENABLE_CONTEXTUAL_RAG,
    RAW_DIR, 
    PROCESSED_DIR
)

def print_header():
    """Print CLI header"""
    print("FOCUS:" + "=" * 58 + "FOCUS:")
    print("STARTING:   LlamaIndex Enhanced Document Ingestion CLI   STARTING:")
    print("FOCUS:" + "=" * 58 + "FOCUS:")

def print_configuration():
    """Print current configuration"""
    print(f"\n Current Configuration:")
    print(f"   Raw Directory: {RAW_DIR}")
    print(f"   Processed Directory: {PROCESSED_DIR}")
    print(f"   Chunking Strategy: {CHUNKING_STRATEGY}")
    print(f"   Table Extraction: {'SUCCESS: Enabled' if ENABLE_TABLE_EXTRACTION else 'ERROR: Disabled'}")
    print(f"   Image Extraction: {'SUCCESS: Enabled' if ENABLE_IMAGE_EXTRACTION else 'ERROR: Disabled'}")
    print(f"   Contextual RAG: {'SUCCESS: Enabled' if ENABLE_CONTEXTUAL_RAG else 'ERROR: Disabled'}")
    
    # Check for files
    if os.path.exists(RAW_DIR):
        files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.pdf', '.docx'))]
        print(f"   Files Found: {len(files)}")
        for f in files[:5]:  # Show first 5
            print(f"     - {f}")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more")
    else:
        print(f"   WARNING: Raw directory not found: {RAW_DIR}")

def run_enhanced_ingestion(args):
    """Run the enhanced ingestion process"""
    print_header()
    print_configuration()
    
    if args.dry_run:
        print(f"\nCHECKING: DRY RUN MODE - No actual ingestion will be performed")
        print(f"   Configuration looks good! Ready for ingestion.")
        return True
    
    print(f"\nSTARTING: Starting Enhanced Document Ingestion...")
    start_time = time.time()
    
    try:
        # Set environment variables if provided via CLI
        if args.chunking_strategy:
            os.environ['CHUNKING_STRATEGY'] = args.chunking_strategy
        if args.enable_tables:
            os.environ['ENABLE_TABLE_EXTRACTION'] = 'true'
        if args.enable_images:
            os.environ['ENABLE_IMAGE_EXTRACTION'] = 'true'
        if args.enable_contextual_rag:
            os.environ['ENABLE_CONTEXTUAL_RAG'] = 'true'
        
        # Run ingestion
        index = ingest_documents_enhanced()
        
        total_time = time.time() - start_time
        
        if index:
            print(f"\n Enhanced Ingestion Completed Successfully!")
            print(f"   Total Time: {total_time:.2f} seconds")
            print(f"   Vector Store: llamaindex_enhanced (PostgreSQL)")
            print(f"   Strategy Used: {os.getenv('CHUNKING_STRATEGY', 'basic')}")
            return True
        else:
            print(f"\nERROR: Enhanced Ingestion Failed")
            return False
            
    except KeyboardInterrupt:
        print(f"\n Ingestion cancelled by user")
        return False
    except Exception as e:
        print(f"\nERROR: Error during enhanced ingestion: {e}")
        return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Enhanced LlamaIndex Document Ingestion with Advanced Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INITIALIZING: Available Chunking Strategies:
  basic            - Simple sentence splitting (default)
  structure_aware  - Markdown-aware chunking with document structure
  semantic         - Semantic similarity-based chunking
  hierarchical     - Multi-level hierarchical chunking  
  contextual_rag   - Advanced RAG with metadata extraction

FOCUS: Examples:
  # Basic ingestion
  python ingest_enhanced_cli.py

  # Structure-aware chunking with table extraction
  python ingest_enhanced_cli.py --chunking-strategy structure_aware --enable-tables

  # Full-featured contextual RAG
  python ingest_enhanced_cli.py --chunking-strategy contextual_rag --enable-tables --enable-images --enable-contextual-rag

  # Dry run to check configuration
  python ingest_enhanced_cli.py --dry-run
        """
    )
    
    # Chunking options
    parser.add_argument(
        '--chunking-strategy', 
        choices=['basic', 'structure_aware', 'semantic', 'hierarchical', 'contextual_rag'],
        help='Chunking strategy to use (default: from environment or basic)'
    )
    
    # Feature flags
    parser.add_argument(
        '--enable-tables', 
        action='store_true',
        help='Enable table extraction from documents'
    )
    parser.add_argument(
        '--enable-images', 
        action='store_true',
        help='Enable image extraction from documents'
    )
    parser.add_argument(
        '--enable-contextual-rag', 
        action='store_true',
        help='Enable advanced contextual RAG features'
    )
    
    # Utility options
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show configuration without running ingestion'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Run the enhanced ingestion
    success = run_enhanced_ingestion(args)
    
    if success:
        print(f"\nSUCCESS: Enhanced ingestion process completed successfully!")
    else:
        print(f"\nERROR: Enhanced ingestion process failed!")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
