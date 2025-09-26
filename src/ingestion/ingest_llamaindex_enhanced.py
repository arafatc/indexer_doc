#!/usr/bin/env python3
"""
Enhanced LlamaIndex-based document ingestion with advanced chunking strategies.
Combines the best of both worlds: original docling functionality + LlamaIndex capabilities.
"""

import os
import time
import warnings
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Load environment variables
load_dotenv()

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser, 
    HierarchicalNodeParser,
    MarkdownNodeParser
)
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, MetadataMode

# Import embeddings and LLMs with error handling for different versions
try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    try:
        from llama_index.embeddings import OllamaEmbedding
    except ImportError:
        print("Warning: OllamaEmbedding not available. Install llama-index-embeddings-ollama")
        OllamaEmbedding = None

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    try:
        from llama_index.embeddings import OpenAIEmbedding
    except ImportError:
        print("Warning: OpenAIEmbedding not available. Install llama-index-embeddings-openai")
        OpenAIEmbedding = None

try:
    from llama_index.vector_stores.postgres import PGVectorStore
except ImportError:
    try:
        from llama_index.vector_stores import PGVectorStore
    except ImportError:
        print("Warning: PGVectorStore not available. Install llama-index-vector-stores-postgres")
        PGVectorStore = None

try:
    from llama_index.storage.docstore.postgres import PostgresDocumentStore
except ImportError:
    try:
        from llama_index.storage.docstore import PostgresDocumentStore
    except ImportError:
        print("Warning: PostgresDocumentStore not available. Install llama-index-storage-docstore-postgres")
        PostgresDocumentStore = None

try:
    from llama_index.llms.ollama import Ollama
except ImportError:
    try:
        from llama_index.llms import Ollama
    except ImportError:
        print("Warning: Ollama LLM not available. Install llama-index-llms-ollama")
        Ollama = None

try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    try:
        from llama_index.llms import OpenAI
    except ImportError:
        print("Warning: OpenAI LLM not available. Install llama-index-llms-openai")
        OpenAI = None

# Document processing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# ===================== CONFIGURATION =====================
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
PROCESSED_MD_DIR = os.path.join(PROCESSED_DIR, 'md')
PROCESSED_JSON_DIR = os.path.join(PROCESSED_DIR, 'json')

SUPPORTED_EXTENSIONS = ['.pdf', '.docx']

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")  # "ollama" or "openai"

# Gemma/Ollama settings - Production optimized
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# OpenAI settings (backup)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# PostgreSQL/PGVector settings
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "rag_db")

# Enhanced Chunking Settings
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "basic").lower()
# Options: "basic", "structure_aware", "semantic", "hierarchical", "contextual_rag"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Advanced features flags
ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
ENABLE_IMAGE_EXTRACTION = os.getenv("ENABLE_IMAGE_EXTRACTION", "false").lower() == "true"
ENABLE_CONTEXTUAL_RAG = os.getenv("ENABLE_CONTEXTUAL_RAG", "false").lower() == "true"

# Embedding dimensions based on model
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536, 
    "text-embedding-3-large": 3072,
    "nomic-embed-text:latest": 768,
    "nomic-embed-text:v1.5": 768
}

# Add externalized EMBEDDING_DIM with fallback to dynamic detection
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

def get_embedding_dim():
    """Get embedding dimension - prioritize env variable, fallback to model-based detection"""
    # Check for explicit override first
    if os.getenv("EMBEDDING_DIM"):
        return EMBEDDING_DIM
    
    # Otherwise use model-based detection
    if EMBEDDING_PROVIDER == "openai":
        return EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536)
    else:
        return EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 768)

def get_files(directory, extensions):
    """Get all supported files from directory"""
    files = []
    for fname in os.listdir(directory):
        if any(fname.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, fname))
    return files

def setup_llm_and_embedding():
    """Initialize LLM and embedding models based on configuration"""
    print(f"Setting up LLM Provider: {LLM_PROVIDER}")
    print(f"Setting up Embedding Provider: {EMBEDDING_PROVIDER}")
    
    # Initialize LLM
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI")
        if OpenAI is None:
            raise ImportError("OpenAI LLM not available. Install: pip install llama-index-llms-openai")
        llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        print(f"Using OpenAI LLM: {OPENAI_MODEL}")
    else:
        if Ollama is None:
            raise ImportError("Ollama LLM not available. Install: pip install llama-index-llms-ollama")
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=600.0)
        print(f"Using Ollama LLM: {LLM_MODEL} (timeout: 600s)")
    
    # Initialize embedding model
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        if OpenAIEmbedding is None:
            raise ImportError("OpenAI Embedding not available. Install: pip install llama-index-embeddings-openai")
        embedding = OpenAIEmbedding(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        print(f"Using OpenAI embedding: {OPENAI_EMBEDDING_MODEL}")
    else:
        if OllamaEmbedding is None:
            raise ImportError("Ollama Embedding not available. Install: pip install llama-index-embeddings-ollama")
        embedding = OllamaEmbedding(
            model_name=EMBEDDING_MODEL, 
            base_url=OLLAMA_BASE_URL,
            request_timeout=600.0
        )
        print(f"Using Ollama embedding: {EMBEDDING_MODEL} (timeout: 600s)")
    
    return llm, embedding

def setup_storage(chunking_strategy=None):
    """Initialize vector store and document store"""
    embed_dim = get_embedding_dim()
    print(f"Setting up storage with embedding dimension: {embed_dim}")
    print(f"STORAGE: Received strategy parameter: {chunking_strategy}")
    print(f"STORAGE: Global CHUNKING_STRATEGY: {CHUNKING_STRATEGY}")
    
    # Check for required components
    if PGVectorStore is None:
        raise ImportError("PGVectorStore not available. Install: pip install llama-index-vector-stores-postgres")
    if PostgresDocumentStore is None:
        raise ImportError("PostgresDocumentStore not available. Install: pip install llama-index-storage-docstore-postgres")
    
    # Strategy-specific table names to avoid mixing
    strategy = chunking_strategy or CHUNKING_STRATEGY
    strategy_suffix = strategy.lower()
    print(f"STORAGE: Using final strategy: {strategy}")
    vector_table = f"llamaindex_enhanced_{strategy_suffix}"
    docstore_table = f"llamaindex_enhanced_docstore_{strategy_suffix}"
    
    print(f"INFO: Using strategy-specific tables:")
    print(f"   Vector Store: {vector_table}")
    print(f"   Doc Store: {docstore_table}")
    
    # Vector store for embeddings
    vector_store = PGVectorStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=vector_table,  # Strategy-specific table
        embed_dim=embed_dim,
        perform_setup=True
    )
    
    # Document store for metadata
    doc_store = PostgresDocumentStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=docstore_table,  # Strategy-specific table
        perform_setup=True
    )
    
    # Storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=doc_store
    )
    
    print("SUCCESS: Storage context initialized with strategy separation")
    return storage_context

def ensure_processed_directories():
    """Ensure processed output directories exist (from original)"""
    try:
        Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROCESSED_MD_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROCESSED_JSON_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified processed directories:")
        print(f"  Base: {PROCESSED_DIR}")
        print(f"  MD: {PROCESSED_MD_DIR}")
        print(f"  JSON: {PROCESSED_JSON_DIR}")
    except Exception as e:
        print(f"Error creating directories: {e}")
        raise

def process_document_with_docling_enhanced(file_path):
    """Enhanced docling processing with table/image extraction"""
    source_filename = os.path.basename(file_path)
    
    # Configure DocumentConverter with enhanced settings
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Disable OCR to avoid issues
    pipeline_options.do_table_structure = ENABLE_TABLE_EXTRACTION  # SUCCESS: Enhanced: Table extraction
    pipeline_options.table_structure_options.do_cell_matching = ENABLE_TABLE_EXTRACTION
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    try:
        print(f"  Processing {source_filename} with enhanced docling...")
        result = converter.convert(file_path)
        doc = result.document
        
        # Extract enhanced content
        full_text = doc.export_to_text() if hasattr(doc, 'export_to_text') else str(doc)
        
        # SUCCESS: Enhanced: Extract structured elements with citation info
        structured_elements = []
        page_mappings = {}  # Track content to page number mapping
        
        if hasattr(doc, 'main_text') and doc.main_text:
            current_page = 1
            for idx, element in enumerate(doc.main_text):
                # Extract page information if available
                page_num = getattr(element, 'page', current_page)
                if hasattr(element, 'page') and element.page:
                    current_page = element.page
                
                element_data = {
                    'text': element.text if hasattr(element, 'text') else str(element),
                    'type': element.label if hasattr(element, 'label') else 'text',
                    'bbox': getattr(element, 'bbox', None),  # SUCCESS: Bounding box for images/tables
                    'page_number': page_num,  # SUCCESS: Citation: Page number for reference
                    'element_id': f"{source_filename}_elem_{idx}",  # SUCCESS: Citation: Unique element ID
                    'source_file': source_filename  # SUCCESS: Citation: Source document
                }
                structured_elements.append(element_data)
                
                # Build page mapping for citation purposes
                if element_data['text'].strip():
                    page_mappings[idx] = {
                        'page': page_num,
                        'text_start': element_data['text'][:100],  # First 100 chars for matching
                        'source': source_filename
                    }
                structured_elements.append(element_data)
        
        # SUCCESS: Enhanced: Save markdown and JSON files (preserved from original)
        save_document_as_markdown(doc, file_path, source_filename)
        save_document_as_json(doc, file_path, source_filename, structured_elements)
        
        if not full_text or full_text.strip() == "":
            full_text = f"Content from {source_filename} - text extraction incomplete"
        
        print(f"  SUCCESS: Extracted {len(full_text)} characters from {source_filename}")
        print(f"  SUCCESS: Found {len(structured_elements)} structured elements")
        
        return full_text, structured_elements
        
    except Exception as e:
        print(f"  ERROR: Error processing {file_path}: {e}")
        return f"Error processing {source_filename}: {str(e)[:200]}", []

def save_document_as_markdown(doc, file_path, source_filename):
    """Save processed document as markdown file (from original)"""
    try:
        Path(PROCESSED_MD_DIR).mkdir(parents=True, exist_ok=True)
        base_name = os.path.splitext(source_filename)[0]
        md_filename = f"{base_name}.md"
        md_filepath = os.path.join(PROCESSED_MD_DIR, md_filename)
        
        if hasattr(doc, 'export_to_markdown'):
            markdown_content = doc.export_to_markdown()
        elif hasattr(doc, 'export_to_text'):
            text_content = doc.export_to_text()
            markdown_content = f"# {base_name}\n\n{text_content}"
        else:
            markdown_content = f"# {base_name}\n\n{str(doc)}"
        
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"    [SUCCESS] Saved markdown: {md_filename}")
        return md_filepath
        
    except Exception as e:
        print(f"    [ERROR] Failed to save markdown for {source_filename}: {e}")
        return None

def save_document_as_json(doc, file_path, source_filename, structured_elements):
    """Save processed document metadata as JSON file (enhanced from original)"""
    try:
        Path(PROCESSED_JSON_DIR).mkdir(parents=True, exist_ok=True)
        base_name = os.path.splitext(source_filename)[0]
        json_filename = f"{base_name}.json"
        json_filepath = os.path.join(PROCESSED_JSON_DIR, json_filename)
        
        # Enhanced metadata
        doc_data = {
            "source_file": source_filename,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text_content": doc.export_to_text() if hasattr(doc, 'export_to_text') else str(doc),
            "chunking_strategy": CHUNKING_STRATEGY,
            "table_extraction_enabled": ENABLE_TABLE_EXTRACTION,
            "image_extraction_enabled": ENABLE_IMAGE_EXTRACTION,
            "contextual_rag_enabled": ENABLE_CONTEXTUAL_RAG,
            "structured_elements": structured_elements,  # SUCCESS: Enhanced: Structure info
            "metadata": {
                "file_path": file_path,
                "llm_provider": LLM_PROVIDER,
                "embedding_provider": EMBEDDING_PROVIDER
            }
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            import json
            json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        print(f"    [SUCCESS] Saved enhanced JSON metadata: {json_filename}")
        return json_filepath
        
    except Exception as e:
        print(f"    [ERROR] Failed to save JSON for {source_filename}: {e}")
        return None

def create_enhanced_chunking_pipeline(llm, embedding, chunking_strategy=None):
    """Create enhanced chunking pipeline with multiple strategies and citation preservation"""
    effective_strategy = chunking_strategy or CHUNKING_STRATEGY
    print(f"\nINITIALIZING: Initializing chunking strategy: {effective_strategy}")
    
    transformations = []
    
    # SUCCESS: Strategy 1: Basic Sentence Splitting (Default)
    if effective_strategy == "basic":
        node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            include_metadata=True,  # SUCCESS: Citation: Preserve metadata in chunks
            include_prev_next_rel=True  # SUCCESS: Citation: Track chunk relationships
        )
        transformations.append(node_parser)
        print(f"  NOTE: Basic chunking: {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    
    # SUCCESS: Strategy 2: Structure-Aware (Markdown-based)  
    elif effective_strategy == "structure_aware":
        # First parse markdown structure, then split
        md_parser = MarkdownNodeParser(include_metadata=True)
        sentence_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            include_metadata=True,
            include_prev_next_rel=True
        )
        transformations.extend([md_parser, sentence_splitter])
        print(f"  BUILDING: Structure-aware chunking with markdown parsing")
    
    # SUCCESS: Strategy 3: Semantic Chunking
    elif effective_strategy == "semantic":
        semantic_parser = SemanticSplitterNodeParser(
            buffer_size=1,  # Number of sentences to group
            breakpoint_percentile_threshold=95,  # Semantic similarity threshold
            embed_model=embedding,
            include_metadata=True  # SUCCESS: Citation: Preserve metadata
        )
        transformations.append(semantic_parser)
        print(f"  FOCUS: Semantic chunking with similarity threshold")
    
    # SUCCESS: Strategy 4: Hierarchical Chunking
    elif effective_strategy == "hierarchical":
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128],  # Multi-level chunks
            chunk_overlap=CHUNK_OVERLAP,
            include_metadata=True  # SUCCESS: Citation: Preserve metadata in hierarchy
        )
        transformations.append(hierarchical_parser)
        print(f"  PROCESSING: Hierarchical chunking: [2048, 512, 128] chars")
    
    # SUCCESS: Strategy 5: Contextual RAG (Advanced metadata extraction)
    elif effective_strategy == "contextual_rag":
        # Start with sentence splitting
        node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        transformations.append(node_parser)
        
        # Add advanced extractors for contextual information (optimized for performance)
        extractors = [
            TitleExtractor(nodes=3, llm=llm),  # Extract titles from context (reduced nodes)
            QuestionsAnsweredExtractor(questions=2, llm=llm),  # Generate questions (reduced count)
            SummaryExtractor(summaries=["self"], llm=llm),  # Chunk summaries (self only for speed)
            KeywordExtractor(keywords=5, llm=llm)  # Extract keywords (reduced count)
        ]
        transformations.extend(extractors)
        print(f"  ANALYZING: Contextual RAG with optimized metadata extraction")
        print(f"    - Title extraction (3 nodes), Q&A generation (2 questions), summaries (self), keywords (5)")
    
    else:
        raise ValueError(f"Unknown chunking strategy: {effective_strategy}")
    
    # SUCCESS: Always add embedding at the end
    transformations.append(embedding)
    
    # Create ingestion pipeline with citation preservation
    pipeline = IngestionPipeline(transformations=transformations)
    
    # Add custom post-processing for citation preservation
    def enhance_chunks_with_citations(nodes):
        """Enhance each chunk with proper citation metadata"""
        enhanced_nodes = []
        for node in nodes:
            if hasattr(node, 'metadata') and node.metadata:
                # Add chunk-specific citation info
                original_metadata = node.metadata.copy()
                chunk_text = getattr(node, 'text', '')
                
                # Enhanced citation metadata for each chunk
                citation_metadata = {
                    'chunk_id': getattr(node, 'node_id', f"chunk_{hash(chunk_text)}"),
                    'chunk_index': len(enhanced_nodes),
                    'source_citation': f"{original_metadata.get('source_document', 'Unknown Document')}",
                    'page_reference': f"Approx. page {(len(enhanced_nodes) // 10) + 1}",  # Rough page estimate
                    'citation_format': f"{original_metadata.get('source_document', 'Document')} ({original_metadata.get('document_type', 'PDF')})",
                    'retrieval_metadata': {
                        'chunk_length': len(chunk_text),
                        'strategy': original_metadata.get('chunking_strategy', effective_strategy),
                        'timestamp': original_metadata.get('processed_at', ''),
                        'extraction_method': 'docling_enhanced'
                    }
                }
                
                # Merge citation info with original metadata
                node.metadata.update(citation_metadata)
            
            enhanced_nodes.append(node)
        
        print(f"  SUCCESS: Enhanced {len(enhanced_nodes)} chunks with citation metadata")
        return enhanced_nodes
    
    # Store the citation enhancement function for later use
    pipeline._citation_enhancer = enhance_chunks_with_citations
    
    print(f"  SUCCESS: Pipeline created with {len(transformations)} transformations + citation support")
    
    return pipeline

def create_enhanced_documents(files, chunking_strategy=None):
    """Create LlamaIndex documents with enhanced metadata"""
    effective_strategy = chunking_strategy or CHUNKING_STRATEGY
    print(f"\nDOCUMENT: Creating enhanced documents from {len(files)} files with strategy: {effective_strategy}")
    documents = []
    
    for file_path in files:
        print(f"\n  Processing: {os.path.basename(file_path)}")
        
        # Use enhanced docling processing
        content, structured_elements = process_document_with_docling_enhanced(file_path)
        
        if not content:
            print(f"    WARNING: No content extracted from {file_path}")
            continue
        
        # SUCCESS: Enhanced metadata with citation support
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": os.path.splitext(file_path)[1],
            "file_size": os.path.getsize(file_path),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunking_strategy": effective_strategy,
            "total_structured_elements": len(structured_elements),
            "has_tables": any(el.get('type', '').lower() in ['table', 'table-cell'] for el in structured_elements),
            "has_images": any(el.get('type', '').lower() in ['figure', 'image', 'picture'] for el in structured_elements),
            "content_length": len(content),
            
            # SUCCESS: Citation handling metadata
            "source_document": os.path.basename(file_path),
            "document_type": os.path.splitext(file_path)[1].upper().replace('.', ''),
            "total_pages": max([el.get('page_number', 1) for el in structured_elements], default=1),
            "citation_id": f"doc_{hash(file_path)}",
            "extraction_method": "docling_enhanced"
        }
        
        # SUCCESS: Add structured elements as separate metadata fields with citations
        if ENABLE_TABLE_EXTRACTION and structured_elements:
            tables = [el for el in structured_elements if el.get('type', '').lower() in ['table', 'table-cell']]
            if tables:
                metadata["table_count"] = len(tables)
                metadata["table_preview"] = tables[0].get('text', '')[:200] if tables else ""
                metadata["table_citations"] = [
                    {
                        "page": table.get('page_number', 'Unknown'),
                        "element_id": table.get('element_id', ''),
                        "source": table.get('source_file', os.path.basename(file_path))
                    } for table in tables[:5]  # First 5 tables
                ]
        
        if ENABLE_IMAGE_EXTRACTION and structured_elements:
            images = [el for el in structured_elements if el.get('type', '').lower() in ['figure', 'image', 'picture']]
            if images:
                metadata["image_count"] = len(images)
                metadata["image_info"] = [{"type": img.get('type'), "bbox": img.get('bbox')} for img in images[:3]]
                metadata["image_citations"] = [
                    {
                        "page": img.get('page_number', 'Unknown'),
                        "element_id": img.get('element_id', ''),
                        "source": img.get('source_file', os.path.basename(file_path)),
                        "type": img.get('type', 'image')
                    } for img in images[:5]  # First 5 images
                ]
        
        # Create LlamaIndex document
        doc = Document(
            text=content,
            metadata=metadata,
            id_=f"doc_{hash(file_path)}"  # Unique ID for each document
        )
        
        documents.append(doc)
        print(f"    SUCCESS: Created document with {len(content)} chars and {len(metadata)} metadata fields")
    
    print(f"  SUCCESS: Created {len(documents)} enhanced documents")
    return documents

def ingest_documents_enhanced(chunking_strategy=None):
    """Enhanced document ingestion with multiple chunking strategies"""
    print("STARTING: Starting Enhanced LlamaIndex Document Ingestion")
    print("=" * 60)
    
    # Use the provided strategy or fall back to global
    effective_strategy = chunking_strategy or CHUNKING_STRATEGY
    print(f"STRATEGY: Using chunking strategy: {effective_strategy}")
    
    # Setup
    ensure_processed_directories()
    llm, embedding = setup_llm_and_embedding()
    storage_context = setup_storage(effective_strategy)
    
    # Get files
    files = get_files(RAW_DIR, SUPPORTED_EXTENSIONS)
    if not files:
        print(f"ERROR: No supported files found in {RAW_DIR}")
        print(f"   Looking for: {SUPPORTED_EXTENSIONS}")
        return None
    
    print(f"FOLDER: Found {len(files)} files to process:")
    for f in files:
        print(f"   - {os.path.basename(f)}")
    
    # Create documents with enhanced processing
    documents = create_enhanced_documents(files, chunking_strategy=effective_strategy)
    
    if not documents:
        print("ERROR: No documents created")
        return None
    
    # Create enhanced chunking pipeline
    pipeline = create_enhanced_chunking_pipeline(llm, embedding, effective_strategy)
    
    # Process documents through pipeline
    print(f"\nPROCESSING: Processing {len(documents)} documents through enhanced pipeline...")
    start_time = time.time()
    
    try:
        # Run ingestion pipeline with progress tracking
        print(f"  INFO: Running pipeline with {effective_strategy} strategy...")
        print(f"  INFO: This may take several minutes for large documents with metadata extraction...")
        nodes = pipeline.run(documents=documents)
        
        # SUCCESS: Apply citation enhancement to all chunks
        if hasattr(pipeline, '_citation_enhancer'):
            nodes = pipeline._citation_enhancer(nodes)
        
        processing_time = time.time() - start_time
        print(f"  SUCCESS: Pipeline processing completed in {processing_time:.2f}s")
        print(f"  INFO: Generated {len(nodes)} enhanced chunks with citations")
        
        # Create index and store
        print(f"\nSAVING: Creating enhanced vector index...")
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            embed_model=embedding
        )
        
        # Enhanced statistics
        print(f"\nCOMPLETE: Enhanced Ingestion Complete!")
        print(f"   Strategy: {CHUNKING_STRATEGY}")
        print(f"   Documents: {len(documents)}")
        print(f"   Chunks: {len(nodes)}")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Average chunks per doc: {len(nodes)/len(documents):.1f}")
        
        # SUCCESS: Feature analysis
        table_count = sum(1 for doc in documents if doc.metadata.get('has_tables', False))
        image_count = sum(1 for doc in documents if doc.metadata.get('has_images', False))
        print(f"   Documents with tables: {table_count}")
        print(f"   Documents with images: {image_count}")
        
        # Sample chunk preview with citation info
        if nodes:
            sample_chunk = nodes[0]
            print(f"\nCHECKING: Sample enhanced chunk with citations:")
            print(f"   Text: {sample_chunk.text[:100]}...")
            print(f"   Source: {sample_chunk.metadata.get('source_citation', 'Unknown')}")
            print(f"   Citation: {sample_chunk.metadata.get('citation_format', 'No citation')}")
            print(f"   Chunk ID: {sample_chunk.metadata.get('chunk_id', 'No ID')}")
            print(f"   All metadata keys: {list(sample_chunk.metadata.keys())}")
        
        return index
        
    except Exception as e:
        error_message = str(e)
        if "ReadTimeout" in error_message or "timeout" in error_message.lower():
            print(f"ERROR: Timeout occurred during processing!")
            print(f"  This can happen with large documents or slow LLM responses.")
            print(f"  Consider:")
            print(f"    1. Using a simpler strategy (e.g., 'simple' instead of 'contextual_rag')")
            print(f"    2. Reducing document size")
            print(f"    3. Checking Ollama server performance")
        else:
            print(f"ERROR: Enhanced ingestion failed: {e}")
        
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return None

def format_citation_from_metadata(node_metadata):
    """
    Format a proper citation from node metadata for use in RAG responses.
    
    Args:
        node_metadata: Dictionary containing chunk metadata
        
    Returns:
        str: Formatted citation string
    """
    source_doc = node_metadata.get('source_document', 'Unknown Document')
    doc_type = node_metadata.get('document_type', '')
    page_ref = node_metadata.get('page_reference', '')
    chunk_id = node_metadata.get('chunk_id', '')
    
    # Format citation based on available information
    citation_parts = [source_doc]
    
    if doc_type:
        citation_parts.append(f"({doc_type})")
    
    if page_ref:
        citation_parts.append(f"- {page_ref}")
    
    if chunk_id:
        citation_parts.append(f"[{chunk_id[:8]}]")  # Short chunk ID
    
    return " ".join(citation_parts)

def extract_citations_from_response(response_nodes):
    """
    Extract and format citations from RAG response nodes.
    
    Args:
        response_nodes: List of nodes returned from RAG query
        
    Returns:
        list: List of formatted citations
    """
    citations = []
    seen_sources = set()
    
    for node in response_nodes:
        if hasattr(node, 'metadata') and node.metadata:
            citation = format_citation_from_metadata(node.metadata)
            source = node.metadata.get('source_document', 'Unknown')
            
            # Avoid duplicate citations from same source
            if source not in seen_sources:
                citations.append(citation)
                seen_sources.add(source)
    
    return citations

def main():
    """Main entry point for enhanced ingestion"""
    
    # Add command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='LlamaIndex Enhanced Document Ingestion')
    parser.add_argument('--strategy', type=str, 
                       choices=['basic', 'structure_aware', 'semantic', 'hierarchical', 'contextual_rag'],
                       help='Chunking strategy to use (overrides CHUNKING_STRATEGY env var)')
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Chunk size (overrides CHUNK_SIZE env var)')
    parser.add_argument('--chunk-overlap', type=int, default=None,
                       help='Chunk overlap (overrides CHUNK_OVERLAP env var)')
    parser.add_argument('--enable-tables', action='store_true', default=None,
                       help='Enable table extraction (overrides ENABLE_TABLE_EXTRACTION env var)')
    parser.add_argument('--disable-tables', action='store_true', default=None,
                       help='Disable table extraction (overrides ENABLE_TABLE_EXTRACTION env var)')
    
    args = parser.parse_args()
    
    # Override environment variables with command line args
    global CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, ENABLE_TABLE_EXTRACTION
    
    if args.strategy:
        CHUNKING_STRATEGY = args.strategy.lower()
    if args.chunk_size is not None:
        CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap is not None:
        CHUNK_OVERLAP = args.chunk_overlap
    if args.enable_tables:
        ENABLE_TABLE_EXTRACTION = True
    elif args.disable_tables:
        ENABLE_TABLE_EXTRACTION = False
    
    print("FOCUS: LlamaIndex Enhanced Document Ingestion System")
    print("=" * 60)
    print(f"Raw directory: {RAW_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Chunking strategy: {CHUNKING_STRATEGY}")
    print(f"Table extraction: {'SUCCESS:' if ENABLE_TABLE_EXTRACTION else 'ERROR:'}")
    print(f"Image extraction: {'SUCCESS:' if ENABLE_IMAGE_EXTRACTION else 'ERROR:'}")
    print(f"Contextual RAG: {'SUCCESS:' if ENABLE_CONTEXTUAL_RAG else 'ERROR:'}")
    print("=" * 60)
    
    try:
        # Pass the updated strategy (either from CLI or global)
        strategy_to_use = args.strategy.lower() if args.strategy else CHUNKING_STRATEGY
        index = ingest_documents_enhanced(strategy_to_use)
        
        if index:
            print(f"\n Enhanced ingestion successful!")
            print(f"   Use the enhanced 'llamaindex_enhanced' table for querying")
            return True
        else:
            print(f"\nERROR: Enhanced ingestion failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Fatal error in enhanced ingestion: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
