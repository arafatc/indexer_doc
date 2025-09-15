#!/usr/bin/env python3
"""
Tests for enhanced LlamaIndex document ingestion with advanced chunking strategies.
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import enhanced ingestion functions
from ingestion.ingest_llamaindex_enhanced import (
    get_files,
    setup_llm_and_embedding,
    setup_storage,
    create_enhanced_chunking_pipeline,
    create_enhanced_documents,
    process_document_with_docling_enhanced,
    get_embedding_dim,
    ensure_processed_directories
)

class TestEnhancedIngestion:
    """Test class for enhanced ingestion functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.test_dir = tempfile.mkdtemp()
        self.test_files = []
        
        # Create test PDF content
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, 'wb') as f:
            f.write(b"Test PDF content")
        self.test_files.append(test_pdf)
        
        # Create test DOCX content
        test_docx = os.path.join(self.test_dir, "test.docx")
        with open(test_docx, 'wb') as f:
            f.write(b"Test DOCX content")
        self.test_files.append(test_docx)
    
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_get_files(self):
        """Test file discovery functionality"""
        files = get_files(self.test_dir, ['.pdf', '.docx'])
        assert len(files) == 2
        assert any('test.pdf' in f for f in files)
        assert any('test.docx' in f for f in files)
    
    def test_get_files_no_match(self):
        """Test file discovery with no matching extensions"""
        files = get_files(self.test_dir, ['.txt', '.md'])
        assert len(files) == 0
    
    def test_embedding_dimensions(self):
        """Test embedding dimension detection"""
        # Test OpenAI dimensions
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'openai', 'OPENAI_EMBEDDING_MODEL': 'text-embedding-ada-002'}):
            dim = get_embedding_dim()
            assert dim == 1536
        
        # Test Ollama dimensions
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'ollama', 'EMBEDDING_MODEL': 'nomic-embed-text:latest'}):
            dim = get_embedding_dim()
            assert dim == 768
    
    @patch('ingestion.ingest_llamaindex_enhanced.OpenAIEmbedding')
    @patch('ingestion.ingest_llamaindex_enhanced.OpenAI')
    def test_setup_llm_and_embedding_openai(self, mock_openai_llm, mock_openai_embed):
        """Test OpenAI LLM and embedding setup"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'EMBEDDING_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test-key'
        }):
            llm, embedding = setup_llm_and_embedding()
            mock_openai_llm.assert_called_once()
            mock_openai_embed.assert_called_once()
    
    @patch('ingestion.ingest_llamaindex_enhanced.OllamaEmbedding')
    @patch('ingestion.ingest_llamaindex_enhanced.Ollama')
    def test_setup_llm_and_embedding_ollama(self, mock_ollama_llm, mock_ollama_embed):
        """Test Ollama LLM and embedding setup"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'ollama',
            'EMBEDDING_PROVIDER': 'ollama'
        }):
            llm, embedding = setup_llm_and_embedding()
            mock_ollama_llm.assert_called_once()
            mock_ollama_embed.assert_called_once()
    
    def test_ensure_processed_directories(self):
        """Test creation of processed directories"""
        with patch('ingestion.ingest_llamaindex_enhanced.PROCESSED_DIR', self.test_dir):
            with patch('ingestion.ingest_llamaindex_enhanced.PROCESSED_MD_DIR', os.path.join(self.test_dir, 'md')):
                with patch('ingestion.ingest_llamaindex_enhanced.PROCESSED_JSON_DIR', os.path.join(self.test_dir, 'json')):
                    ensure_processed_directories()
                    assert os.path.exists(os.path.join(self.test_dir, 'md'))
                    assert os.path.exists(os.path.join(self.test_dir, 'json'))

class TestChunkingStrategies:
    """Test different chunking strategies"""
    
    def setup_method(self):
        """Setup mock LLM and embedding for tests"""
        self.mock_llm = MagicMock()
        self.mock_embedding = MagicMock()
    
    def test_basic_chunking_strategy(self):
        """Test basic chunking strategy"""
        with patch.dict(os.environ, {'CHUNKING_STRATEGY': 'basic'}):
            pipeline = create_enhanced_chunking_pipeline(self.mock_llm, self.mock_embedding)
            assert pipeline is not None
            # Should have SentenceSplitter + Embedding
            assert len(pipeline.transformations) == 2
    
    def test_structure_aware_chunking_strategy(self):
        """Test structure-aware chunking strategy"""
        with patch.dict(os.environ, {'CHUNKING_STRATEGY': 'structure_aware'}):
            pipeline = create_enhanced_chunking_pipeline(self.mock_llm, self.mock_embedding)
            assert pipeline is not None
            # Should have MarkdownParser + SentenceSplitter + Embedding
            assert len(pipeline.transformations) == 3
    
    def test_semantic_chunking_strategy(self):
        """Test semantic chunking strategy"""
        with patch.dict(os.environ, {'CHUNKING_STRATEGY': 'semantic'}):
            pipeline = create_enhanced_chunking_pipeline(self.mock_llm, self.mock_embedding)
            assert pipeline is not None
            # Should have SemanticSplitter + Embedding
            assert len(pipeline.transformations) == 2
    
    def test_hierarchical_chunking_strategy(self):
        """Test hierarchical chunking strategy"""
        with patch.dict(os.environ, {'CHUNKING_STRATEGY': 'hierarchical'}):
            pipeline = create_enhanced_chunking_pipeline(self.mock_llm, self.mock_embedding)
            assert pipeline is not None
            # Should have HierarchicalParser + Embedding
            assert len(pipeline.transformations) == 2
    
    def test_contextual_rag_chunking_strategy(self):
        """Test contextual RAG chunking strategy"""
        with patch.dict(os.environ, {'CHUNKING_STRATEGY': 'contextual_rag'}):
            pipeline = create_enhanced_chunking_pipeline(self.mock_llm, self.mock_embedding)
            assert pipeline is not None
            # Should have SentenceSplitter + 4 extractors + Embedding = 6 total
            assert len(pipeline.transformations) == 6
    
    def test_invalid_chunking_strategy(self):
        """Test invalid chunking strategy raises error"""
        with patch.dict(os.environ, {'CHUNKING_STRATEGY': 'invalid_strategy'}):
            with pytest.raises(ValueError, match="Unknown chunking strategy"):
                create_enhanced_chunking_pipeline(self.mock_llm, self.mock_embedding)

class TestDocumentProcessing:
    """Test document processing functionality"""
    
    def setup_method(self):
        """Setup test files"""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test files"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('ingestion.ingest_llamaindex_enhanced.DocumentConverter')
    def test_process_document_with_docling_enhanced(self, mock_converter_class):
        """Test enhanced document processing with docling"""
        # Create test file
        test_file = os.path.join(self.test_dir, "test.pdf")
        with open(test_file, 'wb') as f:
            f.write(b"Test PDF content")
        
        # Mock converter and result
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        mock_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.export_to_text.return_value = "Extracted text content"
        mock_doc.main_text = [
            MagicMock(text="Sample text", label="paragraph", bbox=[0, 0, 100, 20]),
            MagicMock(text="Table data", label="table", bbox=[0, 20, 100, 40])
        ]
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        # Test processing
        with patch.dict(os.environ, {
            'ENABLE_TABLE_EXTRACTION': 'true',
            'ENABLE_IMAGE_EXTRACTION': 'false'
        }):
            with patch('ingestion.ingest_llamaindex_enhanced.save_document_as_markdown'):
                with patch('ingestion.ingest_llamaindex_enhanced.save_document_as_json'):
                    content, elements = process_document_with_docling_enhanced(test_file)
        
        assert content == "Extracted text content"
        assert len(elements) == 2
        assert elements[0]['text'] == "Sample text"
        assert elements[1]['text'] == "Table data"
        assert elements[1]['type'] == "table"
    
    @patch('ingestion.ingest_llamaindex_enhanced.process_document_with_docling_enhanced')
    def test_create_enhanced_documents(self, mock_process_doc):
        """Test enhanced document creation"""
        # Setup mock
        test_file = os.path.join(self.test_dir, "test.pdf")
        with open(test_file, 'wb') as f:
            f.write(b"Test content")
        
        mock_process_doc.return_value = (
            "Extracted text content",
            [
                {'text': 'Table data', 'type': 'table', 'bbox': [0, 0, 100, 20]},
                {'text': 'Image caption', 'type': 'figure', 'bbox': [0, 20, 100, 40]}
            ]
        )
        
        # Test document creation
        with patch.dict(os.environ, {
            'CHUNKING_STRATEGY': 'basic',
            'ENABLE_TABLE_EXTRACTION': 'true',
            'ENABLE_IMAGE_EXTRACTION': 'true'
        }):
            documents = create_enhanced_documents([test_file])
        
        assert len(documents) == 1
        doc = documents[0]
        
        # Check basic metadata
        assert doc.metadata['file_name'] == 'test.pdf'
        assert doc.metadata['chunking_strategy'] == 'basic'
        assert doc.metadata['total_structured_elements'] == 2
        
        # Check enhanced metadata
        assert doc.metadata['has_tables'] == True
        assert doc.metadata['has_images'] == True
        assert doc.metadata['table_count'] == 1
        assert doc.metadata['image_count'] == 1
        assert 'table_preview' in doc.metadata
        assert 'image_info' in doc.metadata

class TestIntegration:
    """Integration tests for the complete enhanced pipeline"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('ingestion.ingest_llamaindex_enhanced.setup_storage')
    @patch('ingestion.ingest_llamaindex_enhanced.setup_llm_and_embedding')  
    @patch('ingestion.ingest_llamaindex_enhanced.process_document_with_docling_enhanced')
    @patch('ingestion.ingest_llamaindex_enhanced.VectorStoreIndex')
    def test_full_enhanced_ingestion_pipeline(self, mock_index, mock_process_doc, mock_setup_llm, mock_setup_storage):
        """Test the complete enhanced ingestion pipeline"""
        # Setup mocks
        mock_llm = MagicMock()
        mock_embedding = MagicMock()
        mock_setup_llm.return_value = (mock_llm, mock_embedding)
        
        mock_storage_context = MagicMock()
        mock_setup_storage.return_value = mock_storage_context
        
        mock_process_doc.return_value = (
            "Sample document content for testing the enhanced pipeline",
            [{'text': 'Table content', 'type': 'table'}]
        )
        
        mock_index_instance = MagicMock()
        mock_index.return_value = mock_index_instance
        
        # Create test file
        test_file = os.path.join(self.test_dir, "test.pdf")
        with open(test_file, 'wb') as f:
            f.write(b"Test content")
        
        # Test ingestion with different strategies
        strategies = ['basic', 'structure_aware', 'semantic', 'hierarchical']
        
        for strategy in strategies:
            with patch.dict(os.environ, {
                'CHUNKING_STRATEGY': strategy,
                'ENABLE_TABLE_EXTRACTION': 'true'
            }):
                with patch('ingestion.ingest_llamaindex_enhanced.RAW_DIR', self.test_dir):
                    with patch('ingestion.ingest_llamaindex_enhanced.ensure_processed_directories'):
                        from ingestion.ingest_llamaindex_enhanced import ingest_documents_enhanced
                        
                        result = ingest_documents_enhanced()
                        assert result is not None
                        mock_index.assert_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
