import pandas as pd
import requests
import chardet
from pathlib import Path
from typing import Tuple, Optional
import logging

class DataIngestionAgent:
    """Agent responsible for data ingestion from CSV files or URLs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_csv(self, source: str, is_url: bool = False) -> Tuple[pd.DataFrame, dict]:
        """
        Load CSV data from file path or URL
        
        Args:
            source: File path or URL to CSV
            is_url: Whether source is a URL
            
        Returns:
            Tuple of (dataframe, metadata)
        """
        metadata = {
            'source': source,
            'is_url': is_url,
            'encoding': None,
            'rows': 0,
            'columns': 0,
            'success': False,
            'error': None
        }
        
        try:
            if is_url:
                df, encoding = self._load_from_url(source)
            else:
                df, encoding = self._load_from_file(source)
                
            metadata.update({
                'encoding': encoding,
                'rows': len(df),
                'columns': len(df.columns),
                'success': True,
                'column_names': list(df.columns)
            })
            
            self.logger.info(f"✅ Successfully loaded CSV: {metadata['rows']} rows, {metadata['columns']} columns")
            return df, metadata
            
        except Exception as e:
            metadata['error'] = str(e)
            self.logger.error(f"❌ Failed to load CSV: {e}")
            raise
    
    def _load_from_url(self, url: str) -> Tuple[pd.DataFrame, str]:
        """Load CSV from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Detect encoding
        detected = chardet.detect(response.content)
        encoding = detected['encoding'] or 'utf-8'
        
        # Try to read with detected encoding
        try:
            df = pd.read_csv(url, encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            df = pd.read_csv(url, encoding='latin-1')
            encoding = 'latin-1'
            
        return df, encoding
    
    def _load_from_file(self, file_path: str) -> Tuple[pd.DataFrame, str]:
        """Load CSV from local file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Detect encoding
        with open(path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] or 'utf-8'
        
        # Try to read with detected encoding
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1
            df = pd.read_csv(file_path, encoding='latin-1')
            encoding = 'latin-1'
            
        return df, encoding