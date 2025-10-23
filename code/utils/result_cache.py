"""
result_cache.py
Core caching utilities for saving and loading model results
Handles parameter verification and metadata management
"""

import pickle
import numpy as np
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
import gzip

class ResultCache:
    """Main class for handling model result caching"""
    
    def __init__(self, cache_dir="results/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir.parent / "metadata.json"
        self.index_file = self.cache_dir.parent / "cache_index.json"
        
    def get_cache_key(self, parameters):
        """Generate unique cache key from parameters"""
        # Sort parameters for consistent hashing
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    def save_model_results(self, results, model_type, parameters, compress=True):
        """
        Save model results with metadata
        
        Args:
            results: Model results dictionary
            model_type: 'full_model' or 'no_hc_model' or 'comparison'
            parameters: Dictionary of model parameters
            compress: Whether to compress the saved file
        """
        # Generate cache key
        cache_key = self.get_cache_key(parameters)
        
        # Create filename
        filename = f"{model_type}_{cache_key}.pkl"
        if compress:
            filename += ".gz"
        
        filepath = self.cache_dir / filename
        
        # Prepare data to save
        cache_data = {
            'results': results,
            'parameters': parameters,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'cache_key': cache_key,
            'version': '1.0'
        }
        
        # Save data
        if compress:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
        
        # Update metadata and index
        self._update_metadata(cache_key, model_type, parameters, filename)
        
        print(f"Saved {model_type} results to cache: {filename}")
        return cache_key
    
    def load_model_results(self, model_type, parameters=None, cache_key=None):
        """
        Load model results from cache
        
        Args:
            model_type: 'full_model' or 'no_hc_model' or 'comparison'
            parameters: Parameters to match (if cache_key not provided)
            cache_key: Specific cache key to load
            
        Returns:
            results dictionary or None if not found
        """
        if cache_key is None:
            if parameters is None:
                raise ValueError("Must provide either parameters or cache_key")
            cache_key = self.get_cache_key(parameters)
        
        # Try both compressed and uncompressed versions
        for compress in [True, False]:
            filename = f"{model_type}_{cache_key}.pkl"
            if compress:
                filename += ".gz"
            
            filepath = self.cache_dir / filename
            
            if filepath.exists():
                try:
                    if compress:
                        with gzip.open(filepath, 'rb') as f:
                            cache_data = pickle.load(f)
                    else:
                        with open(filepath, 'rb') as f:
                            cache_data = pickle.load(f)
                    
                    # Verify parameters if provided
                    if parameters is not None:
                        if not self.check_cache_validity(cache_data['parameters'], parameters):
                            print(f"Warning: Parameter mismatch for cached {model_type}")
                            return None
                    
                    print(f"Loaded {model_type} results from cache: {filename}")
                    return cache_data['results']
                    
                except Exception as e:
                    print(f"Error loading cache file {filename}: {e}")
                    continue
        
        return None
    
    def check_cache_validity(self, cached_params, current_params, tolerance=1e-10):
        """
        Check if cached parameters match current parameters
        
        Args:
            cached_params: Parameters from cached results
            current_params: Current model parameters
            tolerance: Numerical tolerance for float comparison
            
        Returns:
            True if parameters match, False otherwise
        """
        # Check if all keys match
        if set(cached_params.keys()) != set(current_params.keys()):
            return False
        
        # Check each parameter value
        for key in cached_params:
            cached_val = cached_params[key]
            current_val = current_params[key]
            
            # Handle different types
            if isinstance(cached_val, (int, float)) and isinstance(current_val, (int, float)):
                if abs(cached_val - current_val) > tolerance:
                    return False
            else:
                if cached_val != current_val:
                    return False
        
        return True
    
    def _update_metadata(self, cache_key, model_type, parameters, filename):
        """Update metadata and index files"""
        # Load existing metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Add new entry
        metadata[cache_key] = {
            'model_type': model_type,
            'parameters': parameters,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'file_size': os.path.getsize(self.cache_dir / filename)
        }
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update index
        self._update_index()
    
    def _update_index(self):
        """Update cache index for quick lookups"""
        if not self.metadata_file.exists():
            return
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create index by model type
        index = {}
        for cache_key, info in metadata.items():
            model_type = info['model_type']
            if model_type not in index:
                index[model_type] = []
            
            index[model_type].append({
                'cache_key': cache_key,
                'timestamp': info['timestamp'],
                'parameters': info['parameters']
            })
        
        # Save index
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def list_cached_results(self, model_type=None):
        """
        List available cached results
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of cached result information
        """
        if not self.metadata_file.exists():
            return []
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        results = []
        for cache_key, info in metadata.items():
            if model_type is None or info['model_type'] == model_type:
                results.append({
                    'cache_key': cache_key,
                    'model_type': info['model_type'],
                    'timestamp': info['timestamp'],
                    'file_size': info['file_size'],
                    'parameters': info['parameters']
                })
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results
    
    def clear_cache(self, model_type=None, cache_key=None, older_than_days=None):
        """
        Clear cached results
        
        Args:
            model_type: Clear specific model type (optional)
            cache_key: Clear specific cache key (optional)
            older_than_days: Clear results older than N days (optional)
        """
        if not self.metadata_file.exists():
            return
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        keys_to_remove = []
        
        for key, info in metadata.items():
            should_remove = False
            
            # Check filters
            if cache_key and key == cache_key:
                should_remove = True
            elif model_type and info['model_type'] == model_type:
                should_remove = True
            elif older_than_days:
                timestamp = datetime.fromisoformat(info['timestamp'])
                age_days = (datetime.now() - timestamp).days
                if age_days > older_than_days:
                    should_remove = True
            elif cache_key is None and model_type is None and older_than_days is None:
                # Clear all if no filters specified
                should_remove = True
            
            if should_remove:
                keys_to_remove.append(key)
                # Remove file
                filepath = self.cache_dir / info['filename']
                if filepath.exists():
                    filepath.unlink()
                    print(f"Removed cache file: {info['filename']}")
        
        # Update metadata
        for key in keys_to_remove:
            del metadata[key]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update index
        self._update_index()
        
        print(f"Cleared {len(keys_to_remove)} cached results")
    
    def get_cache_stats(self):
        """Get cache usage statistics"""
        if not self.metadata_file.exists():
            return {'total_files': 0, 'total_size': 0, 'by_type': {}}
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        stats = {
            'total_files': len(metadata),
            'total_size': sum(info['file_size'] for info in metadata.values()),
            'by_type': {}
        }
        
        # Group by model type
        for info in metadata.values():
            model_type = info['model_type']
            if model_type not in stats['by_type']:
                stats['by_type'][model_type] = {'count': 0, 'size': 0}
            
            stats['by_type'][model_type]['count'] += 1
            stats['by_type'][model_type]['size'] += info['file_size']
        
        return stats


def create_parameter_dict(Kai_n, Kai_e, e_l, e_h, lambda_param, r, w, discount, 
                         delta_h=0, sigma_z=0.001, sigma_y=0.001, delta_corr=0.5, 
                         no_hc_model=False):
    """
    Create standardized parameter dictionary for caching
    
    Args:
        All model parameters
        
    Returns:
        Dictionary with all parameters for cache key generation
    """
    return {
        'Kai_n': float(Kai_n),
        'Kai_e': float(Kai_e),
        'e_l': float(e_l),
        'e_h': float(e_h),
        'lambda_param': float(lambda_param),
        'r': float(r),
        'w': float(w),
        'discount': float(discount),
        'delta_h': float(delta_h),
        'sigma_z': float(sigma_z),
        'sigma_y': float(sigma_y),
        'delta_corr': float(delta_corr),
        'no_hc_model': bool(no_hc_model)
    }


# Global cache instance
_cache = None

def get_cache():
    """Get global cache instance"""
    global _cache
    if _cache is None:
        _cache = ResultCache()
    return _cache
