"""
cache_manager.py
Cache management utilities for listing, comparing, and managing cached results
"""

import json
from datetime import datetime
from pathlib import Path
import argparse
from result_cache import get_cache

class CacheManager:
    """Manager for cache operations and utilities"""
    
    def __init__(self):
        self.cache = get_cache()
    
    def list_results(self, model_type=None, detailed=False):
        """
        List cached results with optional filtering
        
        Args:
            model_type: Filter by model type
            detailed: Show detailed parameter information
        """
        results = self.cache.list_cached_results(model_type)
        
        if not results:
            print("No cached results found.")
            return
        
        print(f"\n{'='*80}")
        print(f"CACHED RESULTS ({len(results)} found)")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            timestamp = datetime.fromisoformat(result['timestamp'])
            size_mb = result['file_size'] / (1024 * 1024)
            
            print(f"\n{i}. {result['model_type'].upper()}")
            print(f"   Cache Key: {result['cache_key']}")
            print(f"   Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   File Size: {size_mb:.2f} MB")
            
            if detailed:
                print(f"   Parameters:")
                for key, value in result['parameters'].items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.6f}")
                    else:
                        print(f"     {key}: {value}")
    
    def compare_parameters(self, cache_key1, cache_key2):
        """Compare parameters between two cached results"""
        results = self.cache.list_cached_results()
        
        # Find results by cache keys
        result1 = next((r for r in results if r['cache_key'] == cache_key1), None)
        result2 = next((r for r in results if r['cache_key'] == cache_key2), None)
        
        if not result1:
            print(f"Cache key {cache_key1} not found")
            return
        if not result2:
            print(f"Cache key {cache_key2} not found")
            return
        
        params1 = result1['parameters']
        params2 = result2['parameters']
        
        print(f"\n{'='*80}")
        print(f"PARAMETER COMPARISON")
        print(f"{'='*80}")
        print(f"Result 1: {result1['model_type']} ({cache_key1})")
        print(f"Result 2: {result2['model_type']} ({cache_key2})")
        print(f"{'='*80}")
        
        # Get all parameter keys
        all_keys = set(params1.keys()) | set(params2.keys())
        
        differences = []
        
        print(f"{'Parameter':<15} {'Result 1':<15} {'Result 2':<15} {'Match':<10}")
        print(f"{'-'*60}")
        
        for key in sorted(all_keys):
            val1 = params1.get(key, 'MISSING')
            val2 = params2.get(key, 'MISSING')
            
            # Format values
            if isinstance(val1, float):
                val1_str = f"{val1:.6f}"
            else:
                val1_str = str(val1)
            
            if isinstance(val2, float):
                val2_str = f"{val2:.6f}"
            else:
                val2_str = str(val2)
            
            # Check if they match
            if val1 == val2:
                match = "✓"
            else:
                match = "✗"
                differences.append(key)
            
            print(f"{key:<15} {val1_str:<15} {val2_str:<15} {match:<10}")
        
        if differences:
            print(f"\nDifferences found in: {', '.join(differences)}")
        else:
            print(f"\nAll parameters match!")
    
    def show_cache_stats(self):
        """Display cache usage statistics"""
        stats = self.cache.get_cache_stats()
        
        print(f"\n{'='*60}")
        print(f"CACHE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Files: {stats['total_files']}")
        print(f"Total Size: {stats['total_size'] / (1024*1024):.2f} MB")
        
        if stats['by_type']:
            print(f"\nBy Model Type:")
            print(f"{'Type':<20} {'Files':<10} {'Size (MB)':<15}")
            print(f"{'-'*45}")
            
            for model_type, type_stats in stats['by_type'].items():
                size_mb = type_stats['size'] / (1024*1024)
                print(f"{model_type:<20} {type_stats['count']:<10} {size_mb:<15.2f}")
    
    def clear_cache_interactive(self):
        """Interactive cache clearing"""
        print(f"\n{'='*60}")
        print(f"CACHE CLEARING OPTIONS")
        print(f"{'='*60}")
        print("1. Clear all cached results")
        print("2. Clear by model type")
        print("3. Clear specific cache key")
        print("4. Clear results older than N days")
        print("5. Cancel")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            confirm = input("Clear ALL cached results? (yes/no): ").strip().lower()
            if confirm == 'yes':
                self.cache.clear_cache()
            else:
                print("Cancelled.")
        
        elif choice == '2':
            print("\nAvailable model types:")
            results = self.cache.list_cached_results()
            types = set(r['model_type'] for r in results)
            for i, model_type in enumerate(sorted(types), 1):
                print(f"{i}. {model_type}")
            
            type_choice = input("Enter model type: ").strip()
            if type_choice in types:
                confirm = input(f"Clear all {type_choice} results? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.cache.clear_cache(model_type=type_choice)
                else:
                    print("Cancelled.")
            else:
                print("Invalid model type.")
        
        elif choice == '3':
            cache_key = input("Enter cache key: ").strip()
            if cache_key:
                confirm = input(f"Clear cache key {cache_key}? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.cache.clear_cache(cache_key=cache_key)
                else:
                    print("Cancelled.")
            else:
                print("Invalid cache key.")
        
        elif choice == '4':
            try:
                days = int(input("Clear results older than how many days? "))
                confirm = input(f"Clear results older than {days} days? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.cache.clear_cache(older_than_days=days)
                else:
                    print("Cancelled.")
            except ValueError:
                print("Invalid number of days.")
        
        elif choice == '5':
            print("Cancelled.")
        
        else:
            print("Invalid choice.")
    
    def find_matching_results(self, parameters):
        """Find cached results matching given parameters"""
        cache_key = self.cache.get_cache_key(parameters)
        results = self.cache.list_cached_results()
        
        matching = [r for r in results if r['cache_key'] == cache_key]
        
        if matching:
            print(f"\nFound {len(matching)} matching results for parameters:")
            for result in matching:
                timestamp = datetime.fromisoformat(result['timestamp'])
                print(f"  {result['model_type']} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No matching cached results found for these parameters.")
        
        return matching


def main():
    """Command-line interface for cache management"""
    parser = argparse.ArgumentParser(description="Manage model result cache")
    parser.add_argument('--list', action='store_true', help='List cached results')
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    parser.add_argument('--type', type=str, help='Filter by model type')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--clear', action='store_true', help='Interactive cache clearing')
    parser.add_argument('--compare', nargs=2, metavar=('KEY1', 'KEY2'), 
                       help='Compare parameters between two cache keys')
    
    args = parser.parse_args()
    
    manager = CacheManager()
    
    if args.list:
        manager.list_results(model_type=args.type, detailed=args.detailed)
    elif args.stats:
        manager.show_cache_stats()
    elif args.clear:
        manager.clear_cache_interactive()
    elif args.compare:
        manager.compare_parameters(args.compare[0], args.compare[1])
    else:
        # Interactive mode
        print(f"\n{'='*60}")
        print(f"CACHE MANAGER - INTERACTIVE MODE")
        print(f"{'='*60}")
        print("1. List cached results")
        print("2. Show cache statistics")
        print("3. Clear cache")
        print("4. Compare parameters")
        print("5. Exit")
        
        while True:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                detailed = input("Show detailed info? (y/n): ").strip().lower() == 'y'
                model_type = input("Filter by model type (or press Enter for all): ").strip()
                if not model_type:
                    model_type = None
                manager.list_results(model_type=model_type, detailed=detailed)
            
            elif choice == '2':
                manager.show_cache_stats()
            
            elif choice == '3':
                manager.clear_cache_interactive()
            
            elif choice == '4':
                key1 = input("Enter first cache key: ").strip()
                key2 = input("Enter second cache key: ").strip()
                if key1 and key2:
                    manager.compare_parameters(key1, key2)
                else:
                    print("Invalid cache keys.")
            
            elif choice == '5':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()
