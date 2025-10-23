"""
cached_analysis.py
Quick analysis and visualization using cached results
Allows instant review without re-solving models
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

import os
from pathlib import Path

# Ensure we're in the correct directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from result_cache import get_cache, create_parameter_dict

def load_baseline_results():
    """Load baseline model results from cache"""
    cache = get_cache()
    
    # Define baseline parameters
    baseline_params = create_parameter_dict(
        Kai_n=2.34/3*0.25,
        Kai_e=2.34*0.5/3*0.25,
        e_l=1/6*3*2,
        e_h=1/3*3*2,
        lambda_param=0.4,
        r=0,
        w=1,
        discount=0.95,
        delta_h=0,
        sigma_z=0.001,
        sigma_y=0.001,
        delta_corr=0.5,
        no_hc_model=False
    )
    
    # Try to load full model results
    full_results = cache.load_model_results('full_model', baseline_params)
    
    # Try to load no-HC model results
    no_hc_params = baseline_params.copy()
    no_hc_params['no_hc_model'] = True
    no_hc_results = cache.load_model_results('no_hc_model', no_hc_params)
    
    return full_results, no_hc_results, baseline_params

def analyze_cached_results(full_results, no_hc_results, show_plots=True):
    """Analyze cached results and generate insights"""
    
    if full_results is None or no_hc_results is None:
        print("Error: Could not load cached results. Please run models first.")
        return None
    
    print("="*80)
    print("CACHED RESULTS ANALYSIS")
    print("="*80)
    
    # Extract key data
    choice_2d = full_results['choice_2d']
    saving_full = full_results['optimal_saving_2d']
    saving_no_hc = no_hc_results['optimal_saving_2d']
    zgrid = full_results['zgrid']
    hgrid = full_results['hgrid']
    
    # Compute saving difference
    saving_diff = saving_full - saving_no_hc
    
    # Identify education regions
    el_mask = np.logical_or(
        np.abs(choice_2d - 2) < 0.1,  # Choice 2: n=0, e=e_l
        np.abs(choice_2d - 5) < 0.1   # Choice 5: n=1, e=e_l
    )
    
    eh_mask = np.abs(choice_2d - 3) < 0.1  # Choice 3: n=0, e=e_h
    
    # Extract regional differences
    el_space_diff = saving_diff[el_mask]
    eh_space_diff = saving_diff[eh_mask]
    
    # Print summary statistics
    print("\n1. OVERALL STATISTICS:")
    print("-" * 40)
    print(f"Full Model - Average Saving: {np.mean(saving_full):.4f}")
    print(f"No-HC Model - Average Saving: {np.mean(saving_no_hc):.4f}")
    print(f"Average Difference: {np.mean(saving_diff):.4f}")
    print(f"Relative Difference: {(np.mean(saving_diff)/np.mean(saving_no_hc))*100:.2f}%")
    
    print("\n2. EDUCATION REGION ANALYSIS:")
    print("-" * 40)
    print(f"e_l regions ({len(el_space_diff)} points):")
    print(f"  Mean difference: {np.mean(el_space_diff):.4f}")
    print(f"  Std deviation: {np.std(el_space_diff):.4f}")
    print(f"  Range: [{np.min(el_space_diff):.4f}, {np.max(el_space_diff):.4f}]")
    
    print(f"e_h regions ({len(eh_space_diff)} points):")
    print(f"  Mean difference: {np.mean(eh_space_diff):.4f}")
    print(f"  Std deviation: {np.std(eh_space_diff):.4f}")
    print(f"  Range: [{np.min(eh_space_diff):.4f}, {np.max(eh_space_diff):.4f}]")
    
    # Regional analysis
    print("\n3. REGIONAL BREAKDOWN:")
    print("-" * 40)
    Z, H = np.meshgrid(zgrid, hgrid, indexing='ij')
    
    low_h_mask = H < 2.0
    mid_h_mask = (H >= 2.0) & (H <= 4.5)
    high_h_mask = H > 4.5
    
    regions = [
        ("Low HC (h<2)", low_h_mask),
        ("Mid HC (2<=h<=4.5)", mid_h_mask),
        ("High HC (h>4.5)", high_h_mask)
    ]
    
    for region_name, mask in regions:
        if np.sum(mask) > 0:
            region_diff = saving_diff[mask]
            print(f"{region_name}: Mean = {np.mean(region_diff):.4f}, "
                  f"Std = {np.std(region_diff):.4f}, Points = {np.sum(mask)}")
    
    # Create visualization if requested
    if show_plots:
        create_cached_visualization(full_results, no_hc_results, saving_diff, 
                                  el_mask, eh_mask, zgrid, hgrid)
    
    return {
        'saving_diff': saving_diff,
        'el_mask': el_mask,
        'eh_mask': eh_mask,
        'el_space_diff': el_space_diff,
        'eh_space_diff': eh_space_diff,
        'full_results': full_results,
        'no_hc_results': no_hc_results
    }

def create_cached_visualization(full_results, no_hc_results, saving_diff, 
                               el_mask, eh_mask, zgrid, hgrid):
    """Create visualization from cached results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    Z, H = np.meshgrid(zgrid, hgrid, indexing='ij')
    choice_2d = full_results['choice_2d']
    saving_full = full_results['optimal_saving_2d']
    saving_no_hc = no_hc_results['optimal_saving_2d']
    
    # 1. Education regions
    ax1 = axes[0, 0]
    im1 = ax1.contourf(H.T, Z.T, choice_2d.T, levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
                      colors=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum'], alpha=0.7)
    
    el_z, el_h = Z[el_mask], H[el_mask]
    eh_z, eh_h = Z[eh_mask], H[eh_mask]
    ax1.scatter(el_h, el_z, c='green', s=10, alpha=0.8, label='e_l regions')
    ax1.scatter(eh_h, eh_z, c='red', s=10, alpha=0.8, label='e_h regions')
    
    ax1.axvline(x=2.0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Human Capital (h)')
    ax1.set_ylabel('Productivity (z)')
    ax1.set_title('Education Choice Regions')
    ax1.legend()
    ax1.set_ylim([0, 2])
    
    # 2. Saving difference heatmap
    ax2 = axes[0, 1]
    im2 = ax2.contourf(H.T, Z.T, saving_diff.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=ax2, label='Saving Difference')
    ax2.set_xlabel('Human Capital (h)')
    ax2.set_ylabel('Productivity (z)')
    ax2.set_title('Saving Difference (Full - No-HC)')
    ax2.set_ylim([0, 2])
    
    # 3. Full model saving
    ax3 = axes[0, 2]
    im3 = ax3.contourf(H.T, Z.T, saving_full.T, levels=20, cmap='viridis')
    plt.colorbar(im3, ax=ax3, label='Optimal Saving')
    ax3.set_xlabel('Human Capital (h)')
    ax3.set_ylabel('Productivity (z)')
    ax3.set_title('Full Model Optimal Saving')
    ax3.set_ylim([0, 2])
    
    # 4. el_space comparison
    ax4 = axes[1, 0]
    if np.sum(el_mask) > 0:
        el_full = saving_full[el_mask]
        el_no_hc = saving_no_hc[el_mask]
        ax4.scatter(el_no_hc, el_full, alpha=0.6, s=20, color='green')
        min_val = min(np.min(el_no_hc), np.min(el_full))
        max_val = max(np.max(el_no_hc), np.max(el_full))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax4.set_xlabel('No-HC Model Saving')
    ax4.set_ylabel('Full Model Saving')
    ax4.set_title('e_l Regions Comparison')
    ax4.grid(True, alpha=0.3)
    
    # 5. eh_space comparison
    ax5 = axes[1, 1]
    if np.sum(eh_mask) > 0:
        eh_full = saving_full[eh_mask]
        eh_no_hc = saving_no_hc[eh_mask]
        ax5.scatter(eh_no_hc, eh_full, alpha=0.6, s=20, color='red')
        min_val = min(np.min(eh_no_hc), np.min(eh_full))
        max_val = max(np.max(eh_no_hc), np.max(eh_full))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax5.set_xlabel('No-HC Model Saving')
    ax5.set_ylabel('Full Model Saving')
    ax5.set_title('e_h Regions Comparison')
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution comparison
    ax6 = axes[1, 2]
    ax6.hist(saving_diff.flatten(), bins=40, alpha=0.7, color='blue', label='All regions', density=True)
    if np.sum(el_mask) > 0:
        ax6.hist(saving_diff[el_mask], bins=30, alpha=0.7, color='green', 
                label='e_l regions', density=True)
    if np.sum(eh_mask) > 0:
        ax6.hist(saving_diff[eh_mask], bins=20, alpha=0.7, color='red', 
                label='e_h regions', density=True)
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    ax6.set_xlabel('Saving Difference')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution of Differences')
    ax6.legend()
    
    plt.suptitle('Cached Results Analysis: Full Model vs No-HC Model', fontsize=14)
    plt.tight_layout()
    plt.show()

def list_available_cache():
    """List available cached results"""
    cache = get_cache()
    results = cache.list_cached_results()
    
    if not results:
        print("No cached results found.")
        print("Please run the models first to generate cached results.")
        return
    
    print("\nAvailable cached results:")
    print("-" * 60)
    
    for result in results:
        from datetime import datetime
        timestamp = datetime.fromisoformat(result['timestamp'])
        size_mb = result['file_size'] / (1024 * 1024)
        
        print(f"{result['model_type']:<15} {result['cache_key']:<12} "
              f"{timestamp.strftime('%Y-%m-%d %H:%M'):<16} {size_mb:.1f} MB")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Analyze cached model results")
    parser.add_argument('--no-plots', action='store_true', help='Skip visualization')
    parser.add_argument('--list', action='store_true', help='List available cached results')
    parser.add_argument('--cache-key', type=str, help='Use specific cache key')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_cache()
        return
    
    print("Loading cached results...")
    
    try:
        full_results, no_hc_results, params = load_baseline_results()
        
        if full_results is None or no_hc_results is None:
            print("\nNo cached results found for baseline parameters.")
            print("Available cached results:")
            list_available_cache()
            print("\nTo generate results, run: python TwoPeriod.py")
            return
        
        # Analyze results
        analysis = analyze_cached_results(full_results, no_hc_results, 
                                        show_plots=not args.no_plots)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - Results loaded from cache!")
        print("="*80)
        
    except Exception as e:
        print(f"Error analyzing cached results: {e}")
        print("Please ensure cached results exist and are valid.")

if __name__ == "__main__":
    main()
