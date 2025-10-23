"""
quick_comparison.py
Focused analysis of education regions and saving comparisons
Simplified version for immediate analysis
Now supports result caching for faster repeated analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from TwoPeriod import run_both_models
import seaborn as sns
import argparse
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

try:
    from result_cache import get_cache, create_parameter_dict
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    print("Warning: Caching not available. Install required dependencies or check utils path.")

def analyze_education_regions_and_savings(use_cache=True, save_cache=True, force_recompute=False):
    """
    Quick analysis of education regions and saving differences
    
    Args:
        use_cache: Try to load results from cache first
        save_cache: Save results to cache after computation
        force_recompute: Force recomputation even if cache exists
    """
    
    results_full = None
    results_no_hc = None
    
    # Define baseline parameters for caching
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
    ) if CACHING_AVAILABLE else None
    
    # Try to load from cache first
    if use_cache and CACHING_AVAILABLE and not force_recompute:
        print("Checking cache for existing results...")
        cache = get_cache()
        
        # Try to load full model results
        results_full = cache.load_model_results('full_model', baseline_params)
        
        # Try to load no-HC model results
        no_hc_params = baseline_params.copy()
        no_hc_params['no_hc_model'] = True
        results_no_hc = cache.load_model_results('no_hc_model', no_hc_params)
        
        if results_full is not None and results_no_hc is not None:
            print("✓ Successfully loaded results from cache!")
        else:
            print("Cache miss - will compute results...")
    
    # Compute results if not loaded from cache
    if results_full is None or results_no_hc is None:
        print("Running both models for comparison...")
        results_full, results_no_hc = run_both_models()
        
        # Save to cache if requested
        if save_cache and CACHING_AVAILABLE:
            print("Saving results to cache...")
            cache = get_cache()
            
            try:
                cache.save_model_results(results_full, 'full_model', baseline_params)
                no_hc_params = baseline_params.copy()
                no_hc_params['no_hc_model'] = True
                cache.save_model_results(results_no_hc, 'no_hc_model', no_hc_params)
                print("✓ Results saved to cache!")
            except Exception as e:
                print(f"Warning: Could not save to cache: {e}")
    
    # Extract key data
    choice_2d = results_full['choice_2d']
    saving_full = results_full['optimal_saving_2d']
    saving_no_hc = results_no_hc['optimal_saving_2d']
    zgrid = results_full['zgrid']
    hgrid = results_full['hgrid']
    
    # Compute saving difference
    saving_diff = saving_full - saving_no_hc
    
    # Identify education regions
    print("\nIdentifying education choice regions...")
    
    # el_space: where choice is 2 (n=0, e=e_l) or 5 (n=1, e=e_l)
    el_mask = np.logical_or(
        np.abs(choice_2d - 2) < 0.1,  # Choice 2: n=0, e=e_l
        np.abs(choice_2d - 5) < 0.1   # Choice 5: n=1, e=e_l
    )
    
    # eh_space: where choice is 3 (n=0, e=e_h)
    eh_mask = np.abs(choice_2d - 3) < 0.1  # Choice 3: n=0, e=e_h
    
    # Extract saving differences in each region
    el_space_diff = saving_diff[el_mask]
    eh_space_diff = saving_diff[eh_mask]
    el_space_full = saving_full[el_mask]
    el_space_no_hc = saving_no_hc[el_mask]
    eh_space_full = saving_full[eh_mask]
    eh_space_no_hc = saving_no_hc[eh_mask]
    
    print(f"Education regions identified:")
    print(f"  el_space: {np.sum(el_mask)} grid points")
    print(f"  eh_space: {np.sum(eh_mask)} grid points")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create meshgrid for plotting
    Z, H = np.meshgrid(zgrid, hgrid, indexing='ij')
    
    # 1. Education choice regions
    ax1 = axes[0, 0]
    im1 = ax1.contourf(H.T, Z.T, choice_2d.T, levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
                      colors=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum'], alpha=0.7)
    
    # Highlight education regions
    el_z, el_h = Z[el_mask], H[el_mask]
    eh_z, eh_h = Z[eh_mask], H[eh_mask]
    
    ax1.scatter(el_h, el_z, c='green', s=15, alpha=0.8, label='e_l regions', edgecolors='darkgreen')
    ax1.scatter(eh_h, eh_z, c='red', s=15, alpha=0.8, label='e_h regions', edgecolors='darkred')
    
    # Add sectoral boundaries
    ax1.axvline(x=2.0, color='black', linestyle='--', linewidth=2, label='h_M = 2.0')
    ax1.axvline(x=4.5, color='black', linestyle='--', linewidth=2, label='h_H = 4.5')
    
    ax1.set_xlabel('Human Capital (h)')
    ax1.set_ylabel('Productivity (z)')
    ax1.set_title('Education Choice Regions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 2])
    
    # 2. Saving difference heatmap
    ax2 = axes[0, 1]
    im2 = ax2.contourf(H.T, Z.T, saving_diff.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=ax2, label='Saving Difference (Full - No-HC)')
    ax2.set_xlabel('Human Capital (h)')
    ax2.set_ylabel('Productivity (z)')
    ax2.set_title('Optimal Saving Difference Heatmap')
    ax2.set_ylim([0, 2])
    
    # 3. Full model saving heatmap
    ax3 = axes[0, 2]
    im3 = ax3.contourf(H.T, Z.T, saving_full.T, levels=20, cmap='viridis')
    plt.colorbar(im3, ax=ax3, label='Optimal Saving')
    ax3.set_xlabel('Human Capital (h)')
    ax3.set_ylabel('Productivity (z)')
    ax3.set_title('Full Model Optimal Saving')
    ax3.set_ylim([0, 2])
    
    # 4. Scatter plot for el_space
    ax4 = axes[1, 0]
    if len(el_space_full) > 0:
        ax4.scatter(el_space_no_hc, el_space_full, alpha=0.6, s=30, color='green', edgecolors='darkgreen')
        # Add 45-degree line
        min_val = min(np.min(el_space_no_hc), np.min(el_space_full))
        max_val = max(np.max(el_space_no_hc), np.max(el_space_full))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax4.set_xlabel('No-HC Model Saving')
        ax4.set_ylabel('Full Model Saving')
        ax4.set_title('Saving Comparison in e_l Regions')
        ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot for eh_space
    ax5 = axes[1, 1]
    if len(eh_space_full) > 0:
        ax5.scatter(eh_space_no_hc, eh_space_full, alpha=0.6, s=30, color='red', edgecolors='darkred')
        # Add 45-degree line
        min_val = min(np.min(eh_space_no_hc), np.min(eh_space_full))
        max_val = max(np.max(eh_space_no_hc), np.max(eh_space_full))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax5.set_xlabel('No-HC Model Saving')
        ax5.set_ylabel('Full Model Saving')
        ax5.set_title('Saving Comparison in e_h Regions')
        ax5.grid(True, alpha=0.3)
    
    # 6. Histogram of saving differences
    ax6 = axes[1, 2]
    ax6.hist(saving_diff.flatten(), bins=30, alpha=0.7, color='blue', label='All regions')
    if len(el_space_diff) > 0:
        ax6.hist(el_space_diff, bins=20, alpha=0.7, color='green', label='e_l regions')
    if len(eh_space_diff) > 0:
        ax6.hist(eh_space_diff, bins=20, alpha=0.7, color='red', label='e_h regions')
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    ax6.set_xlabel('Saving Difference (Full - No-HC)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Saving Differences')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("DETAILED COMPARISON ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print("\n1. OVERALL SAVING STATISTICS:")
    print("-" * 50)
    print(f"Full Model - Average: {np.mean(saving_full):.4f}, Std: {np.std(saving_full):.4f}")
    print(f"No-HC Model - Average: {np.mean(saving_no_hc):.4f}, Std: {np.std(saving_no_hc):.4f}")
    print(f"Difference - Average: {np.mean(saving_diff):.4f}, Std: {np.std(saving_diff):.4f}")
    print(f"Relative Difference: {(np.mean(saving_diff)/np.mean(saving_no_hc))*100:.2f}%")
    
    # Education region statistics
    print("\n2. EDUCATION REGION ANALYSIS:")
    print("-" * 50)
    
    if len(el_space_diff) > 0:
        print(f"e_l Regions ({len(el_space_diff)} points):")
        print(f"  Saving difference - Mean: {np.mean(el_space_diff):.4f}, Std: {np.std(el_space_diff):.4f}")
        print(f"  Saving difference - Range: [{np.min(el_space_diff):.4f}, {np.max(el_space_diff):.4f}]")
        print(f"  Full model saving - Mean: {np.mean(el_space_full):.4f}")
        print(f"  No-HC model saving - Mean: {np.mean(el_space_no_hc):.4f}")
        
        # Percentage of points where full model saves less
        pct_less = np.mean(el_space_diff < 0) * 100
        print(f"  % of points where full model saves less: {pct_less:.1f}%")
    
    if len(eh_space_diff) > 0:
        print(f"\ne_h Regions ({len(eh_space_diff)} points):")
        print(f"  Saving difference - Mean: {np.mean(eh_space_diff):.4f}, Std: {np.std(eh_space_diff):.4f}")
        print(f"  Saving difference - Range: [{np.min(eh_space_diff):.4f}, {np.max(eh_space_diff):.4f}]")
        print(f"  Full model saving - Mean: {np.mean(eh_space_full):.4f}")
        print(f"  No-HC model saving - Mean: {np.mean(eh_space_no_hc):.4f}")
        
        # Percentage of points where full model saves less
        pct_less = np.mean(eh_space_diff < 0) * 100
        print(f"  % of points where full model saves less: {pct_less:.1f}%")
    
    # Regional comparison
    print("\n3. REGIONAL COMPARISON:")
    print("-" * 50)
    
    # Analyze by human capital levels
    low_h_mask = H < 2.0  # Low sector
    mid_h_mask = (H >= 2.0) & (H <= 4.5)  # Middle sector
    high_h_mask = H > 4.5  # High sector
    
    for region_name, mask in [("Low HC (h<2)", low_h_mask), 
                             ("Mid HC (2<=h<=4.5)", mid_h_mask), 
                             ("High HC (h>4.5)", high_h_mask)]:
        if np.sum(mask) > 0:
            region_diff = saving_diff[mask]
            print(f"{region_name}: Mean diff = {np.mean(region_diff):.4f}, "
                  f"Std = {np.std(region_diff):.4f}, Points = {np.sum(mask)}")
    
    print("\n" + "="*80)
    
    # Return results for further analysis
    return {
        'results_full': results_full,
        'results_no_hc': results_no_hc,
        'saving_diff': saving_diff,
        'el_mask': el_mask,
        'eh_mask': eh_mask,
        'el_space_diff': el_space_diff,
        'eh_space_diff': eh_space_diff,
        'zgrid': zgrid,
        'hgrid': hgrid
    }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Quick comparison analysis with caching support")
    parser.add_argument('--no-cache', action='store_true', help='Disable cache loading')
    parser.add_argument('--no-save', action='store_true', help='Disable cache saving')
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation even if cache exists')
    
    args = parser.parse_args()
    
    # Run analysis with specified options
    results = analyze_education_regions_and_savings(
        use_cache=not args.no_cache,
        save_cache=not args.no_save,
        force_recompute=args.force_recompute
    )
    
    return results

if __name__ == "__main__":
    results = main()
