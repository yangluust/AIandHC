"""
Test script for TwoPeriod.py
Validates key outputs and functionality
"""

import numpy as np
from TwoPeriod import main

def test_model():
    """Test the Aiyagari model with human capital"""
    print("Running Aiyagari model tests...")
    
    # Run the model
    results = main()
    
    # Basic validation tests
    assert results['V1'].shape == (101, 101), "V1 should be 101x101"
    assert results['choice'].shape == (101, 101), "choice should be 101x101"
    assert results['cstar'].shape == (101, 101), "cstar should be 101x101"
    assert results['EV_2'].shape == (101, 101), "EV_2 should be 101x101"
    
    # Check that choices are in valid range (1-5)
    assert np.all(results['choice'] >= 1) and np.all(results['choice'] <= 5), \
        "All choices should be between 1 and 5"
    
    # Check that consumption is non-negative
    assert np.all(results['cstar'] >= 0), "All consumption choices should be non-negative"
    
    # Check that value functions are finite
    assert np.all(np.isfinite(results['V1'])), "All V1 values should be finite"
    assert np.all(np.isfinite(results['EV_2'])), "All EV_2 values should be finite"
    
    # Print some summary statistics
    print(f"Choice distribution:")
    for i in range(1, 6):
        count = np.sum(results['choice'] == i)
        percentage = count / results['choice'].size * 100
        print(f"  Choice {i}: {count} ({percentage:.1f}%)")
    
    print(f"Average consumption: {np.mean(results['cstar']):.4f}")
    print(f"Max consumption: {np.max(results['cstar']):.4f}")
    print(f"Min consumption: {np.min(results['cstar']):.4f}")
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    test_model()
