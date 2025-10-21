# Aiyagari Model with Human Capital - Python Implementation

This repository contains a Python translation of the MATLAB code `TwoPeriod.m`, implementing an Aiyagari model with human capital accumulation.

## Overview

The model solves a two-period optimization problem where agents make decisions about:
- Consumption
- Labor supply (work vs. not work)
- Human capital investment (education levels: none, low, high)

## Files

- `TwoPeriod.py` - Main model implementation (translated from MATLAB)
- `TwoPeriod.m` - Original MATLAB code
- `test_model.py` - Test script to validate the model
- `requirements.txt` - Python dependencies

## Model Features

### Parameters
- **Kai_n**: Cost of working
- **Kai_e**: Cost of education
- **e_l, e_h**: Low and high education effort levels
- **lambda_param**: Skill premium parameter
- **r**: Interest rate
- **w**: Wage rate
- **discount**: Discount factor

### Grids
- **Asset grid (a)**: 101 points from 0 to 1.4
- **Productivity grid (z)**: 101 points with lognormal distribution
- **Human capital grid (h)**: 101 points from 0 to 6

### Decision Variables
The model considers 5 choices:
1. **n=0, e=0**: Don't work, no education
2. **n=0, e=e_l**: Don't work, low education
3. **n=0, e=e_h**: Don't work, high education
4. **n=1, e=0**: Work, no education
5. **n=1, e=e_l**: Work, low education

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Run the main model:
```bash
python TwoPeriod.py
```

This will:
- Solve the two-period optimization problem
- Display a scatter plot showing optimal choices in (h,z) space
- Print key economic variables and bounds

### Run tests:
```bash
python test_model.py
```

This validates the model outputs and provides summary statistics.

## Output

The model produces:
- **Value functions**: V1 (period 1) and EV_2 (expected period 2)
- **Policy functions**: Optimal choices and consumption levels
- **Visualization**: Scatter plot of optimal decisions
- **Economic variables**: Various bounds and thresholds

### Key Results Printed:
- `zupper_fast_L`: Upper bound for fast learning
- `zlower_fast_L`: Lower bound for fast learning  
- `zupper_slow_L`: Upper bound for slow learning
- `zupper_non_L`: Upper bound for non-learning
- `lambda_lowb`: Lower bound for lambda parameter
- `z2_cutoff`: Productivity cutoff for period 2

## Model Structure

1. **Setup**: Define parameters, grids, and wage matrix
2. **Period 2**: Solve backward for expected value function
3. **Period 1**: Solve for optimal choices using dynamic programming
4. **Analysis**: Compute economic bounds and visualize results

## Differences from MATLAB

- Uses NumPy arrays instead of MATLAB matrices
- Scipy for statistical functions and interpolation
- Matplotlib for plotting instead of MATLAB's plot functions
- 0-based indexing (Python) vs 1-based indexing (MATLAB)
- Added error handling for edge cases (e.g., log(0))

## Author

Translated from MATLAB code by YKL, October 19, 2024
