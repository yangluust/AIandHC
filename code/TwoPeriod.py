"""
TwoPeriod.py
Aiyagari model with human capital - Python translation
Translated from MATLAB code TwoPeriod.m
Oct 19, 2024 By YKL (translated to Python)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, multivariate_normal
from scipy.interpolate import interp1d

def main():
    """Main function to run the Aiyagari model with human capital"""
    
    # Parameters
    Kai_n = 2.34/3*0.25
    Kai_e = 2.34*0.5/3*0.25
    e_l = 1/6*3*2
    e_h = 1/3*3*2
    hbar = 6
    delta_h = 0
    
    lambda_param = 0.4
    r = 0
    w = 1
    discount = 0.95
    
    # Asset grid
    a_upper = 1 + lambda_param
    a_lower = 0
    n_a = 101
    # Create the asset grid (agrid) with n_a points from a_lower to a_upper.
    # The .reshape(-1, 1) turns the resulting 1D array into a 2D column vector of shape (n_a, 1).
    # This is similar to how MATLAB represents column vectors.
    # In this context, reshape(-1, 1) is used to make the array explicitly two-dimensional,
    # which can help with broadcasting and maintaining consistency when combining agrid
    # with other matrices or vectors. Without reshape, agrid would be a 1D array of shape (n_a,).
    agrid = np.linspace(a_lower, a_upper, n_a).reshape(-1, 1)  # n_a x 1
    
    # Bivariate productivity shocks (z for income, y for human capital)
    zscaleupper = 4
    # final period productivity cutoff for working decision
    zbar_Fcn = lambda a: (np.exp(Kai_n) - 1) * (1 + r) * a / w
    z_lower = 1e-4
    z_upper = zscaleupper * zbar_Fcn(a_upper) / (1 - lambda_param)
    n_z = 51  # Reduced for computational efficiency with bivariate grid
    zgrid = np.linspace(z_lower, z_upper, n_z)  # z for labor income
    
    # y shock has same bounds as z shock
    y_lower = z_lower
    y_upper = z_upper
    n_y = 51  # Same size as z grid
    ygrid = np.linspace(y_lower, y_upper, n_y)  # y for human capital formation
    
    # Bivariate distribution parameters
    mean_z = 0
    mean_y = 0
    sigma_z = 0.001
    sigma_y = 0.001  # Same variance as z
    delta_corr = 0.5  # Correlation coefficient between z and y (can be adjusted)
    
    # Create bivariate lognormal distribution
    # For lognormal: if X ~ LN(μ, σ²), then ln(X) ~ N(μ, σ²)
    mean_vec = [mean_z, mean_y]
    cov_matrix = [[sigma_z**2, delta_corr * sigma_z * sigma_y],
                  [delta_corr * sigma_z * sigma_y, sigma_y**2]]
    
    # Create joint probability matrix
    zy_prob = np.zeros((n_z, n_y))
    for iz in range(n_z):
        for iy in range(n_y):
            # Evaluate bivariate normal density at log(z), log(y)
            log_z = np.log(zgrid[iz])
            log_y = np.log(ygrid[iy])
            zy_prob[iz, iy] = multivariate_normal.pdf([log_z, log_y], mean_vec, cov_matrix)
    
    # Normalize to sum to 1
    zy_prob = zy_prob / np.sum(zy_prob)
    
    # Compute marginal probabilities
    prob_z = np.sum(zy_prob, axis=1)  # Marginal probability of z
    prob_y = np.sum(zy_prob, axis=0)  # Marginal probability of y
    
    # Human capital grid
    h_lower = 0
    h_upper = hbar
    n_h = 101
    hgrid = np.linspace(h_lower, h_upper, n_h)  # 1 x n_h
    
    h_M = 2
    h_H = 4.5
    
    # w*z*x(h) matrix n_z x n_h (z for labor income only)
    wzx = np.zeros((n_z, n_h))
    for iz in range(n_z):
        for ih in range(n_h):
            if hgrid[ih] < h_M:
                wzx[iz, ih] = w * zgrid[iz] * (1 - lambda_param)
            elif hgrid[ih] > h_H:
                wzx[iz, ih] = w * zgrid[iz] * (1 + lambda_param)
            else:
                wzx[iz, ih] = w * zgrid[iz]
    
    # Period-2 Value Function n_a x n_h
    # Now allow optimal choice: work vs. not work
    EV_2 = np.zeros((n_a, n_h))
    for ia in range(n_a):
        avalue = agrid[ia, 0]
        # Value if working: log(w*z*x(h) + (1+r)a) - Kai_n
        V_work = np.log(wzx + (1 + r) * avalue) - Kai_n
        # Value if not working: log((1+r)a)
        if (1 + r) * avalue > 0:
            V_notwork = np.log((1 + r) * avalue)
        else:
            V_notwork = -np.inf  # Very low utility for zero consumption

        # For each z,h, take max of working and not working utility
        V_2 = np.maximum(V_work, V_notwork)
        
        # Expected value over bivariate (z,y) distribution
        # Since Period-2 only depends on z (for income), we sum over y
        EV_2temp = np.zeros(n_h)
        marginal_prob_z = np.sum(zy_prob, axis=1)  # shape: n_z
        for ih in range(n_h):
            EV_2temp[ih] = np.sum(V_2[:, ih] * marginal_prob_z)  # Expected over z
        EV_2[ia, :] = EV_2temp
    
    # Period-1 Decision Rule - now with bivariate (z,y) shocks
    V1 = np.zeros((n_z, n_y, n_h))  # Expanded to include y dimension
    choice = np.zeros((n_z, n_y, n_h), dtype=int)
    cstar = np.zeros((n_z, n_y, n_h))
    optimal_saving = np.zeros((n_z, n_y, n_h))  # Store optimal asset holdings
    
    h2_0 = (1 - delta_h) * hgrid
    
    # avalue needs to be such that if households know z'=1, it will for sure
    # work even if they have the highest possible asset next period in equilibrium
    avalue = a_upper / 2 * 2
    c1_upper = (w * z_upper * (1 + lambda_param) + (1 + r) * avalue + 
                w * 1 * (1 + lambda_param) / (1 + r)) / (1 + discount)
    a2_upper = w * z_upper * (1 + lambda_param) + (1 + r) * avalue - c1_upper
    z2_cutoff = zbar_Fcn(a2_upper)
    # needs z'=1>z2_cutoff
    
    a2_nowork = (1 + r) * avalue
    
    # Condition for e_h, e_l, Kai_e and Kai_n
    LHS = np.exp(Kai_e * e_h / (1 + discount))
    RHS = (np.exp(Kai_e * e_l / (1 + discount)) / 
           (np.exp(Kai_e * e_l / (1 + discount)) + np.exp(Kai_n / (1 + discount)) - 
            np.exp((Kai_e * e_l + Kai_n) / (1 + discount))))
    
    n_c = 1001
    cgrid_nowork = np.linspace(0, a2_nowork, n_c)
    cstartemp = np.zeros(5)
    
    for ih in range(n_h):
        for iz in range(n_z):
            for iy in range(n_y):
                # z affects current labor income, y affects human capital transitions
                a2_work = wzx[iz, ih] + (1 + r) * avalue
                
                # Find closest indices for human capital transitions (using y shock)
                ih2_0 = np.argmin(np.abs(hgrid - h2_0[ih]))
                ih2_l = np.argmin(np.abs(hgrid - (h2_0[ih] + ygrid[iy] * e_l)))  # Use y for HC
                ih2_h = np.argmin(np.abs(hgrid - (h2_0[ih] + ygrid[iy] * e_h)))  # Use y for HC
                
                # Ensure indices are within bounds
                ih2_0 = np.clip(ih2_0, 0, n_h - 1)
                ih2_l = np.clip(ih2_l, 0, n_h - 1)
                ih2_h = np.clip(ih2_h, 0, n_h - 1)
                
                # n=0, e=0
                assets_remaining = a2_nowork - cgrid_nowork
                assets_remaining = np.clip(assets_remaining, a_lower, a_upper)
                
                interp_func = interp1d(agrid.flatten(), EV_2[:, ih2_0], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                obj1 = np.log(np.maximum(cgrid_nowork, 1e-10)) + discount * interp_func(assets_remaining)
                val1_idx = np.argmax(obj1)
                val1 = obj1[val1_idx]
                cstartemp[0] = cgrid_nowork[val1_idx]
                
                # n=0, e=e_l
                interp_func = interp1d(agrid.flatten(), EV_2[:, ih2_l], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                obj2 = (np.log(np.maximum(cgrid_nowork, 1e-10)) - Kai_e * e_l + 
                       discount * interp_func(assets_remaining))
                val2_idx = np.argmax(obj2)
                val2 = obj2[val2_idx]
                cstartemp[1] = cgrid_nowork[val2_idx]
                
                # n=0, e=e_h
                interp_func = interp1d(agrid.flatten(), EV_2[:, ih2_h], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                obj3 = (np.log(np.maximum(cgrid_nowork, 1e-10)) - Kai_e * e_h + 
                       discount * interp_func(assets_remaining))
                val3_idx = np.argmax(obj3)
                val3 = obj3[val3_idx]
                cstartemp[2] = cgrid_nowork[val3_idx]
                
                # Working options
                cgrid_work = np.linspace(0, a2_work, n_c)
                assets_remaining_work = a2_work - cgrid_work
                assets_remaining_work = np.clip(assets_remaining_work, a_lower, a_upper)
                
                # n=1, e=0
                interp_func = interp1d(agrid.flatten(), EV_2[:, ih2_0], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                obj4 = (np.log(np.maximum(cgrid_work, 1e-10)) - Kai_n + 
                       discount * interp_func(assets_remaining_work))
                val4_idx = np.argmax(obj4)
                val4 = obj4[val4_idx]
                cstartemp[3] = cgrid_work[val4_idx]
                
                # n=1, e=e_l
                interp_func = interp1d(agrid.flatten(), EV_2[:, ih2_l], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                obj5 = (np.log(np.maximum(cgrid_work, 1e-10)) - Kai_n - Kai_e * e_l + 
                       discount * interp_func(assets_remaining_work))
                val5_idx = np.argmax(obj5)
                val5 = obj5[val5_idx]
                cstartemp[4] = cgrid_work[val5_idx]
                
                # Choose over 5 options
                values = [val1, val2, val3, val4, val5]
                best_choice = np.argmax(values)
                V1[iz, iy, ih] = values[best_choice]  # Now includes y dimension
                choice[iz, iy, ih] = best_choice + 1  # MATLAB uses 1-based indexing
                cstar[iz, iy, ih] = cstartemp[best_choice]
                
                # Compute optimal saving based on choice
                optimal_consumption = cstartemp[best_choice]
                if best_choice < 3:  # Choices 1-3: not working (n=0)
                    total_resources = a2_nowork
                else:  # Choices 4-5: working (n=1)
                    total_resources = a2_work
                optimal_saving[iz, iy, ih] = total_resources - optimal_consumption
    
    # Aggregate over y dimension for plotting and analysis
    # Check that prob_y sums to one
    prob_y_sum = np.sum(prob_y)
    assert np.isclose(prob_y_sum, 1.0), f"prob_y does not sum to 1 (sum={prob_y_sum})"
    
    # Weighted average of choices and savings for each (z, h) pair
    choice_2d = np.zeros((n_z, n_h))
    optimal_saving_2d = np.zeros((n_z, n_h))
    
    for iz in range(n_z):
        for ih in range(n_h):
            choices_y = choice[iz, :, ih]  # shape: (n_y,)
            choices_y = choice[iz, :, ih]  # shape: (n_y,)
            savings_y = optimal_saving[iz, :, ih]  # shape: (n_y,)
            
            weighted_avg_choice = np.sum(choices_y * prob_y)
            weighted_avg_saving = np.sum(savings_y * prob_y)
            
            choice_2d[iz, ih] = weighted_avg_choice
            optimal_saving_2d[iz, ih] = weighted_avg_saving
    
    z1, h1 = np.where(choice_2d == 1)
    z2, h2 = np.where(choice_2d == 2)
    z3, h3 = np.where(choice_2d == 3)
    z4, h4 = np.where(choice_2d == 4)
    z5, h5 = np.where(choice_2d == 5)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(hgrid[h1], zgrid[z1], c='black', s=1, label='n=0, e=0')
    plt.scatter(hgrid[h2], zgrid[z2], c='green', s=1, label='n=0, e=e_l')
    plt.scatter(hgrid[h3], zgrid[z3], c='red', s=1, label='n=0, e=e_h')
    plt.scatter(hgrid[h4], zgrid[z4], c='yellow', s=1, label='n=1, e=0')
    plt.scatter(hgrid[h5], zgrid[z5], c='magenta', s=1, label='n=1, e=e_l')
    plt.ylim([0, 2])
    plt.xlabel('Human Capital (h)')
    plt.ylabel('Productivity (z)')
    plt.title('Optimal Choices in (h,z) Space (Aggregated over y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Final calculations
    futurewage = w * 1 / (1 + r)
    zupper_fast_L = (((np.exp(Kai_e * e_l / (1 + discount)) * lambda_param * 
                      (np.exp(Kai_e * e_l / (1 + discount)) - 1)**(-1) - 1) * 
                     futurewage - (1 + r) * avalue) / w)
    zlower_fast_L = ((np.exp(Kai_n / (1 + discount)) - 1) * 
                    ((1 + r) * avalue + futurewage) / w)
    zupper_slow_L = (((np.exp((Kai_n - Kai_e * e_h) / (1 + discount)) - 1) * 
                     ((1 + r) * avalue + futurewage) + lambda_param * futurewage) / w)
    zupper_non_L = ((np.exp(Kai_n / (1 + discount)) - 1) * 
                   ((1 + r) * avalue + futurewage * (1 - lambda_param)) / w)
    
    # Lower bound of lambda
    lambda_lowb = ((np.exp(Kai_e * e_h / (1 + discount)) - 1) * 
                  ((1 + r) * avalue + futurewage) / futurewage)
    
    # Print results
    print(f"zupper_fast_L: {zupper_fast_L}")
    print(f"zlower_fast_L: {zlower_fast_L}")
    print(f"zupper_slow_L: {zupper_slow_L}")
    print(f"zupper_non_L: {zupper_non_L}")
    print(f"lambda_lowb: {lambda_lowb}")
    print(f"LHS: {LHS}")
    print(f"RHS: {RHS}")
    print(f"z2_cutoff: {z2_cutoff}")
    
    # Print optimal saving statistics
    print(f"\nOptimal Saving Statistics:")
    print(f"Average optimal saving: {np.mean(optimal_saving_2d):.4f}")
    print(f"Min optimal saving: {np.min(optimal_saving_2d):.4f}")
    print(f"Max optimal saving: {np.max(optimal_saving_2d):.4f}")
    print(f"Std optimal saving: {np.std(optimal_saving_2d):.4f}")
    
    return {
        'V1': V1,
        'choice': choice,
        'choice_2d': choice_2d,  # Aggregated choice for visualization
        'cstar': cstar,
        'optimal_saving': optimal_saving,  # Optimal asset holdings (3D: z,y,h)
        'optimal_saving_2d': optimal_saving_2d,  # Aggregated optimal savings (2D: z,h)
        'EV_2': EV_2,
        'agrid': agrid,
        'zgrid': zgrid,
        'ygrid': ygrid,  # New y grid
        'hgrid': hgrid,
        'wzx': wzx,
        'zy_prob': zy_prob,  # Joint probability matrix
        'prob_z': prob_z,  # Marginal probability of z
        'prob_y': prob_y,  # Marginal probability of y
        'parameters': {
            'Kai_n': Kai_n,
            'Kai_e': Kai_e,
            'e_l': e_l,
            'e_h': e_h,
            'lambda_param': lambda_param,
            'r': r,
            'w': w,
            'discount': discount,
            'delta_corr': delta_corr  # Correlation coefficient
        }
    }

if __name__ == "__main__":
    results = main()
    print("Model solved successfully!")
