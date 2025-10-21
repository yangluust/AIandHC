"""
TwoPeriod.py
Aiyagari model with human capital - Python translation
Translated from MATLAB code TwoPeriod.m
Oct 19, 2024 By YKL (translated to Python)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.interpolate import interp1d

def main():
    """Main function to run the Aiyagari model with human capital"""
    
    # Parameters
    Kai_n = 2.34/3*0.25
    Kai_e = 2.34*0.5/3*0.25
    e_l = 1/6*3*2
    e_h = 1/3*3*2
    hbar = 6
    delta_h = 0.1
    
    lambda_param = 0.4
    r = 0
    w = 1
    discount = 0.95
    
    # Asset grid
    a_upper = 1 + lambda_param
    a_lower = 0
    n_a = 101
    agrid = np.linspace(a_lower, a_upper, n_a).reshape(-1, 1)  # n_a x 1
    
    # Productivity grid
    zscaleupper = 4
    zbar_Fcn = lambda a: (np.exp(Kai_n) - 1) * (1 + r) * a / w
    z_lower = 1e-4
    z_upper = zscaleupper * zbar_Fcn(a_upper) / (1 - lambda_param)
    n_z = 101
    zgrid = np.linspace(z_lower, z_upper, n_z).reshape(-1, 1)  # n_z x 1
    
    # Productivity distribution
    mean_z = 0
    sigma_z = 0.001  # 0.3; 0.01;
    zdensity = lognorm.pdf(zgrid.flatten(), s=sigma_z, scale=np.exp(mean_z))
    z_prob = zdensity / np.sum(zdensity)
    
    # Human capital grid
    h_lower = 0
    h_upper = hbar
    n_h = 101
    hgrid = np.linspace(h_lower, h_upper, n_h)  # 1 x n_h
    
    h_M = 2
    h_H = 4.5
    
    # w*z*x(h) matrix n_z x n_h
    wzx = np.zeros((n_z, n_h))
    for iz in range(n_z):
        for ih in range(n_h):
            if hgrid[ih] < h_M:
                wzx[iz, ih] = w * zgrid[iz, 0] * (1 - lambda_param)
            elif hgrid[ih] > h_H:
                wzx[iz, ih] = w * zgrid[iz, 0] * (1 + lambda_param)
            else:
                wzx[iz, ih] = w * zgrid[iz, 0]
    
    # Period-2 Value Function n_a x n_h
    EV_2 = np.zeros((n_a, n_h))
    for ia in range(n_a):
        avalue = agrid[ia, 0]
        V_2 = np.log(wzx + (1 + r) * avalue) - Kai_n
        # Handle case where avalue might be 0 (avoid log(0))
        if (1 + r) * avalue > 0:
            V_notwork = np.log((1 + r) * avalue)
        else:
            V_notwork = -np.inf  # Very low utility for zero consumption
        V_2[V_2 < V_notwork] = V_notwork
        EV_2temp = V_2.T @ z_prob
        EV_2[ia, :] = EV_2temp  # 1 x n_h
    
    # Period-1 Decision Rule
    V1 = np.zeros((n_z, n_h))
    choice = np.zeros((n_z, n_h), dtype=int)
    cstar = np.zeros((n_z, n_h))
    
    h2_0 = (1 - delta_h) * hgrid
    
    # Asset value calculation
    avalue = a_upper / 2 * 2
    c1_upper = (w * z_upper * (1 + lambda_param) + (1 + r) * avalue + 
                w * 1 * (1 + lambda_param) / (1 + r)) / (1 + discount)
    a2_upper = w * z_upper * (1 + lambda_param) + (1 + r) * avalue - c1_upper
    z2_cutoff = zbar_Fcn(a2_upper)
    
    a2_nowork = (1 + r) * avalue
    a2_work = wzx + (1 + r) * avalue
    
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
            # Find closest indices for human capital transitions
            ih2_0 = np.argmin(np.abs(hgrid - h2_0[ih]))
            ih2_l = np.argmin(np.abs(hgrid - (h2_0[ih] + zgrid[iz, 0] * e_l)))
            ih2_h = np.argmin(np.abs(hgrid - (h2_0[ih] + zgrid[iz, 0] * e_h)))
            
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
            cgrid_work = np.linspace(0, a2_work[iz, ih], n_c)
            assets_remaining_work = a2_work[iz, ih] - cgrid_work
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
            V1[iz, ih] = values[best_choice]
            choice[iz, ih] = best_choice + 1  # MATLAB uses 1-based indexing
            cstar[iz, ih] = cstartemp[best_choice]
    
    # Visualization
    z1, h1 = np.where(choice == 1)
    z2, h2 = np.where(choice == 2)
    z3, h3 = np.where(choice == 3)
    z4, h4 = np.where(choice == 4)
    z5, h5 = np.where(choice == 5)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(hgrid[h1], zgrid[z1, 0], c='black', s=1, label='n=0, e=0')
    plt.scatter(hgrid[h2], zgrid[z2, 0], c='green', s=1, label='n=0, e=e_l')
    plt.scatter(hgrid[h3], zgrid[z3, 0], c='red', s=1, label='n=0, e=e_h')
    plt.scatter(hgrid[h4], zgrid[z4, 0], c='yellow', s=1, label='n=1, e=0')
    plt.scatter(hgrid[h5], zgrid[z5, 0], c='magenta', s=1, label='n=1, e=e_l')
    plt.ylim([0, 2])
    plt.xlabel('Human Capital (h)')
    plt.ylabel('Productivity (z)')
    plt.title('Optimal Choices in (h,z) Space')
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
    
    return {
        'V1': V1,
        'choice': choice,
        'cstar': cstar,
        'EV_2': EV_2,
        'agrid': agrid,
        'zgrid': zgrid,
        'hgrid': hgrid,
        'wzx': wzx,
        'parameters': {
            'Kai_n': Kai_n,
            'Kai_e': Kai_e,
            'e_l': e_l,
            'e_h': e_h,
            'lambda_param': lambda_param,
            'r': r,
            'w': w,
            'discount': discount
        }
    }

if __name__ == "__main__":
    results = main()
    print("Model solved successfully!")
