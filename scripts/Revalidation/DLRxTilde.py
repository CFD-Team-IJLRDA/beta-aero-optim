from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def read_tecplot_dat(filename):
    """
    Read a simple Tecplot ASCII file with POINT format.
    
    Parameters
    ----------
    filename : str
        Path to the .dat file
        
    Returns
    -------
    data : numpy.ndarray
        Array with shape (n_points, n_variables) containing the data
    variables : list
        List of variable names
    """
    import numpy as np
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse variable names from first line
    var_line = lines[0].strip()
    variables = var_line.split('=')[1].strip().split()
    variables = [v.strip('"') for v in variables]
    
    # Skip the ZONE line (line 1) and read data starting from line 2
    data = []
    for line in lines[2:]:
        line = line.strip()
        if line:  # Skip empty lines
            values = [float(x) for x in line.split()]
            data.append(values)
    
    data = np.array(data)
    
    return data, variables

data, variables = read_tecplot_dat("./BLADE_LNR-OGV_i323_k1.dat")
data, variables = read_tecplot_dat("./testBlade.dat")
xx = data[:,0]
yy = data[:,1]

CHORD = 0.07*1000
STAGGER = 16.04
xtilde = (xx ** 2 + yy ** 2) ** 0.5 * np.cos(np.arctan2(yy, xx) - np.deg2rad(STAGGER)) / CHORD

LE, TE = np.argmin(xx), np.argmax(xx)

ss_points = np.s_[LE+1:TE-1]
ps_points = list(reversed(np.arange(0, LE))) + list(reversed(np.arange(TE+1, len(xx))))

fig, ax = plt.subplots()
ax.plot(xx[ss_points], yy[ss_points], '-', label="Suction side")
ax.plot(xx[ps_points], yy[ps_points], '-', label="Pressure side")
ax.plot(xx[LE], yy[LE], 'o', label="LE")
ax.plot(xx[TE], yy[TE], 'o', label="TE")
ax.legend()
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

np.argwhere(np.diff(xtilde[ps_points]) == 0)
# Delete duplicate point
xtilde_ps = xtilde[ps_points]
xx_ps = xx[ps_points]
xtilde_ps = np.delete(xtilde_ps, 140)
xx_ps = np.delete(xx_ps, 140)
yy_ps = np.delete(yy[ps_points], 140)

bsplsuctionside = make_interp_spline(xtilde[ss_points], xx[ss_points], k=3)
bsplsuctionside_y = make_interp_spline(xtilde[ss_points], yy[ss_points], k=3)
bsplpressureside = make_interp_spline(xtilde_ps, xx_ps, k=3)
bsplpressureside_y = make_interp_spline(xtilde_ps, yy_ps, k=3)

# Delete duplicate point
xtilde_ps = xtilde[ps_points]
xx_ps = xx[ps_points]
xtilde_ps = np.delete(xtilde_ps, 140)
xx_ps = np.delete(xx_ps, 140)
yy_ps = np.delete(yy[ps_points], 140)

xtilde_ss = xtilde[ss_points]
xx_ss = xx[ss_points]
xtilde_ss = np.delete(xtilde_ss, 140)
xx_ss = np.delete(xx_ss, 140)
yy_ss = np.delete(yy[ss_points], 140)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(xtilde_ss, xx_ss/CHORD, 'o', label="Suction side")
plt.plot([0,1],[0,1], 'k:', label="y=x")
plt.subplot(1, 2, 2)
plt.plot(xtilde_ps, xx_ps/CHORD, 'o', label="Pressure side")
plt.plot([0,1],[0,1], 'k:', label="y=x")
plt.show()
