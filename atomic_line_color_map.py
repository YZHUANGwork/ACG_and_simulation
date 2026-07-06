import numpy as np

from scipy.interpolate import interp1d
import pandas as pd
import astropy.units as u

data = np.loadtxt('ciexyz31_1nm.txt', delimiter=',')#https://www.site.uottawa.ca/~edubois/mdsp/data/index.html
wl_table, xbar_table, ybar_table, zbar_table = data[:,0], data[:,1], data[:,2], data[:,3]

xbar_interp = interp1d(wl_table, xbar_table, kind='cubic', bounds_error=False, fill_value=0)
ybar_interp = interp1d(wl_table, ybar_table, kind='cubic', bounds_error=False, fill_value=0)
zbar_interp = interp1d(wl_table, zbar_table, kind='cubic', bounds_error=False, fill_value=0)


def _xyz_to_srgb(x, y, z, normalize=True, target_luminance=1.0):
    """
    Shared XYZ -> gamma-corrected sRGB step.
 
    normalize=True  (default, matches original behavior): rescale so Y=1
        (max brightness) before converting -- always gives the most
        SATURATED version of the color, but different intensity ratios
        that share the same chromaticity direction will collapse to the
        same clipped RGB (e.g. many red:blue ratios all -> pure magenta).
 
    normalize=False: keep Y at its actual summed value (scaled by
        target_luminance) -- lets you see real brightness/shade
        differences between mixtures instead of everything clipping to
        the same saturated color.
    """
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
 
    if normalize:
        x_n = x / y
        z_n = z / y
        y_n = 1.0
    else:
        scale = target_luminance / y
        x_n = x * scale
        y_n = y * scale
        z_n = z * scale
 
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
 
    r_linear = M[0][0]*x_n + M[0][1]*y_n + M[0][2]*z_n
    g_linear = M[1][0]*x_n + M[1][1]*y_n + M[1][2]*z_n
    b_linear = M[2][0]*x_n + M[2][1]*y_n + M[2][2]*z_n
 
    rgb = []
    for c in (r_linear, g_linear, b_linear):
        c = max(c, 0.0)
        if c <= 0.0031308:
            c_gamma = 12.92 * c
        else:
            c_gamma = 1.055 * c**(1/2.4) - 0.055
        c_gamma = min(max(c_gamma, 0.0), 1.0)
        rgb.append(c_gamma)
 
    return np.array(rgb)

def wavelength_to_rgb_CIE1931(wavelength_nm):
    """
    Convert a single (monochromatic) wavelength in nm to sRGB.
    Wavelengths outside the visible range map to black.

    Parameters
    ----------
    wavelength_nm : float or astropy Quantity (length units)

    Returns
    -------
    np.ndarray of shape (3,): [R, G, B], each in [0, 1]
    """
    if isinstance(wavelength_nm, u.Quantity):
        wl_nm = wavelength_nm.to(u.nm).value
    else:
        wl_nm = wavelength_nm

    x = float(xbar_interp(wl_nm))
    y = float(ybar_interp(wl_nm))
    z = float(zbar_interp(wl_nm))

    return _xyz_to_srgb(x, y, z)


def get_atom_obs_wl(atom):

    file = atom+'_atomic_lines1.txt'#https://physics.nist.gov/PhysRefData/ASD/lines_form.html
    print(file)
    raw = pd.read_csv(file, sep='\t', header=None)

    all_headers = raw[raw[0] == 'element'].index.tolist()
    header_types = raw.loc[all_headers, 2].tolist()
    blocks = list(zip(all_headers, all_headers[1:] + [len(raw)]))

    #select b/w obs_wl_air(nm) and obs_wl_vac(nm)
    air_blocks = [(s, e) for (s, e), t in zip(blocks, header_types) if t == 'obs_wl_air(nm)']
    air_start, air_end = air_blocks[0]

    air_block = raw.iloc[air_start+1:air_end].copy()
    air_block.columns = raw.iloc[air_start].tolist()   # plain list, no stray .name

    valid_air_block = air_block[~air_block['Aki(s^-1)'].isna()]
    
    #valid_air_block['intens']=  valid_air_block['intens'].astype(str).str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0].astype(float)
    
    return valid_air_block

def calc_BR(df_input):
    ionized_stages = sorted(list(set(df_input['sp_num'])))
    
    df_output = df_input
    df_output = df_input[['element', 'sp_num', 'obs_wl_air(nm)','intens', 'Aki(s^-1)']].copy()
    df_output['intens'] = df_input['intens'].astype(str).str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0].astype(float)
    df_output['BR'] = None


    for ionized_stage in ionized_stages:
        select = df_input['sp_num']==ionized_stage
        sub_block = df_output[select].copy()
        A_tot = sum([float(i) for i in sub_block['Aki(s^-1)']])/u.s
        tau_tot = 1/A_tot
        prob = np.array([float(i)/u.s/A_tot for i in sub_block['Aki(s^-1)']])
        df_output['BR'][select] = prob
    
    return df_output

def calc_rgb(df_input):
    #select visible light range match boundary in #https://www.site.uottawa.ca/~edubois/mdsp/data/index.html     
    obs_df_output = df_input[[float(v)>380 and float(v) < 830 for v in df_input['obs_wl_air(nm)']]]
    rgbs =np.array([wavelength_to_rgb_CIE1931(float(wl)*u.nm) for wl in obs_df_output['obs_wl_air(nm)']])

    obs_df_output['r'] = rgbs.T[0]
    obs_df_output['g'] = rgbs.T[1]
    obs_df_output['b'] = rgbs.T[2]

    return obs_df_output