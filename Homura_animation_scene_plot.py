import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, Counter
from types import SimpleNamespace

import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import cg_draw_figure as cgf

import TRAJECTORY_CentralForce as SOLN
import Homura_scene_plot as HOMURA_SCENE

rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100

fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=8, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.03


head_center_row = max(r for r,c in hex_rc_arr)//2
head_center_col = max(c for r,c in hex_rc_arr)//2
hex_colors = cgc.select_normal_color([True]*len(hex_colors), cgc.hex_to_rgb("#e43939"), np.ones(3)*sigma_color*3)  
char_dict_ = {
    'Soul Gem': HOMURA_SCENE.draw_char(max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2, 1, 'Soul Gem'),

             }
for dict_ in char_dict_.values(): 
    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear') 

ORBITS = [
    dict(e=0.0, a=1.0 * u.AU),
    dict(e=0.5, a=1.0 * u.AU),
#    dict(e=0.8, a=1.0 * u.AU),
    dict(e=1.0, a=1 * u.AU),   # a = periapsis distance q for a parabola
#    dict(e=1.2, a=2* u.AU),
#    dict(e=5, a=2 * u.AU),
        dict(e=10, a=2 * u.AU),
    dict(e=15, a=2 * u.AU),
]
NUM_FRAMES = 80
all_solutions = []
for params in ORBITS:
    orbit = SOLN.KeplerOrbit(eccentricity=params['e'], semi_major_axis=params['a'], mu=const.GM_sun)
    nu_shape, r_shape, x_shape, y_shape = orbit.shape(num_points=850)

    if params['e']<1.:
        DOMAIN_W_SCALE = 2
        DOMAIN_H_SCALE = 2
        t_min, t_max = 0.0, 2 * np.pi / orbit.mean_motion()
    else:
        DOMAIN_W_SCALE = 1
        DOMAIN_H_SCALE = 0.8
        nu_edge = nu_shape[-1]
        t_edge = orbit.true_anomaly_to_time(nu_edge)[0]
        t_min, t_max = -t_edge, t_edge

    t_anim = np.linspace(t_min, t_max, NUM_FRAMES)
    traj = orbit.trajectory_over_time(t_anim, t_peri=0.0)

    
    x = orbit.to_unit(traj['x'], u.AU).value
    y = orbit.to_unit(traj['y'], u.AU).value
    #X, Y = x_shape, y_shape
    X, Y = x, y
    grid_hex_rc = cg.fn2grid(X, Y, DOMAIN_W_SCALE ,DOMAIN_H_SCALE, 
                          max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2, 0, 0, 
                              detail_info, square = False, EPS_DOMAIN = 1e-6)
    VALID_grid_hex_rc = [(r,c) for r,c in grid_hex_rc if (r,c) in hex_rc_arr]
    
    print(len(VALID_grid_hex_rc), params)
    select_shape = cg.select_mask(list(set(grid_hex_rc)), hex_rc_arr)
    hex_colors[select_shape] =[0,1,0]
    all_solutions.append([list(set(grid_hex_rc))])

    
PARAMS = [
    dict(b_km=-8000, v_inf_kms=10, r0_scale = 3),
    dict(b_km=-10000, v_inf_kms=5, r0_scale = 3),
   
]

    
#b_km = 8000
#v_inf_kms = 10
NUM_FRAMES = 100
#r0_scale = 3
for PARAM in PARAMS:
    b_km = PARAM['b_km']
    v_inf_kms = PARAM['v_inf_kms']
    r0_scale = PARAM['r0_scale']
    r0 = (-abs(r0_scale* b_km), b_km, 0) * u.km
    v0 = (v_inf_kms, 0, 0) * u.km / u.s
    t_span = (0, abs(2 * r0_scale * b_km) / v_inf_kms) * u.s
 
    scattering = SOLN.CentralForceScattering(k=const.GM_earth, force_type='attractive')
    sol = scattering.integrate_trajectory(r0, v0, t_span, num_points=NUM_FRAMES)

    DOMAIN_W_SCALE = 1.
    DOMAIN_H_SCALE = 1.
    grid_hex_rc = cg.fn2grid(sol.y[0]/ 1000 ,  sol.y[1]/ 1000, DOMAIN_W_SCALE ,DOMAIN_H_SCALE, 
                              max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2, 0, 0, 
                                  detail_info, square = False, EPS_DOMAIN = 1e-6)

    select_shape = cg.select_mask(list(set(grid_hex_rc)), hex_rc_arr)
    hex_colors[select_shape] =[1,1,0]


pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'cg'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Homura_animation_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')