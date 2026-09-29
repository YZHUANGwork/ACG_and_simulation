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
sigma_color = 0.02



hex_colors = cgc.select_normal_color([True]*len(hex_colors), cgc.hex_to_rgb("#e43939"), np.ones(3)*sigma_color*3)  

char_dict_ = {'bkgd': HOMURA_SCENE.draw_bkgd(max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2, 5),
                'Soul Gem': HOMURA_SCENE.draw_char(max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2, 1, 'Soul Gem'),}

        
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
            
bkgd_base_hex_colors = hex_colors.copy()

ORBITS = [
    dict(e=0.5, a=1.0 * u.AU),
    dict(e=1.0, a=1 * u.AU),   # a = periapsis distance q for a parabola
    dict(e=10, a=2 * u.AU),
]
NUM_FRAMES = 100
grid_hex_rc_ORBITS = []
for ORBIT in ORBITS:
    orbit = SOLN.KeplerOrbit(eccentricity=ORBIT['e'], semi_major_axis=ORBIT['a'], mu=const.GM_sun)
    nu_shape, r_shape, x_shape, y_shape = orbit.shape(num_points=850)

    if ORBIT['e']<1.:
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
    
    #select_shape = cg.select_mask(list(set(grid_hex_rc)), hex_rc_arr)
    #hex_colors[select_shape] =[0,1,0]
    VALID_grid_hex_rc = [(r,c) for r,c in grid_hex_rc if (r,c) in hex_rc_arr]
    grid_hex_rc_ORBITS.append(VALID_grid_hex_rc)

PARAMS = [
    dict(b_km=-8000, v_inf_kms=10, r0_scale = 3),
    dict(b_km=-10000, v_inf_kms=5, r0_scale = 3),
   
]
NUM_FRAMES = 200
grid_hex_rc_PARAMS = []
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
    VALID_grid_hex_rc = [(r,c) for r,c in grid_hex_rc if (r,c) in hex_rc_arr]
    grid_hex_rc_PARAMS.append(VALID_grid_hex_rc)
 
    #select_shape = cg.select_mask(list(set(grid_hex_rc)), hex_rc_arr)
    #hex_colors[select_shape] =[1,1,0]
    

# ── UPDATE ──────────────────────────────────────────────────────────────
Phase0_end = 10           # intro frames that just hold the base scene
Phase_LILIA_start = Phase0_end
Phase_LILIA2_start = Phase0_end+10
Phase_CLARADOLLS_start = Phase0_end+10
Phase_Homulilyv1_start = Phase0_end+20

Phase_Homulily_start = Phase0_end+40
Phase_Homulily_start2 = Phase_Homulily_start+len(grid_hex_rc_PARAMS[0])
total_frames = Phase0_end + max(Phase_LILIA_start+len(grid_hex_rc_ORBITS[0]), 
                                Phase_CLARADOLLS_start+len(grid_hex_rc_ORBITS[1]),
                               Phase_Homulilyv1_start+len(grid_hex_rc_ORBITS[2]),
                                 Phase_Homulily_start+len(grid_hex_rc_PARAMS[0]),
                                 Phase_Homulily_start2+len(grid_hex_rc_PARAMS[1])
                               )+len(grid_hex_rc_PARAMS[1])+len(grid_hex_rc_PARAMS[1])
print("total_frames", total_frames)
reached_v2 = False
def update(snapshot):
    if snapshot < Phase0_end:
        current_hex_colors = bkgd_base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        
        current_hex_colors = bkgd_base_hex_colors.copy()
        char_dict_ = {
                'Soul Gem': HOMURA_SCENE.draw_char(max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2, 1, 'Soul Gem'),}

                      
        for dict_ in char_dict_.values(): 
            for key in dict_.keys():
                part  = dict_.get(key)[0]
                colors = dict_.get(key)[1] 
                if len(colors) == 1:

                    select_part= cg.select_mask(part,hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*2) 
                else:
                    current_hex_colors = cgc.color_row_gradient(part, 
                              colors[0],colors[1],
                              hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear') 

        
        if snapshot >=Phase_LILIA_start:
            LILIA_grid_hex_rc = grid_hex_rc_ORBITS[0]
            
            
            LILIA_snapshot = (snapshot - Phase_LILIA_start) % len(LILIA_grid_hex_rc)
            center_row, center_col = LILIA_grid_hex_rc[LILIA_snapshot]
            current_dict = HOMURA_SCENE.draw_char(center_row, center_col, 2, 'LILIA')
            for key in current_dict.keys():
                part = current_dict.get(key)[0]
                colors = current_dict.get(key)[1]
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*2)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                       colors[0], colors[1],
                                       hex_rc_arr, current_hex_colors, sort='row',
                                       sigma_color=sigma_color, end_weight=0.01, mode='linear')
        
        if snapshot >=Phase_CLARADOLLS_start:
            CLARADOLLS_grid_hex_rc = grid_hex_rc_ORBITS[1]
            
            CLARADOLLS_snapshot = (snapshot - Phase_CLARADOLLS_start) % len(CLARADOLLS_grid_hex_rc)
            center_row, center_col = CLARADOLLS_grid_hex_rc[CLARADOLLS_snapshot]
            current_dict = HOMURA_SCENE.draw_char(center_row, center_col, 2, 'CLARA DOLLS')
            for key in current_dict.keys():
                part = current_dict.get(key)[0]
                colors = current_dict.get(key)[1]
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*2)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                       colors[0], colors[1],
                                       hex_rc_arr, current_hex_colors, sort='row',
                                       sigma_color=sigma_color, end_weight=0.01, mode='linear')
        
        if snapshot >=Phase_Homulilyv1_start:
            Homulilyv1_grid_hex_rc = grid_hex_rc_ORBITS[2]
            
            Homulilyv1_snapshot = (snapshot - Phase_Homulilyv1_start) % len(Homulilyv1_grid_hex_rc)
            
            center_row, center_col = Homulilyv1_grid_hex_rc[Homulilyv1_snapshot]
            direction = 'left' if center_row<= max(r for r,c in hex_rc_arr)//2 else 'right'
            version = 'v1'
            shift_row = 22
            current_dict = HOMURA_SCENE.draw_char(center_row-shift_row, center_col, 1, 'Homulily '+version+' '+direction)
            for key in current_dict.keys():
                part = current_dict.get(key)[0]
                colors = current_dict.get(key)[1]
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*2)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                       colors[0], colors[1],
                                       hex_rc_arr, current_hex_colors, sort='row',
                                       sigma_color=sigma_color, end_weight=0.01, mode='linear')
        
        if snapshot >=Phase_Homulily_start:
            Homulily_grid_hex_rc = grid_hex_rc_PARAMS[0]
            
            Homulily_snapshot = (snapshot - Phase_Homulily_start) % len(Homulily_grid_hex_rc)
            center_row, center_col = Homulily_grid_hex_rc[Homulily_snapshot]
            direction = 'right'
            version = 'v1' if center_col<= max(c for r,c in hex_rc_arr)//2 else 'v2'
            shift_row = 22
            current_dict = HOMURA_SCENE.draw_char(center_row-shift_row, center_col, 1, 'Homulily '+version+' '+direction)
            for key in current_dict.keys():
                part = current_dict.get(key)[0]
                colors = current_dict.get(key)[1]
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*2)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                       colors[0], colors[1],
                                       hex_rc_arr, current_hex_colors, sort='row',
                                       sigma_color=sigma_color, end_weight=0.01, mode='linear')
        
        if snapshot >=Phase_Homulily_start2:
            Homulily2_grid_hex_rc = grid_hex_rc_PARAMS[1]
            
            Homulily2_snapshot = (snapshot - Phase_Homulily_start2) % len(Homulily2_grid_hex_rc)
            
            center_row, center_col = Homulily2_grid_hex_rc[Homulily2_snapshot]
            direction = 'left'if center_row<= max(r for r,c in hex_rc_arr)//2 else 'right'
            global reached_v2
            if center_col > max(c for r, c in hex_rc_arr)//2:
                reached_v2 = True
            version = 'v2' if reached_v2 else 'v1'
            shift_row = 22
            current_dict = HOMURA_SCENE.draw_char(center_row-shift_row, center_col, 1, 'Homulily '+version+' '+direction)
            for key in current_dict.keys():
                part = current_dict.get(key)[0]
                colors = current_dict.get(key)[1]
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*2)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                       colors[0], colors[1],
                                       hex_rc_arr, current_hex_colors, sort='row',
                                       sigma_color=sigma_color, end_weight=0.01, mode='linear')
       
        
        pc.set_facecolor(current_hex_colors)
    return (pc,)



total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'Homura_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")