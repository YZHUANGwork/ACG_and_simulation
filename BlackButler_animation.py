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
import os
import re
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import cg_draw_figure as cgf
import SOLN_AngularMomentum_conserve as SOLN

import BlackButler_scene_plot as B_SCENE
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
char_dict_ = {
    'bkdg': B_SCENE.draw_bkgd(B_SCENE.ciel_loc, B_SCENE.doll_loc,B_SCENE.joker_loc, hex_rc_arr),
     'joker':  B_SCENE.draw_char(B_SCENE.joker_loc[0], B_SCENE.joker_loc[1], 'joker')
    
}

for char, dict_ in char_dict_.items(): 
    for key in dict_.keys():
        
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        
        m = re.search(r"sigmafactor\s*=\s*([\d.]+)", str(colors))
        sigmafactor = float(m.group(1)) if m else 1.

        is_single_with_modifier = len(colors) == 2 and isinstance(colors[1], str) and 'sigmafactor' in colors[1]
         
        
        is_striped = len(colors) == 3 and isinstance(colors[-1], str) and 'stripe' in colors[-1]
        if is_striped:
            select_part= cg.select_mask(part,hex_rc_arr)
            stripe_target_colors = cgc.color_stripe(hex_rc_arr, STRIPE_WIDTH=1, STRIPE_PALETTE=[np.array(colors[0]), np.array(colors[1])])
            hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color*2)[0]
                                                    for c in stripe_target_colors[select_part] 
                                                   ])
            
            continue
        if len(colors) == 1 or is_single_with_modifier:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*sigmafactor) 
            
        else:
            if len(colors) == len(hex_rc_arr):
                print('FULLY DEFINED COLORS')
                select_part = cg.select_mask(part, hex_rc_arr)
                
                for idx in np.where(select_part)[0]:
                    hex_colors[idx] = cgc.select_normal_color([True], colors[idx], np.ones(3) * sigma_color*sigmafactor)[0]
            else:
                sort_info = colors[-1]
                if 'hex' in str(sort_info):
                    print('sored by hex')
                    
                    
                    center =sort_info[0]
                    n_layers = sort_info[1]
                    start_n = sort_info[2]
                    end_n=sort_info[3]
                    if 'sigmoid' in str(sort_info):
                        mode = 'sigmoid'
                    else:
                        mode = 'linear'
                        
                    print(mode)
                    hex_colors = cgc.color_hex_gradient(part,colors[0],colors[1], hex_rc_arr, 
                            hex_colors, center, n_layers,
                            sigma_color=sigma_color*sigmafactor, end_weight=0.01, period=None, mode=mode,
                            start_n=start_n, end_n=end_n)
                elif 'col' in sort_info:
                    print(key)
                    print('col gradient')
                    if 'max' in sort_info:
                        pivot = 'max'
                    elif 'min' in sort_info:
                        pivot = 'min'
                    else:
                        pivot = None
                    if 'sigmoid' in sort_info:
                        mode = 'sigmoid'
                    else:
                        mode = 'linear'
                    print('mode', mode)
                    hex_colors = cgc.color_row_gradient(part, 
                            colors[0],colors[1],
                            hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color*sigmafactor, 
                                                        end_weight= 0.01, mode = mode, pivot=pivot ) 

                else:
                    print(key)
                    print('row gradient')
                    
                    if 'sigmoid' in sort_info:
                        mode = 'sigmoid'
                    else:
                        mode = 'linear'
                        
                    if 'max' in sort_info:
                        pivot = 'max'
                    elif 'min' in sort_info:
                        pivot = 'min'
                    else:
                        pivot = None
                        
                    print('mode', mode, 'pivot', pivot)

                    hex_colors = cgc.color_row_gradient(part, 
                            colors[0],colors[1],
                            hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*sigmafactor, 
                                                        end_weight= 0.01, mode = mode, pivot=pivot )
background_hex_colors = hex_colors.copy()

                    
def draw_animtaion(snapshot, detail_info, version, other_info):
    if version == 'ciel':
        
        [theta_spinning_sticks_constant] = other_info
        foreshorten_scale = np.cos(theta_spinning_sticks_constant[snapshot])
        if foreshorten_scale >= 0:
            dict_ = B_SCENE.draw_char(B_SCENE.ciel_loc[0], B_SCENE.ciel_loc[1], 'ciel front')
        else:
            dict_ = B_SCENE.draw_char(B_SCENE.ciel_loc[0], B_SCENE.ciel_loc[1],  'ciel back')
        
        dict_modified = {}
        for key in dict_.keys():
            part_raw, colors = dict_.get(key)[0], dict_.get(key)[1]
            modified_part = cg.foreshorten_hex_shape(part_raw, B_SCENE.ciel_loc, abs(foreshorten_scale), detail_info)
            dict_modified[key] = [modified_part, colors]
        print(version)
            
    elif 'doll' in version:
        [theta_arr, long_flags] = other_info
        foreshorten_scale = np.cos(theta_arr[snapshot])
        long = long_flags[snapshot]
        extra_version = 'v2' if long else 'v1'
        
        if foreshorten_scale >= 0:
            dict_ = B_SCENE.draw_char(B_SCENE.doll_loc[0], B_SCENE.doll_loc[1], 'doll '+extra_version+' front')
        else:
            dict_ = B_SCENE.draw_char(B_SCENE.doll_loc[0], B_SCENE.doll_loc[1],  'doll '+extra_version+' back')
        dict_modified = {}
        for key in dict_.keys():
            part_raw, colors = dict_.get(key)[0], dict_.get(key)[1]
            modified_part = cg.foreshorten_hex_shape(part_raw, B_SCENE.doll_loc, abs(foreshorten_scale), detail_info)
            dict_modified[key] = [modified_part, colors]
            
    elif 'joker stick' in version:   
        [grid_hex_rc_COM, THETA_all] = other_info
        COM_LOC = grid_hex_rc_COM[snapshot]
        STICK_HALF_LEN = 4 

        coords0 = cgd.horizontal_lines([(COM_LOC[0], (COM_LOC[1] - STICK_HALF_LEN, COM_LOC[1] + STICK_HALF_LEN))])
        coords_theta = cg.rotate_hex_shape(coords0, COM_LOC, THETA_all[snapshot], detail_info)
        
        coords_COLOR = [cgc.hex_to_rgb("#9f601e"), cgc.hex_to_rgb("#4f9f1e"), 'col' ]
        end_sphere_locs = [(min(r for r,c in coords_theta), min(c for r,c in coords_theta if r==min(r for r,c in coords_theta))),
                          (max(r for r,c in coords_theta), max(c for r,c in coords_theta if r==max(r for r,c in coords_theta))),]
        spheres = []
        for end_sphere_loc in end_sphere_locs:
            spheres+=cg.hex_neighbours_n(end_sphere_loc[0], end_sphere_loc[1], n=1)
        spheres_COLOR = [cgc.hex_to_rgb("#f2ff00")]
        dict_modified = {"coords": [coords_theta, coords_COLOR],
                         "spheres": [spheres, spheres_COLOR],
                        }
    elif 'ciel stick' in version:   
        [grid_hex_rc_COM, THETA_all] = other_info
        COM_LOC = grid_hex_rc_COM[snapshot]
        STICK_HALF_LEN = 3 

        coords0 = cgd.horizontal_lines([(COM_LOC[0], (COM_LOC[1] - STICK_HALF_LEN, COM_LOC[1] + STICK_HALF_LEN))])
        coords_theta = cg.rotate_hex_shape(coords0, COM_LOC, THETA_all[snapshot], detail_info)
        
        coords_COLOR = [[0,0,1], cgc.hex_to_rgb("#173a07"), 'col' ]
        end_sphere_locs = [(min(r for r,c in coords_theta), min(c for r,c in coords_theta if r==min(r for r,c in coords_theta))),
                          (max(r for r,c in coords_theta), max(c for r,c in coords_theta if r==max(r for r,c in coords_theta))),]
        spheres = []
        for end_sphere_loc in end_sphere_locs:
            spheres+=cg.hex_neighbours_n(end_sphere_loc[0], end_sphere_loc[1], n=1)
        spheres_COLOR = [cgc.hex_to_rgb("#f63b2f"),'sigmafactor=3']  
        dict_modified = {"coords": [coords_theta, coords_COLOR],
                         "spheres": [spheres, spheres_COLOR],
                        }
        
    return dict_modified
            

def draw_soln_functions(version, other_info ):
    if version == 'joker stick':
        [PIVOT_loc, sticks] = other_info
        
        

        YCM, THETA = [], []
        for s in sticks:
            t_flight = 2 * s.vy0 / s.g
            t_i = np.linspace(0, t_flight, frames_per_case) * u.s
            x_cm, y_cm, theta = s.trajectory(t_i)
            YCM.append(y_cm)
            THETA.append(theta)
        YCM_all = np.concatenate(YCM)
        THETA_all = np.concatenate(THETA)

        grid_hex_rc_COM = cg.fn2grid(
            np.zeros_like(YCM_all), YCM_all,
            DOMAIN_W_SCALE=2, DOMAIN_H_SCALE=1.1,
            PIVOT_ROW=PIVOT_loc[0], PIVOT_COL=PIVOT_loc[1],
            PIVOT_ROW_X=0, PIVOT_COL_Y=YCM_all.min(),
            detail_info=detail_info, square=False,
        )
        result = [grid_hex_rc_COM, THETA_all]
        
    elif version == 'ciel stick':
        
        [PIVOT_loc, flight_frames] = other_info
        XCM, YCM, THETA = [], [], []
        x0 = 0.0 * u.m
        for i, n_f in enumerate(flight_frames):
            vx0 = (2.0 if i % 2 == 0 else -2.0) * u.m / u.s
            bs = SOLN.RotatingStick(m=0.2 * u.kg, l=0.3 * u.m, x0=x0,
                                     vx0=vx0, vy0=4.0 * u.m / u.s,
                                     theta0=0.0 * u.rad, omega0=35.0 * u.rad / u.s)
            t_b = np.linspace(0, 2 * bs.vy0 / bs.g, n_f) * u.s
            bx_cm, by_cm, btheta = bs.trajectory(t_b)
            XCM.append(bx_cm); YCM.append(by_cm); THETA.append(btheta)
            x0 = bx_cm[-1] * u.m

        XCM_all = np.concatenate(XCM)
        YCM_all = np.concatenate(YCM)
        THETA_all = np.concatenate(THETA)

        XCM_all = np.concatenate(XCM)
        YCM_all = np.concatenate(YCM)
        THETA_all = np.concatenate(THETA)
        
        grid_hex_rc_COM = cg.fn2grid(
                                XCM_all, YCM_all,
                                DOMAIN_W_SCALE=5.5, DOMAIN_H_SCALE=1.8,
                                PIVOT_ROW=PIVOT_loc[0], PIVOT_COL=PIVOT_loc[1],
                                PIVOT_ROW_X=0.5 * (XCM_all.min() + XCM_all.max()), PIVOT_COL_Y=YCM_all.min(),
                                detail_info=detail_info, square=False,
                            )
        result = [grid_hex_rc_COM, THETA_all]
    return result

frames_per_case = 60

stick_cases = [
                dict(vx0=0.0, vy0=4.0, theta0=0.0, omega0=25.0),
                dict(vx0=0.0, vy0=3.5, theta0=-2., omega0=15.0),
                dict(vx0=0.0, vy0=5.0, theta0=1.5, omega0=-20.0),
                dict(vx0=0.0, vy0=3.0, theta0=-0.5, omega0=30.0),
                dict(vx0=0.0, vy0=4.5, theta0=1.0, omega0=-32.0),
            ]
sticks = [SOLN.RotatingStick(m=0.2 * u.kg, l=0.3 * u.m,
                                      vx0=c['vx0'] * u.m / u.s, vy0=c['vy0'] * u.m / u.s,
                                      theta0=c['theta0'] * u.rad, omega0=c['omega0'] * u.rad / u.s)
                  for c in stick_cases]

n_stick_frames = frames_per_case * len(sticks)




# ── UPDATE ──────────────────────────────────────────────────────────────
total_seconds = 18
Phase_jokersticks = 0
Phase_DOLL = 5

Phase_CIEL = 10
Phase_cielsticks = Phase_CIEL
total_frames = Phase_jokersticks + n_stick_frames
print("total_frames", total_frames)
frame_dt = total_seconds / total_frames

SpinStick = SOLN.SpinningStick()
theta_spinning_sticks_constant = SpinStick.omega_initial * frame_dt * np.arange(total_frames)

T_half = np.pi / (SpinStick.omega_initial * frame_dt)   # render frames per half spin-turn
n_bounces = int(np.ceil((total_frames - Phase_cielsticks) / T_half)) + 1
landing_targets = [int(round((k + 1) * T_half)) for k in range(n_bounces)]
flight_frames = [landing_targets[0] - Phase_cielsticks + 1] + [
    landing_targets[k] - landing_targets[k - 1] for k in range(1, n_bounces)
]

result_joker_stick = draw_soln_functions('joker stick', [B_SCENE.STICK_LOC_joker, sticks] )
result_ciel_stick = draw_soln_functions('ciel stick', [(B_SCENE.STICK_ROW_ciel, B_SCENE.ciel_loc[1] ), flight_frames] )

seg_dur, n_segments = 4.0, 8
theta_spinning_sticks_longshort = np.empty(total_frames)
long_flags_longshort = np.empty(total_frames, dtype=bool)
for seg, idx in enumerate(np.array_split(np.arange(total_frames), n_segments)):
    t_local = np.linspace(0.0, seg_dur, len(idx)) * u.s
    fast = seg % 2 == 1            # fast spin == arms pulled in == "short"
    long_flags_longshort[idx] = not fast
    theta_spinning_sticks_longshort[idx] = (SpinStick.theta_final(t_local) if fast
                                             else SpinStick.theta_initial(t_local))

    
def update(snapshot):
    current_hex_colors = background_hex_colors.copy()
    
    current_jokersticks_snapshot = snapshot - Phase_jokersticks
    current_cielsticks_snapshot = snapshot - Phase_cielsticks
    
    current_CIEL_snapshot = snapshot-Phase_CIEL
    current_DOLL_snapshot = snapshot-Phase_DOLL
    
    if snapshot>=Phase_jokersticks:
        
        current_dict = {'JOKER STICK': draw_animtaion(current_jokersticks_snapshot, detail_info, 'joker stick', 
                                                      result_joker_stick),
                       
                       }
        
    
    
    if snapshot>=Phase_DOLL:
        
        current_dict = {'JOKER STICK': draw_animtaion(current_jokersticks_snapshot, detail_info, 'joker stick', 
                                                      result_joker_stick),
                       'doll': draw_animtaion(current_DOLL_snapshot, detail_info, 'doll', 
                                               [theta_spinning_sticks_longshort, long_flags_longshort]), 
                        
                       }
        
    if snapshot>=Phase_CIEL:
        
        current_dict = {'JOKER STICK': draw_animtaion(current_jokersticks_snapshot, detail_info, 'joker stick', 
                                                      result_joker_stick),
                       'ciel': draw_animtaion(current_CIEL_snapshot, detail_info, 'ciel', [theta_spinning_sticks_constant]), 
                        'ciel STICK': draw_animtaion(current_cielsticks_snapshot, detail_info, 'ciel stick', 
                                                      result_ciel_stick),
                       
                        'doll': draw_animtaion(current_DOLL_snapshot, detail_info, 'doll', 
                                               [theta_spinning_sticks_longshort, long_flags_longshort]), 
                       }
        
    for char, dict_ in current_dict.items(): 
        for key in dict_.keys():

            part  = dict_.get(key)[0]
            
            colors = dict_.get(key)[1] 

            m = re.search(r"sigmafactor\s*=\s*([\d.]+)", str(colors))
            sigmafactor = float(m.group(1)) if m else 1.

            is_single_with_modifier = len(colors) == 2 and isinstance(colors[1], str) and 'sigmafactor' in colors[1]


            is_striped = len(colors) == 3 and isinstance(colors[-1], str) and 'stripe' in colors[-1]
            if is_striped:
                select_part= cg.select_mask(part,hex_rc_arr)
                stripe_target_colors = cgc.color_stripe(hex_rc_arr, STRIPE_WIDTH=1, 
                                                        STRIPE_PALETTE=[np.array(colors[0]), np.array(colors[1])])
                current_hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color*2)[0]
                                                        for c in stripe_target_colors[select_part] 
                                                       ])

                continue
            if len(colors) == 1 or is_single_with_modifier:

                select_part= cg.select_mask(part,hex_rc_arr)
                current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*sigmafactor) 

            else:
                if len(colors) == len(hex_rc_arr):
                    print('FULLY DEFINED COLORS')
                    select_part = cg.select_mask(part, hex_rc_arr)

                    for idx in np.where(select_part)[0]:
                        current_hex_colors[idx] = cgc.select_normal_color([True], colors[idx], np.ones(3) * sigma_color*sigmafactor)[0]
                else:
                    sort_info = colors[-1]
                    if 'hex' in str(sort_info):
                        print('sored by hex')
                        print(part)

                        center =sort_info[0]
                        n_layers = sort_info[1]
                        start_n = sort_info[2]
                        end_n=sort_info[3]
                        if 'sigmoid' in str(sort_info):
                            mode = 'sigmoid'
                        else:
                            mode = 'linear'

                        print(mode)
                        current_hex_colors = cgc.color_hex_gradient(part,colors[0],colors[1], hex_rc_arr, 
                                current_hex_colors, center, n_layers,
                                sigma_color=sigma_color*sigmafactor, end_weight=0.01, period=None, mode=mode,
                                start_n=start_n, end_n=end_n)
                    elif 'col' in sort_info:
                        print(key)
                        print('col gradient')
                        if 'max' in sort_info:
                            pivot = 'max'
                        elif 'min' in sort_info:
                            pivot = 'min'
                        else:
                            pivot = None
                        if 'sigmoid' in sort_info:
                            mode = 'sigmoid'
                        else:
                            mode = 'linear'
                        print('mode', mode)
                        current_hex_colors = cgc.color_row_gradient(part, 
                                colors[0],colors[1],
                                hex_rc_arr, current_hex_colors, sort = 'col', sigma_color = sigma_color*sigmafactor, 
                                                            end_weight= 0.01, mode = mode, pivot=pivot ) 

                    else:
                        print(key)
                        print('row gradient')

                        if 'sigmoid' in sort_info:
                            mode = 'sigmoid'
                        else:
                            mode = 'linear'

                        if 'max' in sort_info:
                            pivot = 'max'
                        elif 'min' in sort_info:
                            pivot = 'min'
                        else:
                            pivot = None

                        print('mode', mode, 'pivot', pivot)

                        current_hex_colors = cgc.color_row_gradient(part, 
                                colors[0],colors[1],
                                hex_rc_arr, current_hex_colors, sort = 'row', sigma_color = sigma_color*sigmafactor, 
                                                            end_weight= 0.01, mode = mode, pivot=pivot )

  
    pc.set_facecolor(current_hex_colors)
    return (pc,)



FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'BlackButler_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")