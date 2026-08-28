import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import astropy.units as u
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import expected_value as EXP

import  DIAGRAM_periodigram_timeseries as DIAG

rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info


sigma_color = 0.02
rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}


def draw_kuze(head_center_row, head_center_col, n_head):
    head, neck, body, pelvis, thigh, raw_body_detail= cgd.draw_body(head_center_row, head_center_col, n = n_head)
    raw_body = head+ neck+ body+ pelvis+ thigh#+arm
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail
    
    face_raw =cgd.draw_trapezoid(head_center_row, 
                               center_col-1, 
                               center_col+1, 
                               head_center_row+n_head, slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                  bend_left = 'right', bend_right = 'left')[-2]
    face_remove =cgd.horizontal_lines([(head_center_row, (center_col+1, center_col+n_head))])
    face  =[(r,c ) for r,c in face_raw if (r,c) not in face_remove]
    
    
    remove_hair = [(head_center_row, center_col-n_head), (head_center_row, center_col+n_head)]
    add_hair_raw = cg.hex_neighbours_n(head_center_row, center_col, n=n_head+1, 
                                                   keep_origin = False, return_frontier=True)[-1]
    
    add_hair = cgd.verticle_line(head_center_row+1, 
                                 min(c for r,c in head if r == head_center_row+1), max(r for r,c in pelvis), bend = 'right'
                            )+cgd.verticle_line(head_center_row, 
                                                center_col+n_head, max(r for r,c in pelvis), bend = 'right'
                            )+[(r,c) for r,c in add_hair_raw if r <head_center_row and c >=center_col-1]
    
    hair = [(r,c) for r,c in head if (r,c) not in remove_hair ]+add_hair
    
    body_top_r = min(r for r, c in body)
    body_bot_r = max(r for r, c in body)
    
    remove_body = [(body_center_row, center_col-n_head), (body_center_row, center_col+n_head)]
    tops = [(r,c) for r,c in body if r <body_bot_r]+neck
    
    belt = cgd.draw_trapezoid(body_bot_r, 
                               min(c for r,c in body if r == body_bot_r)-1, 
                               max(c for r,c in body if r == body_bot_r)+1, 
                               max(r for r,c in pelvis), slope_left = 'inf', slope_right = 'inf', direction = 'lr',
                  bend_left = 'right', bend_right = 'left')[-2]
    
    right_shoulder_joint = cg.hex_neighbours_n(body_top_r, max(c for r,c in body if r ==body_top_r)+1, n=1, 
                                                   keep_origin = True, return_frontier=False)
    right_brachium_raw = cgd.draw_trapezoid(max(r for r,c in right_shoulder_joint), 
                               min(c for r,c in right_shoulder_joint if r == max(r for r,c in right_shoulder_joint)), 
                               max(c for r,c in right_shoulder_joint if r == max(r for r,c in right_shoulder_joint)), 
                               max(r for r,c in pelvis), slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                  bend_left = 'right', bend_right = 'right')[-2]
    remove_right_brachium = [(max(r for r,c in right_brachium_raw),
                              min(c for r,c in right_brachium_raw if r == max(r for r,c in right_brachium_raw)) ),
                            (max(r for r,c in right_brachium_raw)-1,
                              min(c for r,c in right_brachium_raw if r == max(r for r,c in right_brachium_raw)-1) )]
    right_brachium  =[(r,c ) for r,c in right_brachium_raw if (r,c) not in remove_right_brachium]
    
    right_forearm = cgd.draw_trapezoid(max(r for r,c in right_brachium), 
                               min(c for r,c in right_brachium if r == max(r for r,c in right_brachium)), 
                               max(c for r,c in right_brachium if r == max(r for r,c in right_brachium)), 
                               max(r for r,c in thigh), slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                  bend_left = 'right', bend_right = 'right')[-2]
    
    hand = cgd.draw_trapezoid(max(r for r,c in right_forearm), 
                               min(c for r,c in right_forearm if r == max(r for r,c in right_forearm))+1, 
                               min(c for r,c in right_forearm if r == max(r for r,c in right_forearm))+3, 
                               max(r for r,c in right_forearm)+n_head+n_head, slope_left = '0.5', slope_right = '0.5', direction = 'll',
                  bend_left = 'right', bend_right = 'right')[-2]
    
    extra_sleeves =cgd.draw_trapezoid(max(r for r,c in right_forearm)-1, 
                               min(c for r,c in right_forearm if r == max(r for r,c in right_forearm)-1), 
                               max(c for r,c in right_forearm if r == max(r for r,c in right_forearm)-1), 
                               max(r for r,c in thigh)+n_head+n_head, slope_left = '0.5', slope_right = 'inf', direction = 'rl',
                  bend_left = 'right', bend_right = 'right')[-2]+cgd.verticle_line(body_top_r, 
                                 max(c for r,c in right_shoulder_joint if r == body_top_r), max(r for r,c in pelvis), bend = 'right'
                            )
    sleeves =right_shoulder_joint+right_brachium+right_forearm+extra_sleeves
    
    
    hakama_raw = cg.hex_neighbours_n(thigh_center_row-1, center_col-n_head-n_head-n_head//2, n=n_head, 
                                                   keep_origin = True, return_frontier=False
                                    )+cg.hex_neighbours_n(thigh_center_row-2, center_col-n_head-n_head-n_head//2-n_head-n_head, n=n_head, 
                                                   keep_origin = True, return_frontier=False
                                    )+cg.hex_neighbours_n(thigh_center_row+n_head, 
                                                          center_col-n_head-n_head-n_head-n_head//2-n_head-n_head, n=n_head, 
                                                   keep_origin = True, return_frontier=False
                                    )+thigh
    
    
    hakama_part1 = cgd.draw_trapezoid(thigh_center_row+n_head, 
                               min(c for r,c in hakama_raw if r == thigh_center_row+n_head)+n_head+n_head, 
                               max(c for r,c in hakama_raw if r == thigh_center_row+n_head)-n_head, 
                               thigh_center_row-n_head, slope_left = 'inf', slope_right = 'inf', direction = 'rl',
                  bend_left = 'right', bend_right = 'left')[-2]+cgd.draw_trapezoid(thigh_center_row+n_head, 
                               min(c for r,c in hakama_raw if r == thigh_center_row+n_head)+n_head+n_head, 
                               max(c for r,c in hakama_raw if r == thigh_center_row+n_head), 
                               max(r for r,c in hakama_raw), slope_left = 'inf', slope_right = '1.5', direction = 'rl',
                  bend_left = 'right', bend_right = 'left')[-2]
    hakama = hakama_raw+hakama_part1
    return { 
            
        #'raw_body': [raw_body, [[0,0,0]]],
        'hair': [list(set(hair)), [[0,0,0]]],
        'face': [list(set(face)), [cgc.hex_to_rgb("#fee9d2")]],
        
        
        'belt': [ list(set(belt)), [[0.2,0.2,0.2]]],
       
        'tops': [list(set(tops)), [[0.5,0.5,0.5]]],
        
        'hakama': [list(set(hakama)), [[0,0,0]]],
        'hand': [list(set(hand)), [cgc.hex_to_rgb("#fee9d2")]],
        
        'sleeves': [list(set(sleeves)), [[0,0,1]]],
            }
def draw_bkgd(start_row, start_left_col, start_right_col, end_row, hex_rc_arr):
    left_LINE, _, _, _, water_region, _ = cgd.draw_trapezoid(start_row, start_left_col, start_right_col, end_row, 
                                      slope_left = '8/3', slope_right = '0.5', direction = 'lr',
                  bend_left = 'right', bend_right = 'right')
    wall_region = [(r,c) for r,c in hex_rc_arr if (r,c) not in water_region]
    return { 
       
        'wall': [list(set(wall_region)), [[0,0,0]]],
         'left_LINE': [list(set(left_LINE)), [cgc.hex_to_rgb("#717a8d"), cgc.hex_to_rgb("#132240")]],
        'water': [list(set(water_region)), [cgc.hex_to_rgb("#717a8d"), cgc.hex_to_rgb("#132240")]],
       }

def compute_color_copys(N_in_bin, noise_type_num, Nbkg_in_bin=0., seed=1, threshold=0.1):
    """
    Rebuild the recurrence-plot-derived color_copys list for a given
    DIAG_full configuration. Same blending pipeline as the original
    static color_copys -- only the DIAG_full inputs vary, so this can
    be called with a different variant each animation frame.
    """
    noise_type = 'none' if noise_type_num == 0 else 'poisson'
    DIAG_variant = DIAG.TimeSeriesPeriodogram(Ad=0.03, Phase=0 * u.deg,
                     N_in_bin=N_in_bin, Nbkg_in_bin=Nbkg_in_bin,
                     P=Period, T=5 * u.yr, dt=5*u.day,
                     freqs=np.linspace(f0.value, 1, 2000)/u.s,
                     noise_type=noise_type, threshold=threshold,
                     seed=seed)

    _, recurrence_matrix_v = DIAG_variant.recurrence_plot()
    t_v = DIAG_variant.time_bin_centers.to(u.day).value
    i_idx, j_idx = np.where(recurrence_matrix_v)
    X_v = t_v[i_idx]
    Y_v = t_v[j_idx]

    DOMAIN_W = X_v.max() - X_v.min()
    DOMAIN_H = Y_v.max() - Y_v.min()
    canvas_physical_x_range = (0, DOMAIN_W)
    canvas_physical_y_range = (0, DOMAIN_H)

    px0, py0 = cg.hex_center_pixel(22, 0, detail_info)
    OFFSET_X = px0 * DOMAIN_W / IMG_W_SCENE - X_v.min()
    OFFSET_Y = DOMAIN_H * (1.0 - py0 / IMG_H_SCENE) - Y_v.min()

    grid_hex_rc_v = cg.world_metres_to_hex_index(
        X_v + OFFSET_X, Y_v + OFFSET_Y, detail_info,
        canvas_physical_x_range=canvas_physical_x_range,
        canvas_physical_y_range=canvas_physical_y_range,
    )
    hex_to_grid_idx_v = {}
    for flat_i, rc in enumerate(grid_hex_rc_v):
        hex_to_grid_idx_v.setdefault(rc, []).append(flat_i)
    hex_pos_to_grid_idx_v = {
        rc_to_idx[rc]: idxs
        for rc, idxs in hex_to_grid_idx_v.items()
        if rc in rc_to_idx
    }

    max_count = max((len(idxs) for idxs in hex_pos_to_grid_idx_v.values()), default=1)

    color_copys_v = []
    for bkgd_color, blend_color in zip(bkgd_colors, blend_colors):
        base_layer = np.full((len(hex_rc_arr), 3), bkgd_color, dtype=float)
        hex_idx_to_color_v = {}
        for hex_idx in range(len(hex_rc_arr)):
            idxs = hex_pos_to_grid_idx_v.get(hex_idx, [])
            a = len(idxs) / max_count
            blend_weight = RECURRENCE_ALPHA * a
            hex_idx_to_color_v[hex_idx] = (
                (1 - blend_weight) * base_layer[hex_idx]
                + blend_weight * np.array(blend_color)
            )
        layer = np.array([
            cgc.select_normal_color([True], hex_idx_to_color_v[i], np.ones(3) * sigma_color)[0]
            for i in range(len(hex_rc_arr))
        ])
        color_copys_v.append(layer)

    return color_copys_v
#------read solution-----------------------
Period = 1*u.yr
f0 = (1/Period).to(1/u.s)
DIAG_full = DIAG.TimeSeriesPeriodogram(Ad=0.03,
                 Phase=0 * u.deg,
                 N_in_bin=10, Nbkg_in_bin=0.,
                 P=Period, T=5 * u.yr, dt=5*u.day,
                 freqs=np.linspace(f0.value,1, 2000)/u.s  ,
                 noise_type='none',threshold = 0.1,
                 seed=1)

power = DIAG_full.run_periodogram().value
_, recurrence_matrix = DIAG_full.recurrence_plot()
t  = DIAG_full.time_bin_centers.to(u.day).value
i_idx, j_idx = np.where(recurrence_matrix)
X = t[i_idx]
Y = t[j_idx]

rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}


DOMAIN_W = X.max() - X.min()
DOMAIN_H = Y.max() - Y.min()
CANVAS_PHYSICAL_W = DOMAIN_W
CANVAS_PHYSICAL_H = DOMAIN_H
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
 
px0, py0 = cg.hex_center_pixel(22, 0, detail_info)
OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X.min()
OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y.min()

grid_hex_rc = cg.world_metres_to_hex_index(
    X + OFFSET_X, Y + OFFSET_Y, detail_info,
    canvas_physical_x_range=canvas_physical_x_range,
    canvas_physical_y_range=canvas_physical_y_range,
)
hex_to_grid_idx = {}
for flat_i, rc in enumerate(grid_hex_rc):
    hex_to_grid_idx.setdefault(rc, []).append(flat_i)
hex_pos_to_grid_idx = {
    rc_to_idx[rc]: idxs
    for rc, idxs in hex_to_grid_idx.items()
    if rc in rc_to_idx
}

bkgd_colors = [cgc.hex_to_rgb("#010049"), cgc.hex_to_rgb("#f1f1f1"), 
               [0,0,1], cgc.hex_to_rgb("#e4f3fa" ), [0,0,0] ]
blend_colors = [cgc.hex_to_rgb("#0601d1" ), cgc.hex_to_rgb("#9291c6" ), 
                cgc.hex_to_rgb("#03008b" ) , cgc.hex_to_rgb("#a3abbb" ), cgc.hex_to_rgb("#04225b" )  ]
color_copys = []

RECURRENCE_ALPHA = 1   

max_count = max((len(idxs) for idxs in hex_pos_to_grid_idx.values()), default=1)
for bkgd_color, blend_color in zip(bkgd_colors, blend_colors):
    hex_colors[[True]*len(hex_rc_arr)] =bkgd_color
    RECURRENCE_FILL_COLOR = blend_color
    hex_idx_to_color = {}
    for hex_idx in range(len(hex_rc_arr)):
        idxs = hex_pos_to_grid_idx.get(hex_idx, [])
        a = len(idxs) / max_count            # 0 at no recurrence, up to 1 at max density
        blend_weight = RECURRENCE_ALPHA * a  # 0 at a=0 (fully base), rising toward RECURRENCE_ALPHA
        hex_idx_to_color[hex_idx] = (
            (1 - blend_weight) * hex_colors[hex_idx]
            + blend_weight * np.array(RECURRENCE_FILL_COLOR)
        )

    hex_colors = np.array([
                cgc.select_normal_color([True], hex_idx_to_color[i], np.ones(3) * sigma_color)[0]
                for i in range(len(hex_rc_arr))
            ])
    color_copys.append(hex_colors.copy())    

kuze_center = (6, 30)
n_head = 2

char_dict_ = {'bkgd': draw_bkgd(0, 34, 34, 22, hex_rc_arr),
    'kuze reika': draw_kuze(kuze_center[0], kuze_center[1], n_head),
             }
foreground_points = []
for char_name, dict_ in char_dict_.items(): 

    for key in dict_.keys():
        part  = dict_.get(key)[0]
        if char_name == 'kuze reika':
            foreground_points.extend([(r,c) for r,c in part if r <17])
        colors = dict_.get(key)[1] 
        select_part= cg.select_mask(part,hex_rc_arr)
        if len(colors) == 1:
            if key in ["hakama"]:
                hex_colors[select_part] = color_copys[0][select_part]
            elif key in ['sleeves']:
                 hex_colors[select_part] = color_copys[1][select_part]
            elif key in ['belt']: 
                hex_colors[select_part] = color_copys[2][select_part]
            elif key in ['tops']: 
                hex_colors[select_part] = color_copys[3][select_part]
            elif key in ['wall']: 
                hex_colors[select_part] = color_copys[-1][select_part]
            else:
                hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01,mode='linear') 

foreground_mask = cg.select_mask(foreground_points, hex_rc_arr)

animation_base_hex_colors = hex_colors.copy()  
start_coord = sorted([(r,c) for r, c in char_dict_.get('bkgd').get('left_LINE')[0] if (r,c) in hex_rc_arr], reverse=True)
full_start_coord = [(i, -1) for i in np.arange(22,  start_coord[0][0], -1)]+start_coord
print(full_start_coord)

#------read solution-----------------------
power = DIAG_full.run_periodogram().value
freqs = DIAG_full.freqs.value

X_freqs = freqs
Y_freqs = power



#time series
grid_hex_rc_rows = dict()
river_frames = 0
for s, start_coord in enumerate(full_start_coord):
    DIAG_idvl = DIAG.TimeSeriesPeriodogram(Ad=0.03,
                 Phase=10 * u.deg,
                 N_in_bin=10, Nbkg_in_bin=5,
                 P=Period, T=10 * u.yr, dt=30*u.day,
                 freqs=np.linspace(f0.value,1, 2000)/u.s  ,
                 noise_type='poisson',threshold = 0.1,
                 seed=(s+1)*10)
    
    t  = DIAG_idvl.time_bin_centers.to(u.day).value
    counts = DIAG_idvl.add_noise()
    
    X = t
    Y = counts
    DOMAIN_W = X.max() - X.min()
    DOMAIN_H = Y.max() - Y.min()
    CANVAS_PHYSICAL_H = DOMAIN_H*4
    CANVAS_PHYSICAL_W = DOMAIN_W
    canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
    canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)

    px0, py0 = cg.hex_center_pixel(start_coord[0], start_coord[1], detail_info)

    OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
    OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]

    grid_hex_rc = cg.world_metres_to_hex_index(
        X + OFFSET_X, Y + OFFSET_Y, detail_info,
        canvas_physical_x_range=canvas_physical_x_range,
        canvas_physical_y_range=canvas_physical_y_range,
    )
    valid_grid_hex_rc = [(r,c) for r,c in grid_hex_rc if (r,c) in hex_rc_arr]
    grid_hex_rc_rows[s]=valid_grid_hex_rc
    #river_frames+=len(valid_grid_hex_rc)
river_frames = max(s + len(grid_hex_rc_rows[s]) for s in grid_hex_rc_rows)


#POWER 
X = X_freqs
Y = Y_freqs

DOMAIN_W = X.max() - X.min()
DOMAIN_H = Y.max() - Y.min()
CANVAS_PHYSICAL_H = DOMAIN_H*0.7

CANVAS_PHYSICAL_W = DOMAIN_W
    
print(DOMAIN_W, DOMAIN_H)
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)

px0, py0 = cg.hex_center_pixel(22, 0, detail_info)

OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X.min()
OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y.min()


N_Y_SAMPLES = river_frames//3
frac = np.linspace(0.0, 1.0, N_Y_SAMPLES)
X_samples = np.repeat(X, N_Y_SAMPLES)
Y_samples = np.concatenate([frac * Y[i] for i in range(len(X))])
frac_flat = np.tile(frac, len(X))
 
grid_hex_rc_samples = np.array(cg.world_metres_to_hex_index(
    X_samples + OFFSET_X, Y_samples + OFFSET_Y, detail_info,
    canvas_physical_x_range=canvas_physical_x_range,
    canvas_physical_y_range=canvas_physical_y_range,
))
 
print(grid_hex_rc_samples[0])

# ── UPDATE ──────────────────────────────────────────────────────────────
Phase0_end_part1 = 5           # intro frames that just hold the base scene
Phase0_end= 10         # intro frames that just hold the base scene
total_seconds = 18.0
print(river_frames)
alpha = 0.5
total_frames = Phase0_end+river_frames+N_Y_SAMPLES#+len(X)#N_Y_SAMPLES
visited_idx_rows = {s: [] for s in grid_hex_rc_rows}
rows_colors = cgc.gradient_sequence_colors(max(len(v) for v in grid_hex_rc_rows.values()) , [0,0,0], [0,0,1], mode='linear')  # or 'sigmoid'

visited_color_rows = [[] for _ in grid_hex_rc_rows] 

visited_idx = []
def update(snapshot):
    if snapshot < Phase0_end_part1:
        current_hex_colors = animation_base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot - Phase0_end_part1
        current_hex_colors = animation_base_hex_colors.copy()
        N_in_bin = rng.uniform(5, 20)                    # random float in [5, 20)
        noise_type_num = int(rng.integers(0, 2))   
        
        color_copys_frame = compute_color_copys(N_in_bin, noise_type_num, Nbkg_in_bin=0., seed=500+current_snapshot, threshold=0.2)

    
        for char_name, dict_ in char_dict_.items():
            for key in dict_.keys():
                part = dict_.get(key)[0]
                
                select_part = cg.select_mask(part, hex_rc_arr)
                if key in ['wall']:
                    current_hex_colors[select_part] = color_copys_frame[-1][select_part]
        character_colored = current_hex_colors.copy() 
        
        if snapshot>=Phase0_end:
            river_snapshot = snapshot - Phase0_end
            
            #time series
            for s, grid_hex_rc in grid_hex_rc_rows.items():
                if river_snapshot >= s:
                    local_idx = river_snapshot - s
                    if local_idx < len(grid_hex_rc):         
                        rc = grid_hex_rc[local_idx]
                        idx = rc_to_idx.get(rc)
                        if idx is not None:
                            visited_idx_rows[s].append(idx)
                            visited_color_rows[s].append(rows_colors[local_idx])

            for s in grid_hex_rc_rows:
                visited_idx_row = visited_idx_rows[s]
                visited_color_row = visited_color_rows[s]
                if visited_idx_row:
                    new_colors = np.array([
                        cgc.select_normal_color([True], c, np.ones(3) * sigma_color)[0]
                        for c in visited_color_row
                    ])
                    background = animation_base_hex_colors[visited_idx_row]   # or animation_base_hex_colors
                    current_hex_colors[visited_idx_row] = alpha * new_colors + (1 - alpha) * background

            current_hex_colors[foreground_mask] = animation_base_hex_colors[foreground_mask]  # keep silhouette on top
            pc.set_facecolor(current_hex_colors)
            
            power_start_frame = river_frames-N_Y_SAMPLES//10
            if river_snapshot  >= power_start_frame:
                #POWER 
                power_current_hex_colors = river_snapshot -power_start_frame
                growth = min((power_current_hex_colors + 1) / N_Y_SAMPLES, 1.0)   # 0 -> 1

                mask = frac_flat <= growth
                active_rc = set(map(tuple, grid_hex_rc_samples[mask]))
                select = cg.select_mask(list(active_rc), hex_rc_arr)
                current_hex_colors[select] = cgc.select_normal_color(select, [0, 0, 1], np.ones(3)*sigma_color*3)  
                pc.set_facecolor(current_hex_colors)


    return (pc,)
total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'fatalframe_animation.mp4')
 
print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")
 
