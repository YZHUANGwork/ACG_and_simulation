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
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import cg_draw_figure as cgf

import SOLUTION_Schrodinger as SOLN
import HF_scene_plot as HF_SCENE
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
HEX_R_start = 22
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=HEX_R_start, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.03


hex_colors= cgc.select_normal_color([True]*len(hex_colors), cgc.hex_to_rgb("#ebeaff"), np.ones(3)*sigma_color)  #stripe_bkgd_colors


n_head = 3
head_center_row = 5
head_center_col = 17
anchor_x, anchor_y = cg.hex_center_pixel(head_center_row, head_center_col, detail_info)
#---------------animation_base_hex_colors_phase1----------------------------
target_STRIPE_PALETTE = [np.array([0,0,0]),cgc.hex_to_rgb("#bb0000"),]
stripe_target_colors = cgc.color_stripe(hex_rc_arr, 
                                        STRIPE_WIDTH = 1 ,
                                        STRIPE_PALETTE = target_STRIPE_PALETTE)#cgc.hex_to_rgb("#bb0000")
stripe_skin_colors = cgc.color_stripe(hex_rc_arr, 
                                        STRIPE_WIDTH = 1 ,
                                        STRIPE_PALETTE = [cgc.hex_to_rgb("#fee9d2"),cgc.hex_to_rgb("#bb0000"), ])

dict_, _ = HF_SCENE.draw_Sakura(head_center_row, head_center_col, n_head)
for key in dict_.keys():
    part  = dict_.get(key)[0]
    colors = dict_.get(key)[1] 
    if len(colors) == 1:
        
        select_part= cg.select_mask(part,hex_rc_arr)
        if key == 'raw_body':
            hex_colors[select_part] = stripe_target_colors[select_part] 
        elif key == 'stripe_skin':
            hex_colors[select_part] = stripe_skin_colors[select_part] 
        else:
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
    else:
        hex_colors = cgc.color_row_gradient(part, 
                                    colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, 
                                                end_weight= 0.01, mode = 'linear') 
animation_base_hex_colors = hex_colors.copy()
final_phase2_colors = animation_base_hex_colors.copy()  

#------ solution PARAMES-----------------------
PARAMS= {
     "LEFT1":{"x_start": -1., "x_end": 0., "E": 1., "A":0.3, "B":0., "W_SCALE":5, "H_SCALE":5},#LEFT SLEEVES,
     "LEFT2":{"x_start": -1., "x_end": 0., "E": 2.5, "A":0.5, "B":0.,"W_SCALE":4, "H_SCALE":6 },#LEFT SLEEVES,
   
    
     "LEFT3":{"x_start": -2., "x_end": 0., "E": 1.5, "A":1., "B":0.,"W_SCALE":3, "H_SCALE":6 },
    "LEFT4":{"x_start": -2., "x_end": 0., "E":3. , "A":2., "B":0., "W_SCALE":6, "H_SCALE":3},
    "LEFT5":{"x_start": -2., "x_end": 0., "E":0.5 , "A":0.5, "B":0., "W_SCALE":5, "H_SCALE":3},
    

     "RIGHT1":{"x_start": 0., "x_end": 1., "E": 1, "A":0., "B":0.3,"W_SCALE":5, "H_SCALE":5 },#RIGHT SLEEVES
    "RIGHT2":{"x_start": 0., "x_end": 1., "E": 2.5, "A":0., "B":0.5,"W_SCALE":6, "H_SCALE":4  },#RIGHT SLEEVES
    
    "RIGHT3":{"x_start": 0., "x_end": 2., "E": 3., "A":0., "B":1., "W_SCALE":3, "H_SCALE":6 },
    "RIGHT4":{"x_start": 0., "x_end": 2., "E": 1.5, "A":0., "B":2., "W_SCALE":2, "H_SCALE":8},
    "RIGHT5":{"x_start": 0., "x_end": 2., "E": 0.5, "A":0., "B":0.5, "W_SCALE":5, "H_SCALE":5},
    
}


sol_well = SOLN.InfiniteWell(
    m=const.m_e * const.c ** 2,
    x=np.linspace(-0.3, 1.3, 300) * u.nm,
    t=0 * u.fs,
    L=1 * u.nm, n_max=6,
)
_, energies, states = sol_well.solve(t=0 * u.fs)
x_well = sol_well.x.to_value(u.nm)
Lv = sol_well.L.to_value(u.nm)
X = x_well

DOMAIN_H_ref = max(state.real.max() - state.real.min() for state in states)
CANVAS_PHYSICAL_H = DOMAIN_H_ref  # computed ONCE, same for every state

grid_hex_rc_wells = []
max_len_wells = 0
for s, state in enumerate(states):
    Y = state.real
    DOMAIN_W = X.max() - X.min()
    CANVAS_PHYSICAL_W = DOMAIN_W
    canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
    canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
 
    px0, py0 = cg.hex_center_pixel(max(r for r, c in hex_rc_arr)//2, 0, detail_info)
    OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
    OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]
    
    VALID_X = (X > 0.) & (X <= 1.)
    grid_hex_rc = cg.world_metres_to_hex_index(
            X[VALID_X] + OFFSET_X, Y[VALID_X] + OFFSET_Y, detail_info,
            canvas_physical_x_range=canvas_physical_x_range,
            canvas_physical_y_range=canvas_physical_y_range,
        )
    # left-to-right (ascending column) so the final phase can reveal each
    # state's curve one hex per snapshot, walking left to right
    VALID_grid_hex_rc = sorted(dict.fromkeys((r, c) for r, c in grid_hex_rc if (r, c) in hex_rc_arr),key=lambda rc: rc[1])
    grid_hex_rc_wells.append(VALID_grid_hex_rc)
    
    
well_cum_boundaries = []
_running_wells = 0
for _cells in grid_hex_rc_wells:
    _running_wells += max(len(_cells), 1)
    well_cum_boundaries.append(_running_wells)
print("_running_wells", _running_wells) 
def resolve_well_state_and_reveal(current_snapshot3):
    """Map a snapshot index (0-based, within the final phase) to (active_state,
    n_reveal) -- n_reveal counts up by exactly 1 per snapshot within that
    state's own occupancy, resetting to 1 each time the active state changes.
    """
    prev_boundary = 0
    for s, boundary in enumerate(well_cum_boundaries):
        if current_snapshot3 < boundary:
            return s, current_snapshot3 - prev_boundary + 1
        prev_boundary = boundary
    return len(grid_hex_rc_wells) - 1, len(grid_hex_rc_wells[-1])

# ── UPDATE ──────────────────────────────────────────────────────────────
BKGD_SEGS_WIDTH = 5    # row spacing between background wave lines
HEX_R_end = 6.#HEX_R_start-5
len_shrink = int((HEX_R_start - HEX_R_end)/1 + 1)   # smallest HEX_R reached once shrinking stops
print("len_shrink", len_shrink, )
bin_num =600
bin_num_perR = int(bin_num // len_shrink)



_end_fig, _end_ax, _end_patches, _end_hex_colors, _end_centers, _end_hex_rc_arr, _end_pc, _end_detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=HEX_R_end, DPI=DPI, hex_index=HEX_INDEX, z_order_max=z_order_max)
plt.close(_end_fig)
_end_head_row, _end_head_col = cg.pixel_to_nearest_hex(anchor_x, anchor_y, _end_detail_info)
_end_head_row, _end_head_col = int(_end_head_row), int(_end_head_col)
_end_dict_, _ = HF_SCENE.draw_Sakura(_end_head_row, _end_head_col, n_head)
_raw_body_ref = _end_dict_['raw_body'][0]
bottom_row_ref = max(r for r, c in _raw_body_ref)
bottom_col_ref = _end_head_col
delta_row_bottom = bottom_row_ref - _end_head_row
anchor_bottom_x, anchor_bottom_y = cg.hex_center_pixel(bottom_row_ref, bottom_col_ref, _end_detail_info)


def well_x_to_col(x_val, x_domain, hex_rc_arr_, detail_info_):
    """Map a physical x-coordinate (nm) from the well domain to its hex
    column, using the same anchor point (max_row//2, col=0) and projection
    formula as the commented well-curve code. Row is irrelevant here."""
    IMG_W_SCENE_, IMG_H_SCENE_, HEX_R_, dx_, dy_ = detail_info_
    DOMAIN_W = x_domain.max() - x_domain.min()
    CANVAS_PHYSICAL_W = DOMAIN_W
    CANVAS_PHYSICAL_H = dy_   # arbitrary nonzero placeholder -- only column matters
    anchor_row = max(r for r, c in hex_rc_arr_) // 2
    px0, py0 = cg.hex_center_pixel(anchor_row, 0, detail_info_)
    OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE_ - x_domain[0]
    OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE_)
    grid_hex_rc = cg.world_metres_to_hex_index(
        np.array([x_val]) + OFFSET_X, np.array([0.0]) + OFFSET_Y,
        detail_info_,
        canvas_physical_x_range=(0, CANVAS_PHYSICAL_W),
        canvas_physical_y_range=(0, CANVAS_PHYSICAL_H))
    row_, col_ = grid_hex_rc[0]
    return int(col_.value) if hasattr(col_, 'value') else int(col_)

 
def build_frame_data(current_HEX_R):
    chunk_index = HEX_R_start - current_HEX_R   # 0 for the first/largest HEX_R, 1 for the next, ...
    chunk_start = chunk_index * bin_num_perR
    chunk_end = min(chunk_start + bin_num_perR, bin_num)
 
    _fig, _ax, patches_, hex_colors_, _centers, hex_rc_arr_, _pc, detail_info_ = cg.make_hex_scene(
        IMG_W=1280, IMG_H=720, HEX_R=current_HEX_R, DPI=DPI,
        hex_index=HEX_INDEX, z_order_max=z_order_max)
    plt.close(_fig)
 
    head_row, head_col = cg.pixel_to_nearest_hex(anchor_x, anchor_y, detail_info_)
    head_row, head_col = int(head_row), int(head_col)
    dict_here, stripe_pts = HF_SCENE.draw_Sakura(head_row, head_col, n_head)
 
    curves = {}
    static_curves = {}
    for key, pt in stripe_pts.items():
        PARAM = PARAMS[key]
        sol_free = SOLN.PlaneWave(
            m=const.m_e * const.c ** 2,
            x=np.linspace(PARAM["x_start"], PARAM["x_end"], bin_num) * u.nm,
            t=0 * u.fs, E=PARAM["E"] * u.eV, A=PARAM["A"], B=PARAM["B"])
        phi_free = sol_free.solve(t=0 * u.fs)
        x_free = sol_free.x
        row, col = pt
        if col < head_col:
            X, Y = x_free[::-1], phi_free[::-1].real
        else:
            X, Y = x_free, phi_free.real
 
        DOMAIN_W = X.max() - X.min()
        DOMAIN_H = Y.max() - Y.min()
        CANVAS_PHYSICAL_H = DOMAIN_H*PARAM["H_SCALE"]
        CANVAS_PHYSICAL_W = DOMAIN_W*PARAM["W_SCALE"]
        
        px0, py0 = cg.hex_center_pixel(pt[0], pt[1], detail_info_)
        OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
        OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]
        canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
        canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
 
        # prior chunks (already fully revealed in earlier HEX_R blocks) -- static
        X_static, Y_static = X[:chunk_start], Y[:chunk_start]
        if len(X_static) > 0:
            grid_hex_rc_static = cg.world_metres_to_hex_index(
                X_static + OFFSET_X, Y_static + OFFSET_Y, detail_info_,
                canvas_physical_x_range=canvas_physical_x_range,
                canvas_physical_y_range=canvas_physical_y_range)
            static_curves[key] = list(dict.fromkeys(
                (r, c.value) for r, c in grid_hex_rc_static if (r, c.value) in hex_rc_arr_))
        else:
            static_curves[key] = []
 
        # this HEX_R's own chunk -- progressively revealed
        X_chunk, Y_chunk = X[chunk_start:chunk_end], Y[chunk_start:chunk_end]
        grid_hex_rc = cg.world_metres_to_hex_index(
            X_chunk + OFFSET_X, Y_chunk + OFFSET_Y, detail_info_,
            canvas_physical_x_range=canvas_physical_x_range,
            canvas_physical_y_range=canvas_physical_y_range)
        curves[key] = list(dict.fromkeys((r, c.value) for r, c in grid_hex_rc if (r, c.value) in hex_rc_arr_))
    max_len = max((len(v) for v in curves.values()), default=0)
    
    
    # background wave lines -- alternating rows starting from the left or
    # right canvas edge, same static/current-chunk split as the limb curves.
    # Rebuilt fresh each HEX_R since the grid's own row/col range changes.
    max_row_ = max(r for r, c in hex_rc_arr_)
    max_col_ = max(c for r, c in hex_rc_arr_)
    seg_rows = np.arange(0, max_row_, BKGD_SEGS_WIDTH)
    BKGD_start_pts = []
    for i in range(len(seg_rows)):
        if i % 2 == 0:
            BKGD_start_pts.append((seg_rows[i], -1))
        else:
            BKGD_start_pts.append((seg_rows[i], max_col_))
 
    curves_bkgd = {}
    static_curves_bkgd = {}
    for i, BKGD_start_pt in enumerate(BKGD_start_pts):
        row, col = BKGD_start_pt
        if col > max_col_ // 2:   # start from right boundary
            sol_bkgd = SOLN.PlaneWave(
                m=const.m_e * const.c ** 2,
                x=np.linspace(-10, 0., bin_num) * u.nm,
                t=0 * u.fs, E=2 * u.eV, A=0., B=1.)
            phi_bkgd = sol_bkgd.solve(t=0 * u.fs)
            x_bkgd = sol_bkgd.x
            X_b, Y_b = x_bkgd[::-1], phi_bkgd[::-1].real
        else:
            sol_bkgd = SOLN.PlaneWave(
                m=const.m_e * const.c ** 2,
                x=np.linspace(0, 10, bin_num) * u.nm,
                t=0 * u.fs, E=2 * u.eV, A=1., B=0.)
            phi_bkgd = sol_bkgd.solve(t=0 * u.fs)
            x_bkgd = sol_bkgd.x
            X_b, Y_b = x_bkgd, phi_bkgd.real
 
        DOMAIN_W_b = X_b.max() - X_b.min()
        DOMAIN_H_b = Y_b.max() - Y_b.min()
        CANVAS_PHYSICAL_H_b = DOMAIN_H_b * BKGD_SEGS_WIDTH *2
        CANVAS_PHYSICAL_W_b = DOMAIN_W_b
        canvas_physical_x_range_b = (0, CANVAS_PHYSICAL_W_b)
        canvas_physical_y_range_b = (0, CANVAS_PHYSICAL_H_b)
 
        px0_b, py0_b = cg.hex_center_pixel(row, col, detail_info_)
        OFFSET_X_b = px0_b * CANVAS_PHYSICAL_W_b / IMG_W_SCENE - X_b[0]
        OFFSET_Y_b = CANVAS_PHYSICAL_H_b * (1.0 - py0_b / IMG_H_SCENE) - Y_b[0]
 
        X_static_b, Y_static_b = X_b[:chunk_start], Y_b[:chunk_start]
        if len(X_static_b) > 0:
            grid_hex_rc_static_b = cg.world_metres_to_hex_index(
                X_static_b + OFFSET_X_b, Y_static_b + OFFSET_Y_b, detail_info_,
                canvas_physical_x_range=canvas_physical_x_range_b,
                canvas_physical_y_range=canvas_physical_y_range_b)
            static_curves_bkgd[i] = list(dict.fromkeys(
                (r, c.value) for r, c in grid_hex_rc_static_b if (r, c.value) in hex_rc_arr_))
        else:
            static_curves_bkgd[i] = []
 
        X_chunk_b, Y_chunk_b = X_b[chunk_start:chunk_end], Y_b[chunk_start:chunk_end]
        grid_hex_rc_b = cg.world_metres_to_hex_index(
            X_chunk_b + OFFSET_X_b, Y_chunk_b + OFFSET_Y_b, detail_info_,
            canvas_physical_x_range=canvas_physical_x_range_b,
            canvas_physical_y_range=canvas_physical_y_range_b)
        curves_bkgd[i] = list(dict.fromkeys(
            (r, c.value) for r, c in grid_hex_rc_b if (r, c.value) in hex_rc_arr_))
 
    max_len = max((len(v) for v in curves.values()), default=0)
    max_len_bkgd = max((len(v) for v in curves_bkgd.values()), default=0)
    max_len = max(max_len, max_len_bkgd)
    return dict(patches=patches_, hex_rc_arr=hex_rc_arr_, detail_info=detail_info_,
                head_row=head_row, head_col=head_col, dict_=dict_here,
                stripe_pts=stripe_pts, curves=curves, static_curves=static_curves,
                curves_bkgd=curves_bkgd, static_curves_bkgd=static_curves_bkgd, max_len=max_len)

def build_frame_data_phase2(current_HEX_R):
    """Phase2 (HEX_R growing back from HEX_R_end to HEX_R_start): anchor the
    BOTTOM of the body to a fixed pixel instead of the head, so as HEX_R
    grows the character enlarges around a fixed bottom -- the head drifts
    up and eventually off-canvas. Curves are shown fully (no chunking),
    since by Phase1's end everything has already been revealed.
    """
    _fig, _ax, patches_, hex_colors_, _centers, hex_rc_arr_, _pc, detail_info_ = cg.make_hex_scene(
        IMG_W=1280, IMG_H=720, HEX_R=current_HEX_R, DPI=DPI,
        hex_index=HEX_INDEX, z_order_max=z_order_max)
    plt.close(_fig)
 
    bottom_row, bottom_col = cg.pixel_to_nearest_hex(anchor_bottom_x, anchor_bottom_y, detail_info_)
    bottom_row, bottom_col = int(bottom_row), int(bottom_col)
    head_row = bottom_row - delta_row_bottom
    head_col = bottom_col
 
    dict_here, stripe_pts = HF_SCENE.draw_Sakura(head_row, head_col, n_head)
 
    curves_full = {}
    for key, pt in stripe_pts.items():
        PARAM = PARAMS[key]
        sol_free = SOLN.PlaneWave(
            m=const.m_e * const.c ** 2,
            x=np.linspace(PARAM["x_start"], PARAM["x_end"], bin_num) * u.nm,
            t=0 * u.fs, E=PARAM["E"] * u.eV, A=PARAM["A"], B=PARAM["B"])
        phi_free = sol_free.solve(t=0 * u.fs)
        x_free = sol_free.x
        row, col = pt
        if col < head_col:
            X, Y = x_free[::-1], phi_free[::-1].real
        else:
            X, Y = x_free, phi_free.real
 
        DOMAIN_W = X.max() - X.min()
        DOMAIN_H = Y.max() - Y.min()
        CANVAS_PHYSICAL_H = DOMAIN_H*PARAM["H_SCALE"]
        CANVAS_PHYSICAL_W = DOMAIN_W*PARAM["W_SCALE"]
        
        px0, py0 = cg.hex_center_pixel(pt[0], pt[1], detail_info_)
        OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
        OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]
 
        grid_hex_rc = cg.world_metres_to_hex_index(
            X + OFFSET_X, Y + OFFSET_Y, detail_info_,
            canvas_physical_x_range=(0, CANVAS_PHYSICAL_W),
            canvas_physical_y_range=(0, CANVAS_PHYSICAL_H))
        curves_full[key] = list(dict.fromkeys((r, c.value) for r, c in grid_hex_rc if (r, c.value) in hex_rc_arr_))
 
    max_row_ = max(r for r, c in hex_rc_arr_)
    max_col_ = max(c for r, c in hex_rc_arr_)
    seg_rows = np.arange(0, max_row_, BKGD_SEGS_WIDTH)
    BKGD_start_pts = []
    for i in range(len(seg_rows)):
        if i % 2 == 0:
            BKGD_start_pts.append((seg_rows[i], -1))
        else:
            BKGD_start_pts.append((seg_rows[i], max_col_))
 
    curves_bkgd_full = {}
    for i, BKGD_start_pt in enumerate(BKGD_start_pts):
        row, col = BKGD_start_pt
        if col > max_col_ // 2:
            sol_bkgd = SOLN.PlaneWave(
                m=const.m_e * const.c ** 2,
                x=np.linspace(-10, 0., bin_num) * u.nm,
                t=0 * u.fs, E=2 * u.eV, A=0., B=1.)
            phi_bkgd = sol_bkgd.solve(t=0 * u.fs)
            x_bkgd = sol_bkgd.x
            X_b, Y_b = x_bkgd[::-1], phi_bkgd[::-1].real
        else:
            sol_bkgd = SOLN.PlaneWave(
                m=const.m_e * const.c ** 2,
                x=np.linspace(0, 10, bin_num) * u.nm,
                t=0 * u.fs, E=2 * u.eV, A=1., B=0.)
            phi_bkgd = sol_bkgd.solve(t=0 * u.fs)
            x_bkgd = sol_bkgd.x
            X_b, Y_b = x_bkgd, phi_bkgd.real
 
        DOMAIN_H_b = Y_b.max() - Y_b.min()
        CANVAS_PHYSICAL_H_b = DOMAIN_H_b * BKGD_SEGS_WIDTH
        CANVAS_PHYSICAL_W_b = X_b.max() - X_b.min()
        px0_b, py0_b = cg.hex_center_pixel(row, col, detail_info_)
        OFFSET_X_b = px0_b * CANVAS_PHYSICAL_W_b / IMG_W_SCENE - X_b[0]
        OFFSET_Y_b = CANVAS_PHYSICAL_H_b * (1.0 - py0_b / IMG_H_SCENE) - Y_b[0]
 
        grid_hex_rc_b = cg.world_metres_to_hex_index(
            X_b + OFFSET_X_b, Y_b + OFFSET_Y_b, detail_info_,
            canvas_physical_x_range=(0, CANVAS_PHYSICAL_W_b),
            canvas_physical_y_range=(0, CANVAS_PHYSICAL_H_b))
        curves_bkgd_full[i] = list(dict.fromkeys(
            (r, c.value) for r, c in grid_hex_rc_b if (r, c.value) in hex_rc_arr_))
    
    col_x0 = well_x_to_col(0.0, x_well, hex_rc_arr_, detail_info_)
    col_x1 = well_x_to_col(1.0, x_well, hex_rc_arr_, detail_info_)
 
    # actual eigenstate curves, projected onto THIS frame's grid -- the
    # part the commented PHASE INF WELL block never wired in
    # NOTE: this function already uses bare X / CANVAS_PHYSICAL_H as LOCALS
    # for the (unrelated) limb curves above -- using X_well / CANVAS_PHYSICAL_H_well
    # here, sourced explicitly from the untouched module globals x_well /
    # DOMAIN_H_ref, so the well projection can never silently read leftover
    # limb-curve values.
    X_well = x_well
    CANVAS_PHYSICAL_H_well = DOMAIN_H_ref
    anchor_row_well = max(r for r, c in hex_rc_arr_) // 2
    curves_well = {}
    for s, state in enumerate(states):
        Y_well = state.real
        DOMAIN_W_well = X_well.max() - X_well.min()
        CANVAS_PHYSICAL_W_well = DOMAIN_W_well
        canvas_physical_x_range_well = (0, CANVAS_PHYSICAL_W_well)
        canvas_physical_y_range_well = (0, CANVAS_PHYSICAL_H_well)
 
        px0_well, py0_well = cg.hex_center_pixel(anchor_row_well, 0, detail_info_)
        OFFSET_X_well = px0_well * CANVAS_PHYSICAL_W_well / IMG_W_SCENE - X_well[0]
        OFFSET_Y_well = CANVAS_PHYSICAL_H_well * (1.0 - py0_well / IMG_H_SCENE) - Y_well[0]
 
        grid_hex_rc_well = cg.world_metres_to_hex_index(
            X_well + OFFSET_X_well, Y_well + OFFSET_Y_well, detail_info_,
            canvas_physical_x_range=canvas_physical_x_range_well,
            canvas_physical_y_range=canvas_physical_y_range_well)
        curves_well[s] = list(dict.fromkeys(
            (r, c) for r, c in grid_hex_rc_well if (r, c) in hex_rc_arr_))
 
    return dict(patches=patches_, hex_rc_arr=hex_rc_arr_, detail_info=detail_info_,
                head_row=head_row, head_col=head_col, dict_=dict_here,
                stripe_pts=stripe_pts, curves_full=curves_full, curves_bkgd_full=curves_bkgd_full,
                col_x0=col_x0, col_x1=col_x1, curves_well=curves_well)

  
# Build the schedule: for each HEX_R in the shrink sequence, how many
# snapshots does it need to fully reveal its longest curve? HEX_R stays
# fixed for that many snapshots before moving to the next, smaller HEX_R.
HEX_R_values = list(range(HEX_R_start, HEX_R_start - len_shrink, -1))
schedule = []          # [(HEX_R, snapshots_needed), ...]
cum_boundaries = []    # cumulative snapshot count after each HEX_R
_running = 0
for _hr in HEX_R_values:
    _snapshots_needed = max(build_frame_data(_hr)['max_len'], 1)
    schedule.append((_hr, _snapshots_needed))
    _running += _snapshots_needed
    cum_boundaries.append(_running)

    


Phase0_end = 10
SNAPSHOTS_PER_HEXR_SHRINK = 5   # each shrink-phase change holds for this many real snapshots
SNAPSHOTS_PER_HEXR_EXPAND = 8   # each expand-phase change holds for this many real snapshots
Phase1_end = Phase0_end + cum_boundaries[-1] * SNAPSHOTS_PER_HEXR_SHRINK
expand_len = len_shrink * SNAPSHOTS_PER_HEXR_EXPAND   # mirror the shrink phase length, HEX_R_end back up to HEX_R_start
Phase2_end = Phase1_end + expand_len+Phase0_end+Phase0_end
Phase3_end = Phase2_end + _running_wells 
total_frames = Phase3_end+Phase0_end
print("Phase1_ LEN", cum_boundaries[-1] * SNAPSHOTS_PER_HEXR_SHRINK)
print("Phase2_ LEN",expand_len)
print("Phase3_ LEN",_running_wells)
print('tatal frame', total_frames, '| schedule', schedule)
 
def resolve_hexr_and_reveal(current_snapshot):
    """Map a snapshot index (0-based, within Phase1) to (current_HEX_R,
    n_reveal) -- n_reveal counts up by exactly 1 per snapshot, resetting
    to 1 each time HEX_R changes."""
    prev_boundary = 0
    for (hr, _cnt), boundary in zip(schedule, cum_boundaries):
        if current_snapshot < boundary:
            return hr, current_snapshot - prev_boundary + 1
        prev_boundary = boundary
    return schedule[-1][0], schedule[-1][1]

def update(snapshot):
    global pc, final_phase2_colors,  patches  
    if snapshot < Phase0_end:
        current_hex_colors = animation_base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    elif snapshot>=Phase0_end and snapshot<Phase1_end:
        
        current_snapshot = snapshot-Phase0_end
        effective_snapshot = current_snapshot // SNAPSHOTS_PER_HEXR_SHRINK
        current_HEX_R, local_n_reveal = resolve_hexr_and_reveal(effective_snapshot)
        print("current_HEX_R = ", current_HEX_R, "| local_n_reveal =", local_n_reveal)
 
        frame = build_frame_data(current_HEX_R)
        patches = frame['patches']
        current_hex_rc_arr = frame['hex_rc_arr']
        detail_info = frame['detail_info']
        current_head_row, current_head_col = frame['head_row'], frame['head_col']
        dict_ = frame['dict_']
        curves = frame['curves']
        current_hex_colors = np.full((len(current_hex_rc_arr), 3), 0.96)
        current_hex_colors = cgc.select_normal_color([True]*len(current_hex_colors), cgc.hex_to_rgb("#ebeaff"), np.ones(3)*sigma_color/2)
        for _bkgd_key, _bkgd_static_cells in frame['static_curves_bkgd'].items():
            if _bkgd_static_cells:
                _select_bkgd_static = cg.select_mask(_bkgd_static_cells, current_hex_rc_arr)
                current_hex_colors[_select_bkgd_static] = cgc.select_normal_color(_select_bkgd_static, 
                                                                                  cgc.hex_to_rgb("#c6acf0"), 
                                                                                  np.ones(3)*sigma_color*3) 
 
        for _bkgd_key, _bkgd_curve in frame['curves_bkgd'].items():
            _bkgd_n_reveal = min(local_n_reveal, len(_bkgd_curve))
            _bkgd_valid_cells = _bkgd_curve[:_bkgd_n_reveal]
            _select_bkgd = cg.select_mask(_bkgd_valid_cells, current_hex_rc_arr)
            current_hex_colors[_select_bkgd] = cgc.select_normal_color(_select_bkgd, 
                                                                       cgc.hex_to_rgb("#c6acf0"), 
                                                                       np.ones(3)*sigma_color*3) 
            
            
        target_STRIPE_PALETTE = [np.array([0,0,0]),cgc.hex_to_rgb("#bb0000"),]
        stripe_target_colors = cgc.color_stripe(current_hex_rc_arr, 
                                                STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = target_STRIPE_PALETTE)#cgc.hex_to_rgb("#bb0000")
        stripe_skin_colors = cgc.color_stripe(current_hex_rc_arr, 
                                                STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = [cgc.hex_to_rgb("#fee9d2"),cgc.hex_to_rgb("#bb0000"), ])
 
        
        for key in dict_.keys():
            part  = dict_.get(key)[0]
            colors = dict_.get(key)[1] 
            if len(colors) == 1:
 
                select_part= cg.select_mask(part,current_hex_rc_arr)
                if key == 'raw_body':
                    current_hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color)[0]
                                                                for c in stripe_target_colors[select_part]])
                elif key == 'stripe_skin':
                    current_hex_colors[select_part] = stripe_skin_colors[select_part] 
                else:
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
            else:
                current_hex_colors = cgc.color_row_gradient(part, 
                                            colors[0],colors[1],
                                                current_hex_rc_arr, current_hex_colors, sort = 'row', sigma_color = sigma_color, 
                                                        end_weight= 0.01, mode = 'linear') 
 
        for key, static_cells in frame['static_curves'].items():
            stripe_start_pt = frame['stripe_pts'][key]
            color = cgc.color_stripe([stripe_start_pt], STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = target_STRIPE_PALETTE)[0]
            if static_cells:
                select_static = cg.select_mask(static_cells, current_hex_rc_arr)
                current_hex_colors[select_static] = cgc.select_normal_color(select_static, color, np.ones(3)*sigma_color*3) 
 
        for key, ordered_grid_hex_rc in curves.items():
            stripe_start_pt = frame['stripe_pts'][key]
 
            color = cgc.color_stripe([stripe_start_pt], STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = target_STRIPE_PALETTE)[0]
            
            # n_reveal counts up by exactly 1 per snapshot within this HEX_R's
            # occupancy -- clip per-key since shorter limbs finish revealing early
            n_reveal = min(local_n_reveal, len(ordered_grid_hex_rc))
            VALID_grid_hex_rc = ordered_grid_hex_rc[:n_reveal]
 
            select_curve = cg.select_mask(VALID_grid_hex_rc, current_hex_rc_arr)
            current_hex_colors[select_curve] = cgc.select_normal_color(select_curve, color, np.ones(3)*sigma_color*3) 
 
        pc.remove()  # take the old collection off the real axes
        pc = PatchCollection(patches, facecolor=current_hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
        ax.add_collection(pc)
        pc.set_facecolor(current_hex_colors)
        
    elif snapshot >= Phase1_end and snapshot < Phase2_end:
        # HEX_R grows back from HEX_R_end to HEX_R_start, anchored on the
        # BOTTOM of the body -- head drifts up and off-canvas as it grows.
        current_snapshot2 = snapshot - Phase1_end
        effective_snapshot2 = current_snapshot2 // SNAPSHOTS_PER_HEXR_EXPAND
        current_HEX_R = int(min(HEX_R_end + effective_snapshot2, HEX_R_start))
        print("Phase2 current_HEX_R = ", current_HEX_R)
 
        frame2 = build_frame_data_phase2(current_HEX_R)
        patches = frame2['patches']
        current_hex_rc_arr = frame2['hex_rc_arr']
        dict_ = frame2['dict_']
        current_hex_colors = np.full((len(current_hex_rc_arr), 3), 0.96)
        current_hex_colors = cgc.select_normal_color([True]*len(current_hex_colors), cgc.hex_to_rgb("#ebeaff"), np.ones(3)*sigma_color/2)
        
        for _bkgd_key, _bkgd_curve in frame2['curves_bkgd_full'].items():
            _select_bkgd = cg.select_mask(_bkgd_curve, current_hex_rc_arr)
            current_hex_colors[_select_bkgd] = cgc.select_normal_color(_select_bkgd, 
                                                                       cgc.hex_to_rgb("#c6acf0"), 
                                                                       np.ones(3)*sigma_color*3) 
 
        target_STRIPE_PALETTE = [np.array([0,0,0]),cgc.hex_to_rgb("#bb0000"),]
        stripe_target_colors = cgc.color_stripe(current_hex_rc_arr,
                                                STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = target_STRIPE_PALETTE)
        stripe_skin_colors = cgc.color_stripe(current_hex_rc_arr,
                                                STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = [cgc.hex_to_rgb("#fee9d2"),cgc.hex_to_rgb("#bb0000"), ])
        for key in dict_.keys():
            part = dict_.get(key)[0]
            colors = dict_.get(key)[1]
            if len(colors) == 1:
                select_part = cg.select_mask(part, current_hex_rc_arr)
                if key == 'raw_body':
                    current_hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color)[0]
                                                                for c in stripe_target_colors[select_part]])
                elif key == 'stripe_skin':
                    current_hex_colors[select_part] = stripe_skin_colors[select_part]
                else:
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)
            else:
                current_hex_colors = cgc.color_row_gradient(part,
                                            colors[0], colors[1],
                                            current_hex_rc_arr, current_hex_colors, sort='row', sigma_color=sigma_color,
                                            end_weight=0.01, mode='linear')
 
        for key, curve_full in frame2['curves_full'].items():
            stripe_start_pt = frame2['stripe_pts'][key]
            color = cgc.color_stripe([stripe_start_pt], STRIPE_WIDTH = 1 ,
                                                STRIPE_PALETTE = target_STRIPE_PALETTE)[0]
            select_curve = cg.select_mask(curve_full, current_hex_rc_arr)
            current_hex_colors[select_curve] = cgc.select_normal_color(select_curve, color, np.ones(3)*sigma_color*3) 
        
        # side walls outside the well (c < col_x0 or c > col_x1), filled as a
        # band growing from the bottom of the canvas -- each snapshot adds
        # (max_row/len_shrink) more rows, so by the time HEX_R reaches
        # HEX_R_start again the full canvas height on both sides is colored.
        col_x0 = frame2['col_x0']
        col_x1 = frame2['col_x1']
        max_row_this = max(r for r, c in current_hex_rc_arr)
        band_height = int(round((max_row_this / len_shrink) * (effective_snapshot2 + 1)))
        band_row_start = max(max_row_this - band_height, 0)
        _select_walls = [
            (r >= band_row_start) and (c < col_x0 or c > col_x1)
            for r, c in current_hex_rc_arr
        ]
        current_hex_colors[_select_walls] = cgc.select_normal_color(_select_walls,cgc.hex_to_rgb("#fad5fd"), np.ones(3)*sigma_color)
        #  [0, 1, 0]
        
        pc.remove()
        pc = PatchCollection(patches, facecolor=current_hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
        ax.add_collection(pc)
        pc.set_facecolor(current_hex_colors)
        final_phase2_colors = current_hex_colors.copy() 
    else:
        current_snapshot3 = snapshot - Phase2_end
        active_state, local_n_reveal = resolve_well_state_and_reveal(current_snapshot3)
        current_hex_colors = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color)[0] for c in final_phase2_colors])
        
        
        for s, well_cells in enumerate(grid_hex_rc_wells):
            if s < active_state:
                valid_cells = well_cells          # already finished -- stays fully colored
            elif s == active_state:
                n_reveal = min(local_n_reveal, len(well_cells))
                valid_cells = well_cells[:n_reveal]
            else:
                continue                          # not started yet
            select_well = cg.select_mask(valid_cells, hex_rc_arr)
            color_well = [0, 0, 0] if s % 2 == 0 else cgc.hex_to_rgb("#bb0000")
            current_hex_colors[select_well] = cgc.select_normal_color(select_well, color_well, 
                                                                                  np.ones(3)*sigma_color*3) 
 
        pc.remove()
        pc = PatchCollection(patches, facecolor=current_hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
        ax.add_collection(pc)
        pc.set_facecolor(current_hex_colors)
                             
    return (pc,)



total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'HF_animation.mp4')
 
print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")
