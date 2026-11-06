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
import SOLN_lorentztransform as SOLN

import Kurumi_scene_plot as K_SCENE
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


center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
center_col = max(c for r, c in hex_rc_arr) // 2

light_region = [(r,c) for r,c in hex_rc_arr if (r<=center_row and c<=center_col) or (r>=center_row and c>=center_col)]

dark_region = [(r,c) for r,c in hex_rc_arr if (r<=center_row and c>=center_col) or (r>=center_row and c<=center_col)]


select_light_region= cg.select_mask(light_region,hex_rc_arr)
hex_colors[select_light_region] = cgc.select_normal_color(select_light_region, cgc.hex_to_rgb("#adb7fa"), np.ones(3)*sigma_color*3) 
select_dark_region= cg.select_mask(dark_region,hex_rc_arr)
hex_colors[select_dark_region] = cgc.select_normal_color(select_dark_region, cgc.hex_to_rgb("#2036c4"), np.ones(3)*sigma_color*2) 

animation_background_hex_colors = hex_colors.copy()
hex_rc_index = {rc: i for i, rc in enumerate(hex_rc_arr)}

char_dict_ = {
   "clock": K_SCENE.draw_clock(center_row, center_col, 27),
        'body1': K_SCENE.draw_char(center_row, center_col, '0'),
    'body2': K_SCENE.draw_char(center_row,center_col , '1'),
    'body3': K_SCENE.draw_char(center_row, center_col, '2')
    
}

body_names = ['body1', 'body2', 'body3']   # one-to-one with scenarios, in order
body_pivot, body_height, body_width = {}, {}, {}
body_left_width, body_right_width, body_up_height, body_down_height = {}, {}, {}, {}
body_original_color = {}
body_nearest_left, body_nearest_right, body_nearest_up, body_nearest_down = {}, {}, {}, {}


for char, dict_ in char_dict_.items(): 
    for key in dict_.keys():

        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 

        else:
            sort_info = colors[-1]
            if 'hex' in sort_info:

                hex_colors = cgc.color_hex_gradient(part,colors[0],colors[1], hex_rc_arr, 
                        hex_colors, sort_info[0], sort_info[1],
                        sigma_color=sigma_color, end_weight=0.01, period=None, mode='linear',
                        start_n=1, end_n=None)
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
                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01, mode = mode, pivot=pivot ) 

            else:
                if 'sigmoid' in sort_info:
                    mode = 'sigmoid'
                else:
                    mode = 'linear'
                print('mode', mode)

                hex_colors = cgc.color_row_gradient(part, 
                        colors[0],colors[1],
                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = mode)
    if char =='clock':
        background_hex_colors = hex_colors.copy()

        
    if char in body_names:
        pivot_row, pivot_col = center_row + 10, center_col
        body_pivot[char] = (pivot_row, pivot_col)

        all_body_coords = [rc for part in dict_.values() for rc in part[0]]
        height_body = max(r for r, c in all_body_coords) - min(r for r, c in all_body_coords)
        width_body = max(c for r, c in all_body_coords) - min(c for r, c in all_body_coords)
        body_height[char] = height_body
        body_width[char] = width_body

        left_width = pivot_col - min(c for r, c in all_body_coords)
        right_width = max(c for r, c in all_body_coords) - pivot_col
        up_height = pivot_row - min(r for r, c in all_body_coords)
        down_height = max(r for r, c in all_body_coords) - pivot_row
        body_left_width[char] = left_width
        body_right_width[char] = right_width
        body_up_height[char] = up_height
        body_down_height[char] = down_height

        original_body_color = {rc: hex_colors[hex_rc_index[rc]] for rc in all_body_coords if rc in hex_rc_index}
        body_original_color[char] = original_body_color

        # for expanding: the color nearest the pivot on each side, per row
        # (for L/R) or per col (for U/D) -- the duplicate that fills the
        # new space opened up next to the pivot
        nearest_left_color, nearest_right_color = {}, {}
        nearest_up_color, nearest_down_color = {}, {}
        for (rr, cc), col in original_body_color.items():
            if cc < pivot_col:
                d = pivot_col - cc
                if rr not in nearest_left_color or d < nearest_left_color[rr][0]:
                    nearest_left_color[rr] = (d, col)
            elif cc > pivot_col:
                d = cc - pivot_col
                if rr not in nearest_right_color or d < nearest_right_color[rr][0]:
                    nearest_right_color[rr] = (d, col)
            if rr < pivot_row:
                d = pivot_row - rr
                if cc not in nearest_up_color or d < nearest_up_color[cc][0]:
                    nearest_up_color[cc] = (d, col)
            elif rr > pivot_row:
                d = rr - pivot_row
                if cc not in nearest_down_color or d < nearest_down_color[cc][0]:
                    nearest_down_color[cc] = (d, col)
        body_nearest_left[char] = {rr: col for rr, (d, col) in nearest_left_color.items()}
        body_nearest_right[char] = {rr: col for rr, (d, col) in nearest_right_color.items()}
        body_nearest_up[char] = {cc: col for cc, (d, col) in nearest_up_color.items()}
        body_nearest_down[char] = {cc: col for cc, (d, col) in nearest_down_color.items()}



rng = np.random.default_rng(seed=0)
#clock_region =  cg.hex_neighbours_n(center_row, center_col, n=1, keep_origin = True, return_frontier=False):
_, minute_hand_dict = K_SCENE.draw_divide(center_row, center_col, hex_rc_arr, n_divide=12)
_, second_hand_dict = K_SCENE.draw_divide(center_row, center_col, hex_rc_arr, n_divide=18)

#-------------------READ SOLN-----------------------
scenarios = [
    dict(name='event_1_TIME', x=0. * u.m, y=0.0 * u.m, z=0.0 * u.m, t=60.0 * u.ns,
         beta=[ 0.0,0.8, 0.0]),
    dict(name='event_1_LENGTH', x=0. * u.m, y=20.0 * u.m, z=0.0 * u.m, t=0.0 * u.ns,
         beta=[ 0.0,0.8, 0.0]),
    
    dict(name='event_2_TIME', x=0. * u.m, y=0.0 * u.m, z=0.0 * u.m, t=60.0 * u.ns,
         beta=[0.4, 0.8, 0.0]),
    dict(name='event_2_LENGTH', x=10. * u.m, y=30.0 * u.m, z=0.0 * u.m, t=0.0 * u.ns,
         beta=[0.4, 0.8, 0.0]),
    
    
    dict(name='event_3_TIME', x=0.0 * u.m, y=0.0 * u.m, z=0.0 * u.m, t=60.0 * u.ns,
         beta=[0.9, 0., 0.0]),
    dict(name='event_3_LENGTH', x=10.0 * u.m, y=0.0 * u.m, z=0.0 * u.m, t=60.0 * u.ns,
         beta=[0.9, 0., 0.0]),

    
    
]

results = []
for s in scenarios:
    boost = SOLN.LorentzTransform(s['beta'])
    t_p, x_p, y_p, z_p = boost.transform((s['t'], s['x'], s['y'], s['z']))
    results.append(dict(s, gamma=boost.gamma, t_p=t_p, x_p=x_p, y_p=y_p, z_p=z_p))

# scenarios alternate TIME, LENGTH, one pair per body -- split them apart.
# t' always comes from the TIME event (pure time dilation: x=y=z=0 there,
# so t'=gamma*t exactly, always longer, never sign-flipped). x'/y' always
# come from the LENGTH event (pure length contraction: Delta t=0 there,
# so no simultaneity-offset term, x'/y' stay positive).
results_time = results[0::2]
results_length = results[1::2]

n_frames = 50

frac = np.linspace(0.0, 1.0, n_frames)
 
# phase A, original time: one plain 360 deg sweep representing t itself
# (ratio t/t = 1) -- same for every scenario, body stays unsqueezed
minute_angle_per_frame_orig = (360.0 * frac) % 360.0
second_angle_per_frame_orig = (360.0 * frac * 12.0) % 360.0

minute_angle_per_frame = []   # one array (len n_frames) per body
second_angle_per_frame = []
for r in results_time:
    t_val = r['t'].to_value(u.ns)
    tp_val = r['t_p'].to_value(u.ns)
    angle_max = 360.0 * (tp_val / t_val)          # fixed scalar target for this body

    total_angle = angle_max * frac                 # shape (n_frames,)
    minute_angle_per_frame.append(total_angle % 360.0)
    second_angle_per_frame.append((total_angle * 12.0) % 360.0)

transformed_height_body_per_scenario = []   # one fixed target size per body
transformed_width_body_per_scenario = []
for i, r in enumerate(results_length):
    name = body_names[i]   # this body's own matching LENGTH event
    # with t=0, y_p = gamma*y is the REST length (bigger) -- y is the
    # contracted one. The squeeze ratio we actually want is y/y_p = 1/gamma,
    # not y_p/y (that direction expands instead of contracting).
    transformed_height_body_per_scenario.append(body_height[name] * (r['y'].value / r['y_p'].value if r['y_p'].value else 1.0))
    transformed_width_body_per_scenario.append(body_width[name] * (r['x'].value / r['x_p'].value if r['x_p'].value else 1.0))


total_remove_c_per_scenario = [round(body_left_width[body_names[i]] + body_right_width[body_names[i]] - w)
                                for i, w in enumerate(transformed_width_body_per_scenario)]
total_remove_r_per_scenario = [round(body_up_height[body_names[i]] + body_down_height[body_names[i]] - h)
                                for i, h in enumerate(transformed_height_body_per_scenario)]

# which side (L/R, U/D) goes first within its own axis -- whichever is
# bigger, per body. Col and row each progress at their own pace,
# proportional to p.
start_c_per_scenario = ['L' if body_left_width[n] >= body_right_width[n] else 'R' for n in body_names]
start_r_per_scenario = ['U' if body_up_height[n] >= body_down_height[n] else 'D' for n in body_names]


# ── UPDATE ──────────────────────────────────────────────────────────────
total_seconds = 18

Phase0_end = 20
total_frames = Phase0_end + len(body_names) * (2 * n_frames)   # 3 bodies, each: phase A (original) + phase B (transformed)
print("total_frames", total_frames)

def update(snapshot):
    if snapshot<Phase0_end:
        current_hex_colors = background_hex_colors.copy()
    else:
        current_snapshot = snapshot-Phase0_end
        current_hex_colors = animation_background_hex_colors.copy()    

        scenario_idx = current_snapshot // (2 * n_frames)
        within_scenario = current_snapshot % (2 * n_frames)
        name = body_names[scenario_idx]            # this scenario's one-to-one body
        pivot_row, pivot_col = body_pivot[name]
        original_body_color = body_original_color[name]

        # resolve which phase we're in and what angle it uses BEFORE
        # drawing anything -- the angle has to be known before the hands
        # can be drawn at all.
        if within_scenario < n_frames:
            frame_idx = within_scenario
            current_minute_hand_angle = minute_angle_per_frame_orig[frame_idx]
            current_second_hand_angle = second_angle_per_frame_orig[frame_idx]
        else:
            frame_idx = within_scenario - n_frames
            current_minute_hand_angle = minute_angle_per_frame[scenario_idx][frame_idx]
            current_second_hand_angle = second_angle_per_frame[scenario_idx][frame_idx]

        # hands drawn first, so the body (drawn below) ends up on top of
        # them wherever the two overlap
        current_minute_hand = minute_hand_dict[str(int(np.ceil(((90 - current_minute_hand_angle) % 360) / 30)) * 30 % 360)]
        current_second_hand = second_hand_dict[str(int(np.ceil(((90 - current_second_hand_angle) % 360) / 20)) * 20 % 360)]

        VALID_current_minute_hand = [(r,c) for r,c in current_minute_hand if (r,c) in 
                                     cg.hex_neighbours_n(center_row, center_col, n=18)]

        VALID_current_second_hand = [(r,c) for r,c in current_second_hand if (r,c) in 
                                     cg.hex_neighbours_n(center_row, center_col, n=27)]

        select_minute_hand= cg.select_mask(VALID_current_minute_hand,hex_rc_arr)
        current_hex_colors[select_minute_hand] = cgc.select_normal_color(select_minute_hand, 
                                                                         cgc.hex_to_rgb("#ffcf27"), np.ones(3)*sigma_color*3) 

        select_second_hand= cg.select_mask(VALID_current_second_hand,hex_rc_arr)
        current_hex_colors[select_second_hand] = cgc.select_normal_color(select_second_hand, 
                                                                         cgc.hex_to_rgb("#ff9527"), np.ones(3)*sigma_color*3) 

        if within_scenario < n_frames:
            # phase A: original body, original time -- nothing squeezed,
            # drawn after the hands so it sits on top of them
            for (r, c), color in original_body_color.items():
                idx = hex_rc_index.get((r, c))
                if idx is not None:
                    current_hex_colors[idx] = color

        else:
            # phase B: squeeze toward the transformed size, transformed
            # time -- body drawn after the hands so it sits on top
            nearest_left_color = body_nearest_left[name]
            nearest_right_color = body_nearest_right[name]
            nearest_up_color = body_nearest_up[name]
            nearest_down_color = body_nearest_down[name]
            start_c = start_c_per_scenario[scenario_idx]
            start_r = start_r_per_scenario[scenario_idx]

            total_remove_c = total_remove_c_per_scenario[scenario_idx]
            total_remove_r = total_remove_r_per_scenario[scenario_idx]

            p = frame_idx / (n_frames - 1)
            # col and row progress independently, each proportional to p.
            # This drops the earlier 4-cycle interleaving (col gets 2
            # turns, row 2, repeat) -- that math only works when both
            # totals are >= 0 (shrinking); it breaks for a negative
            # (expanding) total, so scoping back to what's correct for
            # both cases.
            step_c = round(p * total_remove_c)
            step_r = round(p * total_remove_r)

            m_c = (step_c + 1) // 2 if start_c == 'L' else step_c // 2   # L's so far
            n_c = step_c - m_c                                           # R's so far
            m_r = (step_r + 1) // 2 if start_r == 'U' else step_r // 2   # U's so far
            n_r = step_r - m_r                                           # D's so far

            for (r, c), color in original_body_color.items():
                # same formula handles shrink (m>0, pulls in) and expand
                # (m<0, pushes out) -- d<=m only ever true when m>0, so
                # the skip only ever fires during a shrink, never an expand
                if c < pivot_col:
                    d = pivot_col - c
                    if d <= m_c: continue
                    new_c = pivot_col - (d - m_c)
                elif c > pivot_col:
                    d = c - pivot_col
                    if d <= n_c: continue
                    new_c = pivot_col + (d - n_c)
                else:
                    new_c = pivot_col

                if r < pivot_row:
                    d = pivot_row - r
                    if d <= m_r: continue
                    new_r = pivot_row - (d - m_r)
                elif r > pivot_row:
                    d = r - pivot_row
                    if d <= n_r: continue
                    new_r = pivot_row + (d - n_r)
                else:
                    new_r = pivot_row

                idx = hex_rc_index.get((new_r, new_c))
                if idx is not None:
                    current_hex_colors[idx] = color

            # expanding: duplicate the color nearest the pivot into the
            # new space that opened up next to it (instead of a gap)
            if total_remove_c < 0:
                for j in range(1, -m_c + 1):
                    for rr, col in nearest_left_color.items():
                        idx = hex_rc_index.get((rr, pivot_col - j))
                        if idx is not None: current_hex_colors[idx] = col
                for j in range(1, -n_c + 1):
                    for rr, col in nearest_right_color.items():
                        idx = hex_rc_index.get((rr, pivot_col + j))
                        if idx is not None: current_hex_colors[idx] = col
            if total_remove_r < 0:
                for j in range(1, -m_r + 1):
                    for cc, col in nearest_up_color.items():
                        idx = hex_rc_index.get((pivot_row - j, cc))
                        if idx is not None: current_hex_colors[idx] = col
                for j in range(1, -n_r + 1):
                    for cc, col in nearest_down_color.items():
                        idx = hex_rc_index.get((pivot_row + j, cc))
                        if idx is not None: current_hex_colors[idx] = col

        

    pc.set_facecolor(current_hex_colors)
    return (pc,)



FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'Kurumi_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")