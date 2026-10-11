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
import SOLN_waveinterference as SOLN
import DTB_scene_plot as DTB_SCENE
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100

fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=12, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.03


rng = np.random.default_rng(seed=0)
n_drops = 10

xs = rng.uniform(-5, 5, n_drops)
ys = rng.uniform(-5, 5, n_drops)
sources = tuple(np.array([x, y]) * u.m for x, y in zip(xs, ys))

wavelength =(0.8,) * n_drops * u.m 
amplitude = tuple(rng.uniform(2., 3., n_drops) *u.m) # (2.5,) * n_drops * u.m
freq = tuple(rng.uniform(1., 2., n_drops) *u.Hz) 
kick_time =(0 * u.s,)+ tuple(np.sort(rng.uniform(0, 13, n_drops-1)) * u.s)#np.arange(1, 18, 1.2)[:n_drops]* u.s

#
decay_tau = tuple(rng.uniform(0.2, 1, n_drops) * u.s)

is_ripple = (True,) * n_drops

kick_time_source = 13. * u.s
sources_constant = ([-3,-5] * u.m,)
wavelength_constant =(3.0 * u.m,)
amplitude_constant = (2.0 * u.m,)
freq_constant = (3.0 * u.Hz,)
kick_time_constant = (kick_time_source,)
is_ripple_constant =(False,)
decay_tau_constant =(None,)

WAVE = SOLN.WaveInterference2D(
    grid_size=200,
    x_range=[-8, 8] * u.m, y_range=[-8, 8] * u.m,
    sources= tuple(sources)+tuple(sources_constant),
    wavelength=tuple(wavelength)+tuple(wavelength_constant),
    amplitude=tuple(amplitude)+tuple(amplitude_constant),
    freq=tuple(freq)+tuple(freq_constant),
    kick_time=tuple(kick_time)+tuple(kick_time_constant),
    is_ripple=tuple(is_ripple)+tuple(is_ripple_constant),
    decay_tau = tuple(decay_tau)+tuple(decay_tau_constant)
)
    

COLORMAP = [
    (0.0,  '#000309'),   # dark pine forest, base
    (0.1, '#00246c'),   # treeline
    (0.3, '#0048d5'),
    (0.5, '#003295'),# bare grey rock
    (0.7, '#0036a0'), 
    (0.8, '#cea0c4'),#  rock dusted with snow
    (0.9, '#0048d5'),   # snowfield
    (0.95, '#8fb8ff'),   # snowfield
    (1.0,  '#ffffff'),   # summit snow
]


X = WAVE.X.ravel()
Y = WAVE.Y.ravel()
grid_hex_rc_square = cg.fn2grid(
    X, Y,
    DOMAIN_W_SCALE=0.5, DOMAIN_H_SCALE=0.5,
    PIVOT_ROW=min(r for r,c in hex_rc_arr), PIVOT_COL=(min(c for r,c in hex_rc_arr) + max(c for r,c in hex_rc_arr)) // 2,
    PIVOT_ROW_X=0, PIVOT_COL_Y=Y.max(),
    detail_info=detail_info, square=True,
)

grid_hex_rc_squeeze = cg.fn2grid(
    X, Y,
    DOMAIN_W_SCALE=0.95, DOMAIN_H_SCALE=0.95,
    PIVOT_ROW=min(r for r,c in hex_rc_arr), PIVOT_COL=(min(c for r,c in hex_rc_arr) + max(c for r,c in hex_rc_arr)) // 2,
    PIVOT_ROW_X=0, PIVOT_COL_Y=Y.max(),
    detail_info=detail_info, square=False,
)


animation_base_hex_colors = hex_colors.copy() 
              
total_seconds = 18.0
dt = 0.1
t_span = np.arange(0, total_seconds, dt)# time comes first


char_dict_ = {
        'Kirsi': DTB_SCENE.draw_char((max(r for r, c in hex_rc_arr) // 2, max(c for r, c in hex_rc_arr) // 3*2), 'Kirsi', 2),
            '1': DTB_SCENE.draw_char((max(r for r, c in hex_rc_arr) // 3, max(c for r, c in hex_rc_arr) // 5*3), '1', 5),
     '2':DTB_SCENE.draw_char((8, max(c for r, c in hex_rc_arr) // 4), '2', 3),
       
                }

# ── UPDATE ──────────────────────────────────────────────────────────────
FADE_SECONDS = 2.0

total_frames = len(t_span)     
print("total_frames", total_frames)

def update(snapshot):
    current_hex_colors = animation_base_hex_colors.copy()
    current_time = t_span[snapshot]
    VALUE = WAVE.field_at(current_time).ravel()
    if current_time<10:
        grid_hex_rc = grid_hex_rc_square
        dict_ = char_dict_.get('Kirsi')
        alpha_part = min((10 - current_time) / FADE_SECONDS, 1.0)
    else:
        grid_hex_rc = grid_hex_rc_squeeze
        alpha_part = 0.8
        if current_time<=kick_time_source.value:
            dict_ = char_dict_.get('1')
            alpha_part = min((kick_time_source.value - current_time) / FADE_SECONDS, 1.0) * 0.8
        else:
            dict_ = char_dict_.get('2')
        
    hex_positions, blended = cg.average_grid_to_hex_scene(
        grid_hex_rc, hex_rc_arr,VALUE, current_hex_colors, COLORMAP, alpha=1.0, weight_by_value=False
                                                        )
    current_hex_colors[hex_positions] = blended
    
    
    for key in dict_.keys():
        prev_hex_colors = current_hex_colors.copy()

        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            sort_info = colors[-1]
            if 'hex' in sort_info:

                current_hex_colors = cgc.color_hex_gradient(part,colors[0],colors[1], hex_rc_arr, 
                        current_hex_colors, sort_info[0], sort_info[1],
                        sigma_color=sigma_color, end_weight=0.01, period=None, mode='linear',
                        start_n=1, end_n=None)

            else:
                current_hex_colors = cgc.color_row_gradient(part, 
                        colors[0],colors[1],
                        hex_rc_arr, current_hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear') 
        current_hex_colors = (1 - alpha_part) * prev_hex_colors + alpha_part * current_hex_colors
    pc.set_facecolor(current_hex_colors)
    return (pc,)



FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'DTB_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")