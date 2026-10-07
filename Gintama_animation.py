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
import SOLVE_LagrangianMultiplier as SOLN
import Gintama_scene_plot as GINTAMA_SCENE

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

hex_colors = cgc.color_row_gradient(hex_rc_arr, cgc.hex_to_rgb("#ffffff"), cgc.hex_to_rgb("#0341aa"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.01)    

base_hex_colors = hex_colors.copy() 
size = 2 
def f(x, y):
    """Single-peak mountain: summit at (0, 0), falling off in x and y."""
    return 6 * np.exp(-x**2 / 8 - y**2 / 170)

def g_switchback(x, y, A=1.6, k=np.pi / 3):
    """Switchback constraint: x - A*sin(k*y) = 0.
    Swings left/right in x as it descends in y."""
    return x - A * np.sin(k * y)


def g_parabola(x, y, x0=2.5, y_mid=9.0, A=(2.5 + 2.5) / 81.0):
    """Parabolic constraint: x - (x0 - A*(y - y_mid)**2) = 0.
    Bulges toward x0 at y = y_mid, curves back toward the opposite
    side at both the top and the bottom -- a "C" shape."""
    return x - (x0 - A * (y - y_mid) ** 2)


def g_line(x, y, m=-0.15, b=2.0):
    """Straight-line constraint: x - (m*y + b) = 0."""
    return x - (m * y + b)


def g_horizontal(x, y, c=5.0):
    """Horizontal-line constraint: y - c = 0."""
    return y - c

XLIM = (-4, 4)
YLIM = (0, 18)
ZMAX = f(0, 0) 

glines_dict = {'1': g_switchback, '2': g_parabola, '3':g_line, '4':g_horizontal
              }
n_y_lines = 5
    
COLORMAP = [
    (0.0,  '#4a3518'),   # dark pine forest, base
    (0.1, '#2f1f04'),   # treeline
    (0.3, '#22280e'),
    (0.5, '#2a4f42'),# bare grey rock
    (0.75, '#546581'),   # rock dusted with snow
    (0.9,  '#a0b1ce'),   # snowfield
    (1.0,  '#ffffff'),   # summit snow
]

x_fill = np.linspace(*XLIM, 900)
z_env = f(x_fill, 0.0)
grid_points, grid_values = [], []
for xi, hi in zip(x_fill, z_env):
    z_col = np.linspace(0, hi, 150)
    grid_points.append(np.column_stack([np.full_like(z_col, xi), z_col]))
    grid_values.append(z_col)
grid_points = np.concatenate(grid_points)
grid_values = np.concatenate(grid_values)

PARAMS = [
    {"DOMAIN_W_SCALE": 0.6, "DOMAIN_H_SCALE": 0.6, "PIVOT_ROW": 5, "PIVOT_COL": max(c for r, c in hex_rc_arr) // 2,
     "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 1},
    {"DOMAIN_W_SCALE": 1,   "DOMAIN_H_SCALE": 1,   "PIVOT_ROW": 5, "PIVOT_COL": max(c for r, c in hex_rc_arr) // 2,
     "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 2},
    {"DOMAIN_W_SCALE": 1,   "DOMAIN_H_SCALE": 0.8, "PIVOT_ROW": 5, "PIVOT_COL": max(c for r, c in hex_rc_arr) // 2,
     "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 3},
    {"DOMAIN_W_SCALE": 0.8, "DOMAIN_H_SCALE": 1,   "PIVOT_ROW": 5, "PIVOT_COL": max(c for r, c in hex_rc_arr) // 2,
     "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 4},
]

def get_path_direction(x_arr, z_arr, i, slope_threshold_deg=45):
    """
    Direction label for the character's pose, from the path's LOCAL
    heading at index i: "<horizontal> <steepness> <vertical>".
    Robust at both ends of the path, and even if i is passed slightly
    out of range (e.g. an off-by-one from the caller).
    """
    n = len(x_arr)
    if n == 0:
        return "right flat down"          # no path at all -- arbitrary default
    i = max(0, min(i, n - 1))             # clamp i itself into range first
 
    if n == 1:
        return "right flat down"          # a single point has no direction
 
    if i < n - 1:
        j = i + 1                         # normal case: look one step ahead
    else:
        j = i - 1                         # at the LAST point: look one step back instead
 
    dx = x_arr[j] - x_arr[i] if j > i else x_arr[i] - x_arr[j]
    dz = z_arr[j] - z_arr[i] if j > i else z_arr[i] - z_arr[j]
 
    horizontal = "right" if dx >= 0 else "left"
    vertical = "up" if dz >= 0 else "down"
    angle = np.degrees(np.arctan2(abs(dz), abs(dx))) if (dx != 0 or dz != 0) else 0.0
    steepness = "mid" if angle >= slope_threshold_deg else "flat"
 
    return f"{horizontal} {steepness} {vertical}"
def build_scene(PARAM):
    """Everything the original script computed ONCE, for ONE PARAM --
    background mountain fill, grey cross-section lines, the solved
    constraint path, and the critical points. Returns one dict per
    PARAM instead of overwriting shared global state each time."""
    scene_hex_colors = hex_colors.copy()
 
    grid_hex_rc, DOMAIN_W, DOMAIN_H, CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H = cg.fn2grid(
        grid_points[:, 0], grid_points[:, 1],
        PARAM["DOMAIN_W_SCALE"], PARAM["DOMAIN_H_SCALE"],
        PARAM["PIVOT_ROW"], PARAM["PIVOT_COL"], PARAM["PIVOT_ROW_X"], PARAM["PIVOT_COL_Y"], detail_info,
        square=False, EPS_DOMAIN=1e-6, return_DOMAIN=True)
    hex_positions, blended = cg.average_grid_to_hex_scene(
        grid_hex_rc, hex_rc_arr, grid_values, scene_hex_colors,
        colors=COLORMAP, alpha=1.0, vmax=ZMAX, weight_by_value=False)
    scene_hex_colors[hex_positions] = blended
    bkgd_base_hex_colors = scene_hex_colors.copy()   # kept for parity with the original script; update() doesn't read it
 
    grid_hex_rc_GREYs = []
    for i in range(n_y_lines):
        y_val = i / (n_y_lines - 1) * YLIM[1]
        x_line = np.linspace(*XLIM, 140)
        z_line = f(x_line, y_val)
        grid_hex_rc_GREY = cg.fn2grid(
            x_line, z_line, PARAM["DOMAIN_W_SCALE"], PARAM["DOMAIN_H_SCALE"],
            PARAM["PIVOT_ROW"], PARAM["PIVOT_COL"], PARAM["PIVOT_ROW_X"], PARAM["PIVOT_COL_Y"], detail_info,
            square=False, EPS_DOMAIN=1e-6, input_PHYSICAL_size=[CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H])
        grid_hex_rc_GREYs.extend(list(set(grid_hex_rc_GREY)))
        mask = cg.select_mask(list(set(grid_hex_rc_GREY)), hex_rc_arr)
        scene_hex_colors[mask] = cgc.select_normal_color(mask, cgc.hex_to_rgb('#b6aaff'), np.ones(3) * sigma_color * 5)
    animation_base_hex_colors = scene_hex_colors.copy()
 
    g = PARAM["gline"]
    gline = glines_dict.get(str(g))
    model = SOLN.LagrangeMultiplier(f, gline)
    n = 300
    if g == 1:
        model.trace_path(x0=0.0, y0=0.0, s_max=19, n=n)
    elif g == 2:
        model.trace_path(x0=-2.5, y0=0.0, s_max=23, n=n)
    elif g == 3:
        model.trace_path(x0=2.0, y0=0.0, s_max=18.5, n=n)
    elif g == 4:
        model.trace_path(x0=3.0, y0=5.0, s_max=6.0, n=n)
    critical_points = model.find_critical_points()
 
    grid_hex_rc_gpath = cg.fn2grid(
        model.path['x'], model.path['z'], PARAM["DOMAIN_W_SCALE"], PARAM["DOMAIN_H_SCALE"],
        PARAM["PIVOT_ROW"], PARAM["PIVOT_COL"], PARAM["PIVOT_ROW_X"], PARAM["PIVOT_COL_Y"], detail_info,
        square=False, EPS_DOMAIN=1e-6, input_PHYSICAL_size=[CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H])
 
    return {
        'bkgd_base_hex_colors': bkgd_base_hex_colors,
        'animation_base_hex_colors': animation_base_hex_colors,
        'grid_hex_rc_GREYs': grid_hex_rc_GREYs,
        'grid_hex_rc_gpath': grid_hex_rc_gpath,
        'model': model,
        'critical_points': critical_points,
    }
 
 
scenes = [build_scene(P) for P in PARAMS]
 
                

# ── UPDATE ──────────────────────────────────────────────────────────────
Phase0_end = 10
path_lengths = [len(s['grid_hex_rc_gpath']) for s in scenes]
scene_cumulative = np.cumsum([0] + path_lengths)   # [0, 100, 200, 300, 400]
total_frames = int(Phase0_end + scene_cumulative[-1])
print("total_frames", total_frames, " path_lengths", path_lengths)
 
 
def update(snapshot):
    if snapshot < Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot - Phase0_end   # 0..399, spanning ALL 4 scenes

        scene_idx = int(np.searchsorted(scene_cumulative, current_snapshot, side='right') - 1)
        scene_idx = min(scene_idx, len(scenes) - 1)
        local_idx = current_snapshot - int(scene_cumulative[scene_idx])
        scene = scenes[scene_idx]
 
        current_hex_colors = scene['animation_base_hex_colors'].copy()
 
        mask = cg.select_mask(scene['grid_hex_rc_GREYs'], hex_rc_arr)
        current_hex_colors[mask] = cgc.select_normal_color(mask, cgc.hex_to_rgb('#b6aaff'), np.ones(3) * sigma_color * 5)
 
        grid_hex_rc_gpath_reveal = scene['grid_hex_rc_gpath'][:local_idx + 1]
        select_path = cg.select_mask(grid_hex_rc_gpath_reveal, hex_rc_arr)
        current_hex_colors[select_path] = cgc.select_normal_color(select_path, cgc.hex_to_rgb('#533ce1'), np.ones(3) * sigma_color * 5)
 
        location = scene['grid_hex_rc_gpath'][local_idx]
        direction = get_path_direction(scene['model'].path['x'], scene['model'].path['z'], local_idx)
        dict_ = GINTAMA_SCENE.draw_char(location, size, direction)
 
        for key in dict_.keys():
            part = dict_.get(key)[0]
            colors = dict_.get(key)[1]
            if len(colors) == 1:
                select_part = cg.select_mask(part, hex_rc_arr)
                current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3) * sigma_color)
            else:
                sort_info = colors[-1]
                if 'hex' in sort_info:
                    current_hex_colors = cgc.color_hex_gradient(
                        part, colors[0], colors[1], hex_rc_arr, current_hex_colors, sort_info[0], sort_info[1],
                        sigma_color=sigma_color, end_weight=0.01, period=None, mode='linear', start_n=1, end_n=None)
                else:
                    current_hex_colors = cgc.color_row_gradient(
                        part, colors[0], colors[1], hex_rc_arr, current_hex_colors,
                        sort='row', sigma_color=sigma_color, end_weight=0.01, mode='linear')
 
        pc.set_facecolor(current_hex_colors)
    return (pc,)


total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'cg'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'Gintama_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")