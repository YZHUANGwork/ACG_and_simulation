import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
import astropy.units as u
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import expected_value as EXP
import blackbody_color_map as BB
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,_ = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)



sigma_color = 0.01
row_boundary = 10

hex_colors = cgc.color_row_gradient(hex_rc_arr, 
                                    cgc.hex_to_rgb("#fafff4"), cgc.hex_to_rgb("#b1b8a6"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    
 

skyline = row_boundary-1

locs = [(0,2), (-2,16)]
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline+5,
                                                        slope_left = '1.5', slope_right = '1.5', direction = 'lr')
    
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#e9f3f0"), cgc.hex_to_rgb("#c5d1cd"), 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

    

loc = (7, 3)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline+5,
                                                    slope_left = '8/3', slope_right = '8/3', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

loc = (3, 18)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                    slope_left = '1.5', slope_right = '1.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

loc = (3, 30)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                    slope_left = '1.5', slope_right = '1.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    


loc = (1, 10)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                    slope_left = '0.5', slope_right = '0.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    


locs = [(6, 7), (7,9), (6,12), (7,14), (7,16), (7,18), (7,20), (6, 22), (5, 25), (7,28), (7,30), (7,32)]
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr')
    
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#1f614f"), cgc.hex_to_rgb("#1c241e"), 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

    
_, _, start_LINE,_, entire_region, vertex =  cgd.draw_trapezoid(row_boundary, 5, 34, 22, 
                                                                slope_left = '1.5', slope_right = '1.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#151d1b"), cgc.hex_to_rgb("#536864") , 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    
select_skyline = cg.select_mask([(r-1, c) for r,c in start_LINE], hex_rc_arr)
hex_colors[select_skyline] = cgc.select_normal_color(select_skyline, cgc.hex_to_rgb("#595d3e"), np.ones(3)*sigma_color) 

foreground_points = []
locs = [(10,0), (19,14)]
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  22,
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr')
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#234431"), [0,0,0], 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)  
    foreground_points.extend(entire_region)
locs = [(18,8), (20,20)] 
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  22,
                                                        slope_left = '1.5', slope_right = '1.5', direction = 'lr')
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#234431"), [0,0,0], 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color) 
    foreground_points.extend(entire_region)

foreground_mask = cg.select_mask(foreground_points, hex_rc_arr)

base_hex_colors = hex_colors.copy()

K = np.array([[1.0], [0.0]])


T_f = 20000 * u.K  # fade target temperature — must match color_f below
T_min, T_max = 1000, 20000
color_f = np.array(BB.blackbody_to_rgb_CIE1931(T_f))

state = {'locs': [],       # list of (row, col, start_snapshot)
        'color_i': [],
        'color_Q': [],
        'life_steps': [],  # once age passes this, the loc has reached its terminal color
        'layer_history': [],  # list of {'select', 'color', 'birth'}
        }

def T_to_life_steps(T, T_SCALE = T_min*2):
    
    """
    Number of recursive steps for this loc's color to reach the fade
    target T_f, based purely on how far its starting temperature is from
    T_f (farther away = more steps = slower decay). T_SCALE sets how
    many Kelvin correspond to one step.
    """
    delta_T = abs((T - T_f).to(u.K).value)
    return max(delta_T / T_SCALE, 1e-3)

def spawn_loc(snapshot):
    """Pick a random valid hex and a random temperature, and register a new loc starting this snapshot."""
    row, col = hex_rc_arr[rng.integers(len(hex_rc_arr))]
    T = BB.random_temperatures(1, T_min=T_min, T_max=T_max, log_scale=True, rng=rng)[0]
    n_life = T_to_life_steps(T)
 
    state['locs'].append((row, col, snapshot))
    state['color_i'].append(np.array(BB.blackbody_to_rgb_CIE1931(T)))
    state['color_Q'].append(cg.steps_to_Q(n_life, end_weight=0.01))
    state['life_steps'].append(n_life)

size_initial = 1
FADE_FRAMES = 5


n_snapshots = 400
Phase0_end = 10
def update(snapshot):
    if snapshot <Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot-Phase0_end
        spawn_loc(current_snapshot)
        current_hex_colors = base_hex_colors.copy()

        for i, (row, col, start) in enumerate(state['locs']):
            age = current_snapshot - start
            if age < 0:
                continue  # this loc hasn't started growing yet
            if age >= state['life_steps'][i]:
                continue  # reached its terminal color — stop growing/updating, let it fade out

            color_i = state['color_i'][i]
            if age == 0:
                pts = cg.hex_neighbours_n(row, col, n=age+size_initial, keep_origin=True)
            else:
                pts = cg.hex_neighbours_n(row, col, n=age+size_initial, keep_origin=False, return_frontier=True)[-1]

            select_pts = cg.select_mask(pts, hex_rc_arr)
            pts_colors = cgc.select_normal_color(select_pts, color_i, np.ones(3)*sigma_color*3)
            state['layer_history'].append({
                    'select': select_pts,
                    'color': pts_colors,
                    'birth': current_snapshot,
                    'loc': i,
                })

            # Recurse this loc's color_i for the next snapshot.
            V = np.array([color_i, color_f])
            Color_output, _ = EXP.expected_value(state['color_Q'][i], K, V)
            state['color_i'][i] = Color_output[0]

        state['layer_history'] = [ l for l in state['layer_history'] if current_snapshot - l['birth'] < FADE_FRAMES ]

        for l in state['layer_history']:
                age = current_snapshot - l['birth']
                alpha = 1.0 - age / FADE_FRAMES
                base_slice = base_hex_colors[l['select']]
                current_hex_colors[l['select']] = alpha * l['color'] + (1 - alpha) * base_slice
        
        current_hex_colors[foreground_mask] = base_hex_colors[foreground_mask]  # keep silhouette on top
        
        pc.set_facecolor(current_hex_colors)

    return (pc,)
 

total_seconds =18.0

# FPS [1/time] = total_frames / total_seconds
FPS = (n_snapshots / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
 
ani = animation.FuncAnimation(fig, update, frames=int(n_snapshots),
                               interval=time_gap, blit=False)
 
# ── SAVE ──────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'valley_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved → {OUTPUT_FILE}")
#plt.savefig('cg/'+cg_name, dpi=DPI, bbox_inches='tight')
#print('saved')