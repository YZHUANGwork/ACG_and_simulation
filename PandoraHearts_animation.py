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
import Trajectory_foucaultpendulum as TRAJ
import PandoraHearts_scene_plot as PH_SCENE
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

animation_base_hex_colors = hex_colors.copy() 
n_marker = 14
marker_center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
marker_center_col = max(c for r, c in hex_rc_arr) // 2


hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#ffb5b5"), cgc.hex_to_rgb("#cb0000"), hex_rc_arr, hex_colors, 
                                    (marker_center_row, marker_center_col), n_marker+5,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='linear',
                        start_n=0, end_n=None)
hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#cb0000"), [0,0,0], hex_rc_arr, hex_colors, 
                                    (marker_center_row, marker_center_col), max(c for r, c in hex_rc_arr) ,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='sigmoid',
                        start_n=n_marker+5, end_n=None)
background_base_hex_colors = hex_colors.copy() 


rng = np.random.default_rng(seed=0)


char_dict_phase0= {
#          '1':PH_SCENE.draw_char(marker_center_row-n_marker//2-2, marker_center_col-n_marker//2+2, name = 'black alice'),
#    '2':PH_SCENE.draw_char(marker_center_row-n_marker//2-2, marker_center_col+n_marker//2-3, name = 'white alice'),
    'marker':PH_SCENE.draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'outer'),      
                }
char_dict_phase1_1= {
    'white':PH_SCENE.draw_char(marker_center_row-n_marker//2-2, marker_center_col+n_marker//2-3, name = 'white alice'),
    'marker':PH_SCENE.draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'outer'),      
                }
char_dict_phase1_2= {
          'black':PH_SCENE.draw_char(marker_center_row-n_marker//2-2, marker_center_col-n_marker//2+2, name = 'black alice'),
    'white':PH_SCENE.draw_char(marker_center_row-n_marker//2-2, marker_center_col+n_marker//2-3, name = 'white alice'),
    'marker':PH_SCENE.draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'outer'),      
                }
char_dict_phase2_1= {
        
     'echo':PH_SCENE.draw_char(marker_center_row-n_marker//2, marker_center_col+n_marker//3+1, name='echo')    ,
      'marker':PH_SCENE.draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'outer'),      
                }  
char_dict_phase2_2= {
        
     'noise':PH_SCENE.draw_char(marker_center_row-n_marker//2-3, marker_center_col-n_marker//2+2, name='noise') ,  
     'echo':PH_SCENE.draw_char(marker_center_row-n_marker//2, marker_center_col+n_marker//3+1, name='echo')    ,
      'marker':PH_SCENE.draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'outer'),      
                }  

char_dict_phaseend= {
        
      'marker':PH_SCENE.draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'full'),      
                }      
animation_bases=[]
for char_dict_phase in [char_dict_phase0, 
                        char_dict_phase1_1, char_dict_phase1_2, 
                        char_dict_phase2_1, char_dict_phase2_2, 
                        char_dict_phaseend]:
    hex_colors = background_base_hex_colors.copy()
    for char, dict_ in char_dict_phase.items(): 

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

                else:
                    hex_colors = cgc.color_row_gradient(part, 
                            colors[0],colors[1],
                            hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear') 
    animation_bases.append(hex_colors)

    #-------------------READ SOLN-----------------------
precession_period = 2 * np.pi / abs(TRAJ.foucault_pendulum().Omega_z) *u.s  # s, one full 360 deg sweep
n_cycles = 3.
traj = TRAJ.foucault_pendulum(l=50.0*u.m,
                               latitude=30*u.deg,
                               x0_prime=0.0*u.m,            # displaced along Y -> swings vertically on the plot
                               y0_prime=2.0*u.m,
                               vx0_prime=0.0*u.m/u.s,
                               vy0_prime=0.0*u.m/u.s,
                               t_total=precession_period*n_cycles,
                               dt=(precession_period*n_cycles/900)) 


sol = traj.solve_analytic()   # closed-form 
X, Y = sol['x'], sol['y']
t_span = sol['t']


grid_hex_rc_square = cg.fn2grid(
    X, Y,
    DOMAIN_W_SCALE=0.5, DOMAIN_H_SCALE=1,
    PIVOT_ROW=marker_center_row, PIVOT_COL=marker_center_col,
    PIVOT_ROW_X=0, PIVOT_COL_Y=0,
    detail_info=detail_info, square=True,
)


alpha = np.unwrap(2*np.arctan2(X, Y)) / 2           # bob angle from +y; doubling removes the up/down flips
sweep = np.abs(alpha - alpha[0])                    # how far the swing plane has turned (0 -> 2π)
direction = np.sign(alpha[-1] - alpha[0])           # +1 clockwise, -1 counterclockwise

# --- each hex's angle from +y around the pivot ---
rc = np.asarray(hex_rc_arr)
dx = (rc[:, 1] - marker_center_col) * dx_hex_center
dy = (marker_center_row - rc[:, 0]) * dy_hex_center           # row 0 is the top, so +y = fewer rows
hex_angle = (direction * np.arctan2(dx, dy)) % (2*np.pi)

REVEAL_cycles = [ "both","both", "one"]    # "both": up AND down in one swing    "one": half-line from +y only
cycle_bases = [[1, 2], [3, 4], [5]]          # indices into animation_bases

sweep_c = sweep * len(REVEAL_cycles) * 2*np.pi / sweep[-1]    # 0 at the first trajectory frame, exactly 3 turns at the end


# ── UPDATE ──────────────────────────────────────────────────────────────
total_seconds = 18.0

Phase0_end1 = 10
Phase0_end2 = Phase0_end1+10
Phase0_endTOT_START = Phase0_end2 + len(t_span)        
total_frames =    Phase0_endTOT_START+30

print("total_frames", total_frames)

def update(snapshot):
    if snapshot<Phase0_end1:
        current_hex_colors = background_base_hex_colors.copy()
    elif snapshot<Phase0_end2:
        current_hex_colors = animation_bases[0].copy()
    elif snapshot<Phase0_endTOT_START:
        current_snapshot = snapshot-Phase0_end2
        s = sweep_c[current_snapshot]
        cycle = min(int(s // (2*np.pi)), len(REVEAL_cycles) - 1)
        s_in = s - cycle * 2*np.pi                                    # 0 -> 2π inside this cycle
        mode, bases = REVEAL_cycles[cycle], cycle_bases[cycle]
        stage_angle = np.pi if mode == "both" else 2*np.pi            # "both" = two sides at once, so π per base
        stage = min(int(s_in // stage_angle), len(bases) - 1)
        s_stage = s_in - stage * stage_angle
        prev_idx = bases[stage-1] if stage > 0 else (cycle_bases[cycle-1][-1] if cycle > 0 else 0)
        revealed = (hex_angle % np.pi <= s_stage) if mode == "both" else (hex_angle <= s_stage)

        current_hex_colors = animation_bases[prev_idx].copy()
        current_hex_colors[revealed] =[cgc.select_normal_color([True], c, np.ones(3)*sigma_color)[0] 
                                                                      for c in animation_bases[bases[stage]][revealed]]
        

        
        #animation_bases[bases[stage]][revealed]

        loc = grid_hex_rc_square[current_snapshot]
        pendulum = cg.hex_neighbours_n(loc[0], loc[1], n=1)
        select_part = cg.select_mask(pendulum, hex_rc_arr)
        current_hex_colors[select_part] = cgc.select_normal_color(select_part, cgc.hex_to_rgb("#ffcf29"), np.ones(3)*sigma_color*3)
    else:
        current_hex_colors = animation_bases[-1].copy()
    pc.set_facecolor(current_hex_colors)
    return (pc,)



FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'PandoraHearts_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")