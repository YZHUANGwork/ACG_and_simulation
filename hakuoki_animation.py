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
import TRAJECTRY_drag as TRAJECTRY
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)



sigma_color = 0.02

hex_colors = cgc.color_row_gradient(hex_rc_arr, cgc.hex_to_rgb("#b5e7ff"), cgc.hex_to_rgb("#f1faff"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01)    
pure_bkgd = hex_colors.copy()
head_row = 12
row_seg1 = 3
row_seg2 = 4
row_seg3 = 6
center_col = 17
n_head = 1
char_ = []
char_dict = {'hijikata': [(head_row, center_col-3),     "#000000",     ["#bbb6c2", "#6b6770"]],
            'kondou':    [(head_row, center_col+2),     "#333535",     ["#5b5a5c", "#2c2a2e"]],
            'soji':      [(head_row-1, center_col+5),   "#604015",     ["#58775e", "#2b392e"]],
            'saito':     [(head_row-1, center_col-8),   "#450b6e",     ["#000000", "#000000"]],
            'harada':    [(head_row-3, center_col-12),  "#5e0606",     ["#9c9598", "#706469"]],
            'heisuke':    [(head_row-1, center_col+10),  "#755137",     ["#8f9c2a", "#4d5412"]],
            'sannan':   [(head_row-3, center_col-16),  "#000000",     ["#2f2f31", "#000000"]],
            'shinpachi': [(head_row-3, center_col+15),  "#2e2926",     ["#d8e8f0", "#646769"]],}


for key in char_dict.keys():
    
    head_band_color = "#e4e4e4" if key !='shinpachi' else "#1e501a"
    head_center = char_dict.get(key)[0]

    head = cg.hex_neighbours_n(head_center[0], head_center[1],  n=n_head, keep_origin = True, return_frontier=False)
    
    head_band= [(r,c) for r,c in head if r == head_center[0]]
    
    hair = []
    if key =='hijikata':
        hair+=cgd.draw_block((head_center[0],   (head_center[1],head_center[1]) ) ,head_center[0]+4)
        print(hair)
    elif key =='harada':    
        hair+=cgd.draw_block((head_center[0],   (head_center[1],head_center[1]) ) ,head_center[0]+3)
    elif key =='heisuke':    
        hair+=cgd.draw_block((head_center[0]-2,   (head_center[1],head_center[1]) ) ,head_center[0]+6)
    elif key =='sannan': 
        hair+=cgd.draw_triangle(head_center[0], head_center[1], head_center[0]+2,
                                                          slope_left = '0.5', slope_right = '0.5', direction = 'lr')[3]
    
    head_bot_r = max(r for r,c in head)
    if head_center[1] < center_col:
        slope_left = '0.5'
        slope_right = 'inf'
        left_col_idc = -1
        left_down = True
        right_down = False
        op = max
    else:
        slope_left = 'inf' 
        slope_right = '0.5'
        left_col_idc = -2
        left_down = False 
        right_down = True
        op = min
    min_col_idc = -2
    max_col_idc = -1
    
    if key =='harada' or key == 'sannan'  or key == 'shinpachi':  
        row_seg1_final =row_seg1+1
        row_seg2_final =row_seg2+1
    else:
        row_seg1_final =row_seg1
        row_seg2_final =row_seg2

    _, _, _, _, body1, vertex1 =  cgd.draw_trapezoid(head_bot_r,  min(c for r,c in head if r ==head_bot_r), 
                                                          max(c for r,c in head if r ==head_bot_r),head_bot_r+row_seg1_final,
                                                          slope_left = '0.5', slope_right = '0.5', direction = 'lr')
    
    haori_bot_r = head_bot_r+row_seg1_final+row_seg2_final
    _, _, _, _, body2, vertex2  = cgd.draw_trapezoid(head_bot_r+row_seg1_final,  
                                                     vertex1[min_col_idc][1], vertex1[max_col_idc][1],haori_bot_r,
                                                          slope_left = slope_left, slope_right = slope_right, direction = 'lr')
    _, _, _,remove, _= cgd.draw_triangle(haori_bot_r-2, 
                               (min(c for r,c in body2 if r ==haori_bot_r-2)+max(c for r,c in body2 if r ==haori_bot_r-2))//2,  
                               haori_bot_r, slope_left = slope_left, slope_right = slope_right, direction = 'lr')
    
    body2 = [(r,c) for r,c in body2 if (r,c) not in remove]
    
    haori = body1+body2
    marker = [(head_bot_r+row_seg1_final,  op(c for r,c in haori if r == head_bot_r+row_seg1_final)),
             
             (head_bot_r+row_seg1_final+2, op(c for r,c in haori if r == head_bot_r+row_seg1_final+2))]
    
    bottom_2 = []
    if key =='saito':   
        bottom = cgd.draw_block((haori_bot_r-2,   (min(c for r,c in body2 if r == haori_bot_r-2)+1,
                                               max(c for r,c in body2 if r == haori_bot_r-2)-1
                                              )
                           ) ,22)
        scarf = [(r,c) for r,c in haori if r == head_bot_r+1]
        select_scarf = cg.select_mask(scarf, hex_rc_arr)
    elif key =='soji':   
        bottom = cgd.draw_block((haori_bot_r-2,   (min(c for r,c in body2 if r == haori_bot_r-2),
                                               max(c for r,c in body2 if r == haori_bot_r-2)-1
                                              )
                           ) ,20)
        
        bottom_2 = cgd.draw_trapezoid(20, min(c for r,c in bottom if r == 20), max(c for r,c in bottom if r == 20),22,
                                                          slope_left = '0.5', slope_right = '0.5', direction = 'rl')[-2]
        print('bottom_2 = ', bottom_2)

    elif key =='heisuke':
        bottom = cgd.draw_block((haori_bot_r-2,   (min(c for r,c in body2 if r == haori_bot_r-2),
                                               max(c for r,c in body2 if r == haori_bot_r-2)
                                              )
                           ) ,haori_bot_r)
        bottom_2 = cgd.draw_block((haori_bot_r-1, (min(c for r,c in body2 if r == haori_bot_r-1)+1,
                                               min(c for r,c in body2 if r == haori_bot_r-1)+2)) ,22
                               )+cgd.draw_block((haori_bot_r-1, (min(c for r,c in body2 if r == haori_bot_r-1)+3,
                                               min(c for r,c in body2 if r == haori_bot_r-1)+4)) ,22
                               )
    elif   key == 'shinpachi':  
        bottom = cgd.draw_block((haori_bot_r-2,   (min(c for r,c in body2 if r == haori_bot_r-2),
                                               max(c for r,c in body2 if r == haori_bot_r-2)
                                              )
                           ) ,haori_bot_r+1)
        bottom_2 = cgd.draw_trapezoid(haori_bot_r-1,  min(c for r,c in body2 if r == haori_bot_r-1), 
                                                      max(c for r,c in body2 if r == haori_bot_r-1),22,
                                                          slope_left = '0.5', slope_right = '0.5', direction = 'rl')[-2]
   
    else:
        
        bottom = cgd.draw_block((haori_bot_r-2,   (min(c for r,c in body2 if r == haori_bot_r-2),
                                               max(c for r,c in body2 if r == haori_bot_r-2)
                                              )
                           ) ,22)
    
    select_marker = cg.select_mask(marker, hex_rc_arr)
    select_haori = cg.select_mask(haori, hex_rc_arr)
    select_head_band = cg.select_mask(head_band, hex_rc_arr) 
    select_bot_region = cg.select_mask(bottom, hex_rc_arr)
    select_bot_region2 = cg.select_mask(bottom_2, hex_rc_arr)
    
    select_head = cg.select_mask(head, hex_rc_arr)
    select_hair = cg.select_mask(hair, hex_rc_arr)
    
  
    if key =='soji':   
        hex_colors[select_bot_region2] =cgc.select_normal_color(select_bot_region2, [0.8, 0.8, 0.8], np.ones(3)*sigma_color) 
    elif  key =='heisuke':   
        hex_colors[select_bot_region2] = cgc.select_normal_color(select_bot_region2, cgc.hex_to_rgb("#143747"), np.ones(3)*sigma_color) 
    elif key =='shinpachi':   
        hex_colors[select_bot_region2] = cgc.select_normal_color(select_bot_region2, cgc.hex_to_rgb("#a8adb9"), np.ones(3)*sigma_color) 
    hex_colors = cgc.color_row_gradient(bottom,  
                                                          cgc.hex_to_rgb(char_dict.get(key)[-1][-1]) , 
                                                          cgc.hex_to_rgb(char_dict.get(key)[-1][0]),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color,   end_weight= 0.01)

    
   
    
    hex_colors = cgc.color_row_gradient(haori,  cgc.hex_to_rgb("#b5e0ff") , cgc.hex_to_rgb("#48cdcd" ),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color,   end_weight= 0.1)

    if key =='saito': 
        hex_colors[select_scarf] = cgc.select_normal_color(select_scarf, [1,1,1], np.ones(3)*sigma_color) 
    hex_colors[select_marker] = [1,1,1]    
    hex_colors[select_head] = cgc.select_normal_color(select_head, cgc.hex_to_rgb(char_dict.get(key)[1]), np.ones(3)*sigma_color*3) 
    
    hex_colors[select_hair] = cgc.select_normal_color(select_hair, cgc.hex_to_rgb(char_dict.get(key)[1]), np.ones(3)*sigma_color*3) 
    hex_colors[select_head_band] = cgc.hex_to_rgb(head_band_color)
    if key =='heisuke':    
        hex_colors[select_hair] = cgc.select_normal_color(select_hair, cgc.hex_to_rgb(char_dict.get(key)[1]), np.ones(3)*sigma_color*3) 
        
    char_.extend(head)
    char_.extend(head_band)
    char_.extend(hair)
    char_.extend(haori)
    char_.extend(bottom)
    char_.extend(bottom_2)
    
    

base_hex_colors = hex_colors.copy()

char_hex = list(set(char_))
select_char_hex = cg.select_mask(char_hex, hex_rc_arr)
sol      = TRAJECTRY.main(N_in=100, seed=42)
N        = TRAJECTRY.N          # capture immediately — main() overwrites these globals on next call
t_spawn  = TRAJECTRY.t_spawn
x0       = TRAJECTRY.x0
z0       = TRAJECTRY.z0

sol2     = TRAJECTRY.main(N_in=100, seed=7)
N2       = TRAJECTRY.N
t_spawn2 = TRAJECTRY.t_spawn
x0_2     = TRAJECTRY.x0

canvas_physical_x_range = (min(x0), max(x0))
canvas_physical_y_range = (0, max(z0)*0.95)



first_ground_row = 22

FALLING_color_range = [[0.90, 0.72, 0.74], [0.98, 0.84, 0.86]]
select_FALL = np.ones(N, dtype=bool)
color_fall  = cgc.select_uniform_color(select_FALL, FALLING_color_range[0], FALLING_color_range[1])  # (N,3)

FALLING_color_range2 = [[0.95, 0.88, 0.89], [0.99, 0.95, 0.96]]
select_FALL2 = np.ones(N2, dtype=bool)
color_fall2  = cgc.select_uniform_color(select_FALL2, FALLING_color_range2[0], FALLING_color_range2[1])

# ── WORLD -> HEX MAPPING (ALL FRAMES, ALL PETALS) ────────────────────────
n_steps = sol.t.shape[0]
R_vec = sol.y[0:3*N, :].reshape(3, N, -1)   # (3, N, n_steps) m

HORIZONTAL_traj = R_vec[0].ravel()
VERTICAL_traj   = R_vec[2].ravel()
hex_coord = cg.world_metres_to_hex_index(HORIZONTAL_traj, VERTICAL_traj, detail_info,
                                        canvas_physical_x_range = canvas_physical_x_range,
                                        canvas_physical_y_range = canvas_physical_y_range)
rc_to_idx   = {rc: i for i, rc in enumerate(hex_rc_arr)}
flat_idx  = np.array([rc_to_idx.get(rc, -1) for rc in hex_coord])
hex_row_arr = np.array([rc[0] for rc in hex_rc_arr])
idx_per_frame = flat_idx.reshape(N, n_steps)
Z_arr = R_vec[2].T


n_steps2 = sol2.t.shape[0]
R_vec2 = sol2.y[0:3*N2, :].reshape(3, N2, -1)   # (3, N2, n_steps2) m
 
HORIZONTAL_traj2 = R_vec2[0].ravel()
VERTICAL_traj2   = R_vec2[2].ravel()
hex_coord2 = cg.world_metres_to_hex_index(HORIZONTAL_traj2, VERTICAL_traj2, detail_info,
                                        canvas_physical_x_range = canvas_physical_x_range,
                                        canvas_physical_y_range = canvas_physical_y_range)
flat_idx2       = np.array([rc_to_idx.get(rc, -1) for rc in hex_coord2])
idx_per_frame2  = flat_idx2.reshape(N2, n_steps2)
Z_arr2 = R_vec2[2].T


Phase0_end = 10
n_snapshots = n_steps+Phase0_end
def update(snapshot):
    if snapshot <Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot-Phase0_end
        current_hex_colors = base_hex_colors.copy()
        
        # layer 2: sol2 — falls behind the character
        idx2    = idx_per_frame2[:, current_snapshot]
        valid2  = idx2 >= 0
        row_now2 = np.where(valid2, hex_row_arr[np.clip(idx2, 0, None)], 10**6)
        spawned2 = sol2.t[current_snapshot] >= t_spawn2
        falling2 = spawned2 & valid2 & (Z_arr2[current_snapshot] > 0) & (row_now2 <= first_ground_row)
        current_hex_colors[idx2[falling2]] = color_fall2[falling2]
 

        # layer 3: character — redrawn on top so sol2 never occludes it
        #current_hex_colors[select_char_hex] = base_hex_colors[select_char_hex]
        alpha = 1.0 - current_snapshot / max(n_steps - 1, 1)
        current_hex_colors[select_char_hex] = (
            alpha * base_hex_colors[select_char_hex] + (1 - alpha) * pure_bkgd[select_char_hex]
        )
        

        idx   = idx_per_frame[:, current_snapshot]
        valid   = idx   >= 0   # False if this trajectory point fell outside the calibrated grid
        row_now  = np.where(valid,   hex_row_arr[np.clip(idx,   0, None)], 10**6)
        

        spawned = sol.t[current_snapshot] >= t_spawn
        falling = spawned & valid & (Z_arr[current_snapshot] > 0) & (row_now <= first_ground_row)
        current_hex_colors[idx[falling]] = color_fall[falling]
 
        
        pc.set_facecolor(current_hex_colors)
    return (pc,)

total_seconds =30.0

# FPS [1/time] = total_frames / total_seconds
FPS = (n_snapshots / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
 
ani = animation.FuncAnimation(fig, update, frames=int(n_snapshots),
                               interval=time_gap, blit=False)
 
# ── SAVE ──────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'hakuoki_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved → {OUTPUT_FILE}")
#plt.savefig('cg/'+cg_name, dpi=DPI, bbox_inches='tight')
#print('saved')