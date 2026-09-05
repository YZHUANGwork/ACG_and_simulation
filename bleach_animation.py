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
from scipy.interpolate import interp1d
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import expected_value as EXP

import SIM_montecarlo as SIM
from scipy.stats import maxwell, norm
import bleach_scene_plot as BLEACH_SCENE
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

sigma_color = 0.03


head_center_row = 12
head_center_col = 14
n_head = 3
N = 500

#---------------animation_base_hex_colors_phase1----------------------------

urahara_dict, body_info_dict = BLEACH_SCENE.draw_urahara(head_center_row, head_center_col, n_head)
urahara_dict['skin'][1] = [cgc.hex_to_rgb("#84807b"),  cgc.hex_to_rgb("#fee9d2")]
urahara_dict['hair'][1] = [cgc.hex_to_rgb("#ffe791"), cgc.hex_to_rgb("#fff4cf")]

bkgd = BLEACH_SCENE.draw_bkgd(head_center_row, head_center_col, 30)
bkgd['shells_odd'][1] = [cgc.hex_to_rgb("#87ff00")]     #[1,1,1]
bkgd['shells_even'][1] = [cgc.hex_to_rgb("#d60600")]#cgc.hex_to_rgb("#5a8930")

foreground_points_phase1 = []
char_points = []
char_dict_phase1 = {'bkgd': bkgd, 'urahara': urahara_dict,
             'clothing': BLEACH_SCENE.draw_urahara_clothing(body_info_dict, N, detail_info, 'mode0'),
              
}
for dict_key in char_dict_phase1.keys(): 
    dict_ = char_dict_phase1.get(dict_key)
    for key in dict_.keys():
        part  = list(set(dict_.get(key)[0]))
        colors = dict_.get(key)[1] 
        
        if key == 'hair':
            foreground_points_phase1.extend(part)
        
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*3) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'sigmoid') 

sigma_color = 0.03


foreground_mask_phase1 = cg.select_mask(foreground_points_phase1, hex_rc_arr)            
animation_base_hex_colors_phase1 = hex_colors.copy()

#---------------animation_base_hex_colors_phase2----------------------------

skyline = head_center_row+n_head
select_upper_bkgd = [r <skyline and ((r - 1) // 2) % 2 == 0 for r,c in hex_rc_arr]
select_bottom_bkgd = [r >=skyline for r,c in hex_rc_arr]
hex_colors[select_upper_bkgd] = cgc.select_normal_color(select_upper_bkgd, cgc.hex_to_rgb("#d60600") , np.ones(3) * sigma_color*5)
hex_colors[select_bottom_bkgd] = cgc.select_normal_color(select_bottom_bkgd, cgc.hex_to_rgb("#87ff00") , np.ones(3) * sigma_color*5)
bkgd_phase2_hex_colors = hex_colors.copy()
       
foreground_points_phase2 = []
char_dict_phase2 = {'urahara': urahara_dict,
             'clothing': BLEACH_SCENE.draw_urahara_clothing(body_info_dict, N, detail_info, 'mode3'),
              
}
for dict_key in char_dict_phase2.keys(): 
    dict_ = char_dict_phase2.get(dict_key)
    for key in dict_.keys():
        part  = list(set(dict_.get(key)[0]))
        if key in ['left green', 'right green']:
            colors = [cgc.hex_to_rgb("#3b6632")]
        else:
            colors = dict_.get(key)[1] 
        foreground_points_phase2.extend(part)
        
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'sigmoid') 

foreground_mask_phase2 = cg.select_mask(foreground_points_phase2, hex_rc_arr)            
animation_base_hex_colors_phase2 = hex_colors.copy()



#---------------animation_base_hex_colors_phase3----------------------------

hex_colors = bkgd_phase2_hex_colors.copy()
benihime_dict = BLEACH_SCENE.draw_benihime(body_info_dict, N, detail_info, 'mode1')
benihime_dict['left red'][1] = [cgc.hex_to_rgb("#4b000e")]
char_dict_phase3 = {
                    'benihime': benihime_dict,
              'urahara': urahara_dict,
             'clothing': BLEACH_SCENE.draw_urahara_clothing(body_info_dict, N, detail_info, 'mode3'),
#    'benihime arm up': BLEACH_SCENE.draw_benihime(body_info_dict, N, detail_info, 'arm up'),
              
}
for dict_key in char_dict_phase3.keys(): 
    dict_ = char_dict_phase3.get(dict_key)
    for key in dict_.keys():
        if key == 'forearms':
            continue
        part  = list(set(dict_.get(key)[0]))
        if key in ['left green', 'right green']:
            colors = [cgc.hex_to_rgb("#3b6632")]
        else:
        
            colors = dict_.get(key)[1] 
        
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'sigmoid') 
animation_base_hex_colors_phase3 = hex_colors.copy()

rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}

STRIPE_WIDTH = 1                   # how many hex columns wide each stripe is
STRIPE_PALETTE = [
    np.array([0.9, 0.9, 0.9]),   
    cgc.hex_to_rgb("#3b6632"),  
]
 
stripe_target_colors = np.zeros((len(hex_rc_arr), 3))
for i, (row_i, col_i) in enumerate(hex_rc_arr):
    stripe_idx = (col_i // STRIPE_WIDTH) % len(STRIPE_PALETTE)
    stripe_target_colors[i] = STRIPE_PALETTE[stripe_idx]
    

mode1 = BLEACH_SCENE.draw_urahara_clothing(body_info_dict, N, detail_info, 'mode1')

SCATTER_SETS = [
    (mode1['left white'][0], Counter(mode1['left white'][0])),
    (mode1['right white'][0], Counter(mode1['right white'][0])),
]            

mode2 = BLEACH_SCENE.draw_urahara_clothing(body_info_dict, N, detail_info, 'mode2')
grid_hex_rc_hat = mode2['hat'][0]
final_counts_hat = Counter(grid_hex_rc_hat)         

SCATTER_SETS_green = [
    (mode2['left green'][0], Counter(mode2['left green'][0])),
    (mode2['right green'][0], Counter(mode2['right green'][0])),
] 

mode0 = BLEACH_SCENE.draw_urahara_clothing(body_info_dict, N, detail_info, 'mode0')

SCATTER_SETS_black = [
    (mode0['left black'][0], Counter(mode0['left black'][0])),
    (mode0['right black'][0], Counter(mode0['right black'][0])),
]            
  
print(np.array(mode0['left black'][0]).shape, np.array(mode2['left green'][0]).shape)


benihime_mode1 = BLEACH_SCENE.draw_benihime(body_info_dict, N, detail_info, 'mode1')
benihime_armup = BLEACH_SCENE.draw_benihime(body_info_dict, N, detail_info, 'up')

benihime_raw = BLEACH_SCENE.draw_benihime(body_info_dict, N, detail_info, 'raw')

SCATTER_SETS_benihime_part1 = [
    (benihime_raw['left red'][0], Counter(benihime_raw['left red'][0]), benihime_raw['left red'][1][0]),
    (benihime_raw['right red'][0], Counter(benihime_raw['right red'][0]), benihime_raw['right red'][1][0]),
    (benihime_raw['face'][0], Counter(benihime_raw['face'][0]), benihime_raw['face'][1][0]),
    (benihime_mode1['left shoulder joint'][0], Counter(benihime_mode1['left shoulder joint'][0]), 
     benihime_mode1['left shoulder joint'][1][0]),
    (benihime_mode1['right shoulder joint'][0], Counter(benihime_mode1['right shoulder joint'][0]), 
     benihime_mode1['right shoulder joint'][1][0]),
]            


SCATTER_SETS_benihime_part2 = [
    (benihime_raw['left brachium'][0], benihime_raw['left brachium'][1][0]),
    (benihime_raw['right brachium'][0], benihime_raw['right brachium'][1][0]),]

SCATTER_SETS_benihime_part3 = [
    (benihime_mode1['left elbow joint'][0], Counter(benihime_mode1['left elbow joint'][0]) ,benihime_mode1['left elbow joint'][1][0]),
    (benihime_mode1['right elbow joint'][0],Counter(benihime_mode1['right elbow joint'][0]) , benihime_mode1['right elbow joint'][1][0]),
] 
SCATTER_SETS_benihime_part4 = [
    (benihime_raw['left hair'][0], benihime_raw['left hair'][1][0] ),
    (benihime_raw['right hair'][0], benihime_raw['right hair'][1][0]  ),
] 
    
SCATTER_SETS_benihime_part5 = [
    (benihime_raw['left hair extra'][0], benihime_raw['left hair extra'][1][0]),
    (benihime_raw['right hair extra'][0], benihime_raw['right hair extra'][1][0]),]

SCATTER_SETS_benihime_part6 = [
    (benihime_armup['left forearm'][0], benihime_armup['left forearm'][1][0] ),
    (benihime_armup['right forearm'][0], benihime_armup['right forearm'][1][0] ),]


# ── UPDATE ──────────────────────────────────────────────────────────────
Phase0_end= 120         # intro frames that just hold the base scene
total_seconds = 18.0
Phase_hat_start = Phase0_end+N
Phase_green_start = Phase_hat_start+N//3*2
Phase2_end = Phase_green_start+N+Phase0_end
Phase_benihime_part1_start = Phase2_end+Phase0_end
Phase_benihime_part2_start = Phase_benihime_part1_start+N//3*2
Phase_benihime_part3_start = Phase_benihime_part2_start+N//3*2
Phase_benihime_part4_start = Phase_benihime_part3_start+N//3
Phase_benihime_part5_start = Phase_benihime_part4_start+N//3
Phase_benihime_part6_start = Phase_benihime_part5_start+N
total_frames = Phase_benihime_part6_start+N+Phase0_end
print('tatal frame', total_frames)
def update(snapshot):
    if snapshot % (total_frames // 10) == 0:
        print(f"{snapshot * 100 // total_frames}% done")
        
        
    if snapshot < Phase0_end:
        current_hex_colors = animation_base_hex_colors_phase1.copy()
        pc.set_facecolor(current_hex_colors)
    elif snapshot>= Phase0_end and snapshot<Phase2_end:
        current_snapshot = snapshot - Phase0_end
        current_hex_colors = animation_base_hex_colors_phase1.copy()
        
        n_revealed = current_snapshot + 1  
        for grid_hex_rc, final_counts in SCATTER_SETS:
            running_counts = Counter(grid_hex_rc[:n_revealed])
            for rc, running_n in running_counts.items():
                idx = rc_to_idx.get(rc)
                if idx is None:
                    continue  # point landed outside the drawn hex grid, skip

                alpha = running_n / final_counts[rc]
                background = animation_base_hex_colors_phase1[idx]
                target = np.array([0.95, 0.95, 0.95])
                mean_color = alpha * target + (1 - alpha) * background
                current_hex_colors[idx] = cgc.select_normal_color(
                    [True], mean_color, np.ones(3) * sigma_color)[0]
        
        current_hex_colors[foreground_mask_phase1] = animation_base_hex_colors_phase1[foreground_mask_phase1]  # keep silhouette on top
        
        if snapshot>= Phase_hat_start:
            HAT_snapshot = snapshot - Phase_hat_start
            n_revealed_HAT = HAT_snapshot + 1  
            running_counts = Counter(grid_hex_rc_hat[:n_revealed_HAT])
            for rc, running_n in running_counts.items():
                idx = rc_to_idx.get(rc)
                if idx is None:
                    continue

                alpha = running_n / final_counts_hat[rc] 
                background = current_hex_colors[idx]
                target = stripe_target_colors[idx]
                mean_color = alpha * target + (1 - alpha) * background
                current_hex_colors[idx] = cgc.select_normal_color(
                    [True], mean_color, np.ones(3) * sigma_color)[0]
        
        if snapshot>= Phase_green_start:
            green_snapshot = snapshot - Phase_green_start
            n_revealed_green = green_snapshot + 1  
            for grid_hex_rc, final_counts in SCATTER_SETS_green:
                running_counts = Counter(grid_hex_rc[:n_revealed_green])
                for rc, running_n in running_counts.items():
                    idx = rc_to_idx.get(rc)
                    if idx is None:
                        continue 
                    alpha = running_n / final_counts[rc]  # 0 -> none landed yet, 1 -> all landed
                    background = current_hex_colors[idx]
                    target = np.array(cgc.hex_to_rgb("#3b6632"))
                    mean_color = alpha * target + (1 - alpha) * background
                    #current_hex_colors[idx] = alpha * target + (1 - alpha) * background
                    current_hex_colors[idx] = cgc.select_normal_color(
                        [True], mean_color, np.ones(3) * sigma_color)[0]
                    
        
        pc.set_facecolor(current_hex_colors)
        
        
    elif snapshot>= Phase2_end and snapshot<Phase_benihime_part1_start:
        current_hex_colors = animation_base_hex_colors_phase2.copy()
        for grid_hex_rc, final_counts in SCATTER_SETS_green:
            running_counts = Counter(grid_hex_rc)
            for rc, running_n in running_counts.items():
                idx = rc_to_idx.get(rc)
                if idx is None:
                    continue 
                
                background = current_hex_colors[idx]
                target = np.array(cgc.hex_to_rgb("#3b6632"))
                mean_color = target 
                #current_hex_colors[idx] = alpha * target + (1 - alpha) * background
                current_hex_colors[idx] = cgc.select_normal_color(
                    [True], mean_color, np.ones(3) * sigma_color)[0]
                    
        pc.set_facecolor(current_hex_colors)
        
    else:
        current_hex_colors = animation_base_hex_colors_phase2.copy()
        for grid_hex_rc, final_counts in SCATTER_SETS_green:
            running_counts = Counter(grid_hex_rc)
            for rc, running_n in running_counts.items():
                idx = rc_to_idx.get(rc)
                if idx is None:
                    continue 
                
                background = current_hex_colors[idx]
                target = np.array(cgc.hex_to_rgb("#3b6632"))
                mean_color = target 
                #current_hex_colors[idx] = alpha * target + (1 - alpha) * background
                current_hex_colors[idx] = cgc.select_normal_color(
                    [True], mean_color, np.ones(3) * sigma_color)[0]
                
        #face, clothing, shoulder joint 
        if snapshot>=Phase_benihime_part1_start:
            
            part1_snapshot = snapshot - Phase_benihime_part1_start
            n_revealed_part1 = part1_snapshot + 1  
            for grid_hex_rc, final_counts, targe_color in SCATTER_SETS_benihime_part1:
                valid_rc = [(r,c) for r,c in grid_hex_rc[:n_revealed_part1] if (r,c) not in foreground_points_phase2]
                running_counts = Counter(valid_rc)
                for rc, running_n in running_counts.items():
                    idx = rc_to_idx.get(rc)
                    if idx is None:
                        continue 
                    alpha = running_n / final_counts[rc]  # 0 -> none landed yet, 1 -> all landed
                    background = current_hex_colors[idx]
                    target = np.array(targe_color)
                    mean_color = alpha * target + (1 - alpha) * background
                    current_hex_colors[idx] = cgc.select_normal_color(
                        [True], mean_color, np.ones(3) * sigma_color)[0]

        #brachium
        if snapshot>= Phase_benihime_part2_start:
            part2_snapshot = snapshot - Phase_benihime_part2_start
            n_revealed_part2 = part2_snapshot + 1  
            for grid_hex_rc, targe_color in SCATTER_SETS_benihime_part2:
                #revealed_points = grid_hex_rc[:n_revealed_part3]  # raw points, min row first
                for r, c in grid_hex_rc[:n_revealed_part2]:
                    for rc in BLEACH_SCENE.brachium_region(r, c):
                        idx = rc_to_idx.get(rc)
                        if idx is None:
                            continue
                        current_hex_colors[idx] = cgc.select_normal_color(
                            [True], np.array(targe_color), np.ones(3) * sigma_color)[0]
                        
        #elbow joint
        if snapshot>= Phase_benihime_part3_start:
            part3_snapshot = snapshot - Phase_benihime_part3_start
            n_revealed_part3 = part3_snapshot + 1  
            for grid_hex_rc, final_counts, targe_color in SCATTER_SETS_benihime_part3:
                valid_rc = [(r,c) for r,c in grid_hex_rc[:n_revealed_part3] if (r,c) not in foreground_points_phase2]
                running_counts = Counter(valid_rc)
                for rc, running_n in running_counts.items():

                    idx = rc_to_idx.get(rc)
                    if idx is None:
                        continue 
                    alpha = running_n / final_counts[rc]  # 0 -> none landed yet, 1 -> all landed
                    background = current_hex_colors[idx]
                    target = np.array(targe_color)
                    mean_color = alpha * target + (1 - alpha) * background
                    current_hex_colors[idx] = cgc.select_normal_color(
                        [True], mean_color, np.ones(3) * sigma_color)[0]
        #hair
        if snapshot>= Phase_benihime_part4_start:
            part4_snapshot = snapshot - Phase_benihime_part4_start
            
            n_revealed_part4 = part4_snapshot + 1  
            for grid_hex_rc, targe_color in SCATTER_SETS_benihime_part4:
                revealed_points = grid_hex_rc[:n_revealed_part4]  # raw points, min row first
                for r, c in revealed_points:
                    for rc in BLEACH_SCENE.hair_region(r, c):

                        idx = rc_to_idx.get(rc)
                        if idx is None:
                            continue
                        current_hex_colors[idx] = cgc.select_normal_color(
                            [True], np.array(targe_color), np.ones(3) * sigma_color*3)[0]
        
        
        
        
        #extra hair
        if snapshot>= Phase_benihime_part5_start:
            part5_snapshot = snapshot - Phase_benihime_part5_start
            n_revealed_part5 = part5_snapshot + 1  
            for grid_hex_rc, targe_color in SCATTER_SETS_benihime_part5:
                valid_rc = [(r,c) for r,c in grid_hex_rc[:n_revealed_part5] if (r,c) not in foreground_points_phase2]
                for rc in valid_rc:
                    idx = rc_to_idx.get(rc)
                    if idx is None:
                        continue
                    current_hex_colors[idx] = cgc.select_normal_color(
                        [True], np.array(targe_color), np.ones(3) * sigma_color*3)[0]
        
        #arm up
        if snapshot>= Phase_benihime_part6_start:
            #current_hex_colors = animation_base_hex_colors_phase3.copy()
            part6_snapshot = snapshot - Phase_benihime_part6_start
            n_revealed_part6 = part6_snapshot + 1  
            for grid_hex_rc, targe_color in SCATTER_SETS_benihime_part6:
                
                for rc in grid_hex_rc[:n_revealed_part6]:
                    idx = rc_to_idx.get(rc)
                    if idx is None:
                        continue
                    current_hex_colors[idx] = cgc.select_normal_color(
                        [True], np.array(targe_color), np.ones(3) * sigma_color)[0]
                  
        pc.set_facecolor(current_hex_colors)
    return (pc,)
total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'bleach_animation.mp4')
 
print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")
 
