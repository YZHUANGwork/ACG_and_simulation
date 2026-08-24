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

import Uzi_scene_plot as UZI_SCENE
import  TRAJECTORY_pseudoforce as TRAJ

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

n_shell = 5
n_egg = 1
snail_shell_side_row = 12
snail_shell_side_col = 8

snail_shell_top_row = 5
snail_shell_top_col = 14

spiral_center = (6,3)
n_head = 3
head_center = (15, 27)
char_dict_ = {'snail_side': UZI_SCENE.draw_snail(snail_shell_side_row, snail_shell_side_col, n_shell, n_egg, view = 'side'),
              'snail_top': UZI_SCENE.draw_snail(snail_shell_top_row, snail_shell_top_col, n_shell, n_egg, view = 'top'),
              'pplspiral': UZI_SCENE.draw_pplspiral(spiral_center[0], spiral_center[1], n_head = 1),
              'ppl': UZI_SCENE.draw_char(head_center[0], head_center[1], n_head, start = 'head'),
#              'uzi1': draw_uzi(0, 29, 10)
             }
bkgd = cgd.draw_trapezoid(0, 28, 34, 22,  slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                      bend_left = 'left', bend_right = 'right')[-2]
select_bkgd= cg.select_mask(bkgd,hex_rc_arr)
hex_colors[select_bkgd] = [0,0,0]

for dict_ in char_dict_.values(): 

    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*0.) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01) 


base_hex_colors = hex_colors.copy()


def draw_char(start_center_row, start_center_col, n_head, start = 'head', view = 'up'):
    head, neck, body, pelvis, thigh, details = cgd.draw_body(start_center_row, start_center_col, n = n_head, start = start)
    [center_col, head_center_row, body_center_row, thigh_center_row]  = details
    
    
    if view == 'up':
        body_top_r = min(r for r, c in body)
        body_bottom_r = max(r for r, c in body)
        remove_head = [(head_center_row, min(c for r,c in head if r ==head_center_row)), 
                       (head_center_row, max(c for r,c in head if r ==head_center_row)), ]
        add_hair = cgd.verticle_line(head_center_row, min(c for r,c in head if r ==head_center_row),head_center_row+3, bend = 'right'
                                    )+cgd.verticle_line(head_center_row, max(c for r,c in head if r ==head_center_row),head_center_row+3, 
                                                        bend = 'left'
                                    )
        hair = [(r,c) for r,c in head+add_hair if (r,c) not in remove_head]
        face = cgd.draw_trapezoid(head_center_row+1, 
                                  min(c for r,c in hair if r ==head_center_row+1)+1, 
                                  max(c for r,c in hair if r ==head_center_row+1)-1, head_center_row+n_head, 
                                      slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                          bend_left = 'left', bend_right = 'right')[-2]


        shoulder_extensions = math.ceil(n_head/2)

        left_shoulders, right_shoulders = [], []
        for shoulder_extension in range(shoulder_extensions):
            left_shoulders+=cgd.draw_slope_0p5_diagonal(body_top_r, min(c for r, c in body if r ==body_top_r)-shoulder_extension-1, 
                                                 body_center_row-shoulder_extension, 
                                                left_down=True, right_down=False, left_up=False, right_up=False)
            right_shoulders+=cgd.draw_slope_0p5_diagonal(body_top_r, max(c for r, c in body if r ==body_top_r)+shoulder_extension+1, 
                                                  body_center_row-shoulder_extension, 
                                                 left_down=False, right_down=True , left_up= False, right_up=False )
        hair_color = [1,1,1]
        char = face+neck+body+left_shoulders+right_shoulders
    elif view == 'down':
        head = thigh
        neck =  pelvis
        head_center_row = thigh_center_row
        
        body_top_r = max(r for r, c in body)
        body_bottom_r = min(r for r, c in body)
        remove_head = [(head_center_row, min(c for r,c in head if r ==head_center_row)), 
                       (head_center_row, max(c for r,c in head if r ==head_center_row)), ]
        add_hair = cgd.verticle_line(head_center_row, min(c for r,c in head if r ==head_center_row),head_center_row+3, bend = 'right'
                                    )+cgd.verticle_line(head_center_row, max(c for r,c in head if r ==head_center_row),head_center_row+3, 
                                                        bend = 'left'
                                    )
        hair = [(r,c) for r,c in head if (r,c) not in remove_head]
        
        face_part1 = cgd.draw_trapezoid(head_center_row-n_head , 
                                  min(c for r,c in hair if r == head_center_row-n_head)+1, 
                                  max(c for r,c in hair if r == head_center_row-n_head)-1, head_center_row-1, 
                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr',
                          bend_left = 'left', bend_right = 'right')[-2]
        face_part2 = cgd.draw_block((max(r for r, c in face_part1), 
                                     ( min(c for r,c in face_part1 if r == max(r for r, c in face_part1)),
                                      max(c for r,c in face_part1 if r == max(r for r, c in face_part1)))), max(r for r, c in head)-1)
        face = face_part1+face_part2
        shoulder_extensions = math.ceil(n_head/2)

        left_shoulders, right_shoulders = [], []
        for shoulder_extension in range(shoulder_extensions):
            left_shoulders+=cgd.draw_slope_0p5_diagonal(body_top_r, min(c for r, c in body if r ==body_top_r)-shoulder_extension-1, 
                                                 body_center_row-shoulder_extension, 
                                                left_down=False, right_down=False, left_up=True , right_up=False)
            right_shoulders+=cgd.draw_slope_0p5_diagonal(body_top_r, max(c for r, c in body if r ==body_top_r)+shoulder_extension+1, 
                                                  body_center_row-shoulder_extension, 
                                                 left_down=False, right_down= False, left_up= False, right_up= True )
        
        hair_color = [0,0,0]
        char = neck+body+left_shoulders+right_shoulders+face
    return {
            'g': [hair, [hair_color]],
           'a': [char, [[0.7,0.7,0.7]]],
      
           
            }

hex_colors[[True]*len(hex_rc_arr)] = [1,1,1] 
bkgd = cgd.draw_trapezoid(0, 24, 34, 22,  slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                      bend_left = 'left', bend_right = 'right')[-2]
select_bkgd= cg.select_mask(bkgd,hex_rc_arr)
hex_colors[select_bkgd] = [0,0,0]


n_head = 3
head_center_up = (15, 27)
head_center_opposite = (7, 7)
char_dict_ = {'a': draw_char(head_center_up[0], head_center_up[1], n_head, start = 'head', view = 'up'),
              'b': draw_char(head_center_opposite[0], head_center_opposite[1], n_head, start = 'thigh', view = 'down'),
             }
for dict_ in char_dict_.values(): 

    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*0.) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01)  
animation_base_hex_colors = hex_colors.copy()  

#------read solution-----------------------

traj_eye = TRAJ.pseudo_force(OMEGA_z = 1*u.rad / u.s ,
                         a_real_prime = [0, 0, 0]*u.m/u.s/u.s,
                         latitude=90*u.deg, 
                         v0_prime=[-15,-15.,  0.0]*u.m/u.s,
                          t_total=40*u.s, dt=0.05*u.s)
sol_eye = traj_eye.solve()

x, y, z = sol_eye.y[0], sol_eye.y[1], sol_eye.y[2]


PARAMS = [
     {"OMEGA_z": 1.5*u.rad / u.s, "latitude": 3*u.deg, "v0_prime": [10.,0.,  0]*u.m/u.s, "r0_prime":[0.0, 0.0, 0.0]*u.m, 
      "start_row": 6, "start_col":10 },
    
    {"OMEGA_z": 0.01*u.rad / u.s, "latitude": 3*u.deg, "v0_prime": [5.,0.,  0.0]*u.m/u.s, "r0_prime":[0.0, 0.0, 100.0]*u.m, 
     "start_row": 10, "start_col":9 },
    {"OMEGA_z": 0.5*u.rad / u.s, "latitude": 30*u.deg, "v0_prime": [0.,0.,  -10.0]*u.m/u.s,"r0_prime":[0.0, 0.0, 100.0]*u.m, 
     "start_row": 10, "start_col":7 },
    {"OMEGA_z": -0.5*u.rad / u.s, "latitude": 30*u.deg, "v0_prime": [0.,0.,  -10.0]*u.m/u.s,"r0_prime":[0.0, 0.0, 100.0]*u.m, 
     "start_row": 6, "start_col":5 },
    
]

rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}   # build once, outside the loop

grid_hex_rc_i_hairs = []
snapshot_lengths = []

for param in PARAMS:
    traj_hair = TRAJ.pseudo_force(OMEGA_z = param["OMEGA_z"],
                         a_real_prime = [0, 0, -10]*u.m/u.s/u.s,
                         latitude=param["latitude"],
                         v0_prime=param["v0_prime"],
                         r0_prime=param["r0_prime"],
                          t_total=1.*u.s, dt=0.01*u.s)


    sol_hair = traj_hair.solve()
    x, y, z = sol_hair.y[0], sol_hair.y[1], sol_hair.y[2]
    X = x
    Y = z

    snapshot_lengths.append(len(sol_hair.t))
    print(len(sol_hair.t))
    DOMAIN_W = X.max() - X.min()
    DOMAIN_H = Y.max() - Y.min()
    CANVAS_PHYSICAL_H = DOMAIN_H
    
    CANVAS_PHYSICAL_W = DOMAIN_W*2
    
    print(DOMAIN_W, DOMAIN_H)
    canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
    canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)

    px0, py0 = cg.hex_center_pixel(param["start_row"], param["start_col"], detail_info)

    OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
    OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]

    rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}
    grid_hex_rc_i = cg.world_metres_to_hex_index(
            X + OFFSET_X, Y + OFFSET_Y, detail_info,
            canvas_physical_x_range=canvas_physical_x_range,
            canvas_physical_y_range=canvas_physical_y_range,
        )
    #print(grid_hex_rc_i)
    grid_hex_rc_i_hairs.append(grid_hex_rc_i)
max_snapshot = max(snapshot_lengths)

#hair uzi 
grid_hex_rc_i_hairuzi = []
snapshot_lengths_revert = []
PARAMS_revert = [
     {"OMEGA_z": 2.*u.rad / u.s, "latitude": 60*u.deg, "v0_prime": [15,15.,  0.0]*u.m/u.s, "t_total":8*u.s, 
      "plot_X_idc": 0, "plot_Y_idc": 2, "zoom_in_fac" :4,
      "start_row":6, "start_col":14 },
    
    {"OMEGA_z": 2.*u.rad / u.s, "latitude": 90*u.deg, "v0_prime": [15,15.,  0.0]*u.m/u.s, "t_total":8*u.s, 
      "plot_X_idc": 0, "plot_Y_idc": 1,"zoom_in_fac" :4,
      "start_row":17, "start_col":8 },
    
    {"OMEGA_z": 2.*u.rad / u.s, "latitude":-50*u.deg, "v0_prime":[-15,-15.,  0.0]*u.m/u.s, "t_total":8*u.s, 
      "plot_X_idc": 0, "plot_Y_idc": 1,"zoom_in_fac" :4,
      "start_row":13, "start_col":4 },
       
    {"OMEGA_z": 2.*u.rad / u.s, "latitude":70*u.deg, "v0_prime":[-15,-15.,  0.0]*u.m/u.s, "t_total":12*u.s, 
      "plot_X_idc": 0, "plot_Y_idc": 1,"zoom_in_fac" :2,
      "start_row":8, "start_col":4 },
    
     {"OMEGA_z": 3.*u.rad / u.s, "latitude":70*u.deg, "v0_prime":[-15,-15.,  0.0]*u.m/u.s, "t_total":10*u.s, 
      "plot_X_idc": 0, "plot_Y_idc": 1,"zoom_in_fac" :2,
      "start_row":12, "start_col":13 },
       
]
for param in PARAMS_revert:
    traj_uzi = TRAJ.pseudo_force(OMEGA_z = param["OMEGA_z"],
                         a_real_prime = [0, 0, 0]*u.m/u.s/u.s,
                         latitude=param["latitude"],
                         v0_prime=param["v0_prime"],
                         r0_prime=[0.0, 0.0, 0.0]*u.m,
                          t_total=param["t_total"], dt=0.05*u.s)
    sol_uzi= traj_uzi.solve()
    X = sol_uzi.y[param["plot_X_idc"]][::-1]
    Y = sol_uzi.y[param["plot_Y_idc"]][::-1]

    snapshot_lengths_revert.append(len(sol_uzi.t))
  
    DOMAIN_W = X.max() - X.min()
    DOMAIN_H = Y.max() - Y.min()
    CANVAS_PHYSICAL_H = DOMAIN_H*param["zoom_in_fac"]
    CANVAS_PHYSICAL_W = CANVAS_PHYSICAL_H * (IMG_W_SCENE / IMG_H_SCENE)
    
    canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
    canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)

    px0, py0 = cg.hex_center_pixel(param["start_row"], param["start_col"], detail_info)
    OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
    OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]

    rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}
    grid_hex_rc_i = cg.world_metres_to_hex_index(
            X + OFFSET_X, Y + OFFSET_Y, detail_info,
            canvas_physical_x_range=canvas_physical_x_range,
            canvas_physical_y_range=canvas_physical_y_range,
        )
    #print(grid_hex_rc_i)
    grid_hex_rc_i_hairuzi.append(grid_hex_rc_i)
max_snapshot_revert = max(snapshot_lengths_revert)
print(grid_hex_rc_i_hairuzi)

# ── PROJECT ONTO THE HEX GRID: physical grid -> pixel -> hex index
x_right, y_right = sol_eye.y[0], sol_eye.y[1]
#right uzi eye
DOMAIN_W = x_right.max() - x_right.min()
DOMAIN_H = y_right.max() - y_right.min()
print(DOMAIN_W, DOMAIN_H)
CANVAS_PHYSICAL_H = DOMAIN_H*0.65

CANVAS_PHYSICAL_W = CANVAS_PHYSICAL_H * (IMG_W_SCENE / IMG_H_SCENE)#DOMAIN_W*2# 
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)


START_HEX_ROW, START_HEX_COL = head_center_up[0], head_center_up[1]  # target hex cell for the jet's (x0, z0)
px0, py0 = cg.hex_center_pixel(START_HEX_ROW, START_HEX_COL, detail_info)
 
OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - x_right[0]
OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - y_right[0]

rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}
grid_hex_rc_i_right = cg.world_metres_to_hex_index(
        x_right + OFFSET_X, y_right + OFFSET_Y, detail_info,
        canvas_physical_x_range=canvas_physical_x_range,
        canvas_physical_y_range=canvas_physical_y_range,
    )

# ── UPDATE ──────────────────────────────────────────────────────────────
Phase0_end = 30           # intro frames that just hold the base scene
total_seconds = 18.0
sec = 3
sec2 = 4

left_max_snapshots = max_snapshot+max(snapshot_lengths_revert[:sec]
                                           )+max(snapshot_lengths_revert[sec:sec2])+max(snapshot_lengths_revert[sec2:])
right_max_snapshots = len(sol_eye.t)
total_frames = Phase0_end + max(left_max_snapshots, right_max_snapshots)

visited_idx_right = []
right_traj_colors = cgc.gradient_sequence_colors(right_max_snapshots, [0,0,0], [1,0,0], mode='linear')  # or 'sigmoid'
visited_color_right = []

visited_idx_hairs = [[] for _ in grid_hex_rc_i_hairs] 
hairs_traj_colors = cgc.gradient_sequence_colors(max_snapshot, [0,0,0], cgc.hex_to_rgb("#2b2a76"), mode='linear')  # or 'sigmoid'
visited_color_hairs = [[] for _ in grid_hex_rc_i_hairs] 

visited_idx_revert_list_part1 = [[] for _ in grid_hex_rc_i_hairuzi[:sec]] 
part1_colors = cgc.gradient_sequence_colors(max(snapshot_lengths_revert[:sec]), [0,0,0], cgc.hex_to_rgb("#28259d"), mode='linear')  # or 
visited_colors_part1 = [[] for _ in grid_hex_rc_i_hairuzi[:sec]] 



visited_idx_revert_list_part2 = [[] for _ in grid_hex_rc_i_hairuzi[sec:sec2]] 
part2_colors = cgc.gradient_sequence_colors(max(snapshot_lengths_revert[sec:sec2]), [0,0,0], cgc.hex_to_rgb("#0d2ed8"), mode='linear')  # or 
visited_colors_part2 = [[] for _ in grid_hex_rc_i_hairuzi[sec:sec2]] 


visited_idx_revert_list_part3 = [[] for _ in grid_hex_rc_i_hairuzi[sec2:]] 
part3_colors = cgc.gradient_sequence_colors(max(snapshot_lengths_revert[sec2:]), [0,0,0], cgc.hex_to_rgb("#3858ff"), mode='linear')  # or 
visited_colors_part3 = [[] for _ in grid_hex_rc_i_hairuzi[sec2:]] 


def update(snapshot):
    if snapshot < Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(base_hex_colors)
    else:
        current_snapshot = snapshot - Phase0_end
        current_hex_colors = animation_base_hex_colors.copy()
        
        
        if current_snapshot<=max_snapshot:
            
            for i, grid_hex_rc_i in enumerate(grid_hex_rc_i_hairs):
                if current_snapshot < len(grid_hex_rc_i):   # this trajectory hasn't ended yet
                    rc = grid_hex_rc_i[current_snapshot]
                    idx = rc_to_idx.get(rc)
                    if idx is not None:                      # skip points outside the hex grid
                        visited_idx_hairs[i].append(idx)
                        visited_color_hairs[i].append(hairs_traj_colors[current_snapshot])
                        
        for visited_idx_hair, visited_color_hair in zip(visited_idx_hairs, visited_color_hairs):
            if visited_idx_hair:
                current_hex_colors[visited_idx_hair] = np.array([
                    cgc.select_normal_color([True], c, np.ones(3) * sigma_color)[0]
                    for c in visited_color_hair
                ]) 
                #current_hex_colors[visited_idx] = [1,0,0]

        pc.set_facecolor(current_hex_colors)
        
        hair_uzi_start_part1 = max_snapshot//3
        if current_snapshot >hair_uzi_start_part1:
            revert_snapshot = current_snapshot - hair_uzi_start_part1
            
            for i, grid_hex_rc_i in enumerate(grid_hex_rc_i_hairuzi[:sec]):
                if revert_snapshot < len(grid_hex_rc_i):   # this trajectory hasn't ended yet
                    rc = grid_hex_rc_i[revert_snapshot]
                    idx = rc_to_idx.get(rc)
                    if idx is not None:                      # skip points outside the hex grid
                        visited_idx_revert_list_part1[i].append(idx)
                        visited_colors_part1[i].append(part1_colors[revert_snapshot])
                        
            for visited_idx_part1, visited_color_part1 in zip(visited_idx_revert_list_part1, visited_colors_part1):
                if visited_idx_part1:
                    current_hex_colors[visited_idx_part1] = np.array([
                        cgc.select_normal_color([True], c, np.ones(3) * sigma_color)[0]
                        for c in visited_color_part1
                    ]) 
            pc.set_facecolor(current_hex_colors)
        
        hair_uzi_start_part2 = max_snapshot
        if current_snapshot>hair_uzi_start_part2:
            revert_snapshot = current_snapshot - hair_uzi_start_part2
            
            for i, grid_hex_rc_i in enumerate(grid_hex_rc_i_hairuzi[sec:sec2]):
                if revert_snapshot < len(grid_hex_rc_i):   # this trajectory hasn't ended yet
                    rc = grid_hex_rc_i[revert_snapshot]
                    idx = rc_to_idx.get(rc)
                    if idx is not None:                      # skip points outside the hex grid
                        visited_idx_revert_list_part2[i].append(idx)
                        visited_colors_part2[i].append(part2_colors[revert_snapshot])
                        
            for visited_idx_part2, visited_color_part2 in zip(visited_idx_revert_list_part2, visited_colors_part2):
                if visited_idx_part2:
                    current_hex_colors[visited_idx_part2] = np.array([
                        cgc.select_normal_color([True], c, np.ones(3) * sigma_color)[0]
                        for c in visited_color_part2
                    ]) 
            pc.set_facecolor(current_hex_colors)
       
        hair_uzi_start_part3 = hair_uzi_start_part2 + max(snapshot_lengths_revert[sec:sec2])
        if current_snapshot>hair_uzi_start_part3:
            revert_snapshot = current_snapshot - hair_uzi_start_part3
            
            for i, grid_hex_rc_i in enumerate(grid_hex_rc_i_hairuzi[sec2:]):
                if revert_snapshot < len(grid_hex_rc_i):   # this trajectory hasn't ended yet
                    rc = grid_hex_rc_i[revert_snapshot]
                    idx = rc_to_idx.get(rc)
                    if idx is not None:                      # skip points outside the hex grid
                        visited_idx_revert_list_part3[i].append(idx)
                        visited_colors_part3[i].append(part3_colors[revert_snapshot])
                        
            for visited_idx_part3, visited_color_part3 in zip(visited_idx_revert_list_part3, visited_colors_part3):
                if visited_idx_part3:
                    current_hex_colors[visited_idx_part3] = np.array([
                        cgc.select_normal_color([True], c, np.ones(3) * sigma_color)[0]
                        for c in visited_color_part3
                    ]) 
            pc.set_facecolor(current_hex_colors)
            
        #right part
        if current_snapshot<=right_max_snapshots:
            rc = grid_hex_rc_i_right[current_snapshot]
            idx = rc_to_idx.get(rc)
            if idx is not None:              
                visited_idx_right.append(idx)
                visited_color_right.append(right_traj_colors[current_snapshot])
        if visited_idx_right:
            current_hex_colors[visited_idx_right] = np.array([
                cgc.select_normal_color([True], c, np.ones(3) * sigma_color)[0]
                for c in visited_color_right
            ])
        pc.set_facecolor(current_hex_colors)
        
    return (pc,)
total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
 
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'Uzi_animation.mp4')
 
print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")
 