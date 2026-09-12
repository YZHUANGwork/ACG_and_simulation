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
from types import SimpleNamespace


import astropy.units as u
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import expected_value as EXP
import cg_draw_figure as cgf
rng = np.random.default_rng(12)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,_ = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)

sigma_color = 0.03

#
n_head = 3
head_center_row = 5
head_center_col = 18


target_STRIPE_PALETTE = [np.array([0,0,0]),cgc.hex_to_rgb("#bb0000"),]
stripe_target_colors = cgc.color_stripe(hex_rc_arr, 
                                        STRIPE_WIDTH = 1 ,
                                        STRIPE_PALETTE = target_STRIPE_PALETTE)#cgc.hex_to_rgb("#bb0000")
stripe_skin_colors = cgc.color_stripe(hex_rc_arr, 
                                        STRIPE_WIDTH = 1 ,
                                        STRIPE_PALETTE = [cgc.hex_to_rgb("#fee9d2"),cgc.hex_to_rgb("#bb0000"), ])
stripe_bkgd_colors = cgc.color_stripe(hex_rc_arr, 
                                        STRIPE_WIDTH = 3 ,
                                        STRIPE_PALETTE = [[0,0,0],cgc.hex_to_rgb("#ebeaff"), ])##ffed00#ffc700#240047

skyline = head_center_row+n_head
select_upper_bkgd = [r <skyline for r,c in hex_rc_arr]
select_bottom_bkgd = [r >=skyline for r,c in hex_rc_arr]
hex_colors = cgc.color_row_gradient(hex_colors[select_upper_bkgd], 
                                        cgc.hex_to_rgb("#d6006d"),cgc.hex_to_rgb("#ffe8f4"),
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*5, 
                                                end_weight= 0.01, mode = 'linear') 
hex_colors= cgc.select_normal_color([True]*len(hex_colors), cgc.hex_to_rgb("#ebeaff"), np.ones(3)*sigma_color*3)  #stripe_bkgd_colors

def draw_Sakura(head_center_row, head_center_col, n_head):
    
    head_raw, raw_body,  line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict,upperarm_dict, forearm_dict, hand_dict,  plot_temp = cgf.draw_human_body(head_center_row, head_center_col, n_head)

    
    ln = SimpleNamespace(**line_dict)
    btlj = SimpleNamespace(**botlimb_joint_dict)
    uplj = SimpleNamespace(**upperlimb_joint_dict)
    tr = SimpleNamespace(**torso_dict) 
    
    skin = [(r,c) for r,c in raw_body if r <= ln.jaw_end_r]
    stripe_skin = thigh_dict['thigh']+calf_dict['calf']+feet_dict['feet']+btlj.full_joints+forearm_dict["forearm"]+[(r,c) for r,c in skin if c >head_center_col and r >= ln.jaw_end_r-1]+hand_dict["hand"]
    stripe_clothing = [(r,c) for r,c in raw_body if (r,c) not in stripe_skin]
    
    left_arm_start_r = min(r for r,c in forearm_dict["left_forearm"])-1
    right_arm_start_r = min(r for r,c in forearm_dict["right_forearm"])-1
    
    left_pelvis_start_r = min(r for r,c in thigh_dict['left_thigh'])-2
    right_pelvis_start_r= min(r for r,c in thigh_dict['right_thigh'])-2
    
    left_leg_start_r  = min(r for r,c in btlj.full_joints)-1
    right_leg_start_r  = min(r for r,c in btlj.full_joints)-1
    
    
    stripe_start_pts = {"LEFT1":(left_arm_start_r, min(c for r,c in stripe_clothing if r == left_arm_start_r)),
                        
                        "RIGHT1":(right_arm_start_r, max(c for r,c in stripe_clothing if r == right_arm_start_r)),
                        
                        "LEFT2":(left_arm_start_r, max(c for r,c in uplj.left_elbow_joint if r == left_arm_start_r)),
                        
                        "RIGHT2":(right_arm_start_r, min(c for r,c in uplj.right_elbow_joint if r == right_arm_start_r)),
                        
                      
                        "LEFT3":(left_leg_start_r, min(c for r,c in stripe_clothing if r == left_leg_start_r)),
                       
                        "RIGHT3":(right_leg_start_r, max(c for r,c in stripe_clothing if r == right_leg_start_r)),
                        
                        
                        "LEFT4":(left_pelvis_start_r, min(c for r,c in stripe_clothing if r == left_pelvis_start_r)),
                      
                        "RIGHT4":(right_pelvis_start_r, max(c for r,c in stripe_clothing if r ==right_pelvis_start_r)),
                        
                        "LEFT5":(max(r for r,c in stripe_clothing), min(c for r,c in stripe_clothing 
                                                                        if r == max(r for r,c in stripe_clothing))),
                       
                        "RIGHT5":(max(r for r,c in stripe_clothing), max(c for r,c in stripe_clothing 
                                                                         if r == max(r for r,c in stripe_clothing))),
                        
                      
                        
                       
                       }
    hair_part1 = [(r,c) for r,c in cg.hex_neighbours_n(head_center_row,head_center_col, n=n_head, keep_origin = True) 
                  if r < head_center_row]
    hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                   min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                   ln.shoulder_joint_r, bend = 'right')
    hair_back = cgd.draw_block((max(r for r,c in hair_part1), 
                                (min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)), 
                                 max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)))
                               ), ln.breastbone_end_r)
    hair_part3 = [(r,c) for r,c in hair_back if (r,c) not in raw_body]
    hair_part4 = cgd.horizontal_lines([(head_center_row, (head_center_col, head_center_col+n_head-2))])
    hair =hair_part1 +hair_part2+hair_part3+hair_part4
    
    ribbon = cgd.draw_triangle(head_center_row, max(c for r,c in raw_body if r==head_center_row)+1, 
                       head_center_row+2, 
                               slope_left = 'inf', slope_right = '0.5', direction = 'rr',
                  bend_left = 'right', bend_right = 'right')[-2]
    
    
  
    keys = [ "raw_body",  "skin", "stripe_skin", "hair", "ribbon"]
    colors = [[[0,0,0]], 
              [cgc.hex_to_rgb("#fee9d2")],
              [cgc.hex_to_rgb("#fee9d2")],
              [cgc.hex_to_rgb("#bab4dc"), cgc.hex_to_rgb("#6500c8")],
              [cgc.hex_to_rgb("#eb009b")],
             ]
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}, stripe_start_pts


dict_, stripe_start_pts = draw_Sakura(head_center_row, head_center_col, n_head)


for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            if key == 'raw_body':
                
                hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color*2)[0]
                                                    for c in stripe_target_colors[select_part] 
                                                   ])
 
            elif key == 'stripe_skin':
                hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color)[0]
                                                    for c in stripe_skin_colors[select_part]  
                                                   ])
                
                
            else:
                hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, 
                                                end_weight= 0.01, mode = 'linear') 
            
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'HF_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')