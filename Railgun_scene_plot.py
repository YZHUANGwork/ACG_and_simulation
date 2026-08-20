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



sigma_color = 0.02

skyline = 20

Handrail_row = 18
ground_center_col = 20

ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
sky = [(r,c) for r,c in hex_rc_arr if r <skyline]
hex_colors = cgc.color_row_gradient(sky, cgc.hex_to_rgb("#6100ff"),cgc.hex_to_rgb("#13149b"),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.1)   
hex_colors = cgc.color_row_gradient(ground, cgc.hex_to_rgb("#8f809c"),cgc.hex_to_rgb("#a5a5a5"),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.1)   


buildings = cgd.draw_block((15, (0, 8)), skyline
                          )+cgd.draw_block((17, (8, 10)), skyline
                          )+cgd.draw_block((14, (10, 15)), skyline
                          )+cgd.draw_block((12, (15, 17)), skyline
                          )+cgd.draw_block((16, (17, 21)), skyline
                          )+cgd.draw_block((10, (21, 24)), skyline
                          )+cgd.draw_block((12, (24, 27)), skyline
                          )+cgd.draw_block((15, (27, 32)), skyline
                          )
hex_colors =cgc.color_row_gradient(buildings, cgc.hex_to_rgb("#fffc00"), cgc.hex_to_rgb("#171f68"),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05)    


Handrail = cgd.horizontal_lines([(Handrail_row, (0, 34)), 
                                 (Handrail_row+1, (0, 34))])+cgd.draw_block((Handrail_row, (0, 2)), 22
                                                             )+cgd.draw_block((Handrail_row, (4, 6)), 22
                                                             )+cgd.draw_block((Handrail_row, (8, 10)), 22
                                                             )+cgd.draw_block((Handrail_row, (12, 14)), 22
                                                             )+cgd.draw_block((Handrail_row, (16, 18)), 22
                                                             )+cgd.draw_block((Handrail_row, (20, 22)), 22
                                                             )+cgd.draw_block((Handrail_row, (24, 26)), 22
                                                             )+cgd.draw_block((Handrail_row, (28, 30)), 22
                                                             )+cgd.draw_block((Handrail_row, (32, 34)), 22
                                                             )
hex_colors =cgc.color_row_gradient(Handrail, cgc.hex_to_rgb("#fffd59"), cgc.hex_to_rgb("#060606"),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05)    

def draw_shoulder(body,body_center_row, n):
    
    body_top_r = min(r for r, c in body)
    body_bottom_r = max(r for r, c in body)
    
    shoulder_extensions = math.ceil(n/2)
    
    
    left_arm_tot = []
    right_arm_tot = []
    left_shoulders, right_shoulders = [], []
    for shoulder_extension in range(shoulder_extensions):
        left_shoulders+=cgd.draw_slope_0p5_diagonal(body_top_r, min(c for r, c in body if r ==body_top_r)-shoulder_extension-1, 
                                             body_center_row-shoulder_extension, 
                                            left_down=True, right_down=False, left_up=False, right_up=False)
        right_shoulders+=cgd.draw_slope_0p5_diagonal(body_top_r, max(c for r, c in body if r ==body_top_r)+shoulder_extension+1, 
                                              body_center_row-shoulder_extension, 
                                             left_down=False, right_down=True , left_up= False, right_up=False )
    
    left_elbow_r = body_bottom_r+1
    left_brachium = cgd.draw_trapezoid(body_center_row, 
                                            min(c for r,c in left_shoulders if r == body_center_row), 
                                            min(c for r,c in body if r == body_center_row), 
                                            body_bottom_r+1, slope_left = 'inf', slope_right = 'inf', direction = 'lr',
                  bend_left = 'left', bend_right = 'left')[-2]
    
    right_elbow_r = body_center_row-n-1
    right_brachium = cgd.draw_trapezoid(body_center_row, 
                                            max(c for r,c in body if r == body_center_row), 
                                            max(c for r,c in right_shoulders if r == body_center_row), 
                                            right_elbow_r, slope_left = 'inf', slope_right = 'inf', direction = 'lr',
                  bend_left = 'right', bend_right = 'right')[-2]
    
    left_forearm = []
    right_wrist_r = right_elbow_r-n
    right_forearm = cgd.draw_trapezoid(right_elbow_r, 
                                            min(c for r,c in right_brachium if r == right_elbow_r), 
                                            max(c for r,c in right_brachium if r == right_elbow_r), 
                                            right_wrist_r, slope_left = '0.5', slope_right = 'inf', direction = 'rl',
                  bend_left = 'right', bend_right = 'right')[-2]
    
    left_hand = []
    right_hand = cg.hex_neighbours_n(right_wrist_r-2, min(c for r,c in right_forearm if r == right_wrist_r), n=1, keep_origin = True, return_frontier=False)
    
   
    left_arm_tot+=left_shoulders
    right_arm_tot+=right_shoulders
  

    return left_shoulders, right_shoulders, left_brachium, right_brachium, left_forearm, right_forearm, left_hand, right_hand
    

def draw_char(center_row, center_col, n_head):
    head, neck, body, pelvis, thigh, raw_body_detail= cgd.draw_body(center_row, center_col, n = n_head)
    neck_r = max(r for r,c in neck)
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail
    
    remove_hair = [
                  (head_center_row, max(c for r,c in head if r == head_center_row))
                  ]
    add_hair = cgd.verticle_line(head_center_row, min(c for r,c in head if r == head_center_row), neck_r+1, bend = 'right'
                                )+cgd.verticle_line(head_center_row+n_head, max(c for r,c in head if r == head_center_row+n_head)-1, 
                                                    neck_r+1, bend = 'left'
                                )+cgd.verticle_line(head_center_row+n_head, min(c for r,c in head if r == head_center_row+n_head), 
                                                    neck_r+1, bend = 'left'
                                )
    
    
    hair  = [(r,c) for r,c in head if (r,c) not in remove_hair]+add_hair
    face = cgd.verticle_line(head_center_row, max(c for r,c in hair if r == head_center_row), head_center_row+n_head, bend = 'left'
                                )+cgd.verticle_line(head_center_row+1, max(c for r,c in hair if r == head_center_row+1), head_center_row+n_head, bend = 'left'
                                )
    valid_face = [(r,c) for r,c in face if (r,c ) in head]
    left_shoulder, right_shoulder, left_brachium, right_brachium, left_forearm, right_forearm, left_hand, right_hand = draw_shoulder(body, body_center_row, n_head)
    
    
    add_sleeves = [(r,c) for r,c in right_brachium if r >=neck_r]
    sleeves = left_shoulder+right_shoulder+[(body_center_row, min(c for r,c in body if r == body_center_row)),
                               (body_center_row, max(c for r,c in body if r == body_center_row))
                              ]+add_sleeves+cgd.verticle_line(body_center_row-n_head, 
                                                              min(c for r,c in right_shoulder if r == body_center_row-n_head), 
                                                              body_center_row-1, bend = 'left')
    remove_body = [(body_center_row+1, max(c for r,c in body if r == body_center_row+1))]
    vest =[(r,c) for r,c in body if (r,c) not in remove_body]
    collar = neck
    arm = left_brachium+right_brachium+left_forearm+right_forearm

    add_hand = cgd.verticle_line(min(r for r,c in right_hand), 
                                 min(c for r,c in right_hand if r == min(r for r,c in right_hand)
                                    ), 
                                 min(r for r,c in right_hand)-1, bend = 'left'
                                )
   
    hand = right_hand+add_hand
    
    return {'b': [arm, [cgc.hex_to_rgb("#fee9d2")]],
            'c': [hand, [cgc.hex_to_rgb("#fee9d2")]],
            'f': [vest, [cgc.hex_to_rgb("#d1bb9a")]],
            'd': [sleeves, [[1,1,1]]],
            'e': [collar, [[1,1,1]]],
            'g': [hair, [cgc.hex_to_rgb("#803c11"), cgc.hex_to_rgb("#f1b995")]],
            'h': [valid_face, [cgc.hex_to_rgb("#fee9d2")]],
            }


n_head = 3
center_row = 12
center_col = 8



  
char_dict_ = {'Misaka': draw_char(center_row, center_col,n_head),
            }
for dict_ in char_dict_.values(): 
    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01, period=1) 

pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Railgun_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')