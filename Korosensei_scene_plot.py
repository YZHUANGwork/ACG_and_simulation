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
import SOLUTION_Schrodinger as SOLN

import astropy.constants as const
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
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=12, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info

sigma_color = 0.03
    
#
n_head = 3
head_center_row = 10
head_center_col = 18


skyline = max(r for r,c in hex_rc_arr)//2
ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
wall = [(r,c) for r,c in hex_rc_arr if r <skyline]
BLACKBOARD = cgd.draw_block((head_center_row-8,  (min(c for r,c in hex_rc_arr)+3,max(c for r,c in hex_rc_arr)-3 
                                                 )
                            ), head_center_row+20)

hex_colors = cgc.color_row_gradient(wall, cgc.hex_to_rgb("#f4f4f4"), cgc.hex_to_rgb("#d6d6d6"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05, mode = 'linear')   
hex_colors = cgc.color_row_gradient(BLACKBOARD, cgc.hex_to_rgb("#262b28"), cgc.hex_to_rgb("#444444"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.05, mode = 'linear')   



hex_colors =cgc.color_row_gradient(ground, cgc.hex_to_rgb("#d6d6d6"), cgc.hex_to_rgb("##343b37"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05, mode = 'linear')       

    

def draw_heads(head_center_row, head_center_col, n_head, mood):
    head_raw = cg.hex_neighbours_n(head_center_row, head_center_col, n=n_head, keep_origin = True, return_frontier=False)
    
    face = cg.hex_neighbours_n(head_center_row, head_center_col, n=n_head-1, keep_origin = False, return_frontier=True)[-1]
    mouth = [(r,c) for r,c in face if r >head_center_row]
    eyes = [(head_center_row-1, min(c for r,c in head_raw if r == head_center_row-1)+1),
            (head_center_row-1, max(c for r,c in head_raw if r == head_center_row-1)-1)
           ]
    hat = cgd.draw_block((head_center_row-n_head, 
                          (min(c for r,c in head_raw if r == head_center_row-n_head),
                          max(c for r,c in head_raw if r == head_center_row-n_head)) ), head_center_row-n_head-2)
    valid_hat = [(r,c) for r,c in hat if (r,c) not in head_raw]
    
    if mood == '1':
        rs = np.arange(head_center_row-n_head,head_center_row+n_head+1,1)
        
        head_odd = []
        head_even = []
        for row in rs:
            if row %2==0:
                head_even+=[(r,c) for r,c in head_raw if r == row]
            else:
                
                head_odd+=[(r,c) for r,c in head_raw if r == row]
        keys = ["head_even", "head_odd", "mouth", "eyes", "valid_hat"]
        colors = [[cgc.hex_to_rgb("#58ad16")], [cgc.hex_to_rgb("#ffd647")],[[1,1,1]], [[1,1,1]], [[0,0,0]]
                 ]
        
    elif mood == '2':
        circle = face
        keys = ["head_raw", "mouth",  "circle", "eyes", "valid_hat"]
        colors = [[cgc.hex_to_rgb("#ff7a00")],[[1,1,1]],[cgc.hex_to_rgb("#df1b00")], [[1,1,1]], [[0,0,0]]
                 ]
        
    elif mood == '3':
        cross = cgd.draw_slope_0p5_diagonal(head_center_row, head_center_col, head_center_row-n_head+1, 
                                            left_down=False, right_down=False, left_up=True, right_up=True
                                           )+cgd.draw_slope_0p5_diagonal(head_center_row, head_center_col, head_center_row+n_head-1, 
                                            left_down=True, right_down=True, left_up=False, right_up=False
                                           )
        keys = ["head_raw", "mouth","cross", "eyes", "valid_hat"]
        colors = [[cgc.hex_to_rgb("#885ca2")],[[1,1,1]],[cgc.hex_to_rgb("#511360")],[[1,1,1]], [[0,0,0]]
                 ]
    elif mood == '4':
        cheek = []
        for eye in eyes:
            cheek+=cgd.draw_triangle(eye[0], eye[1],  eye[0]+1, slope_left = '0.5', slope_right = '0.5', direction = 'lr', 
                 bend_left = 'left', bend_right = 'right')[-2]
            valid_cheek = [(r,c) for r,c in cheek if (r,c) not in eyes]
        keys = ["head_raw", "cheek", "mouth", "eyes", "valid_hat"]
        colors = [[cgc.hex_to_rgb("#ffafd8")], [cgc.hex_to_rgb("#ff3596")],[[1,1,1]], [[1,1,1]], [[0,0,0]]
                 ]
    elif mood == '5':
        lip = cg.hex_neighbours_n(head_center_row+2, head_center_col, n=n_head-2, keep_origin = True, return_frontier=False)
        valid_lip = [(r,c) for r,c in lip if (r,c) not in mouth]
        keys = ["head_raw", "valid_lip", "mouth", "eyes", "valid_hat"]
        colors = [[cgc.hex_to_rgb("#c6c6c6")], [[1,0,0]],[[1,1,1]], [[1,1,1]], [[0,0,0]]
                 ]
        
    local_vars = locals()
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}


def draw_Korosensei(head_center_row, head_center_col, n_head, view = 'front'):
    head_raw, raw_body,  line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict,upperarm_dict, forearm_dict, hand_dict,  plot_temp = cgf.draw_human_body(head_center_row, head_center_col, n_head)

    
    ln = SimpleNamespace(**line_dict)
    tr = SimpleNamespace(**torso_dict)
    
    
    head_size = n_head*2
    head = cg.hex_neighbours_n(head_center_row, head_center_col, n=n_head, keep_origin = True, return_frontier=False)
    face = cg.hex_neighbours_n(head_center_row, head_center_col, n=n_head-1, keep_origin = False, return_frontier=True)[-1]
    #mouth = [(r,c) for r,c in face if r >head_center_row]
    #eyes = [(head_center_row-1, min(c for r,c in head if r == head_center_row-1)+1),
    #        (head_center_row-1, max(c for r,c in head if r == head_center_row-1)-1)
    #       ]
    mouth_raw = [(r,c) for r,c in face if r >head_center_row]
    eyes_raw = [(head_center_row-1, min(c for r,c in head if r == head_center_row-1)+1),
            (head_center_row-1, max(c for r,c in head if r == head_center_row-1)-1)
           ]
    if view == 'side':
        
        mouth = [(r_, c_) for r_, c_ in [(r, c+n_head-1) for r,c in mouth_raw]  if (r_, c_) in head]
        eyes =  [(r_, c_) for r_, c_ in [(r, c+n_head-1) for r,c in eyes_raw]  if (r_, c_) in head]
    else:
        mouth = mouth_raw
        eyes = eyes_raw
        
    hat = cgd.draw_block((head_center_row-n_head, 
                          (min(c for r,c in head if r == head_center_row-n_head),
                          max(c for r,c in head if r == head_center_row-n_head)) ), head_center_row-n_head-2)
    valid_hat = [(r,c) for r,c in hat if (r,c) not in head]
    collar = cgd.draw_trapezoid(head_center_row+n_head, 
                              min(c for r,c in head if r == head_center_row-n_head),#-1, 
                              max(c for r,c in head if r == head_center_row-n_head),#+1, 
                              head_center_row+n_head+n_head,#+n_head-1, 
                              slope_left = '0.5', slope_right = '0.5', direction = 'rl', bend_left = 'left', bend_right = 'right')[-2]
    valid_collar = [(r,c) for r,c in collar if r >head_center_row+n_head]
    
    tie_width = (min(c for r,c in valid_collar if r == head_center_row+n_head+1
                    )+max(c for r,c in valid_collar if r == head_center_row+n_head+1
                         ))/2
    tie_part1 = cgd.draw_block((head_center_row+n_head+1, (math.floor(tie_width), math.ceil(tie_width))),
                          head_center_row+n_head+1+2
                               )
    tie_part2 = cgd.draw_trapezoid(max(r for r,c in tie_part1), 
                              min(c for r,c in tie_part1 if r == max(r for r,c in tie_part1)), 
                              max(c for r,c in tie_part1 if r == max(r for r,c in tie_part1)), 
                              max(r for r,c in tie_part1)+2, 
                              slope_left = '0.5', slope_right = '0.5', direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
    tie_part3 = cgd.draw_trapezoid(max(r for r,c in tie_part2), 
                              min(c for r,c in tie_part2 if r == max(r for r,c in tie_part2)), 
                              max(c for r,c in tie_part2 if r == max(r for r,c in tie_part2)), 
                              max(r for r,c in tie_part2)+2, 
                              slope_left = '0.5', slope_right = '0.5', direction = 'rl', bend_left = 'left', bend_right = 'right')[-2]
                              
    tie = tie_part1+tie_part2+tie_part3
    tie_acc = [(max(r for r,c in tie_part2), (min(c for r,c in tie_part2 if r == max(r for r,c in tie_part2))+ 
                              max(c for r,c in tie_part2 if r == max(r for r,c in tie_part2)))//2)]
    robe_part1 = cgd.draw_trapezoid(max(r for r,c in tr.neck2_shoulder), 
                              min(c for r,c in tr.neck2_shoulder if r == max(r for r,c in tr.neck2_shoulder)), 
                              max(c for r,c in tr.neck2_shoulder if r == max(r for r,c in tr.neck2_shoulder)), 
                              ln.gastrocnemius_end_r, 
                              slope_left = 'inf', slope_right = 'inf', direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
    robe_part2_left = cgd.draw_trapezoid(max(r for r,c in tr.neck2_shoulder), 
                              min(c for r,c in tr.neck2_shoulder if r == max(r for r,c in tr.neck2_shoulder))-n_head, 
                              min(c for r,c in tr.neck2_shoulder if r == max(r for r,c in tr.neck2_shoulder)), 
                              ln.bellybutton_r, 
                              slope_left = '0.5', slope_right = 'inf', direction = 'lr', bend_left = 'left', bend_right = 'right'
                                   )[-2]
    robe_part2_right = cgd.draw_trapezoid(max(r for r,c in tr.neck2_shoulder), 
                              max(c for r,c in tr.neck2_shoulder if r == max(r for r,c in tr.neck2_shoulder)), 
                              max(c for r,c in tr.neck2_shoulder if r == max(r for r,c in tr.neck2_shoulder))+n_head, 
                              ln.bellybutton_r, 
                              slope_left = 'inf', slope_right = '0.5', direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
    
    robe_part2 = robe_part2_left+robe_part2_right
    robe_part2p5_left = cgd.draw_trapezoid(max(r for r,c in robe_part2_left), 
                              min(c for r,c in robe_part2_left if r == max(r for r,c in robe_part2_left)), 
                              max(c for r,c in robe_part2_left if r == max(r for r,c in robe_part2_left)), 
                              max(r for r,c in robe_part2_left)+n_head, 
                              slope_left = '1.5', slope_right = 'inf', direction = 'rl', bend_left = 'left', bend_right = 'right'
                                   )[-2]
    
    robe_part2p5_right = cgd.draw_trapezoid(max(r for r,c in robe_part2_right), 
                              min(c for r,c in robe_part2_right if r == max(r for r,c in robe_part2_right)), 
                              max(c for r,c in robe_part2_right if r == max(r for r,c in robe_part2_right)), 
                              max(r for r,c in robe_part2_right)+n_head,
                              slope_left = 'inf', slope_right = '1.5', direction = 'rl', bend_left = 'left', bend_right = 'right')[-2]
    robe_part2+=robe_part2p5_left+robe_part2p5_right
    
    left_tentacle_r = (min(r for r,c in robe_part2p5_left)+max(r for r,c in robe_part2p5_left)
                      )//2
    right_tentacle_r = (min(r for r,c in robe_part2p5_right)+max(r for r,c in robe_part2p5_right)
                       )//2
    
    tentacle_start_pts = {"LEFT1":(left_tentacle_r, min(c for r,c in robe_part2p5_left if r == left_tentacle_r)),
                        
                        "RIGHT1":(right_tentacle_r, max(c for r,c in robe_part2p5_right if r == right_tentacle_r))
                         }
    
                        
    yellow_stripes_left, yellow_stripes_right = [],[]
    yellow_stripes_starts_r = [min(r for r,c in robe_part2_left),min(r for r,c in robe_part2_left)+1]
    for yellow_stripes_r in yellow_stripes_starts_r:
        yellow_stripes_left += cgd.draw_slope_1p5_diagonal(yellow_stripes_r, 
                                      min(c for r,c in robe_part2 if r == yellow_stripes_r), yellow_stripes_r+3, 
                                      left_down=False, right_down=True, left_up=False, right_up=False
                                      )+cgd.draw_slope_1p5_diagonal(yellow_stripes_r, 
                                      min(c for r,c in robe_part2 if r == yellow_stripes_r)-1, yellow_stripes_r+3, 
                                      left_down=False, right_down=True, left_up=False, right_up=False
                                      )
        yellow_stripes_right += cgd.draw_slope_1p5_diagonal(yellow_stripes_r, 
                                      max(c for r,c in robe_part2 if r == yellow_stripes_r), yellow_stripes_r+3, 
                                      left_down=True, right_down=False, left_up=False, right_up=False
                                      )+cgd.draw_slope_1p5_diagonal(yellow_stripes_r, 
                                      max(c for r,c in robe_part2 if r == yellow_stripes_r)+1, yellow_stripes_r+3, 
                                      left_down=True, right_down=False, left_up=False, right_up=False
                                      )
    yellow_stripes = yellow_stripes_left+yellow_stripes_right
    
    
        
    robe_part2+=cgd.draw_trapezoid(min(r for r,c in robe_part1), 
                              min(c for r,c in robe_part2_left if r ==  min(r for r,c in robe_part1)), 
                              max(c for r,c in robe_part2_right if r == min(r for r,c in robe_part1)), 
                              head_center_row+n_head-1,
                              slope_left = '1.5', slope_right = '1.5', direction = 'rl', bend_left = 'left', bend_right = 'right')[-2]
    
    blue_stripes_left, blue_stripes_right = [],[]
    blue_stripes_starts_r = [max(r for r,c in robe_part2_left)-2,
                                 max(r for r,c in robe_part2_left)-4,
                                 max(r for r,c in robe_part2_left)-6]
    
    for blue_stripes_r in blue_stripes_starts_r:
        blue_stripes_left += cgd.draw_slope_1p5_diagonal(blue_stripes_r, 
                                      min(c for r,c in robe_part2 if r == blue_stripes_r), blue_stripes_r+2, 
                                      left_down=False, right_down=True, left_up=False, right_up=False
                                      )+cgd.draw_slope_1p5_diagonal(blue_stripes_r, 
                                      min(c for r,c in robe_part2 if r == blue_stripes_r)+1, blue_stripes_r+2, 
                                      left_down=False, right_down=True, left_up=False, right_up=False
                                      )
        blue_stripes_right += cgd.draw_slope_1p5_diagonal(blue_stripes_r, 
                                      max(c for r,c in robe_part2 if r == blue_stripes_r), blue_stripes_r+2, 
                                      left_down=True, right_down=False, left_up=False, right_up=False
                                      )+cgd.draw_slope_1p5_diagonal(blue_stripes_r, 
                                      max(c for r,c in robe_part2 if r == blue_stripes_r)-1, blue_stripes_r+2, 
                                      left_down=True, right_down=False, left_up=False, right_up=False
                                      )
        
    blue_stripes = blue_stripes_left+blue_stripes_right
    
    
    
    robe_part3_left = cgd.draw_trapezoid(ln.bellybutton_r, 
                              min(c for r,c in robe_part1 if r == ln.bellybutton_r)-n_head, 
                              min(c for r,c in robe_part1 if r == ln.bellybutton_r), 
                              ln.knee_r, 
                              slope_left = '0.5', slope_right = 'inf', direction = 'lr', bend_left = 'left', bend_right = 'right'
                                   )[-2]
    robe_part3_right = cgd.draw_trapezoid(ln.bellybutton_r, 
                              max(c for r,c in robe_part1 if r == ln.bellybutton_r), 
                              max(c for r,c in robe_part1 if r == ln.bellybutton_r)+n_head, 
                              ln.knee_r, 
                              slope_left = 'inf', slope_right = '0.5', direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
    robe_part3 = robe_part3_left+robe_part3_right
    
    robe_part3p5_left = cgd.draw_trapezoid(max(r for r,c in robe_part3_left), 
                              min(c for r,c in robe_part3_left if r == max(r for r,c in robe_part3_left)), 
                              max(c for r,c in robe_part3_left if r == max(r for r,c in robe_part3_left)), 
                              ln.gastrocnemius_end_r, 
                              slope_left = '8/3', slope_right = 'inf', direction = 'rl', bend_left = 'left', bend_right = 'right'
                                   )[-2]
    
    robe_part3p5_right = cgd.draw_trapezoid(max(r for r,c in robe_part3_right), 
                              min(c for r,c in robe_part3_right if r == max(r for r,c in robe_part3_right)), 
                              max(c for r,c in robe_part3_right if r == max(r for r,c in robe_part3_right)), 
                              ln.gastrocnemius_end_r,
                              slope_left = 'inf', slope_right = '8/3', direction = 'rl', bend_left = 'left', bend_right = 'right')[-2]
    
    robe_part3+=robe_part3p5_left+robe_part3p5_right
    robe = tr.neck2_shoulder+robe_part2+robe_part3
    
    leg_part1 = cgd.draw_block((max(r for r,c in robe_part1), 
                          (min(c for r,c in robe_part1 if r == max(r for r,c in robe_part1)),
                           max(c for r,c in robe_part1 if r == max(r for r,c in robe_part1))
                          ) ), ln.gastrocnemius_end_r+2)
    leg_part2 = cgd.draw_trapezoid(max(r for r,c in leg_part1), 
                              min(c for r,c in leg_part1 if r == max(r for r,c in leg_part1))-n_head-n_head, 
                              min(c for r,c in leg_part1 if r == max(r for r,c in leg_part1))-n_head, 
                              min(r for r,c in robe_part3p5_left), 
                              slope_left = 'inf', slope_right = '0.5', direction = 'rr', bend_left = 'left', bend_right = 'right'
                                  )[-2]+cgd.draw_trapezoid(max(r for r,c in leg_part1), 
                              max(c for r,c in leg_part1 if r == max(r for r,c in leg_part1))+n_head, 
                              max(c for r,c in leg_part1 if r == max(r for r,c in leg_part1))+n_head+n_head, 
                              min(r for r,c in robe_part3p5_left), 
                              slope_left = '0.5', slope_right = 'inf', direction = 'll', bend_left = 'left', bend_right = 'right')[-2]
    leg_part3 = cgd.draw_trapezoid(max(r for r,c in leg_part1), 
                              min(c for r,c in leg_part2 if r == max(r for r,c in leg_part1))-n_head-n_head, 
                              min(c for r,c in leg_part2 if r == max(r for r,c in leg_part1))-n_head, 
                              min(r for r,c in robe_part3p5_left), 
                              slope_left = '0.5', slope_right = '0.5', direction = 'rr', bend_left = 'left', bend_right = 'right'
                                  )[-2]+cgd.draw_trapezoid(max(r for r,c in leg_part1), 
                              max(c for r,c in leg_part2 if r == max(r for r,c in leg_part1))+n_head, 
                              max(c for r,c in leg_part2 if r == max(r for r,c in leg_part1))+n_head+n_head, 
                              min(r for r,c in robe_part3p5_left), 
                              slope_left = '0.5', slope_right = '0.5', direction = 'll', bend_left = 'left', bend_right = 'right'
                                  )[-2]
    
    leg = leg_part1+leg_part2+leg_part3
    keys = ["leg", "robe", "blue_stripes", "robe_part1", "yellow_stripes", "valid_collar", "head", "mouth", "eyes", 
            "valid_hat", "tie", "tie_acc", ]
    colors = [[cgc.hex_to_rgb("#ffd647")], 
              [[0,0,0]],
        [cgc.hex_to_rgb("#22029d")], 
        [cgc.hex_to_rgb("#960000")],
        [cgc.hex_to_rgb("#ffb800")],
        [[1,1,1]],
        [cgc.hex_to_rgb("#ffd647")], [[1,1,1]], [[1,1,1]],
              [[0,0,0]], [[0,0,0]],[cgc.hex_to_rgb("#ffb800")],
             ]
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}, tentacle_start_pts


dict_, tentacle_start_pts = draw_Korosensei(head_center_row, head_center_col, n_head)


for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, 
                                                end_weight= 0.01, mode = 'linear') 
            
head_seg = n_head*3
for col_shift, mood in zip([-head_seg, head_seg, head_seg*2, head_seg*3, head_seg*4],['1', '2', '3', '4', '5'])  :
    head_dict_=draw_heads(head_center_row, head_center_col+col_shift, n_head, mood)
    for key in head_dict_.keys():
        part  = head_dict_.get(key)[0]
        colors = head_dict_.get(key)[1] 
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, 
                                                end_weight= 0.01, mode = 'linear') 
sol_barrier = SOLN.FiniteBarrier(
    m=const.m_e * const.c ** 2,
    x=np.linspace(-1, 4, 300) * u.nm,
    t=0 * u.fs,
    E=1* u.eV, A=1.0, L=0.5* u.nm, V0=1.3* u.eV,
)
phi_barrier, A, B, C, D, F = sol_barrier.solve(t=0 * u.fs)
x_barrier = sol_barrier.x.to_value(u.nm)
Lv_barrier = sol_barrier.L.to_value(u.nm)
L = Lv_barrier

for tentacle_start_pt in tentacle_start_pts.values():
    row, col = tentacle_start_pt
    
    if col > head_center_col:
        X = x_barrier
        Y = phi_barrier.real
    else:
        
        X = x_barrier[::-1]
        Y = phi_barrier.real
        
    DOMAIN_W = X.max() - X.min()
    DOMAIN_H = Y.max() - Y.min()
    CANVAS_PHYSICAL_H = DOMAIN_H*2

    CANVAS_PHYSICAL_W = DOMAIN_W*1.5
    canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
    canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)

    px0, py0 = cg.hex_center_pixel(tentacle_start_pt[0], tentacle_start_pt[1], detail_info)

    OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
    OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]

    rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}
    grid_hex_rc = cg.world_metres_to_hex_index(
            X + OFFSET_X, Y + OFFSET_Y, detail_info,
            canvas_physical_x_range=canvas_physical_x_range,
            canvas_physical_y_range=canvas_physical_y_range,
        )
    VALID_grid_hex_rc = [(r, c) for r,c in grid_hex_rc if (r, c) in hex_rc_arr]
    select_curve = cg.select_mask(list(set(VALID_grid_hex_rc)),hex_rc_arr)
    hex_colors[select_curve] = cgc.select_normal_color(select_curve, cgc.hex_to_rgb("#ffd647"), np.ones(3)*sigma_color*3) 
    
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Korosensei_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')