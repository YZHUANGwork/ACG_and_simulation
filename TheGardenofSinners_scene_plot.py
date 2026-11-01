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

rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100

fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=15, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.03

rng = np.random.default_rng(seed=0)
def draw_split(center_row, center_col, hex_rc_arr):
    left_down_12, right_down_12, left_up_12, right_up_12=True, False, False, True
    left_down_4, right_down_4, left_up_4, right_up_4=False, True , True,False
    
    left_down_5b, right_down_5b, left_up_5b, right_up_5b=False, True , False,False
    left_down_5t, right_down_5t, left_up_5t, right_up_5t=False, False , True,False
    
    split_part1 = cgd.draw_slope_1p5_diagonal(center_row, center_col, center_row+2, 
                                       left_down=left_down_12, right_down=right_down_12, left_up=left_up_12, right_up=right_up_12
          )+cgd.draw_slope_1p5_diagonal(center_row, center_col, center_row-2, 
                                       left_down=left_down_12, right_down=right_down_12, left_up=left_up_12, right_up=right_up_12
          )+cgd.draw_slope_1p5_diagonal(center_row, center_col+1, center_row+2, 
                                       left_down=left_down_12, right_down=right_down_12, left_up=left_up_12, right_up=right_up_12
          )+cgd.draw_slope_1p5_diagonal(center_row, center_col-1, center_row-2, 
                                       left_down=left_down_12, right_down=right_down_12, left_up=left_up_12, right_up=right_up_12
          )
    split_part2 = cgd.draw_slope_0p5_diagonal(min(r for r,c in split_part1) ,
                                       max(c for r,c in split_part1 if r==min(r for r,c in split_part1)),min(r for r,c in split_part1)-3, 
                                       left_down=left_down_12, right_down=right_down_12, left_up=left_up_12, right_up=right_up_12
          )+cgd.draw_slope_0p5_diagonal(max(r for r,c in split_part1) ,
                                       min(c for r,c in split_part1 if r==max(r for r,c in split_part1)),max(r for r,c in split_part1)+3,
                                       left_down=left_down_12, right_down=right_down_12, left_up=left_up_12, right_up=right_up_12
          )
    split_part3 = cgd.verticle_line(min(r for r,c in split_part2), max(c for r,c in split_part2 if r==min(r for r,c in split_part2)), 
                                   min(r for r,c in split_part2)-3, bend = 'right'
         )+cgd.verticle_line(max(r for r,c in split_part2), min(c for r,c in split_part2 if r==max(r for r,c in split_part2)), 
                                   max(r for r,c in split_part2)+3, bend = 'left'
         )
    split_part4 = cgd.draw_slope_0p5_diagonal(min(r for r,c in split_part3) ,
                                    max(c for r,c in split_part3 if r==min(r for r,c in split_part3)),min(r for r,c in split_part3)-4, 
                                       left_down=left_down_4, right_down=right_down_4, left_up=left_up_4, right_up=right_up_4
         )+cgd.draw_slope_0p5_diagonal(max(r for r,c in split_part3) ,
                                      min(c for r,c in split_part3 if r==max(r for r,c in split_part3)),max(r for r,c in split_part3)+4,
                                      left_down=left_down_4, right_down=right_down_4, left_up=left_up_4, right_up=right_up_4
         )
    split_part4_0 = cgd.draw_slope_0p5_diagonal(min(r for r,c in split_part3) ,
                                    max(c for r,c in split_part3 if r==min(r for r,c in split_part3)),min(r for r,c in split_part3)-2, 
                                       left_down=left_down_4, right_down=right_down_4, left_up=left_up_4, right_up=right_up_4
         )+cgd.draw_slope_0p5_diagonal(max(r for r,c in split_part3) ,
                                      min(c for r,c in split_part3 if r==max(r for r,c in split_part3)),max(r for r,c in split_part3)+2,
                                      left_down=left_down_4, right_down=right_down_4, left_up=left_up_4, right_up=right_up_4
         )
    split_part4_1 = cgd.draw_slope_0p5_diagonal(min(r for r,c in split_part4_0) ,
                            max(c for r,c in split_part4_0 if r==min(r for r,c in split_part4_0))-1,min(r for r,c in split_part4_0)-2, 
                                       left_down=left_down_4, right_down=right_down_4, left_up=left_up_4, right_up=right_up_4
         )+cgd.draw_slope_0p5_diagonal(max(r for r,c in split_part4_0) ,
                            min(c for r,c in split_part4_0 if r==max(r for r,c in split_part4_0))+1,max(r for r,c in split_part4_0)+2,
                                      left_down=left_down_4, right_down=right_down_4, left_up=left_up_4, right_up=right_up_4
         )
    split_part5 = cgd.draw_slope_8p3_diagonal(min(r for r,c in split_part4) ,
                                      max(c for r,c in split_part4 if r==min(r for r,c in split_part4)),min(r for r,c in hex_rc_arr), 
                                       left_down=left_down_5t, right_down=right_down_5t, left_up=left_up_5t, right_up=right_up_5t
         )+cgd.draw_slope_8p3_diagonal(max(r for r,c in split_part4) ,
                                      min(c for r,c in split_part4 if r==max(r for r,c in split_part4)),max(r for r,c in hex_rc_arr),
                                      left_down=left_down_5b, right_down=right_down_5b, left_up=left_up_5b, right_up=right_up_5b
         )+cgd.draw_slope_8p3_diagonal(min(r for r,c in split_part4) ,
                                   max(c for r,c in split_part4 if r==min(r for r,c in split_part4))-1,min(r for r,c in hex_rc_arr), 
                                      left_down=left_down_5t, right_down=right_down_5t, left_up=left_up_5t, right_up=right_up_5t
         )+cgd.draw_slope_8p3_diagonal(max(r for r,c in split_part4) ,
                                      min(c for r,c in split_part4 if r==max(r for r,c in split_part4))+1,max(r for r,c in hex_rc_arr),
                                      left_down=left_down_5b, right_down=right_down_5b, left_up=left_up_5b, right_up=right_up_5b
         )+cgd.draw_slope_8p3_diagonal(min(r for r,c in split_part4) ,
                                   max(c for r,c in split_part4 if r==min(r for r,c in split_part4))-2,min(r for r,c in hex_rc_arr), 
                                       left_down=left_down_5t, right_down=right_down_5t, left_up=left_up_5t, right_up=right_up_5t
         )+cgd.draw_slope_8p3_diagonal(max(r for r,c in split_part4) ,
                                      min(c for r,c in split_part4 if r==max(r for r,c in split_part4))+2,max(r for r,c in hex_rc_arr),
                                     left_down=left_down_5b, right_down=right_down_5b, left_up=left_up_5b, right_up=right_up_5b
         )
    
    split = split_part1+split_part2+split_part3+split_part4+split_part5
    split2 = split_part1+split_part2+split_part3+split_part4_0+split_part4_1+split_part5
    return split, split2, split_part1, split_part2, split_part3, split_part4, split_part5

def flip_points(pts, pivot_row=None, pivot_col=None):
    return [(2*pivot_row - r if pivot_row is not None else r,
             2*pivot_col - c - (1 if r % 2 == 1 else 0) if pivot_col is not None else c)
            for r, c in pts]


def draw_char(center_row, center_col, hex_rc_arr, version):

    split, split2, split_part1, split_part2, split_part3, split_part4, split_part5 = draw_split(center_row, center_col, hex_rc_arr)
    shift_col = 2

    left_head_loc = (center_row-4, center_col-4)
    
    head = cg.hex_neighbours_n(left_head_loc[0], left_head_loc[1], n=2) 
    face = [(r,c) for r,c in head if r>left_head_loc[0] and c>=left_head_loc[1]]
    hair = [(r,c) for r,c in head if (r,c) not in face
           ]+cgd.draw_trapezoid(left_head_loc[0], min(c for r,c in head if r==left_head_loc[0]), left_head_loc[1], 
                                max(r for r,c in head)+1, slope_left = 'inf', slope_right = 'inf', direction = 'll',
                  bend_left = 'left', bend_right = 'left')[-2]
    
 
    head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(left_head_loc[0], left_head_loc[1], version="5")
    
    skin = [(r,c) for r,c in face+neck if (r,c) not in hair]
    raw_body_left = [(r,c) for r,c in  raw_body if r<=left_head_loc[0]+6]
    raw_body_left_skeleton = [(r, c-shift_col) for r,c in split if r>left_head_loc[0]+6 and r<=max(r for r,c in split_part4)
                         ]+cgd.draw_slope_0p5_diagonal(max(r for r,c in split_part4),
                                                       max(c for r,c in split if r==max(r for r,c in split_part4))-shift_col,
                                                       max(r for r,c in hex_rc_arr),
                                                       left_down=True, right_down=False, left_up=False, right_up=False)

    sleeves_left_row = max(r for r,c in raw_body_left)+1
    left_arm_split_start = max(r for r,c in raw_body_left)-2
    body_left_skeleton = [min((r,c) for r,c in raw_body_left_skeleton if r==r0) for r0 in {r for r,c in raw_body_left_skeleton}]
    for loc in body_left_skeleton:
        n=1 if loc[0]<max(r for r,c in split_part4) else 2
        raw_body_left+=cg.hex_neighbours_n(loc[0], loc[1], n=n)

    valid_raw_body_left = [(r,c) for r,c in raw_body_left if c < min((c2 for r2,c2 in split2 if r2==r), default=float('inf'))]

    sleeves_left_skeleton_part3 = cgd.verticle_line(sleeves_left_row,
                                                min(c for r,c in valid_raw_body_left if r==sleeves_left_row)-1,
                                                sleeves_left_row+6,  bend = 'left')

    sleeves_left_skeleton_part4 = cgd.draw_slope_1p5_diagonal(max(r for r,c in sleeves_left_skeleton_part3),
                           min(c for r,c in sleeves_left_skeleton_part3 if r==max(r for r,c in sleeves_left_skeleton_part3)),
                           max(r for r,c in sleeves_left_skeleton_part3)+3,
                           left_down=True, right_down=False, left_up=False, right_up=False)
    sleeves_left_skeleton_part5 = cgd.draw_slope_8p3_diagonal(max(r for r,c in sleeves_left_skeleton_part4),
                           min(c for r,c in sleeves_left_skeleton_part4 if r==max(r for r,c in sleeves_left_skeleton_part4)),
                           max(r for r,c in hex_rc_arr),
                           left_down=True, right_down=False, left_up=False, right_up=False)

    sleeves_left_skeleton = sleeves_left_skeleton_part3+sleeves_left_skeleton_part4+sleeves_left_skeleton_part5

    sleeves_left = []
    for loc in sleeves_left_skeleton:
        n=2 if loc[0]>max(r for r,c in hex_rc_arr)-5 else 1
        sleeves_left+=cg.hex_neighbours_n(loc[0], loc[1], n=n)
    valid_sleeves_left = [(r,c) for r,c in sleeves_left if (r,c) not in valid_raw_body_left]

    left_arm_split_start_loc = (left_arm_split_start, max(c for r,c in valid_raw_body_left if r==left_arm_split_start)+1)
    left_arm_split_part1 = cgd.draw_slope_1p5_diagonal(
        left_arm_split_start_loc[0], left_arm_split_start_loc[1], left_arm_split_start_loc[0]-3,
                                   left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_1p5_diagonal(left_arm_split_start_loc[0], left_arm_split_start_loc[1]-1, left_arm_split_start_loc[0]-3,
                                   left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_1p5_diagonal(left_arm_split_start_loc[0], left_arm_split_start_loc[1]+1, left_arm_split_start_loc[0]-2,
                                   left_down=False, right_down=False, left_up=False, right_up=True
                )
    left_arm_split_part2=[min((r,c) for r,c in split if r==r0) for r0 in {r for r,c in split}
                          if r0<=min(row for row, col in left_arm_split_part1) and r0>=min(row for row, col in left_arm_split_part1)-4]
    left_arm_split_part3 = cgd.draw_slope_0p5_diagonal(min(r for r,c in left_arm_split_part2),
                                     min(c for r,c in left_arm_split_part2 if r==min(r for r,c in left_arm_split_part2)),
                                     min(r for r,c in left_arm_split_part2)-2,
                                                       left_down=False, right_down=False, left_up=True, right_up=False
                )
    left_arm_split = left_arm_split_part1+left_arm_split_part2+left_arm_split_part3

    left_hand_split = cg.hex_neighbours_n(min(r for r,c in left_arm_split),
                                          min(c for r,c in left_arm_split if r==min(r for r,c in left_arm_split)), n=1)

    left_blade = [(r,c) for r,c in split2 if r<center_row and (r,c) not in left_hand_split+left_arm_split]
    
    left_kimono_part1 = [(r,c) for r,c in valid_raw_body_left if any(c<=c_ for r_,c_ in body_left_skeleton if r_==r) and (r,c) not in skin+hair]
    left_kimono_part2 = [(r,c) for r,c in valid_raw_body_left if (r,c) not in skin+hair+left_kimono_part1]
    
    if version == 'left':
        
    
        valid_raw_body, body_skeleton, valid_sleeves, sleeves_skeleton, arm_split, hand_split, blade, kimono_part1, kimono_part2 = \
            valid_raw_body_left, body_left_skeleton, valid_sleeves_left, sleeves_left_skeleton, left_arm_split, left_hand_split, left_blade, left_kimono_part1, left_kimono_part2
        region = [(r,c) for r,c in hex_rc_arr if any(c<=c_ for r_,c_ in split2 if r_==r)]
        region_COLOR = [[0,0,0], cgc.hex_to_rgb("#796770"), 'col max linear']
        center_COLOR = [cgc.hex_to_rgb("#534d96") ]
        blade_COLOR = [[0.9,0.9,0.9], [0.8,0.8,0.8], 'row']
        kimono_part1_COLOR = [cgc.hex_to_rgb("#ffd2e9"),cgc.hex_to_rgb("#705d66"), 'col max'] 
        kimono_part2_COLOR = [cgc.hex_to_rgb("#705d66"), cgc.hex_to_rgb("#ffd2e9"), 'col max']
        sleeves_COLOR = [cgc.hex_to_rgb("#705d66"),cgc.hex_to_rgb("#ffd2e9"),  'row']
    else:  # 'right'
        valid_raw_body = flip_points(valid_raw_body_left, center_row, center_col)
        body_skeleton = flip_points(body_left_skeleton, center_row, center_col)
        valid_sleeves = flip_points(valid_sleeves_left, center_row, center_col)
        sleeves_skeleton = flip_points(sleeves_left_skeleton, center_row, center_col)
        arm_split = flip_points(left_arm_split, center_row, center_col)
        hand_split = flip_points(left_hand_split, center_row, center_col)
        blade = flip_points(left_blade, center_row, center_col)
        
        skin = flip_points(skin, center_row, center_col)
        hair = flip_points(hair, center_row, center_col)
        
        kimono_part1 = flip_points(left_kimono_part1, center_row, center_col)
        kimono_part2 = flip_points(left_kimono_part2, center_row, center_col)
        
        region = [(r,c) for r,c in hex_rc_arr if any(c>=c_ for r_,c_ in split2 if r_==r)]
        region_COLOR = [cgc.hex_to_rgb("#ffecf7"), cgc.hex_to_rgb("#dbcfd6"), 'col min linear']
        center_COLOR = [cgc.hex_to_rgb("#d4d0ff")]
        blade_COLOR = [[0.1,0.1,0.1], [0.2,0.2,0.2], 'row']
        kimono_part1_COLOR = [cgc.hex_to_rgb("#ffd2e9"),cgc.hex_to_rgb("#705d66"), 'col min'] 
        kimono_part2_COLOR = [cgc.hex_to_rgb("#705d66"), cgc.hex_to_rgb("#ffd2e9"), 'col min']
        sleeves_COLOR = [cgc.hex_to_rgb("#705d66"),cgc.hex_to_rgb("#ffd2e9"),  'row max']
        
    center = [(center_row, center_col)]
    keys = ["region", #"split2",

            "skin", "hair",  
           
            "kimono_part1","kimono_part2",
           
            "valid_sleeves","arm_split", 
            
            "hand_split", "blade", "center"]
    colors = [region_COLOR,#  [[0,0,0]],
              
              [cgc.hex_to_rgb("#ffefd5")], [cgc.hex_to_rgb("#624616")],
              
              kimono_part2_COLOR, kimono_part2_COLOR,
              
              [cgc.hex_to_rgb("#705d66"),cgc.hex_to_rgb("#ffd2e9"),  'row'],[cgc.hex_to_rgb("#ffd2e9"),cgc.hex_to_rgb("#1b000f"), 'row'],
              [cgc.hex_to_rgb("#ffefd5")], blade_COLOR, 
              center_COLOR, [[0,0,1]], [[1,0,0]], [[0.4, 0.4, 0.4]], [[0.4,1,0]] ]
    local_vars = locals()
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})

   

def draw_obj(center_row, center_col, hex_rc_arr, name ):
    if "building" in name:
        building_skeleton = cgd.draw_block((center_row, (center_col, center_col)), max(r for r,c in hex_rc_arr))
        building = []
        for loc in building_skeleton:
            building+=cg.hex_neighbours_n(loc[0], loc[1], n=5)
        valid_building = [(r,c) for r,c in building if (r,c) in hex_rc_arr]

        window_rows = np.arange(min(r for r,c in building)+2, max(r for r,c in hex_rc_arr), 3)
        window = [(r,c) for r,c in building if r in window_rows]
        valid_window = [(r,c) for r,c in window if (r,c) in hex_rc_arr]

        idx = rng.integers(0, len(valid_window), size=len(valid_window)//6)
        light = [valid_window[i] for i in idx]


        keys = ["valid_building","valid_window",  "light", ]#
        colors = [[cgc.hex_to_rgb("#698486")], [cgc.hex_to_rgb("#2e4f5f")], [cgc.hex_to_rgb("#fec452")]]
    elif "can" in name:
        can_skeleton = cgd.draw_block((center_row, (center_col, center_col)), max(r for r,c in hex_rc_arr))
        can = []
        for loc in can_skeleton:
            can+=cg.hex_neighbours_n(loc[0], loc[1], n=5)
        valid_can = [(r,c) for r,c in can if (r,c) in hex_rc_arr]
        
        head_center_row = center_row+6
        head_center_col =center_col
        head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(head_center_row, head_center_col, version="7")
        valid_body = [(r,c) for r,c in raw_body if r <=max(r_ for r_,c_ in neck)]
        
        blood = cgd.verticle_line(head_center_row, max(c for r,c in head_raw)-2,
                                       max(r for r,c in valid_body), bend = 'right'
                                      )
        hair_part1 = [(r,c) for r,c in head_raw if r < head_center_row]
        hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                       min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head_raw), bend = 'right'
                                      )+cgd.verticle_line(max(r for r,c in hair_part1), 
                                       max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head_raw), bend = 'left'
                                      )
        hair_back = cgd.draw_block((max(r for r,c in hair_part1), 
                                    (min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)), 
                                     max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)))
                                   ), max(r for r,c in valid_can))
        hair_part4 = cgd.verticle_line(head_center_row, head_center_col,
                                       head_center_row+1, bend = 'left'
                                      )
        hair_part3 = [(r,c) for r,c in hair_back if (r,c) not in valid_body]
        hair =hair_part1 +hair_part2+hair_part3+hair_part4
    
        valid_blood = [(r,c) for r,c in blood if (r,c) not in hair]
        
        if 'blood' in name:
            
            keys = ["can",  "valid_body", "hair", "valid_blood"]#
            colors = [[cgc.hex_to_rgb("#d4a85d")], [cgc.hex_to_rgb("#f8d6b2")], [cgc.hex_to_rgb("#d46738"), cgc.hex_to_rgb("#b84818")], 
                      [cgc.hex_to_rgb("#d10000"), cgc.hex_to_rgb("#6b0000"), 'row']]
        else:
            keys = ["can",  "valid_body", "hair"]#
            colors = [[cgc.hex_to_rgb("#d4a85d")], [cgc.hex_to_rgb("#f8d6b2")], [cgc.hex_to_rgb("#d46738"), cgc.hex_to_rgb("#b84818")], ]
            
    elif "Shiki" in name or "Shirazumi" in name:
        head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(center_row, center_col, version="5")
        ln = SimpleNamespace(**line_dict)
 

        head = cg.hex_neighbours_n(center_row, center_col, n=2)
        hair_part1 = [(r,c) for r,c in head_raw if r < center_row]
        hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                       min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head_raw), bend = 'right'
                                      )+cgd.verticle_line(max(r for r,c in hair_part1), 
                                       max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head_raw), bend = 'left'
                                      )+[(center_row, min(c for r,c in head)+1), (center_row, max(c for r,c in head)-1)]
        hair_back = cgd.draw_block((max(r for r,c in hair_part1), 
                                    (min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)), 
                                     max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)))
                                   ), max(r for r,c in head_raw))
        
        hair = [(r,c) for r,c in hair_part1+hair_part2+hair_back if (r,c) not in neck]
        face = [(r,c) for r,c in head if (r,c) not in hair_part1+hair_part2]
        skin_raw = [(r,c) for r,c in neck if (r,c) not in face] +hand_dict["hand"]
        kimono_part1 =  [(r,c) for r,c in cgd.draw_block((ln.shoulder_joint_r, (center_col-2, center_col+2)), ln.foot_r)
                   if (r,c) not in skin_raw]
        kimono_part2 = cgd.draw_block((ln.femoral_head_r-1, 
                                       (min(c for r,c in kimono_part1 if r==ln.femoral_head_r-1)-1, 
                                        max(c for r,c in kimono_part1 if r==ln.femoral_head_r-1)+1)
                                      ), ln.foot_r)
        kimono = kimono_part1+kimono_part2
        
        belt = [(r,c) for r,c in kimono if r==ln.bellybutton_r]
        
        jkt_raw= [(r,c) for r,c in raw_body if (r>=ln.shoulder_joint_r and r <ln.bellybutton_r)
              ]+upperarm_dict["upperarm"]+forearm_dict["forearm"]
        jkt_remove = cgd.verticle_line(ln.shoulder_joint_r, center_col,ln.bellybutton_r, bend = 'right'
                                      )+cgd.verticle_line(ln.shoulder_joint_r, center_col,ln.bellybutton_r, bend = 'left'
                                      )
        jkt = [(r,c) for r,c in jkt_raw if (r,c) not in jkt_remove]
        jkt_collar = cgd.verticle_line(ln.shoulder_joint_r, max(c for r,c in jkt if c<center_col and r==ln.shoulder_joint_r),
                           ln.shoulder_joint_r-1, bend = 'left'
                          )+cgd.verticle_line(ln.shoulder_joint_r, min(c for r,c in jkt if c>center_col and r==ln.shoulder_joint_r),
                           ln.shoulder_joint_r-1, bend = 'right'
                          )
        if 'flip' in name:
            hand_ = 'right_hand'
            left_up = False
            right_up = True
            right_down = False
            left_down=False
            hand_remove = [(max(r for r,c in hand_dict["hand"]), 
                            max(c for r,c in hand_dict["hand"] if r==max(r for r,c in hand_dict["hand"])))]
            knife = [(r,c) for r,c in cgd.draw_slope_0p5_diagonal(min(r for r,c in hand_dict[hand_])+2, 
                                            min(c for r,c in hand_dict[hand_] if r==min(r for r,c in hand_dict[hand_])+2)-1,
                                            min(r for r,c in hand_dict[hand_])-3, 
                                            left_down=left_down, right_down=right_down, left_up=left_up, right_up=right_up)
                 if (r,c) not in hand_dict["hand"]]
        else:
            hand_ =  'left_hand'
            left_up = True
            right_up = False
            right_down = False
            left_down=False
            hand_remove = [(max(r for r,c in hand_dict["hand"]), 
                            min(c for r,c in hand_dict["hand"] if r==max(r for r,c in hand_dict["hand"])))]
        
            knife = [(r,c) for r,c in cgd.draw_slope_0p5_diagonal(min(r for r,c in hand_dict[hand_])+2, 
                                            max(c for r,c in hand_dict[hand_] if r==min(r for r,c in hand_dict[hand_])+2)+1,
                                            min(r for r,c in hand_dict[hand_])-3, 
                                            left_down=left_down, right_down=right_down, left_up=left_up, right_up=right_up)
                 if (r,c) not in hand_dict["hand"]]
        knife_handle = [(r,c) for r,c in knife if r>=min(r for r,c in hand_dict[hand_])]
        
        
        skin = [(r,c) for r,c in skin_raw if (r,c) not in hand_remove]
        
        if "Shiki" in name:
            hair_color = [cgc.hex_to_rgb("#342919")]
            face_color = [cgc.hex_to_rgb("#f8d6b2")]
            skin_color = [cgc.hex_to_rgb("#f8d6b2")]
            
            
        else:
            hair_color = [cgc.hex_to_rgb("#ebe93c")]
            face_color = [cgc.hex_to_rgb("#74604b")]
            skin_color = [cgc.hex_to_rgb("#74604b")]
            
        keys = ["hair", "face", "skin", 
                "kimono",  "belt", "jkt", "jkt_collar", 
                'knife', "knife_handle", ]#
        colors = [hair_color, face_color,skin_color, 
                 
                 [cgc.hex_to_rgb("#65aac4")], [cgc.hex_to_rgb("#2f689d")],  [cgc.hex_to_rgb("#bb0404")],  [[1,1,1]],
                  
                  [[0.7,0.7,0.7]], [cgc.hex_to_rgb("#8b582e")], 
                 ]
            
        
    local_vars = locals()
    if 'flip' in name:
        print('flip')
        for key in keys:
            local_vars[key] = [(2*center_row - r, c) for r, c in local_vars[key]]
            
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})
   

n_marker = 14


marker_center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
marker_center_col = max(c for r, c in hex_rc_arr) // 2

hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#ff0000"), cgc.hex_to_rgb("#006889"), hex_rc_arr, hex_colors, 
                                    (marker_center_row-4, marker_center_col), 18,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='linear',
                        start_n=0, end_n=None)

hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#006889"), cgc.hex_to_rgb("#73ffea"), hex_rc_arr, hex_colors, 
                                    (marker_center_row-4, marker_center_col), max(c for r, c in hex_rc_arr) ,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='sigmoid',
                        start_n=n_marker+5, end_n=None)

char_dict_ = {
     "Shiki": draw_obj(marker_center_row-6, marker_center_col+10, hex_rc_arr, "Shiki flip"),
     "Shirazumi": draw_obj(marker_center_row+6, marker_center_col-10, hex_rc_arr, "Shirazumi"),
     "building": draw_obj(marker_center_row-4, marker_center_col, hex_rc_arr, "building"),

    #     "Aozaki head": draw_obj(marker_center_row-4, marker_center_col+10, hex_rc_arr, "can blood"),

         'left': draw_char(marker_center_row, marker_center_col-12, hex_rc_arr, 'left'),

    'right': draw_char(marker_center_row, marker_center_col+12, hex_rc_arr, 'right')

}

for char, dict_ in char_dict_.items(): 
 
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
            elif 'col' in sort_info:
                print(key)
                print('col gradient')
                if 'max' in sort_info:
                    pivot = 'max'
                elif 'min' in sort_info:
                    pivot = 'min'
                else:
                    pivot = None
                if 'sigmoid' in sort_info:
                    mode = 'sigmoid'
                else:
                    mode = 'linear'
                print('mode', mode)
                hex_colors = cgc.color_row_gradient(part, 
                        colors[0],colors[1],
                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01, mode = mode, pivot=pivot ) 
                
            else:
                if 'max' in sort_info:
                    pivot = 'max'
                elif 'min' in sort_info:
                    pivot = 'min'
                else:
                    pivot = None
                if 'sigmoid' in sort_info:
                    mode = 'sigmoid'
                else:
                    mode = 'linear'
                print('mode', mode)
                
                hex_colors = cgc.color_row_gradient(part, 
                        colors[0],colors[1],
                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = mode, pivot=pivot ) 
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'TheGardenofSinners_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')