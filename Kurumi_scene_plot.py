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
    IMG_W=1280, IMG_H=720, HEX_R=8, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.04

rng = np.random.default_rng(seed=0)

 
def draw_divide(center_row, center_col,region_rc_arr, n_divide=6 ):
    
    divide_6=cgd.draw_slope_0p5_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                         left_down=True, right_down=True, left_up=True, right_up=True
            )+cgd.draw_slope_0p5_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                         left_down=True, right_down=True, left_up=True, right_up=True
            )+cgd.horizontal_lines([(center_row, (min(c for r,c in region_rc_arr), max(c for r,c in region_rc_arr)))])
    
    divide_6angle_dict = {'0':cgd.horizontal_lines([(center_row, (center_col, max(c for r,c in region_rc_arr)))]),
                            '60':cgd.draw_slope_0p5_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                         left_down=False, right_down=False, left_up=False, right_up=True),
                             '120':cgd.draw_slope_0p5_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                         left_down=False, right_down=False, left_up=True, right_up=False),
                             '180':cgd.horizontal_lines([(center_row, (min(c for r,c in region_rc_arr), center_col))]),
                             '240':cgd.draw_slope_0p5_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                         left_down=True, right_down=False, left_up=False, right_up=False),
                             '300':cgd.draw_slope_0p5_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                         left_down=False, right_down=True, left_up=False, right_up=False)
                             
                            }
    if n_divide==6:
        divide = divide_6
        divide_angle_dict = divide_6angle_dict
        
        
    elif n_divide==12:
        divide = divide_6+ cgd.verticle_line(center_row, center_col, min(r for r,c in region_rc_arr), bend = 'left'
                )+cgd.verticle_line(center_row, center_col, min(r for r,c in region_rc_arr), bend = 'right'
                )+cgd.verticle_line(center_row, center_col, max(r for r,c in region_rc_arr), bend = 'left'
                )+cgd.verticle_line(center_row, center_col, max(r for r,c in region_rc_arr), bend = 'right'
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=False, left_up=True, right_up=True
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                                    left_down=True, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col+1, min(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col-1, min(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col+1, max(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col-1, max(r for r,c in region_rc_arr), 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                )
        
        divide_12angle_dict = {
            '30':cgd.draw_slope_1p5_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                 )+cgd.draw_slope_1p5_diagonal(center_row, center_col+1, min(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                ),
            '90': cgd.verticle_line(center_row, center_col, min(r for r,c in region_rc_arr), bend = 'left'
                )+cgd.verticle_line(center_row, center_col, min(r for r,c in region_rc_arr), bend = 'right'
                ),
            '150': cgd.draw_slope_1p5_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                 )+cgd.draw_slope_1p5_diagonal(center_row, center_col-1, min(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                ),
            '210': cgd.draw_slope_1p5_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col-1, max(r for r,c in region_rc_arr), 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                ),
            '270': cgd.verticle_line(center_row, center_col, max(r for r,c in region_rc_arr), bend = 'left'
                )+cgd.verticle_line(center_row, center_col, max(r for r,c in region_rc_arr), bend = 'right'
                ),
            '330': cgd.draw_slope_1p5_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal(center_row, center_col+1, max(r for r,c in region_rc_arr), 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                ),
                                               }
        divide_angle_dict = {**divide_6angle_dict, **divide_12angle_dict}
        
        
    elif n_divide==18:
        divide= divide_6+ cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                       left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                   left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                   left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                   left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col-1, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+1, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+2, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col+1, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+1, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+2, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col+1, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+1, max(r for r,c in region_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+2, max(r for r,c in region_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col-1, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col-1, max(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col-2, max(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                )
        divide_18angle_dict = {
            '20':cgd.draw_slope_8p3_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+1, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+2, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                ),
            '40': cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col-1, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                ),
            '80': cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                       left_down=False, right_down=False, left_up=False, right_up=True),
            '100': cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                   left_down=False, right_down=False, left_up=True, right_up=False),
            '140': cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col+1, min(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                ),
            '160': cgd.draw_slope_8p3_diagonal(center_row, center_col, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+1, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+2, min(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                ),
            '200': cgd.draw_slope_8p3_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+1, max(r for r,c in region_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col+2, max(r for r,c in region_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                ),
            '220': cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col+1, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                ),
            '260': cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                   left_down=True, right_down=False, left_up=False, right_up=False),
            '280': cgd.draw_slope_0p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 3, shift_amount=-1, 
                                                   left_down=False, right_down=True, left_up=False, right_up=False),
            '320': cgd.draw_slope_1p5_diagonal_shift(center_row, center_col, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_1p5_diagonal_shift(center_row, center_col-1, max(r for r,c in region_rc_arr), 4, shift_amount=-2, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                ),
            '340': cgd.draw_slope_8p3_diagonal(center_row, center_col, max(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col-1, max(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                )+cgd.draw_slope_8p3_diagonal(center_row, center_col-2, max(r for r,c in region_rc_arr), 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                )
        }
        divide_angle_dict = {**divide_6angle_dict, **divide_18angle_dict}
        
    else:
        print('n_divide unavailable, it needs to be 6,12,18, ')
        divide = []
        divide_angle_dict = dict()
    return divide, divide_angle_dict

def draw_gear(center_row, center_col, n_gear, version):
    if n_gear>=6 and version == 'solid center':
        gear_raw = cg.hex_neighbours_n(center_row, center_col, n=n_gear)
        holes_centers = cg.get_hex_corners(center_row, center_col, n_gear-2)
        holes = []
        for holes_center in holes_centers:
            holes +=cg.hex_neighbours_n(holes_center[0], holes_center[1], n=1)
        gear = [(r,c) for r,c in gear_raw if(r,c) not in holes]
        
       
    if n_gear>=6 and version == 'hollow center':
        gear_boundary = cg.hex_neighbours_n(center_row, center_col, n=n_gear, keep_origin =False , return_frontier=True
                         )[-1]+cg.hex_neighbours_n(center_row, center_col, n=n_gear+1, keep_origin =False , return_frontier=True
                         )[-1]
        gear_region = cg.hex_neighbours_n(center_row, center_col, n=n_gear-1)
        
        gear_out = [(r,c) for r,c in draw_divide(center_row, center_col,gear_boundary, n_divide=18)[0] if (r,c) in gear_boundary]
        
        gear_in = [(r,c) for r,c in draw_divide(center_row, center_col,gear_region, n_divide=6)[0] if (r,c) in gear_region]
        
        gear = cg.hex_neighbours_n(center_row, center_col, n=n_gear-1, keep_origin =False , return_frontier=True
                                  )[-1]+gear_out+gear_in
        
      
    
    if n_gear<6 and n_gear>3 and version == 'hollow center':
        gear_boundary = cg.hex_neighbours_n(center_row, center_col, n=n_gear, keep_origin =False , return_frontier=True)[-1]
        gear_region = cg.hex_neighbours_n(center_row, center_col, n=n_gear-1)
        
        gear_out = [(r,c) for r,c in draw_divide(center_row, center_col,gear_boundary, n_divide=12)[0] if (r,c) in gear_boundary]
        
        gear_in = [(r,c) for r,c in draw_divide(center_row, center_col,gear_region, n_divide=6)[0] if (r,c) in gear_region]
        
        gear = cg.hex_neighbours_n(center_row, center_col, n=n_gear-1, keep_origin =False , return_frontier=True
                                  )[-1]+gear_out+gear_in
        
     
    if n_gear<=3 and version == 'hollow center':
        gear_boundary = cg.hex_neighbours_n(center_row, center_col, n=n_gear, keep_origin =False , return_frontier=True)[-1]
        gear_region = cg.hex_neighbours_n(center_row, center_col, n=n_gear-1)
        
        gear_out = [(r,c) for r,c in draw_divide(center_row, center_col,gear_boundary, n_divide=6)[0] if (r,c) in gear_boundary]
        
        
        gear = cg.hex_neighbours_n(center_row, center_col, n=n_gear-1, keep_origin =False , return_frontier=True
                                  )[-1]+gear_out
          
    return gear
  

def draw_clock(center_row, center_col, n_clock):
    clock = cg.hex_neighbours_n(center_row, center_col, n=n_clock)
    region_rc_arr = clock
    
    clock_mid = cg.hex_neighbours_n(center_row, center_col, n=n_clock-6)
    clock_in = cg.hex_neighbours_n(center_row, center_col, n=n_clock-14)
    
    
    divide_6,_  = draw_divide(center_row, center_col,region_rc_arr, n_divide=6)
    divide_12,_ = draw_divide(center_row, center_col,region_rc_arr, n_divide=12)
    divide_18,_ = draw_divide(center_row, center_col,region_rc_arr, n_divide=18)
    
    divide_12_tick = [(r,c) for r,c in divide_12 if (r,c) in 
                  cg.hex_neighbours_n(center_row, center_col, n=n_clock-8) and (r,c)  not in 
                  cg.hex_neighbours_n(center_row, center_col, n=n_clock-12)]
    
    gear_bottom_layer = [(r,c) for r,c in draw_gear(center_row-6, center_col-7, 6, 'hollow center'
                                         )+draw_gear(center_row+8, center_col-8, 4, 'hollow center'
                                         )+draw_gear(center_row-4, center_col+8, 7, 'hollow center'
                                         )+draw_gear(center_row+9, center_col+4, 3, 'hollow center'
                                         )if (r,c) in clock_in]
    
    gear_mid_layer = [(r,c) for r,c in draw_gear(center_row-11, center_col, 3, 'hollow center'
                                         )+draw_gear(center_row+12, center_col+2, 5, 'hollow center'
                                         )+draw_gear(center_row, center_col-12, 4, 'hollow center'
                                         )+draw_gear(center_row+4, center_col+10, 4, 'hollow center'
                                         ) if (r,c) in clock_in]
    gear_top_layer = [(r,c) for r,c in draw_gear(center_row, center_col, 6, 'solid center')if (r,c) in clock_in]
    
    gear_center = cg.hex_neighbours_n(center_row, center_col, n=1, keep_origin =False , return_frontier=True
                                  )[-1]
    keys = ["clock", "clock_mid", "clock_in",
            
            "gear_bottom_layer","gear_mid_layer",  "gear_top_layer","gear_center",
            "divide_12_tick", ]#
    colors = [
              [cgc.hex_to_rgb("#b5a746")],[cgc.hex_to_rgb("#ffefa9")], [cgc.hex_to_rgb("#412916")],
        [cgc.hex_to_rgb("#7e6f2c")],[cgc.hex_to_rgb("#b79e31")], [cgc.hex_to_rgb("#decd33")],[cgc.hex_to_rgb("#d5b42e")],
        [cgc.hex_to_rgb("#412916")],[cgc.hex_to_rgb("#decd33")],
        
        
        [[0.7,0.7,0.7]],   [[1,0,0]]
             ]
        
    
    local_vars = locals()
                
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})
   
def draw_char(head_center_row, head_center_col, version):
    head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(head_center_row, head_center_col, version='5')
    ln = SimpleNamespace(**line_dict)
 
    leg_modified = [(r+4,c) for r,c in raw_body if r>= ln.knee_r]
    
    sk_row = ln.femoral_head_r
    sk_raw = cgd.draw_trapezoid(sk_row, 
                                min(c for r,c in thigh_dict['thigh'] if r==sk_row), 
                                max(c for r,c in thigh_dict['thigh'] if r==sk_row), 
                                min(r for r,c in leg_modified), slope_left = '0.5', slope_right = '0.5', direction = 'lr',
                  bend_left = 'left', bend_right = 'right')[-2]
    sk  =[(r,c) for r,c in cg.hex_neighbours_n(ln.bellybutton_r-1, head_center_col, 
                                               n=min(r for r,c in leg_modified)-ln.bellybutton_r) if (r,c) in sk_raw]
    
    raw_body_modified = [(r,c) for r,c in raw_body if r< ln.knee_r]+leg_modified
    
    
    head = cg.hex_neighbours_n(head_center_row, head_center_col, n=2)
    hair_part1 = [(r,c) for r,c in head_raw if r < head_center_row]
    
    head_band = [(r,c) for r,c in cg.hex_neighbours_n(head_center_row, head_center_col, 
                                                            n=3, keep_origin =False , return_frontier=True)[-1]  if (r,c) in hair_part1]
    head_band_frill = cgd.draw_slope_1p5_diagonal(min(r for r,c in head_band)+1, 
                                                  min(c for r,c in head_band if r==min(r for r,c in head_band)+1)+1, 
                                                  min(r for r,c in head_band), 
                                                  left_down=False, right_down=False, left_up=True, right_up=False
                     )+cgd.draw_slope_8p3_diagonal(min(r for r,c in head_band)+1, 
                                                  min(c for r,c in head_band if r==min(r for r,c in head_band)+1)+1, 
                                                  max(r for r,c in head_band), 
                                                  left_down=True, right_down=False, left_up=False, right_up=False
                     )+cgd.draw_slope_1p5_diagonal(min(r for r,c in head_band)+1, 
                                                  max(c for r,c in head_band if r==min(r for r,c in head_band)+1)-1, 
                                                  min(r for r,c in head_band), 
                                                  left_down=False, right_down=False, left_up=False, right_up=True
                     )+cgd.draw_slope_8p3_diagonal(min(r for r,c in head_band)+1, 
                                                  max(c for r,c in head_band if r==min(r for r,c in head_band)+1)-1, 
                                                  max(r for r,c in head_band), 
                                                  left_down=False, right_down=True, left_up=False, right_up=False
                     )+[(min(r for r,c in head_band)-1, min(c for r,c in head_band if r==min(r for r,c in head_band)+1)+1),
                        (min(r for r,c in head_band)-1, max(c for r,c in head_band if r==min(r for r,c in head_band)+1)-1)
                     ]
    left_ponytails_part1 = [(r,c) for r,c in cg.hex_neighbours_n(head_center_row-1, 
                                                            min(c for r,c in hair_part1 if r==head_center_row-1)+1, 
                                                            n=3, keep_origin =False , return_frontier=True)[-1] 
                       if r<=head_center_row and c<head_center_col]
    left_ponytails_part2 = cgd.verticle_line(max(r for r,c in left_ponytails_part1), 
                                   min(c for r,c in left_ponytails_part1 if r==max(r for r,c in left_ponytails_part1)),
                                   max(r for r,c in left_ponytails_part1)+3, bend = 'left'
                                  )
    left_ponytails = left_ponytails_part1+left_ponytails_part2
    right_ponytails_part1 = [(r,c) for r,c in cg.hex_neighbours_n(head_center_row-1, 
                                                            max(c for r,c in hair_part1 if r==head_center_row-1)-2,  
                                                            n=4, keep_origin =False , return_frontier=True)[-1] 
                       if r<=head_center_row and c>head_center_col]
    right_ponytails_part2 = cgd.verticle_line(min(r for r,c in right_ponytails_part1), 
                                   min(c for r,c in right_ponytails_part1 if r==min(r for r,c in right_ponytails_part1)),
                                   min(r for r,c in hair_part1), bend = 'left'
                                  )+cgd.verticle_line(max(r for r,c in right_ponytails_part1), 
                                   min(c for r,c in right_ponytails_part1 if r==max(r for r,c in right_ponytails_part1)),
                                   max(r for r,c in right_ponytails_part1)+6, bend = 'right'
                                  )
    right_ponytails = right_ponytails_part1+right_ponytails_part2
    ponytails = left_ponytails+right_ponytails
    hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                   min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                   max(r for r,c in head), bend = 'right'
                                  )+cgd.verticle_line(max(r for r,c in hair_part1), 
                                   max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                   max(r for r,c in head_raw)+1, bend = 'left'
                                  )+[(head_center_row, head_center_col), (head_center_row, head_center_col+1)]
    hair = [(r,c) for r,c in hair_part1+hair_part2 if (r,c) not in neck]
    face = [(r,c) for r,c in head if (r,c) not in hair_part1+hair_part2]
    
    chocker = [(r,c) for r,c in neck if (r,c) not in head]
    top_region = [(r,c) for r,c in raw_body_modified if r>ln.breastbone_end_r and r<sk_row and (r,c) not in 
                 upperarm_dict["upperarm"]+forearm_dict["forearm"]+hand_dict["hand"]
                 ]
    
    top_frill = [(r,c) for r,c in
                 cg.hex_neighbours_n(min(r for r,c in top_region), min(c for r,c in top_region if r==min(r for r,c in top_region)),  
                                    n=1, keep_origin =False , return_frontier=True
                 )[-1]+cg.hex_neighbours_n(min(r for r,c in top_region), max(c for r,c in top_region if r==min(r for r,c in top_region)),  
                                    n=1, keep_origin =False , return_frontier=True
                 )[-1]  if r<=min(r for r,c in top_region)]
    
    clothing =top_region+sk
    apron = cgd.draw_trapezoid(sk_row-1, min(c for r,c in clothing if r==sk_row-1),min(c for r,c in clothing if r==sk_row-1)+2, 
                           sk_row+1, slope_left = '0.5', slope_right = '0.5', direction = 'll',bend_left = 'left', bend_right = 'right'
          )[-2]+cgd.draw_trapezoid(sk_row-1, max(c for r,c in clothing if r==sk_row-1)-2,max(c for r,c in clothing if r==sk_row-1), 
                           sk_row+1, slope_left = '0.5', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
              )[-2]
    sk_frill_layer1 = cg.hex_neighbours_n(sk_row-4, head_center_col,  n=7, keep_origin =False , return_frontier=True)[-1]
    sk_frill_layer2 = cg.hex_neighbours_n(sk_row-4, head_center_col,  n=10, keep_origin =False , return_frontier=True)[-1]
    sk_frill_layer3 = cg.hex_neighbours_n(sk_row-4, head_center_col,  n=13, keep_origin =False , return_frontier=True)[-1]
    
    sk_frill_raw = [(r,c) for r,c in sk_frill_layer1+sk_frill_layer2+sk_frill_layer3
                    if (r,c) in clothing]
    
    divide_layer2 = draw_divide(sk_row-4, head_center_col,
                              cg.hex_neighbours_n(sk_row-4, head_center_col,  n=12), n_divide=18)[0]
    
    sk_frill_raw+=[(r,c) for r,c in divide_layer2 if (r,c) in 
                     cg.hex_neighbours_n(sk_row-4, head_center_col,  n=9, keep_origin =False , return_frontier=True
                   )[-1]+cg.hex_neighbours_n(sk_row-4, head_center_col,  n=11, keep_origin =False , return_frontier=True
                   )[-1] and (r,c) in clothing]
    divide_layer3 = draw_divide(sk_row-4, head_center_col,
                              cg.hex_neighbours_n(sk_row-4, head_center_col,  n=15), n_divide=18
                    )[0]+draw_divide(sk_row-4, head_center_col,
                              cg.hex_neighbours_n(sk_row-4, head_center_col,  n=15), n_divide=12
                    )[0]
    
    sk_frill_raw+=[(r,c) for r,c in divide_layer3 if (r,c) in 
                     cg.hex_neighbours_n(sk_row-4, head_center_col,  n=14, keep_origin =False , return_frontier=True
                   )[-1]+cg.hex_neighbours_n(sk_row-4, head_center_col,  n=16, keep_origin =False , return_frontier=True
                   )[-1] and r>=max(r for r,c in clothing)]
    
    if version=='0':
        arm_cover = [(r,c) for r,c in forearm_dict["forearm"] if r>=ln.bellybutton_r ]

        left_arm_frill_loc = (max(r for r,c in arm_cover), min(c for r,c in arm_cover if r==max(r for r,c in arm_cover)))
        right_arm_frill_loc = (max(r for r,c in arm_cover), max(c for r,c in arm_cover if r==max(r for r,c in arm_cover)))

        arm_frill = cgd.draw_slope_0p5_diagonal(left_arm_frill_loc[0], left_arm_frill_loc[1], left_arm_frill_loc[0]+1, 
                                                     left_down=True, right_down=True, left_up=True, right_up=True
                        )+cgd.draw_slope_0p5_diagonal(right_arm_frill_loc[0], right_arm_frill_loc[1], right_arm_frill_loc[0]+1, 
                                                     left_down=True, right_down=True, left_up=True, right_up=True
                        )+[(left_arm_frill_loc[0], left_arm_frill_loc[1]-1), (right_arm_frill_loc[0], right_arm_frill_loc[1]+1)]

        gun_left = cgd.draw_slope_8p3_diagonal(max(r for r,c in hand_dict["left_hand"]), 
                                               min(c for r,c in hand_dict["left_hand"] if r== max(r for r,c in hand_dict["left_hand"]))+1,
                                               ln.gastrocnemius_end_r, left_down=True, right_down=False, left_up=False, right_up=False
                        )+cgd.draw_slope_8p3_diagonal(max(r for r,c in hand_dict["left_hand"]), 
                                               min(c for r,c in hand_dict["left_hand"] if r== max(r for r,c in hand_dict["left_hand"])),
                                               ln.gastrocnemius_end_r, left_down=True, right_down=False, left_up=False, right_up=False
                        )+cgd.draw_slope_8p3_diagonal(max(r for r,c in hand_dict["left_hand"]), 
                                               min(c for r,c in hand_dict["left_hand"] if r== max(r for r,c in hand_dict["left_hand"]))-1,
                                               ln.gastrocnemius_end_r, left_down=True, right_down=False, left_up=False, right_up=False
                        )
        gun_right = cgd.draw_slope_1p5_diagonal(max(r for r,c in hand_dict["right_hand"]), 
                                               max(c for r,c in hand_dict["right_hand"] if r== max(r for r,c in hand_dict["right_hand"]))-1,
                                               ln.knee_r, left_down=False, right_down=True, left_up=False, right_up=False
                        )+cgd.draw_slope_1p5_diagonal(max(r for r,c in hand_dict["right_hand"]), 
                                               max(c for r,c in hand_dict["right_hand"] if r== max(r for r,c in hand_dict["right_hand"])),
                                               ln.knee_r, left_down=False, right_down=True, left_up=False, right_up=False
                        )
        keys = ["gun_left", "gun_right", 
                "raw_body_modified", "clothing", "apron","arm_frill",  "arm_cover", 
                "hair","ponytails", "head_band", "head_band_frill", "face", "chocker", "top_frill", "sk_frill_raw", "leg_modified"]#
        colors = [[cgc.hex_to_rgb("#291f06")],[cgc.hex_to_rgb("#946c09")],

                  [cgc.hex_to_rgb("#f8d6b2")],  

                  [cgc.hex_to_rgb("#f66436")], [[0,0,0]],[cgc.hex_to_rgb("#f66436")],[[0,0,0]],


            [[0,0,0]],[[0,0,0]],[cgc.hex_to_rgb("#962400")], [cgc.hex_to_rgb("#f66436")], 
              [cgc.hex_to_rgb("#f8d6b2")], [[0,0,0]], [[0,0,0]], [[0,0,0]],[cgc.hex_to_rgb("#9f8b85"), [0,0,0], 'row'],[[0,0,0]],
                 ]

    elif version=='2':
        gun_left = []
        gun_right =[]
        arm_frill = []
        arm_cover = []
        cut_r = ln.elbow_r+1
        
        modified_left_arm_part1 = cgd.draw_slope_1p5_diagonal(cut_r, min(c for r,c in forearm_dict["forearm"] if r==cut_r), cut_r-3, 
                                                        left_down=False, right_down=False, left_up=True, right_up=False
                           )+cgd.draw_slope_1p5_diagonal(cut_r, min(c for r,c in forearm_dict["forearm"] if r==cut_r)-1, cut_r-2, 
                                                        left_down=False, right_down=False, left_up=True, right_up=False
                           )
        modified_left_arm_part2 = cgd.draw_slope_0p5_diagonal(min(r for r,c in modified_left_arm_part1), 
                                         min(c for r,c in modified_left_arm_part1 if r==min(r for r,c in modified_left_arm_part1)), 
                                         min(r for r,c in modified_left_arm_part1)-3, 
                                         left_down=False, right_down=False, left_up=False, right_up=True
                           )+cgd.verticle_line(min(r for r,c in modified_left_arm_part1),
                                          min(c for r,c in modified_left_arm_part1 if r==min(r for r,c in modified_left_arm_part1)), 
                                          min(r for r,c in modified_left_arm_part1)-3, bend = 'right')
        
        modified_left_arm = modified_left_arm_part1+modified_left_arm_part2
        
        arm_cover = [(r,c) for r,c in forearm_dict["right_forearm"] if r>=ln.bellybutton_r 
                    ]+[(r,c) for r,c in modified_left_arm_part1 if r>ln.breastbone_end_r]
        left_arm_frill_loc = (min(r for r,c in arm_cover), min(c for r,c in arm_cover if r==min(r for r,c in arm_cover)))
        right_arm_frill_loc = (max(r for r,c in arm_cover), max(c for r,c in arm_cover if r==max(r for r,c in arm_cover)))

        arm_frill = cgd.draw_slope_0p5_diagonal(left_arm_frill_loc[0], left_arm_frill_loc[1], left_arm_frill_loc[0]-1, 
                                                     left_down=True, right_down=True, left_up=True, right_up=True
                        )+cgd.draw_slope_0p5_diagonal(right_arm_frill_loc[0], right_arm_frill_loc[1], right_arm_frill_loc[0]+1, 
                                                     left_down=True, right_down=True, left_up=True, right_up=True
                        )+[(left_arm_frill_loc[0], left_arm_frill_loc[1]-1), (right_arm_frill_loc[0], right_arm_frill_loc[1]+1)]
        gun_left = cgd.draw_slope_1p5_diagonal(min(r for r,c in modified_left_arm)+1, 
                                              min(c for r,c in modified_left_arm if r==min(r for r,c in modified_left_arm)+1),
                                              min(r for r,c in modified_left_arm)-3, 
                                              left_down=False, right_down=False, left_up=False, right_up=True
                           )+cgd.draw_slope_1p5_diagonal(min(r for r,c in modified_left_arm)+1, 
                                              min(c for r,c in modified_left_arm if r==min(r for r,c in modified_left_arm)+1)-1,
                                              min(r for r,c in modified_left_arm)-3, 
                                              left_down=False, right_down=False, left_up=False, right_up=True
                           )
        
        gun_right = cgd.draw_slope_8p3_diagonal(max(r for r,c in hand_dict["right_hand"]), 
                                               max(c for r,c in hand_dict["right_hand"] if r== max(r for r,c in hand_dict["right_hand"]))-1,
                                               ln.gastrocnemius_end_r, left_down=False, right_down=True, left_up=False, right_up=False
                        )+cgd.draw_slope_8p3_diagonal(max(r for r,c in hand_dict["right_hand"]), 
                                               max(c for r,c in hand_dict["right_hand"] if r== max(r for r,c in hand_dict["right_hand"])),
                                               ln.gastrocnemius_end_r, left_down=False, right_down=True, left_up=False, right_up=False
                        )+cgd.draw_slope_8p3_diagonal(max(r for r,c in hand_dict["right_hand"]), 
                                               max(c for r,c in hand_dict["right_hand"] if r== max(r for r,c in hand_dict["right_hand"]))+1,
                                               ln.gastrocnemius_end_r, left_down=False, right_down=True, left_up=False, right_up=False
                        )
        
        remove_arm = [(r,c) for r,c in forearm_dict["left_forearm"]+hand_dict["left_hand"] if r>cut_r]
        raw_body_modified = [(r,c) for r,c in raw_body_modified if(r,c) not in remove_arm
                            
                            ]
        keys = [ "gun_right", 
                "raw_body_modified", "clothing", "apron",
                "hair","ponytails", "head_band", "head_band_frill", "face", "chocker", "top_frill", "sk_frill_raw", "leg_modified",
               "gun_left","modified_left_arm", "arm_frill",  "arm_cover", ]#
        colors = [[cgc.hex_to_rgb("#291f06")],

                  [cgc.hex_to_rgb("#f8d6b2")],  

                  [cgc.hex_to_rgb("#f66436")], [[0,0,0]],


            [[0,0,0]],[[0,0,0]],[cgc.hex_to_rgb("#962400")], [cgc.hex_to_rgb("#f66436")], 
              [cgc.hex_to_rgb("#f8d6b2")], [[0,0,0]], [[0,0,0]], [[0,0,0]],[cgc.hex_to_rgb("#9f8b85"), [0,0,0], 'row'],
               [cgc.hex_to_rgb("#946c09")],   [cgc.hex_to_rgb("#f8d6b2")],
                  
                  [cgc.hex_to_rgb("#f66436")],[[0,0,0]],
                 ]

        
    elif version=='1':
        modified_left_arm_part1= cg.hex_neighbours_n(ln.shoulder_joint_r, 
                                                     min(c for r,c in raw_body_modified if r==ln.shoulder_joint_r),  n=1)
        modified_left_arm_part2 = cgd.draw_block((min(r for r,c in modified_left_arm_part1), 
                                     (min(c for r,c in modified_left_arm_part1 if r==min(r for r,c in modified_left_arm_part1)), 
                                      max(c for r,c in modified_left_arm_part1 if r==min(r for r,c in modified_left_arm_part1))
                                     )), min(r for r,c in modified_left_arm_part1)-3)
                                
        modified_left_arm_part3 = cgd.verticle_line(min(r for r,c in modified_left_arm_part2),
                                          min(c for r,c in modified_left_arm_part2 if r==min(r for r,c in modified_left_arm_part2)),
                                          min(r for r,c in modified_left_arm_part2)-4, bend = 'right')
        modified_left_arm_part4 = cgd.draw_slope_0p5_diagonal(min(r for r,c in modified_left_arm_part3), 
                                          min(c for r,c in modified_left_arm_part3 if r==min(r for r,c in modified_left_arm_part3)), 
                                          min(r for r,c in modified_left_arm_part3)-3, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                )+cgd.draw_slope_0p5_diagonal_shift(min(r for r,c in modified_left_arm_part3), 
                                          min(c for r,c in modified_left_arm_part3 if r==min(r for r,c in modified_left_arm_part3)), 
                                          min(r for r,c in modified_left_arm_part3)-3, 2, shift_amount=1, 
                                          left_down=False, right_down=False, left_up=False, right_up=True)
        modified_left_arm = modified_left_arm_part1+modified_left_arm_part2+modified_left_arm_part3+modified_left_arm_part4
        
        modified_right_arm_part1= cgd.horizontal_lines([(ln.shoulder_joint_r+1, 
                                                         
                                                         (max(c for r,c in raw_body_modified if r==ln.shoulder_joint_r+1), 
                                                         max(c for r,c in raw_body_modified if r==ln.shoulder_joint_r+1)+9))])
        modified_right_arm_part2 = cgd.draw_slope_0p5_diagonal(ln.shoulder_joint_r+1, 
                                          max(c for r,c in modified_right_arm_part1 if r==ln.shoulder_joint_r+1), 
                                          ln.shoulder_joint_r-1, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                )+cgd.draw_slope_1p5_diagonal(ln.shoulder_joint_r+1, 
                                          max(c for r,c in modified_right_arm_part1 if r==ln.shoulder_joint_r+1), 
                                          ln.shoulder_joint_r, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                )
        modified_right_arm = modified_right_arm_part1+modified_right_arm_part2
        
        gun_left = cgd.draw_slope_0p5_diagonal_shift(min(r for r,c in modified_left_arm), 
                              min(c for r,c in modified_left_arm if r==min(r for r,c in modified_left_arm)),
                              min(r for r,c in modified_left_arm)-18, 3, shift_amount=-1, 
                              left_down=False, right_down=False, left_up=False, right_up=True)
        gun_right = cgd.horizontal_lines([(min(r for r,c in modified_right_arm), 
                                           (max(c for r,c in modified_right_arm if r==min(r for r,c in modified_right_arm)), 
                                           max(c for r,c in modified_right_arm if r==min(r for r,c in modified_right_arm))+6))])
        
        
        raw_body_modified = [(r,c) for r,c in raw_body_modified if(r,c) not in 
                             upperarm_dict["upperarm"]+forearm_dict["forearm"]+hand_dict["hand"]
                            
                            ]
        left_arm_cover = [(r,c) for r,c in modified_left_arm if r>=min(r for r,c in modified_left_arm_part3) 
                    and r<=min(r for r,c in modified_left_arm_part2)]
        right_arm_cover = [(r,c) for r,c in modified_right_arm if c>=max(c for r,c in raw_body_modified if r==ln.shoulder_joint_r+1)+3
                    and c<=max(c for r,c in raw_body_modified if r==ln.shoulder_joint_r+1)+7]
        arm_cover = left_arm_cover+right_arm_cover
        
        left_arm_frill_loc = (min(r for r,c in left_arm_cover), min(c for r,c in left_arm_cover if r==min(r for r,c in left_arm_cover)))
        right_arm_frill_loc = (min(r for r,c in right_arm_cover), max(c for r,c in right_arm_cover if r==min(r for r,c in right_arm_cover)))
        
        
        arm_frill = cgd.draw_slope_0p5_diagonal(left_arm_frill_loc[0], left_arm_frill_loc[1], left_arm_frill_loc[0]-1, 
                                                     left_down=True, right_down=True, left_up=True, right_up=True
                        )+cgd.draw_slope_0p5_diagonal(right_arm_frill_loc[0], right_arm_frill_loc[1], right_arm_frill_loc[0]-1, 
                                                     left_down=True, right_down=True, left_up=False, right_up=True
                        )+cgd.draw_slope_0p5_diagonal(right_arm_frill_loc[0], right_arm_frill_loc[1], right_arm_frill_loc[0]+1, 
                                                     left_down=False, right_down=True, left_up=False, right_up=True
                        )+[(left_arm_frill_loc[0], left_arm_frill_loc[1]+1), (right_arm_frill_loc[0], right_arm_frill_loc[1]+1)]
        keys = ["gun_left", "gun_right", 
                "raw_body_modified", "clothing", "apron",
                
                
                "hair","ponytails", "head_band", "head_band_frill", "face", "chocker", "top_frill", "sk_frill_raw", "leg_modified",
               
               "modified_left_arm","modified_right_arm", "arm_frill",  "arm_cover", ]#
        colors = [[cgc.hex_to_rgb("#291f06")],[cgc.hex_to_rgb("#946c09")],

                  [cgc.hex_to_rgb("#f8d6b2")],  

                  [cgc.hex_to_rgb("#f66436")], [[0,0,0]],
                  
                  

            [[0,0,0]],[[0,0,0]],[cgc.hex_to_rgb("#962400")], [cgc.hex_to_rgb("#f66436")], 
              [cgc.hex_to_rgb("#f8d6b2")], [[0,0,0]], [[0,0,0]], [[0,0,0]],[cgc.hex_to_rgb("#9f8b85"), [0,0,0], 'row'],
                  
                  [cgc.hex_to_rgb("#f8d6b2")], [cgc.hex_to_rgb("#f8d6b2")], [cgc.hex_to_rgb("#f66436")],[[0,0,0]],
                 ]

    
    local_vars = locals()
                
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})
n_clock = 26


marker_center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
marker_center_col = max(c for r, c in hex_rc_arr) // 2

light_region = [(r,c) for r,c in hex_rc_arr if (r<=marker_center_row and c<=marker_center_col) or (r>=marker_center_row and c>=marker_center_col)]

dark_region = [(r,c) for r,c in hex_rc_arr if (r<=marker_center_row and c>=marker_center_col) or (r>=marker_center_row and c<=marker_center_col)]


select_light_region= cg.select_mask(light_region,hex_rc_arr)
hex_colors[select_light_region] = cgc.select_normal_color(select_light_region, cgc.hex_to_rgb("#adb7fa"), np.ones(3)*sigma_color*3) 
select_dark_region= cg.select_mask(dark_region,hex_rc_arr)
hex_colors[select_dark_region] = cgc.select_normal_color(select_dark_region, cgc.hex_to_rgb("#2036c4"), np.ones(3)*sigma_color*2) 


char_dict_ = {
     "clock": draw_clock(marker_center_row-20, marker_center_col, n_clock),
'body v0': draw_char(marker_center_row+6, marker_center_col, '0'),
    'body v1': draw_char(marker_center_row-14, marker_center_col-35, '2'),
    'body v2': draw_char(marker_center_row-4, marker_center_col+25, '1')
    
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
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Kurumi_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')