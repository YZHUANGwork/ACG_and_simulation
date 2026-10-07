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
import SOLVE_LagrangianMultiplier as SOLN

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
sigma_color = 0.03
Sky_line = max(r for r,c in hex_rc_arr)//2

Sky= [(r,c) for r,c in hex_rc_arr if r <Sky_line]
hex_colors = cgc.color_row_gradient(Sky, cgc.hex_to_rgb("#ffffff"), cgc.hex_to_rgb("#0341aa"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.01)    


def f(x, y):
    """Single-peak mountain: summit at (0, 0), falling off in x and y."""
    return 6 * np.exp(-x**2 / 8 - y**2 / 170)

def g_switchback(x, y, A=1.6, k=np.pi / 3):
    """Switchback constraint: x - A*sin(k*y) = 0.
    Swings left/right in x as it descends in y."""
    return x - A * np.sin(k * y)


def g_parabola(x, y, x0=2.5, y_mid=9.0, A=(2.5 + 2.5) / 81.0):
    """Parabolic constraint: x - (x0 - A*(y - y_mid)**2) = 0.
    Bulges toward x0 at y = y_mid, curves back toward the opposite
    side at both the top and the bottom -- a "C" shape."""
    return x - (x0 - A * (y - y_mid) ** 2)


def g_line(x, y, m=-0.15, b=2.0):
    """Straight-line constraint: x - (m*y + b) = 0."""
    return x - (m * y + b)


def g_horizontal(x, y, c=5.0):
    """Horizontal-line constraint: y - c = 0."""
    return y - c

def draw_char(pivot, size, direction):
    head_center = (pivot[0]-(size*2*3+2), pivot[1])
    body_dict, plot_temp = cgf.draw_tiny_body_(head_center[0], head_center[1], size)
    
    bd = SimpleNamespace(**body_dict)
    raw_body = bd.raw_body
    kimono_part1= cgd.draw_trapezoid(bd.neck_row+1, max(c for r,c in bd.body if r==bd.neck_row+1)-1, 
                                     max(c for r,c in raw_body if r==bd.neck_row+1), bd.body_bottom_r-1, 
                                 slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                                 bend_left = 'left', bend_right = 'right')[-2]
    kimono_part2 = cgd.draw_trapezoid(bd.body_bottom_r, head_center[1], 
                                      max(c for r,c in raw_body if r==bd.body_bottom_r), 
                                      max(r for r,c in raw_body)-2, 
                                 slope_left = 'inf', slope_right = '0.5', direction = 'lr',
                                 bend_left = 'left', bend_right = 'right')[-2]
    kimono_part3 = cgd.draw_trapezoid(bd.body_bottom_r, head_center[1], 
                                      head_center[1]+2, 
                                      max(r for r,c in raw_body)-2, 
                                 slope_left = 'inf', slope_right = 'inf', direction = 'lr',
                                 bend_left = 'right', bend_right = 'left')[-2]
    if 'right' in direction:
        hair = [(r,c) for r,c in bd.head_raw if r <head_center[0] or c <pivot[1]]
    elif 'left' in direction:
        hair = [(r,c) for r,c in bd.head_raw if r <head_center[0] or c >pivot[1]]
    
    kimono = list(set(kimono_part1+[(r,c) for r,c in kimono_part2 if (r,c) in  bd.body+bd.pelvis+bd.leg]+kimono_part3))
    pattern_threshold_col = max(c for r,c in kimono_part1 if r == max(r for r,c in kimono_part1))-2
    pattern = [(r,c) for r,c in kimono_part1 if c >pattern_threshold_col and r == max(r for r,c in kimono_part1)
              ]+[(r,c) for r,c in kimono if r == max(r for r,c in kimono)]
    belt = [(r,c) for r,c in kimono_part1 if c <=pattern_threshold_col and r == max(r for r,c in kimono_part1)
             and (r,c) in raw_body ]
    
    skin = [(r,c) for r,c in bd.head_raw if  (r,c) not in hair
           ]+[(r,c) for r,c in bd.left_upper_arm if r>=bd.body_center_row
             ]+[(r,c) for r,c in bd.right_upper_arm if r>=bd.body_center_row and (r,c) not in kimono
             ]
    if 'right' in direction:
        
        if 'flat down' in direction:
            skateboard_raw = cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+3, 
                             left_down=False, right_down=True, left_up=False, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]-1, max(r for r,c in raw_body)+3, 
                             left_down=False, right_down=True, left_up=False, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]+1, max(r for r,c in raw_body)+2, 
                             left_down=False, right_down=True, left_up=False, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-2, 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )
        elif 'flat up' in direction:
            skateboard_raw = cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-3, 
                             left_down=False, right_down=False, left_up=False, right_up=True
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]-1, max(r for r,c in raw_body)-3, 
                             left_down=False, right_down=False, left_up=False, right_up=True
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]+1, max(r for r,c in raw_body)-2, 
                             left_down=False, right_down=False, left_up=False, right_up=True
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+2, 
                             left_down=True, right_down=False, left_up=False, right_up=False
                             )
        elif 'mid down' in direction:
            skateboard_raw = cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+4, 
                             left_down=False, right_down=True, left_up=False, right_up=False
                             )+cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-2, 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )
        elif 'mid up' in direction:
            skateboard_raw = cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-4, 
                             left_down=False, right_down=False, left_up=False, right_up=True
                             )+cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+2, 
                             left_down=True, right_down=False, left_up=False, right_up=False
                             )
        if 'up' in direction:
            skateboard_head_center = (min( r for r,c in skateboard_raw), 
                                      max(c for r,c in skateboard_raw if r== min(r for r,c in skateboard_raw)))
    
        elif 'down' in direction:
            skateboard_head_center = (max( r for r,c in skateboard_raw), 
                                      max(c for r,c in skateboard_raw if r== max(r for r,c in skateboard_raw)))
    
    elif 'left' in direction:
        if 'flat down' in direction:
            skateboard_raw = cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+3, 
                             left_down=True, right_down=False, left_up=False, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]+1, max(r for r,c in raw_body)+3, 
                             left_down=True, right_down=False, left_up=False, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]-1, max(r for r,c in raw_body)+2, 
                             left_down=True, right_down=False, left_up=False, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-2, 
                             left_down=False, right_down=False, left_up=False, right_up=True
                             )
        elif 'flat up' in direction:
            skateboard_raw = cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-3, 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]-1, max(r for r,c in raw_body)-3, 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1]+1, max(r for r,c in raw_body)-2, 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )+cgd.draw_slope_8p3_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+2, 
                             left_down=False, right_down=True, left_up=False, right_up=False
                             )
            
        elif 'mid down' in direction:
            skateboard_raw = cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+4, 
                             left_down=True, right_down=False, left_up=False, right_up=False
                             )+cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-2, 
                             left_down=False, right_down=False, left_up=False, right_up=True
                             )
        elif 'mid up' in direction:
            skateboard_raw = cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)-4, 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )+cgd.draw_slope_1p5_diagonal(max(r for r,c in raw_body), head_center[1], max(r for r,c in raw_body)+2, 
                             left_down=False, right_down=True, left_up=False, right_up=False
                             )
        if 'up' in direction:    
            skateboard_head_center = (min( r for r,c in skateboard_raw), 
                                  min(c for r,c in skateboard_raw if r== min(r for r,c in skateboard_raw)))
        elif 'down' in direction:
            skateboard_head_center = (max( r for r,c in skateboard_raw), 
                                  min(c for r,c in skateboard_raw if r== max(r for r,c in skateboard_raw)))
    
        
    skateboard_head = cg.hex_neighbours_n(skateboard_head_center[0],skateboard_head_center[1], n=size, keep_origin = True)
    
    if 'right' in direction:
        remove_skateboard_hair = cgd.draw_slope_0p5_diagonal(skateboard_head_center[0], 
                                                         skateboard_head_center[1]+1, 
                                                         skateboard_head_center[0]-1, 
                       left_down=False, right_down=False, left_up=True, right_up=False
                       )
        skateboard_hair = [(r,c) for r,c in skateboard_head if (r <=skateboard_head_center[0] or c <skateboard_head_center[1])
                      and (r,c) not in remove_skateboard_hair]
    elif 'left' in direction:
        remove_skateboard_hair = cgd.draw_slope_0p5_diagonal(skateboard_head_center[0], 
                                                         skateboard_head_center[1]-1, 
                                                         skateboard_head_center[0]-1, 
                       left_down=False, right_down=False, left_up=False, right_up=True
                       )
        skateboard_hair = [(r,c) for r,c in skateboard_head if (r <=skateboard_head_center[0] or c >skateboard_head_center[1])
                      and (r,c) not in remove_skateboard_hair]
    skateboard_face = [(r,c) for r,c in skateboard_head if (r,c) not in skateboard_hair]
    
    skateboard_body = []
    
    if 'flat down' in direction:
        threshold = max(r for r,c in raw_body)+2
    elif 'mid down' in direction:
        threshold = max(r for r,c in raw_body)+1
    elif 'flat up' in direction:
        threshold = max(r for r,c in raw_body)-2
    elif 'mid up' in direction:
        threshold = max(r for r,c in raw_body)-1
    
    for loc in skateboard_raw:
        if 'down' in direction:
            if loc[0]<=threshold:
                part = [(r,c) for r,c in cg.hex_neighbours_n(loc[0], loc[1], n=1, keep_origin = True, return_frontier=False) ]
            else:
                part = [(r,c) for r,c in cg.hex_neighbours_n(loc[0], loc[1], n=2, keep_origin = True, return_frontier=False) 
                       if r !=loc[0]-1]
        elif 'up' in direction:
            if loc[0]>=threshold:
                part = [(r,c) for r,c in cg.hex_neighbours_n(loc[0], loc[1], n=1, keep_origin = True, return_frontier=False) ]
            else:
                part = [(r,c) for r,c in cg.hex_neighbours_n(loc[0], loc[1], n=2, keep_origin = True, return_frontier=False) 
                       if r !=loc[0]-1]
        skateboard_body+=part
    
    if 'right' in direction:
        string_raw = cgd.draw_slope_0p5_diagonal(max(r for r,c in raw_body), head_center[1],
                             max(r for r,c in bd.left_upper_arm), 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )
    elif 'left' in direction:
        string_raw = cgd.draw_slope_0p5_diagonal(max(r for r,c in raw_body), head_center[1],
                             max(r for r,c in bd.left_upper_arm), 
                             left_down=False, right_down=False, left_up=True, right_up=False
                             )
    string = [(r,c) for r,c in string_raw if (r,c) not in    bd.upper_arm] 
    keys = ["skateboard_raw", "skateboard_body","skateboard_hair", "skateboard_face",
            "raw_body", "hair", "kimono", "skin", "pattern", "belt", "string"]
    colors = [[cgc.hex_to_rgb("#f8deb6")],[cgc.hex_to_rgb("#f8deb6")],[[0,0,0]],[cgc.hex_to_rgb("#f8deb6")], 
              
              [[0,0,0]],[[0.98, 0.98, 0.98], cgc.hex_to_rgb("#93dfea")],
              [[0.98, 0.98, 0.98], [0.9, 0.9, 0.9]], 
              [cgc.hex_to_rgb("#f8deb6")],[cgc.hex_to_rgb("#94f6d6")], [[0.2,0.2,0.2]],[[1,1,1]],[[0,1,0]]
              
             ]
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}


size = 2 

XLIM = (-4, 4)
YLIM = (0, 18)
ZMAX = f(0, 0) 

    
COLORMAP = [
    (0.0,  '#4a3518'),   # dark pine forest, base
    (0.1, '#2f1f04'),   # treeline
    (0.3, '#22280e'),
    (0.5, '#2a4f42'),# bare grey rock
    (0.75, '#546581'),   # rock dusted with snow
    (0.9,  '#a0b1ce'),   # snowfield
    (1.0,  '#ffffff'),   # summit snow
]



x_fill = np.linspace(*XLIM, 900)
z_env = f(x_fill, 0.0)
grid_points, grid_values = [], []
for xi, hi in zip(x_fill, z_env):
    z_col = np.linspace(0, hi, 150)
    grid_points.append(np.column_stack([np.full_like(z_col, xi), z_col]))
    grid_values.append(z_col)
grid_points = np.concatenate(grid_points)
grid_values = np.concatenate(grid_values)



PARAMS = [
          {"DOMAIN_W_SCALE": 1, "DOMAIN_H_SCALE": 0.8, "PIVOT_ROW": 2,"PIVOT_COL": max(c for r,c in hex_rc_arr)//5, 
           "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 2,
           'char_info':['right flat up',(16, max(c for r,c in hex_rc_arr)//2+5) ]},
         
          {"DOMAIN_W_SCALE": 1, "DOMAIN_H_SCALE": 1, "PIVOT_ROW": 5,"PIVOT_COL": max(c for r,c in hex_rc_arr)-10, 
           "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 4, 
           'char_info':['left flat down', (6, max(c for r,c in hex_rc_arr)-10)]},
        
          {"DOMAIN_W_SCALE": 2, "DOMAIN_H_SCALE": 1, "PIVOT_ROW": 18,"PIVOT_COL": max(c for r,c in hex_rc_arr)//3*2, 
                 "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 3, 
           'char_info':['right mid down', (40, max(c for r,c in hex_rc_arr)//3*2+10)]},

          {"DOMAIN_W_SCALE": 0.6, "DOMAIN_H_SCALE": 0.8 ,"PIVOT_ROW":13,"PIVOT_COL": max(c for r,c in hex_rc_arr)//2-20, 
                 "PIVOT_ROW_X": 0, "PIVOT_COL_Y": ZMAX, "gline": 1, 
           'char_info':['left mid down', (max(r for r,c in hex_rc_arr)//3*2, max(c for r,c in hex_rc_arr)//2-20)]},
         
         ]
grid_hex_rc_gpaths = []
glines_dict = {'1': g_switchback, '2': g_parabola, '3':g_line, '4':g_horizontal
              }

n_y_lines = 5

for PARAM in PARAMS:
    grid_hex_rc, DOMAIN_W, DOMAIN_H, CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H = cg.fn2grid(grid_points[:, 0], grid_points[:, 1], 
                      PARAM["DOMAIN_W_SCALE"] ,PARAM["DOMAIN_H_SCALE"], 
                      PARAM["PIVOT_ROW"], PARAM["PIVOT_COL"], PARAM["PIVOT_ROW_X"], PARAM["PIVOT_COL_Y"], detail_info, 
            square = False, EPS_DOMAIN = 1e-6, return_DOMAIN= True)    
    hex_positions, blended = cg.average_grid_to_hex_scene(grid_hex_rc, hex_rc_arr, grid_values, hex_colors,
                        colors=COLORMAP, alpha=1.0, vmax=ZMAX, weight_by_value=False)
    hex_colors[hex_positions] = blended

    for i in range(n_y_lines):
        y_val = i / (n_y_lines - 1) * YLIM[1]
        x_line = np.linspace(*XLIM, 140)
        z_line = f(x_line, y_val)

        grid_hex_rc_GREY = cg.fn2grid(x_line, z_line, 
                          PARAM["DOMAIN_W_SCALE"] ,PARAM["DOMAIN_H_SCALE"], 
                      PARAM["PIVOT_ROW"], PARAM["PIVOT_COL"], PARAM["PIVOT_ROW_X"], PARAM["PIVOT_COL_Y"], detail_info, 
                square = False, EPS_DOMAIN = 1e-6, input_PHYSICAL_size = [CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H])  


        mask = cg.select_mask(list(set(grid_hex_rc_GREY)), hex_rc_arr)
        hex_colors[mask] = cgc.select_normal_color(mask, cgc.hex_to_rgb('#b6aaff'), np.ones(3)*sigma_color*5) 

    
    g = PARAM["gline"]
    gline = glines_dict.get(str(g))
    model = SOLN.LagrangeMultiplier(f, gline)
    if g == 1:
        model.trace_path(x0=0.0, y0=0.0, s_max=19, n=600)
    elif g == 2:
        model.trace_path(x0=-2.5, y0=0.0, s_max=23, n=600)   # -x0 default, from g_parabola's signature
    elif g == 3:
        model.trace_path(x0=2.0, y0=0.0, s_max=18.5, n=600)   # b default, from g_line's signature
    elif g == 4:
        model.trace_path(x0=3.0, y0=5.0, s_max=6.0, n=600)   # y0 = c default, from
    
    critical_points = model.find_critical_points()
    
    grid_hex_rc_gpath = cg.fn2grid(model.path['x'], model.path['z'], 
                      PARAM["DOMAIN_W_SCALE"] ,PARAM["DOMAIN_H_SCALE"], 
                      PARAM["PIVOT_ROW"], PARAM["PIVOT_COL"], PARAM["PIVOT_ROW_X"], PARAM["PIVOT_COL_Y"], detail_info, 
            square = False, EPS_DOMAIN = 1e-6, input_PHYSICAL_size = [CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H])  
    select_path = cg.select_mask(grid_hex_rc_gpath, hex_rc_arr)
    hex_colors[select_path] = cgc.select_normal_color(select_path, cgc.hex_to_rgb('#533ce1'), np.ones(3)*sigma_color*5) 
    
    dict_ = draw_char(PARAM["char_info"][1], size,PARAM["char_info"][0])

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

'''




pivot = (max(r for r, c in hex_rc_arr) // 2, max(c for r, c in hex_rc_arr) // 2)
char_dict_ = {
        'right mid up': draw_char((13, max(c for r,c in hex_rc_arr)//3*2
                                  ), size, 'right mid up'),
#        'right mid down': draw_char((max(r for r, c in hex_rc_arr) //2, max(c for r, c in hex_rc_arr) //2), size, 'right mid down'),
        'left flat down': draw_char((10, max(c for r,c in hex_rc_arr)-10
                                    ), size, 'left flat down'),
  'left mid down': draw_char((20, max(c for r,c in hex_rc_arr)//2-20
                             ), size, 'left mid down'),
        'right flat down': draw_char((10,  max(c for r,c in hex_rc_arr)//5
                                     ), size, 'right flat down')
         
 #     'right flat up': draw_char((max(r for r, c in hex_rc_arr) // 3, max(c for r, c in hex_rc_arr) // 3*2), size, 'right flat up'),
  # 'left mid up': draw_char((max(r for r, c in hex_rc_arr) // 3*2, max(c for r, c in hex_rc_arr) // 3*2), size, 'left mid up'),
                
 #     'left flat up': draw_char((max(r for r, c in hex_rc_arr) // 3, max(c for r, c in hex_rc_arr) // 3*2), size, 'left flat up'),
                 }

for dict_ in char_dict_.values(): 
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
'''
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Gintama_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')