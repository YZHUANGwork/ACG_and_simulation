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

rng = np.random.default_rng(42)

DOC = 'pdf'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,_ = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)



sigma_color = 0.02

def draw_snail(center_r, center_c, n_shell, n_egg, view = 'side' ):
    odd_n_shells = np.arange(1,n_shell,2)
    even_n_shells = np.arange(0,n_shell,2)
    
    shells_odd = []
    shells_even = []
    for odd_n_shell in odd_n_shells:
        shells_odd+=cg.hex_neighbours_n(center_r,center_c, n=odd_n_shell, keep_origin = False, return_frontier=True)[-1]
    for even_n_shell in even_n_shells:
        shells_even+=cg.hex_neighbours_n(center_r,center_c, n=even_n_shell, keep_origin = False, return_frontier=True)[-1]
    
    if view == 'side' :
        body = cgd.draw_trapezoid(center_r+n_shell, center_c-n_shell, center_c+n_shell-2, center_r+n_shell+1, 
                                  slope_left = '8/3', slope_right = '1.5', direction = 'lr',
                      bend_left = 'left', bend_right = 'right')[-2]

        tentacle_part1 = cgd.draw_slope_0p5_diagonal(center_r+n_shell, max(c for r,c in body if r ==center_r+n_shell), 
                                               center_r+n_shell-4, left_down=False, right_down=False, left_up=False, right_up=True
                                              )+cgd.draw_slope_0p5_diagonal(center_r+n_shell, 
                                                                            max(c for r,c in body if r ==center_r+n_shell)+1, 
                                               center_r+n_shell-2, left_down=False, right_down=False, left_up=False, right_up=True
                                              )
        tentacle_part2 = cgd.horizontal_lines([(center_r+n_shell-2, (max(c for r,c in tentacle_part1 if r ==center_r+n_shell-2), 
                                                                    max(c for r,c in tentacle_part1 if r ==center_r+n_shell-2)+2))])
        tentacle = tentacle_part1+tentacle_part2

        snail_eggs_row = max(r for r, c in body)+n_egg+n_egg+n_egg
        snail_eggs_minc = min(c for r, c in body)-n_egg-n_egg-1
        snail_eggs_maxc = max(c for r, c in body)
        snail_eggs = []
        for col_egg in np.arange(snail_eggs_minc,snail_eggs_maxc, n_egg+n_egg+n_egg):
            snail_eggs+=cg.hex_neighbours_n(snail_eggs_row,col_egg, n=n_egg, keep_origin =False , return_frontier=True)[0]

        return {
            'a': [shells_odd, [[1,1,1]]],
            'b': [shells_even, [[0,0,0] ]],
            'c': [body, [[0,0,0]]],
            'd': [tentacle, [[0,0,0]]],
            'e': [snail_eggs, [[0,0,0]]],

            }
    elif view == 'top' :
        body_center_line = cgd.draw_slope_0p5_diagonal(center_r, center_c, center_r+n_shell+n_shell//2,
                                                       left_down=False, right_down=True , left_up=False, right_up=False
                                              )+cgd.draw_slope_0p5_diagonal(center_r, center_c, center_r-n_shell-n_shell//2-1,
                                                       left_down=False, right_down= False, left_up=True , right_up=False
                                              )
        body = cgd.draw_triangle(min(r for r,c in body_center_line), 
                                 min(c for r,c in body_center_line if r ==min(r for r,c in body_center_line)),
                                 center_r-n_shell, slope_left = '0.5', slope_right = '1.5', direction = 'rr', 
                 bend_left = 'right', bend_right = 'right')[-2]+cgd.draw_triangle(min(r for r,c in body_center_line), 
                                 min(c for r,c in body_center_line if r ==min(r for r,c in body_center_line)),
                                 center_r, slope_left = 'inf', slope_right = '0.5', direction = 'lr', 
                 bend_left = 'right', bend_right = 'right')[-2]+cgd.draw_trapezoid(max(r for r,c in body_center_line), 
                                 max(c for r,c in body_center_line if r ==max(r for r,c in body_center_line))-1, 
                                 max(c for r,c in body_center_line if r ==max(r for r,c in body_center_line)), 
                                                                                   center_r+n_shell, 
                                 slope_left = '1.5', slope_right = '0.5', direction = 'll',
                  bend_left = 'left', bend_right = 'right')[-2]+cgd.draw_trapezoid(max(r for r,c in body_center_line), 
                                 max(c for r,c in body_center_line if r ==max(r for r,c in body_center_line)), 
                                 max(c for r,c in body_center_line if r ==max(r for r,c in body_center_line)), 
                                                                                   center_r, 
                                 slope_left = '0.5', slope_right = 'inf', direction = 'll',
                  bend_left = 'left', bend_right = 'right')[-2]
        valid_body = [(r,c) for r,c in body if (r,c) not in shells_odd and (r,c) not in shells_even]
        
        tentacle = cgd.draw_slope_0p5_diagonal(max(r for r,c in valid_body), 
                                               min(c for r,c in valid_body if r ==max(r for r,c in valid_body)), 
                                               max(r for r,c in valid_body)+3, 
                                               left_down=False, right_down=True , left_up=False, right_up=False
                                              )+cgd.horizontal_lines([(max(r for r,c in valid_body)-1, (
        max(c for r,c in body_center_line if r ==max(r for r,c in valid_body)-1), 
        max(c for r,c in body_center_line if r ==max(r for r,c in valid_body)-1)+3))])
        

        return {
            'c': [body, [[0,0,0]]],
            'a': [shells_odd, [[1,1,1]]],
            'b': [shells_even, [[0,0,0] ]],
            
            'd': [tentacle, [[0,0,0]]],


            }
def draw_pplspiral(spiral_start_r, spiral_start_c, n_head = 1):
    spiral_bend_r = spiral_start_r+5
    spiral_center_line_part1 = cgd.draw_slope_0p5_diagonal(spiral_start_r, 
                                               spiral_start_c, 
                                               spiral_bend_r,  
                                               left_down=True , right_down= False, left_up=False, right_up=False
                                              )
    
    spiral_center_line_part2 = cgd.draw_slope_0p5_diagonal(spiral_bend_r, 
                                               min(c for r,c in spiral_center_line_part1 if r ==spiral_bend_r), 
                                               max(r for r,c in spiral_center_line_part1)+5,  
                                               left_down= False, right_down=True , left_up=False, right_up=False
                                              )
    
    spiral_center_line = spiral_center_line_part1+spiral_center_line_part2
    spiral_radius = 1
    spiral_diameter = spiral_radius+spiral_radius
    
    neck_white = cgd.verticle_line(spiral_start_r,spiral_start_c,spiral_start_r-4, bend = 'left'
                                       )

    neck_black = cgd.draw_slope_1p5_diagonal(spiral_start_r, spiral_start_c,spiral_start_r-2,
                                            left_down= False, right_down= False, left_up=False , right_up=True )
    
    head_center_white = (min(r for r,c in neck_white), min(c for r,c in neck_white if r ==min(r for r,c in neck_white)))
    head_center_black = (min(r for r,c in neck_black), max(c for r,c in neck_black if r ==min(r for r,c in neck_black)))
    
    head_white = cg.hex_neighbours_n(head_center_white[0],head_center_white[1], 
                                     n=n_head, keep_origin =False , return_frontier=True)[-1]
    
    head_black = cg.hex_neighbours_n(head_center_black[0],head_center_black[1], 
                                     n=n_head, keep_origin =False , return_frontier=True)[-1]
    

    
    row_seg1 = spiral_diameter+spiral_radius
    
    spiral_black_part1 = cgd.horizontal_lines([(spiral_start_r, (spiral_start_c-spiral_radius, spiral_start_c+spiral_radius))]
                                       )+cgd.draw_slope_0p5_diagonal(spiral_start_r, 
                                               spiral_start_c-spiral_radius, 
                                               spiral_start_r+1,  
                                               left_down=True  , right_down=False , left_up=False, right_up=False
                                       )
    
    spiral_black_part2 = cgd.draw_slope_0p5_diagonal(spiral_start_r+row_seg1, 
                                               max(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1)+1, 
                                               spiral_start_r+row_seg1-1,  
                                               left_down= False , right_down= False, left_up=False, right_up=True 
                                       )+cgd.draw_slope_0p5_diagonal(spiral_start_r+row_seg1, 
                                               min(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1)-1, 
                                               spiral_start_r+row_seg1+1,  
                                               left_down= True  , right_down= False, left_up=False, right_up=False 
                                       )+cgd.horizontal_lines([(spiral_start_r+row_seg1, 
                                               (min(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1)-1, 
                                                max(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1)+1))]
                                       )

    
    spiral_black_part3 = cgd.draw_slope_0p5_diagonal(spiral_bend_r, 
                                               max(c for r,c in spiral_center_line if r ==spiral_bend_r)+1, spiral_bend_r+spiral_diameter,  
                                               left_down=True  , right_down= False, left_up=False, right_up= False
                                       )
    
    row_seg2 = spiral_diameter+spiral_diameter
    spiral_black_part4 =cgd.verticle_line(spiral_bend_r+row_seg2, 
                                               max(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg2)+1,
                                               spiral_start_r+row_seg2+3, bend = 'left'
                                       )
    
    spiral_black_part5 = cgd.horizontal_lines([(max(r for r,c in spiral_center_line), 
                                               (max(c for r,c in spiral_center_line if r ==max(r for r,c in spiral_center_line))-5, 
                                                max(c for r,c in spiral_center_line if r ==max(r for r,c in spiral_center_line))-2))]
                                       )
    spiral_black = spiral_black_part1+spiral_black_part2+spiral_black_part3+spiral_black_part4+spiral_black_part5
    
    
    
    spiral_white_part1 = cgd.horizontal_lines([(spiral_start_r+spiral_radius, 
                                                (max(c for r,c in spiral_center_line if r ==spiral_start_r+spiral_radius), 
                                                 max(c for r,c in spiral_center_line if r ==spiral_start_r+spiral_radius)+spiral_radius))]
                                       )+cgd.draw_slope_0p5_diagonal(spiral_start_r+spiral_radius, 
                                               max(c for r,c in spiral_center_line if r ==spiral_start_r+spiral_radius), 
                                               spiral_start_r+spiral_radius+1,  
                                               left_down=True  , right_down= False, left_up=False, right_up= False
                                       )
    

    
    spiral_white_part2 = cgd.draw_slope_0p5_diagonal(spiral_start_r+row_seg1+spiral_radius, 
                                               max(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1+spiral_radius), 
                                               spiral_start_r+row_seg1+spiral_radius+2,  
                                               left_down=True  , right_down= False, left_up=False, right_up=False
                                       )+cgd.horizontal_lines([(spiral_start_r+row_seg1+spiral_radius, 
                                         (max(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1+spiral_radius),
                                          max(c for r,c in spiral_center_line if r ==spiral_start_r+row_seg1+spiral_radius)+1))]
                                       )
    spiral_white_part3 = cgd.draw_slope_0p5_diagonal(spiral_bend_r+spiral_diameter, 
                                               max(c for r,c in spiral_center_line if r ==spiral_bend_r+spiral_diameter), 
                                               spiral_bend_r+spiral_diameter-1,  
                                               left_down= False , right_down= False, left_up=False, right_up=True 
                                       )
    
    spiral_white_part4 = cgd.draw_slope_0p5_diagonal(max(r for r,c in spiral_center_line), 
                                               max(c for r,c in spiral_center_line if r ==max(r for r,c in spiral_center_line))-1, 
                                               max(r for r,c in spiral_center_line)-1,  
                                               left_down= False , right_down= False, left_up=True , right_up= False
                                       )
    spiral_white_part5 = cgd.draw_slope_1p5_diagonal(max(r for r,c in spiral_white_part4), 
                                               max(c for r,c in spiral_white_part4 if r ==max(r for r,c in spiral_white_part4)), 
                                               max(r for r,c in spiral_white_part4)+1,  
                                               left_down=True  , right_down= False, left_up= False, right_up= False
                                       )
    spiral_white = spiral_white_part1+spiral_white_part2+spiral_white_part3+spiral_white_part4#+spiral_white_part5
    
    return {
        'f': [head_white, [[0.7,0.7,0.7]]],
            'g': [head_black, [[0,0,0]]],
            'e': [neck_white, [[0.7,0.7,0.7]]],
            'a': [neck_black, [[0,0,0]]],
'h': [[head_center_black], [[0.7,0.7,0.7]]],
            'i': [[head_center_white], [[0,0,0]]],
         

#                'h': [spiral_center_line, [[0.5,1,0.5]]],
                'd':   [spiral_white, [[0.7,0.7,0.7]]],
        'c':   [spiral_black, [[0,0,0]]],
            }
def draw_uzi(center_r, center_c, n):
    odd_ns = np.arange(1,n,2)
    even_ns = np.arange(0,n,2)
    
    shells_odd = []
    shells_even = []
    for odd_n in odd_ns:
        shells_odd+=cg.hex_neighbours_n(center_r,center_c, n=odd_n, keep_origin = False, return_frontier=True)[-1]
    for even_n in even_ns:
        shells_even+=cg.hex_neighbours_n(center_r,center_c, n=even_n, keep_origin = False, return_frontier=True)[-1]
    return {
            
            'a': [shells_odd, [[0,0,0]]],
            'b': [shells_even, [[1,1,1] ]],
           
            }

def draw_char(start_center_row, start_center_col, n_head, start = 'head'):
    head, neck, body, pelvis, thigh, details = cgd.draw_body(start_center_row, start_center_col, n = n_head, start = start)
    [center_col, head_center_row, body_center_row, thigh_center_row]  = details
    
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
    
    char = face+neck+body+left_shoulders+right_shoulders
    
    uzi_center_front = (head_center_row, center_col)
    uzi_center_back = (head_center_row, max(c for r,c in hair if r ==head_center_row))
    n_uzi = 3
    odd_ns = np.arange(1,n_uzi,2)
    shells_odd_front = []
    shells_odd_back = []
    for odd_n in odd_ns:
        shells_odd_front+=cg.hex_neighbours_n(uzi_center_front[0],uzi_center_front[1],
                                              n=odd_n, keep_origin = False, return_frontier=True)[-1]
    for odd_n in np.arange(8,11,2):
        shells_odd_back+=cg.hex_neighbours_n(uzi_center_back[0],uzi_center_back[1], 
                                             n=odd_n, keep_origin = False, return_frontier=True)[-1]
    
    return {
            'd': [shells_odd_back, [[1,1,1]]],
            'g': [hair, [[1,1,1]]],
           'a': [char, [[0.7,0.7,0.7]]],
            'c': [shells_odd_front, [[0,0,0]]],
#            'b': [shells_even, [[0.9, 0.9, 0.9] ]],
           
            }

n_shell = 5
n_egg = 1
snail_shell_side_row = 12
snail_shell_side_col = 8

snail_shell_top_row = 5
snail_shell_top_col = 14

spiral_center = (6,3)
n_head = 3
head_center = (15, 27)
char_dict_ = {'snail_side': draw_snail(snail_shell_side_row, snail_shell_side_col, n_shell, n_egg, view = 'side'),
              'snail_top': draw_snail(snail_shell_top_row, snail_shell_top_col, n_shell, n_egg, view = 'top'),
              'pplspiral': draw_pplspiral(spiral_center[0], spiral_center[1], n_head = 1),
              'ppl': draw_char(head_center[0], head_center[1], n_head, start = 'head'),
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



pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Uzi_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')