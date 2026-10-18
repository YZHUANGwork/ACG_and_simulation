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
    IMG_W=1280, IMG_H=720, HEX_R=12, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.03

rng = np.random.default_rng(seed=0)

def draw_marker(pivot, n_size, version):
    
    shape = cg.hex_neighbours_n(pivot[0], pivot[1], n=n_size, keep_origin = False, return_frontier=True
                                       )[-1]
    shape_center = cg.hex_neighbours_n(pivot[0], pivot[1], n=1, keep_origin = False, return_frontier=True
                                       )[-1]
    shape+=shape_center
    numbers_black = [] 
    black_locs_part1 = [(min(r for r,c in shape), pivot[1]+math.floor(n_size/2/2)),
                  (min(r for r,c in shape), pivot[1]-math.ceil(n_size/2/2)),
                  (max(r for r,c in shape), pivot[1]+math.floor(n_size/2/2)),
                  (max(r for r,c in shape), pivot[1]-math.ceil(n_size/2/2)),]
    
    black_locs_part2 = [              
                  (min(r for r,c in shape)+math.ceil(n_size/2/2), 
                   min(c for r,c in shape if r==min(r for r,c in shape)+math.ceil(n_size/2/2))),
                  (min(r for r,c in shape)+math.ceil(n_size/2/2), 
                   max(c for r,c in shape if r==min(r for r,c in shape)+math.ceil(n_size/2/2))),
                  
                  (pivot[0]-math.floor(n_size/2/2), 
                   min(c for r,c in shape if r==pivot[0]-math.floor(n_size/2/2))),
                  (pivot[0]-math.floor(n_size/2/2), 
                   max(c for r,c in shape if r==pivot[0]-math.floor(n_size/2/2))),
        
                  (max(r for r,c in shape)-math.floor(n_size/2/2), 
                   min(c for r,c in shape if r==max(r for r,c in shape)-math.floor(n_size/2/2))),
                  (max(r for r,c in shape)-math.floor(n_size/2/2), 
                   max(c for r,c in shape if r==max(r for r,c in shape)-math.floor(n_size/2/2))),
                  
                  (pivot[0]+math.ceil(n_size/2/2), 
                   min(c for r,c in shape if r==pivot[0]+math.ceil(n_size/2/2))),
                  (pivot[0]+math.ceil(n_size/2/2), 
                   max(c for r,c in shape if r==pivot[0]+math.ceil(n_size/2/2))),
        
        
                 ]
    
    black_locs_part3 = [              
                  (min(r for r,c in shape)+math.floor(n_size/2/2), 
                   min(c for r,c in shape if r==min(r for r,c in shape)+math.floor(n_size/2/2))),
                  (min(r for r,c in shape)+math.floor(n_size/2/2), 
                   max(c for r,c in shape if r==min(r for r,c in shape)+math.floor(n_size/2/2))),
                  
                  (pivot[0]-math.ceil(n_size/2/2), 
                   min(c for r,c in shape if r==pivot[0]-math.ceil(n_size/2/2))),
                  (pivot[0]-math.ceil(n_size/2/2), 
                   max(c for r,c in shape if r==pivot[0]-math.ceil(n_size/2/2))),
        
                  (max(r for r,c in shape)-math.ceil(n_size/2/2), 
                   min(c for r,c in shape if r==max(r for r,c in shape)-math.ceil(n_size/2/2))),
                  (max(r for r,c in shape)-math.ceil(n_size/2/2), 
                   max(c for r,c in shape if r==max(r for r,c in shape)-math.ceil(n_size/2/2))),
                  
                  (pivot[0]+math.floor(n_size/2/2), 
                   min(c for r,c in shape if r==pivot[0]+math.floor(n_size/2/2))),
                  (pivot[0]+math.floor(n_size/2/2), 
                   max(c for r,c in shape if r==pivot[0]+math.floor(n_size/2/2))),
        
        
        ]
    
    black_locs = black_locs_part1+black_locs_part2+black_locs_part3
    for black_loc in black_locs_part1:
        number = cgd.verticle_line(black_loc[0], black_loc[1], black_loc[0]-1, bend = 'right'
                                      )+cgd.verticle_line(black_loc[0], black_loc[1], black_loc[0]+1, bend = 'right')
        numbers_black+=number
            
    for black_loc in black_locs_part2:
        if black_loc[1]<pivot[1]: 
            if black_loc[0]<pivot[0]:
                bend = 'left'
            else:
                bend = 'right'
        else:
            if black_loc[0]<pivot[0]:
                bend = 'right'
            else:
                bend = 'left'
        number = cgd.verticle_line(black_loc[0], black_loc[1], black_loc[0]-1, bend = bend)
        numbers_black+=number
            
    for black_loc in black_locs_part3:
        if black_loc[1]<pivot[1]: 
            if black_loc[0]<pivot[0]:
                bend = 'right'
            else:
                bend = 'left'
        else:
            if black_loc[0]<pivot[0]:
                bend = 'left'
            else:
                bend = 'right'
        number = cgd.verticle_line(black_loc[0], black_loc[1], black_loc[0]+1, bend = bend)
        numbers_black+=number
               
    white_locs = [(min(r for r,c in shape), pivot[1]),
                (min(r for r,c in shape), min(c for r,c in shape if r==min(r for r,c in shape))),
                (min(r for r,c in shape), max(c for r,c in shape if r==min(r for r,c in shape))),
                  ((min(r for r,c in shape)+pivot[0])//2, min(c for r,c in shape if r==(min(r for r,c in shape)+pivot[0])//2)),
                  ((min(r for r,c in shape)+pivot[0])//2, max(c for r,c in shape if r==(min(r for r,c in shape)+pivot[0])//2)),
                  
                  (pivot[0], min(c for r,c in shape if r==pivot[0])),
                  (pivot[0], max(c for r,c in shape if r==pivot[0])),
                (max(r for r,c in shape), pivot[1]),
                (max(r for r,c in shape), min(c for r,c in shape if r==max(r for r,c in shape))),
                (max(r for r,c in shape), max(c for r,c in shape if r==max(r for r,c in shape))),
                ((max(r for r,c in shape)+pivot[0])//2, min(c for r,c in shape if r==(max(r for r,c in shape)+pivot[0])//2)),
                ((max(r for r,c in shape)+pivot[0])//2, max(c for r,c in shape if r==(max(r for r,c in shape)+pivot[0])//2)),
                
                 ]
    numbers_white = []
    for white_loc in white_locs:
        number=cg.hex_neighbours_n(white_loc[0], white_loc[1], n=1, keep_origin = False, return_frontier=True
                                       )[-1]
        numbers_white+=number
        if white_loc[1]==pivot[1] :
            if white_loc[0] == min(r for r,c in shape):
                extra = cgd.draw_block((min(r for r,c in number), (min(c for r,c in number if r==min(r for r,c in number)),
                                                                  max(c for r,c in number if r==min(r for r,c in number)))), 
                                       min(r for r,c in number)-3)
            else:
                extra = cgd.draw_block((max(r for r,c in number), (min(c for r,c in number if r==max(r for r,c in number)),
                                                                  max(c for r,c in number if r==max(r for r,c in number)))), 
                                       max(r for r,c in number)+3)
            numbers_white+=extra
        elif white_loc[0]==pivot[0]:
            if white_loc[1] == min(c for r,c in shape):
                extra = cgd.horizontal_lines([(white_loc[0], (min(c for r,c in number if r==white_loc[0])-2,
                                                             min(c for r,c in number if r==white_loc[0]))
                                              )])
            else:
                extra = cgd.horizontal_lines([(white_loc[0], (max(c for r,c in number if r==white_loc[0]),
                                                             max(c for r,c in number if r==white_loc[0])+2)
                                              )])
            numbers_white+=extra
        elif white_loc[0]==min(r for r,c in shape):
            if white_loc[1]< pivot[1]:
                left_up , right_up = True, False
            else:
                left_up , right_up =False, True
            extra = cgd.draw_slope_0p5_diagonal(white_loc[0], white_loc[1], white_loc[0]-2, 
                                                left_down=False, right_down=False, left_up=left_up, right_up=right_up)
            numbers_white+=extra
        elif white_loc[0]==max(r for r,c in shape):
            if white_loc[1]< pivot[1]:
                left_down , right_down = True, False
            else:
                left_down , right_down =False , True
            extra = cgd.draw_slope_0p5_diagonal(white_loc[0], white_loc[1], white_loc[0]+2, 
                                                left_down=left_down, right_down=right_down, left_up=False, right_up=False)
            numbers_white+=extra
        elif white_loc[0]< pivot[0]:
            if white_loc[1]< pivot[1]:
                left_up , right_up = True, False
            else:
                left_up , right_up =False, True
            extra = cgd.draw_slope_1p5_diagonal(white_loc[0], white_loc[1], white_loc[0]-1, 
                                                left_down=False, right_down=False, left_up=left_up, right_up=right_up)
            numbers_white+=extra
        elif white_loc[0]>pivot[0]:
            if white_loc[1]< pivot[1]:
                left_down , right_down = True, False
            else:
                left_down , right_down =False , True
            extra = cgd.draw_slope_1p5_diagonal(white_loc[0], white_loc[1], white_loc[0]+1, 
                                                left_down=left_down, right_down=right_down, left_up=False, right_up=False)
            numbers_white+=extra
    
    
    shape_inner_raw = cg.hex_neighbours_n(pivot[0], pivot[1], n=n_size-3, keep_origin = True)
    shape_inner_boundary =  cg.hex_neighbours_n(pivot[0], pivot[1], n=n_size-3, keep_origin = False, return_frontier=True)[-1]
    
    lines_boundary  = cgd.horizontal_lines([(min(r for r,c in shape_inner_boundary), 
                                     (min(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_raw))+2,
                                      min(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_raw))+math.ceil(n_size/2/2)
                                     )),
                                   (min(r for r,c in shape_inner_boundary), 
                                     (max(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_raw))-math.ceil(n_size/2/2),
                                      max(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_raw))-2
                                     )),
                                   
                                   (max(r for r,c in shape_inner_boundary), 
                                     (min(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_raw)),
                                      min(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_raw))+math.floor(n_size/2/3)
                                     )),
                                   
                                   (max(r for r,c in shape_inner_boundary), 
                                     (max(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_raw))-math.floor(n_size/2/3),
                                      max(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_raw))
                                     )),]
             )+cgd.verticle_line(pivot[0]-math.floor(n_size/2/2), 
                                 min(c for r,c in shape_inner_boundary if r==pivot[0]-math.floor(n_size/2/2)), 
                                 pivot[0]+math.floor(n_size/2/2), bend = 'left'
             )+cgd.draw_slope_0p5_diagonal(min(r for r,c in shape_inner_boundary)+1, 
                                           min(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_boundary)+1), 
                                           min(r for r,c in shape_inner_boundary)+1+math.ceil(n_size/2/2), 
                                           left_down=True, right_down=False, left_up=False, right_up=False
            )+cgd.draw_slope_0p5_diagonal(pivot[0], min(c for r,c in shape_inner_boundary if r==pivot[0]), 
                                          pivot[0]-math.floor(n_size/2/2), left_down=False, right_down=False, left_up=False, right_up=True
            )+cgd.draw_slope_0p5_diagonal(pivot[0], min(c for r,c in shape_inner_boundary if r==pivot[0]), 
                                          pivot[0]+math.floor(n_size/2/2), left_down=False, right_down=True, left_up=False, right_up=False
            )+cgd.draw_slope_0p5_diagonal(max(r for r,c in shape_inner_boundary), 
                                           min(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_boundary)), 
                                           max(r for r,c in shape_inner_boundary)-math.floor(n_size/2/2), 
                                           left_down=False, right_down=False, left_up=True, right_up=False
            )+cgd.draw_slope_0p5_diagonal(max(r for r,c in shape_inner_boundary), 
                                           max(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_boundary)), 
                                           pivot[0]+math.floor(n_size/2/2), 
                                           left_down=False, right_down=False, left_up=False, right_up=True
            )+cgd.draw_slope_0p5_diagonal(pivot[0], max(c for r,c in shape_inner_boundary if r==pivot[0]), 
                                          pivot[0]-math.ceil(n_size/2), left_down=False, right_down=False, left_up=True, right_up=False
            )
    circle_left_raw = cg.hex_neighbours_n(pivot[0]-math.ceil(n_size/3), 
                                      min(c for r,c in shape_inner_boundary if r==pivot[0]-math.ceil(n_size/3))+2,n=2, keep_origin = False, return_frontier=True)[-1]
    circle_left = [(r,c) for r,c in circle_left_raw if c>=min(c for r,c in shape_inner_boundary if r==pivot[0]-math.ceil(n_size/3))+2]           
    circle_left_bot_raw = cg.hex_neighbours_n(pivot[0]+math.ceil(n_size/3), 
                                      min(c for r,c in shape_inner_boundary if r==pivot[0]+math.ceil(n_size/3))-1,n=3, keep_origin = False, return_frontier=True)[-1]
    circle_left_bot_1 = [(r,c) for r,c in circle_left_bot_raw if (r,c) in shape_inner_raw]           
    circle_left_bot_2_raw = cg.hex_neighbours_n(min(r for r,c in circle_left_bot_1), 
                                      max(c for r,c in circle_left_bot_1 if r==min(r for r,c in circle_left_bot_1))+1,n=1, keep_origin = False, return_frontier=True)[-1]
    circle_left_bot_2 = [(r,c) for r,c in circle_left_bot_2_raw 
                         if (r,c) not in [ (min(r for r,c in circle_left_bot_2_raw), 
                                            min(c for r,c in circle_left_bot_2_raw if r==min(r for r,c in circle_left_bot_2_raw)
                                               ))
                                            ]]
    circle_left_bot_3 = cg.hex_neighbours_n(pivot[0]+math.ceil(n_size/3)-1, 
                                      max(c for r,c in circle_left_bot_1 if r==pivot[0]+math.ceil(n_size/3)-1)+1,n=1)
    circle_left_bot = circle_left_bot_1+circle_left_bot_2+circle_left_bot_3
    
    circle_center = cg.hex_neighbours_n(pivot[0], pivot[1],n=1, keep_origin = False, return_frontier=True)[-1]
      
    circle_5 = [(r,c) for r,c in  cg.hex_neighbours_n(max(r for r,c in circle_center), 
                         min(c for r,c in circle_center if r==max(r for r,c in circle_center))-2,n=2,
                                   keep_origin = False, return_frontier=True)[-1] if c<=min(c for r,c in circle_center if r==min(r for r,c in circle_center))-2]
    circle_6 = [(r,c) for r,c in  cg.hex_neighbours_n(min(r for r,c in circle_5), 
                         min(c for r,c in circle_5 if r==min(r for r,c in circle_5))-2,n=2,
                         keep_origin = False, return_frontier=True)[-1] if r>min(r for r,c in circle_5)]
    
    lines_2  = cgd.draw_triangle(pivot[0]+2, pivot[1], pivot[0]+2+2, 
                           slope_left = '0.5', slope_right = '0.5', direction = 'lr', 
                           bend_left = 'left', bend_right = 'right'
                           )[0]+cgd.draw_triangle(pivot[0]+2, pivot[1], pivot[0]+2+2, 
                           slope_left = '0.5', slope_right = '0.5', direction = 'lr', 
                           bend_left = 'left', bend_right = 'right'
                           )[1]+cgd.draw_triangle(pivot[0]+2+2+2, pivot[1], pivot[0]+2+2, 
                           slope_left = '0.5', slope_right = '0.5', direction = 'lr', 
                           bend_left = 'left', bend_right = 'right'
                           )[0]+cgd.draw_triangle(pivot[0]+2+2+2, pivot[1], pivot[0]+2+2, 
                           slope_left = '0.5', slope_right = '0.5', direction = 'lr', 
                           bend_left = 'left', bend_right = 'right'
                           )[1]
    
    lines_3_part1 = cgd.draw_slope_1p5_diagonal(pivot[0], pivot[1], pivot[0]-3, 
                                 left_down=False, right_down=False, left_up=True, right_up=True
                                 )+cgd.draw_slope_1p5_diagonal(pivot[0], pivot[1]+1, pivot[0]-3, 
                                 left_down=False, right_down=False, left_up=True, right_up=True
                                 )+cgd.draw_slope_1p5_diagonal(pivot[0], pivot[1]-1, pivot[0]-3, 
                                 left_down=False, right_down=False, left_up=True, right_up=True
                                 )
    
    circle_midright = [(r,c) for r,c in  cg.hex_neighbours_n(pivot[0]-1, 
                         max(c for r,c in lines_3_part1 if r==pivot[0]-1)+2,n=2,
                         keep_origin = False, return_frontier=True)[-1]  if c > max(c for r,c in lines_3_part1 if r==pivot[0]-1)+1
                      ]
    lines_right_2 = cgd.draw_slope_0p5_diagonal(max(r for r,c in circle_midright), 
                             min(c for r,c in circle_midright if r==max(r for r,c in circle_midright))-1, 
                             max(r for r,c in circle_midright)+2, left_down=True, right_down=False, left_up=False, right_up=False
                    )+ cgd.draw_slope_0p5_diagonal(max(r for r,c in circle_midright), 
                             min(c for r,c in circle_midright if r==max(r for r,c in circle_midright))-1, 
                             max(r for r,c in circle_midright)-1, left_down=False, right_down=False, left_up=False, right_up=True
                    )
    circle_8 = [(r,c) for r,c in cg.hex_neighbours_n(max(r for r,c in lines_right_2), 
                               max(c for r,c in lines_right_2 if r==max(r for r,c in lines_right_2))+2, 
                               n=2, keep_origin = False, return_frontier=True)[-1] if r!=max(r for r,c in lines_right_2)+1]
    
    circle4_raw_left = cgd.draw_slope_0p5_diagonal(min(r for r,c in shape_inner_boundary), 
                                           min(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_boundary)), 
                                           min(r for r,c in shape_inner_boundary)+math.floor(n_size/2/3), 
                                           left_down=False, right_down=True, left_up=False, right_up=False)
    circle4_raw_right = cgd.draw_slope_0p5_diagonal(min(r for r,c in shape_inner_boundary), 
                                           max(c for r,c in shape_inner_boundary if r==min(r for r,c in shape_inner_boundary)), 
                                           min(r for r,c in shape_inner_boundary)+math.floor(n_size/2/3), 
                                           left_down=True, right_down=False, left_up=False, right_up=False)
    
    circle4_left = [(r,c) for r,c in cg.hex_neighbours_n(max(r for r,c in circle4_raw_left), 
                               max(c for r,c in circle4_raw_left if r==max(r for r,c in circle4_raw_left)), 
                               n=2, keep_origin = False, return_frontier=True)[-1] if r!=min(r for r,c in shape_inner_boundary)]
    circle4_right = [(r,c) for r,c in cg.hex_neighbours_n(max(r for r,c in circle4_raw_right), 
                               max(c for r,c in circle4_raw_right if r==max(r for r,c in circle4_raw_right)), 
                               n=2, keep_origin = False, return_frontier=True)[-1] if r!=min(r for r,c in shape_inner_boundary)]
    circle_top =  [(r,c) for r,c in cg.hex_neighbours_n(min(r for r,c in shape_inner_boundary)+1, pivot[1], 
                               n=2, keep_origin = False, return_frontier=True)[-1] if r>=min(r for r,c in shape_inner_boundary)]
    circle4 = circle4_left+circle4_right
    
    circle_right = [(r,c) for r,c in cg.hex_neighbours_n(pivot[0]+1, 
                               max(c for r,c in shape_inner_raw if r==pivot[0]+1)-1, 
                               n=2, keep_origin = False, return_frontier=True)[-1] 
                    if (r,c) in shape_inner_raw and (r,c) not in shape_inner_boundary]
    
    lines_right_1 = cgd.draw_slope_0p5_diagonal(min(r for r,c in circle_right), 
                             min(c for r,c in circle_right if r==min(r for r,c in circle_right)), 
                             max(r for r,c in circle4_right), left_down=False, right_down=False, left_up=True, right_up=False
                    )+cgd.draw_slope_0p5_diagonal(min(r for r,c in circle_right), 
                             min(c for r,c in circle_right if r==min(r for r,c in circle_right))+1, 
                             max(r for r,c in circle4_right)+2, left_down=False, right_down=False, left_up=True, right_up=False
                    )
   
    
    lines_3_part2 = cgd.draw_slope_0p5_diagonal(min(r for r,c in lines_3_part1), 
                                           min(c for r,c in lines_3_part1 if r==min(r for r,c in lines_3_part1)), 
                                           max(r for r,c in circle4), left_down=False, right_down=False, left_up=False, right_up=True
                    )+cgd.draw_slope_0p5_diagonal(min(r for r,c in lines_3_part1), 
                                           max(c for r,c in lines_3_part1 if r==min(r for r,c in lines_3_part1)), 
                                           max(r for r,c in circle4), left_down=False, right_down=False, left_up=True, right_up=False
                    )+cgd.draw_slope_1p5_diagonal(max(r for r,c in circle_top), 
                                           max(c for r,c in circle4_left if r==max(r for r,c in circle_top))+1, 
                                           max(r for r,c in circle_top)+1, left_down=False, right_down=True, left_up=False, right_up=False
                    )+cgd.draw_slope_1p5_diagonal(max(r for r,c in circle_top), 
                                           min(c for r,c in circle4_right if r==max(r for r,c in circle_top))-1, 
                                           max(r for r,c in circle_top)+1, left_down=True, right_down=False, left_up=False, right_up=False
                    )
    lines_3 = lines_3_part1+lines_3_part2
    
    lines_bot_1 = cgd.horizontal_lines([
                                   (max(r for r,c in shape_inner_boundary)-2, 
                         (min(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_boundary)-2)+math.floor(n_size/2/2),
                          max(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_boundary)-2)-math.floor(n_size/2/2)
                                     )
                                   ),])
    lines_bot_2 = cgd.draw_slope_0p5_diagonal(min(r for r,c in lines_bot_1), 
                             min(c for r,c in lines_bot_1 if r==min(r for r,c in lines_bot_1))+1, 
                             max(r for r,c in circle_left_bot_3), left_down=False, right_down=False, left_up=True, right_up=False
                    )+cgd.draw_slope_1p5_diagonal(min(r for r,c in lines_bot_1), 
                             min(c for r,c in lines_bot_1 if r==min(r for r,c in lines_bot_1)), 
                             max(r for r,c in circle_left_bot_1), left_down=False, right_down=False, left_up=True, right_up=False
                    )+cgd.draw_slope_1p5_diagonal(min(r for r,c in lines_bot_1), 
                             min(c for r,c in lines_bot_1 if r==min(r for r,c in lines_bot_1))-1, 
                             max(r for r,c in circle_left_bot_1), left_down=False, right_down=False, left_up=True, right_up=False
                    )+cgd.verticle_line(max(r for r,c in shape_inner_boundary), 
                                        min(c for r,c in shape_inner_boundary if r==max(r for r,c in shape_inner_boundary)), 
                                        max(r for r,c in shape_inner_boundary)-2, bend = 'right'
                   )+cgd.verticle_line(max(r for r,c in lines_bot_1), 
                                        max(c for r,c in lines_bot_1 if r==max(r for r,c in lines_bot_1)), 
                                        max(r for r,c in lines_bot_1)-1, bend = 'right') 
    lines_bot_3 = cgd.draw_slope_0p5_diagonal(max(r for r,c in shape_inner_raw)-1, 
                             pivot[1], max(r for r,c in lines_bot_1)-2, left_down=False, right_down=False, left_up=False, right_up=True
                    )
    lines_bot_4 = cgd.draw_slope_1p5_diagonal(min(r for r,c in lines_bot_3), 
                           min(c for r,c in lines_bot_3 if r==min(r for r,c in lines_bot_3)),  
                           max(r for r,c in circle_right)+1, left_down=False, right_down=False, left_up=False, right_up=True
                    )+cgd.draw_slope_1p5_diagonal(min(r for r,c in lines_bot_3), 
                           min(c for r,c in lines_bot_3 if r==min(r for r,c in lines_bot_3))+1,  
                           max(r for r,c in circle_right)+1, left_down=False, right_down=False, left_up=False, right_up=True
                    )
    region_bot_1 = cg.hex_neighbours_n(min(r for r,c in lines_bot_4), 
                                       max(c for r,c in lines_bot_4 if r==min(r for r,c in lines_bot_4))+1, n=1)
    region_bot_2 = cg.hex_neighbours_n(min(r for r,c in region_bot_1), 
                                       min(c for r,c in region_bot_1 if r==min(r for r,c in region_bot_1)), n=1)
    region_bot = region_bot_1+region_bot_2
    lines_bot = lines_bot_1+lines_bot_2+lines_bot_3+lines_bot_4
    lines_right = lines_right_1+lines_right_2
    
    shape_inner_lines = circle_center+lines_2+lines_3+lines_boundary+circle4+circle_top+circle_left+lines_bot+circle_left_bot+circle_5+circle_6+circle_right+lines_right+circle_midright+region_bot+circle_8+[(pivot[0]-6, pivot[1])]
    
    shape_outer_lines = [(r,c) for r,c in shape+numbers_white+numbers_black if (r,c) not in white_locs+shape_center]
    shape_inner_lines = [(r,c) for r,c in shape_inner_lines+shape_center if (r,c) not in [pivot]]
    if version== 'full':
        keys = [ 'shape_outer_lines' , "shape_inner_lines"]# "shape_inner_raw" ,,"white_locs"
    elif version== 'outer':
        keys = [ 'shape_outer_lines']
    colors = [[[0,0,0]],[[0,0,0]],[[1,0,0]],]#, [[0.5,0.5,0.5]],[[0.98,0.98,0.98]]
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}


def draw_char(head_center_row, head_center_col, name, n_size = 2):
    head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(head_center_row, head_center_col, version="5")
    head = cg.hex_neighbours_n(head_center_row, head_center_col, n=n_size)
    if "alice" in name:
        print(name)
        ln = SimpleNamespace(**line_dict)
        uplj = SimpleNamespace(**upperlimb_joint_dict)
        btlj = SimpleNamespace(**botlimb_joint_dict)

        if name == "black alice":
            version = 'face right'

        elif name == "white alice":
            version = 'face left'
        torso_side_raw = torso_dict["torso"]
        torso_side_remove = [(max(r for r,c in torso_side_raw),min(c for r,c in torso_side_raw if r==max(r for r,c in torso_side_raw))),
                                 (max(r for r,c in torso_side_raw),max(c for r,c in torso_side_raw if r==max(r for r,c in torso_side_raw))),
                                ]
        torso_side = [(r,c) for r,c in torso_side_raw if (r,c) not in torso_side_remove]
        right_upper_limb = uplj.right_shoulder_joint+ upperarm_dict["right_upperarm"]+forearm_dict["right_forearm"]+hand_dict["right_hand"]
        right_upper_limb_shift = [(r, c-3) for r,c in right_upper_limb]


        left_upper_limb = uplj.left_shoulder_joint+ upperarm_dict["left_upperarm"]+forearm_dict["left_forearm"]+hand_dict["left_hand"]
        left_upper_limb_shift = [(r, c+3) for r,c in left_upper_limb]

        right_bot_limb = btlj.right_thigh_joint+ thigh_dict["right_thigh"]+calf_dict['right_calf_raw']+feet_dict["right_foot"]
        right_bot_limb_shift = [(r, c-2) for r,c in right_bot_limb]

        left_bot_limb = btlj.left_thigh_joint+ thigh_dict["left_thigh"]+calf_dict['left_calf_raw']+feet_dict["left_foot"]
        left_bot_limb_shift = [(r, c+2) for r,c in left_bot_limb]


        if version == 'face right':
            upper_limb_side = right_upper_limb_shift
            bot_limb_side = right_bot_limb_shift

            butt_side = cg.hex_neighbours_n(max(r for r,c in torso_side),
                                        min(c for r,c in bot_limb_side if r==max(r for r,c in torso_side)), n=1)

        elif version == 'face left':
            upper_limb_side = left_upper_limb_shift
            bot_limb_side = left_bot_limb_shift

            butt_side = cg.hex_neighbours_n(max(r for r,c in torso_side),
                                        max(c for r,c in bot_limb_side if r==max(r for r,c in torso_side)), n=1)


        bot_limb_side+=butt_side

        raw_body_side = torso_side+upper_limb_side+bot_limb_side

        if name == "white alice":
            hair = [(r,c) for r,c in head if r<head_center_row or c >head_center_col]+[(head_center_row,head_center_col-n_size) ]

            face = [(r,c) for r,c in head if (r,c) not in hair]
            skin = [(r,c) for r,c in torso_side if r<ln.breastbone_end_r-1]
            neckless = [(r,c) for r,c in torso_side if r == max(r for r,c in head)+1]
            tops = [(r,c) for r,c in torso_side if (r,c) not in skin]


            sleeves = [(r,c) for r,c in  upper_limb_side if r<=ln.wrist_r and r >=ln.breastbone_end_r]

            sleeves_extra_1 = cgd.draw_trapezoid(ln.elbow_r, 
                                    min(c for r,c in sleeves if r==ln.elbow_r), 
                                    max(c for r,c in sleeves if r==ln.elbow_r), 
                                    ln.wrist_r, slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            sleeves_extra_2 =cgd.draw_trapezoid(max(r for r,c in sleeves_extra_1), 
                                    min(c for r,c in sleeves_extra_1 if r==max(r for r,c in sleeves_extra_1)), 
                                    max(c for r,c in sleeves_extra_1 if r==max(r for r,c in sleeves_extra_1)), 
                                    max(r for r,c in sleeves_extra_1)+2, slope_left = '1.5', slope_right = 'inf', direction = 'rl',
                                    bend_left = 'left', bend_right = 'right')[-2]
            sleeves+=sleeves_extra_1+sleeves_extra_2

            sk_part1 = cgd.draw_trapezoid(max(r for r,c in torso_side), 
                                    min(c for r,c in bot_limb_side if r==max(r for r,c in torso_side)), 
                                    max(c for r,c in bot_limb_side if r==max(r for r,c in torso_side)), 
                                    ln.sartorius_turning_r, slope_left = '0.5', slope_right = '1.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            sk_part2 = cgd.draw_trapezoid(max(r for r,c in sk_part1), 
                                    max(c for r,c in bot_limb_side if r==max(r for r,c in sk_part1))+1, 
                                    max(c for r,c in sk_part1 if r==max(r for r,c in sk_part1)), 
                                    max(r for r,c in raw_body_side), slope_left = 'inf', slope_right = '0.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]

            sk = sk_part1+sk_part2
            sk_acc = cg.hex_neighbours_n(min(r for r,c in sk_part2), min(c for r,c in sk_part2 if r==min(r for r,c in sk_part2)), n=1
                    )+cg.hex_neighbours_n(min(r for r,c in sk), max(c for r,c in sk if r==min(r for r,c in sk)), n=1
                    )+cg.hex_neighbours_n(max(r for r,c in sk)-2, max(c for r,c in sk if r==max(r for r,c in sk)-2), n=1
                    )
            black_ribbon = [(r,c) for r,c in sleeves if r==min(r for r,c in sleeves)+1
                           ]+[(r,c) for r,c in bot_limb_side if r==ln.foot_r-1 or r== ln.foot_r-3 and (r,c) not in sk
                           ]+[(r,c-1) for r,c in upper_limb_side if r==ln.wrist_r
                           ]
            hair_extra_1 =cgd.draw_trapezoid(head_center_row, 
                                    head_center_col+1, 
                                    max(c for r,c in hair if r==head_center_row), 
                                    ln.breastbone_end_r, slope_left = 'inf', slope_right = 'inf', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            hair_extra_2 = cgd.draw_trapezoid(max(r for r,c in hair_extra_1), 
                                    min(c for r,c in hair_extra_1 if r==max(r for r,c in hair_extra_1)), 
                                    max(c for r,c in hair_extra_1 if r==max(r for r,c in hair_extra_1)), 
                                    ln.femoral_head_r, slope_left = 'inf', slope_right ='0.5',  direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            hair_extra_3 = cgd.draw_trapezoid(max(r for r,c in hair_extra_2), 
                                    min(c for r,c in hair_extra_2 if r==max(r for r,c in hair_extra_2)), 
                                    max(c for r,c in hair_extra_2 if r==max(r for r,c in hair_extra_2)), 
                                    ln.gastrocnemius_end_r, slope_left = 'inf', slope_right = '0.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            hair_extra = [(r,c) for r,c in hair_extra_1+hair_extra_2+hair_extra_3 if (r,c) not in torso_side+upper_limb_side+bot_limb_side]
            hair+=hair_extra
            hair_acc = cgd.draw_slope_1p5_diagonal(head_center_row, head_center_col, head_center_row+1, 
                                                   left_down=False, right_down=True, left_up=False, right_up=False
                      )+cgd.draw_slope_0p5_diagonal(head_center_row, head_center_col, head_center_row+1, 
                                                   left_down=False, right_down=True, left_up=False, right_up=False
                      )
            skin_arm=[(r,c) for r,c in upper_limb_side if (r,c) not in sleeves]+[(r,c) for r,c in raw_body_side if (r,c) not in neckless+sk+sleeves+tops]

            keys = [# "torso_side", "upper_limb_side", "bot_limb_side", 
               "hair","face","skin","neckless", "tops", "skin_arm","sk","sk_acc", "sleeves", "black_ribbon", "hair_acc"]#
            colors = [[cgc.hex_to_rgb("#effffe"), cgc.hex_to_rgb("#c3dce1")],
                      [cgc.hex_to_rgb("#ffe6b7")],[cgc.hex_to_rgb("#ffe6b7")],
                      [cgc.hex_to_rgb("#a1e3f1")],[[1,1,1]],[cgc.hex_to_rgb("#ffe6b7")],
                      [[1,1,1]],[cgc.hex_to_rgb("#a1e3f1")],
                      [cgc.hex_to_rgb("#d9dadb"), cgc.hex_to_rgb("#e2faff")],

                      [[0,0,0]],[[1,1,1]],
                     ]#

        if name == "black alice":
            hair = [(r,c) for r,c in head if r<head_center_row or c <=head_center_col]+[(head_center_row,head_center_col+n_size) ]

            face = [(r,c) for r,c in head if (r,c) not in hair]
            skin = [(r,c) for r,c in torso_side if r<ln.breastbone_end_r]
            print(skin)
            neckless = [(r,c) for r,c in torso_side if r == max(r for r,c in head)+1]
            tops = [(r,c) for r,c in torso_side if (r,c) not in skin]
            sleeves  = [(r,c) for r,c in  upper_limb_side if r<=ln.wrist_r]
            sleeves_extra_1 = cgd.draw_trapezoid(ln.elbow_r, 
                                    min(c for r,c in sleeves if r==ln.elbow_r), 
                                    max(c for r,c in sleeves if r==ln.elbow_r), 
                                    ln.wrist_r, slope_left = 'inf', slope_right = '0.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            sleeves_extra_2 =cgd.draw_trapezoid(max(r for r,c in sleeves_extra_1), 
                                    min(c for r,c in sleeves_extra_1 if r==max(r for r,c in sleeves_extra_1)), 
                                    max(c for r,c in sleeves_extra_1 if r==max(r for r,c in sleeves_extra_1)), 
                                    max(r for r,c in sleeves_extra_1)+2, slope_left = 'inf', slope_right = '1.5', direction = 'rl',
                                    bend_left = 'left', bend_right = 'right')[-2]
            sleeves+=sleeves_extra_1+sleeves_extra_2
            sk_part1 = cgd.draw_trapezoid(max(r for r,c in torso_side), 
                                    min(c for r,c in bot_limb_side if r==max(r for r,c in torso_side)), 
                                    max(c for r,c in bot_limb_side if r==max(r for r,c in torso_side)), 
                                    ln.gastrocnemius_end_r, slope_left = '1.5', slope_right = '0.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]

            sk_part2 = cgd.draw_trapezoid(max(r for r,c in sk_part1), 
                                    min(c for r,c in sk_part1 if r==max(r for r,c in sk_part1)), 
                                    max(c for r,c in sk_part1 if r==max(r for r,c in sk_part1)), 
                                    ln.foot_r, slope_left = '8/3', slope_right = 'inf', direction = 'rl',
                                    bend_left = 'left', bend_right = 'right')[-2]

            sk = sk_part1+sk_part2+[(r,c) for r,c in bot_limb_side if r <=ln.foot_r]
            sk_white_acc_raw= cgd.draw_trapezoid(min(r for r,c in sk)+2, 
                                    min(c for r,c in sk if r==min(r for r,c in sk)+2)+2, 
                                    min(c for r,c in sk if r==min(r for r,c in sk)+2)+3, 
                                    max(r for r,c in sk), slope_left = 'inf', slope_right = '0.5', direction = 'rl',
                                    bend_left = 'left', bend_right = 'right'
                         )[-2]+cgd.draw_trapezoid(min(r for r,c in sk)+2, 
                                    max(c for r,c in sk if r==min(r for r,c in sk)+2), 
                                    max(c for r,c in sk if r==min(r for r,c in sk)+2)-1, 
                                    max(r for r,c in sk), slope_left = 'inf', slope_right = '0.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right'
                         )[-2]

            sk_white_acc = [(r,c) for r,c in sk_white_acc_raw if (r,c) in sk]
            ribbon = cgd.verticle_line(min(r for r,c in sk_white_acc)+2, 
                                       min(c for r,c in sk_white_acc if r==min(r for r,c in sk_white_acc)+2)-1,
                                       min(r for r,c in sk_white_acc)+3, bend = 'left'
                                      )+[(min(r for r,c in sk_white_acc)+2, 
                                       min(c for r,c in sk_white_acc if r==min(r for r,c in sk_white_acc)+2)-2)]

            hair_extra_1 =cgd.draw_trapezoid(head_center_row, 
                                    min(c for r,c in hair if r==head_center_row), 
                                    head_center_col-1, 
                                    ln.breastbone_end_r, slope_left = 'inf', slope_right = 'inf', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            hair_extra_2 = cgd.draw_trapezoid(max(r for r,c in hair_extra_1), 
                                    min(c for r,c in hair_extra_1 if r==max(r for r,c in hair_extra_1)), 
                                    max(c for r,c in hair_extra_1 if r==max(r for r,c in hair_extra_1)), 
                                    ln.femoral_head_r, slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            hair_extra_3 = cgd.draw_trapezoid(max(r for r,c in hair_extra_2), 
                                    min(c for r,c in hair_extra_2 if r==max(r for r,c in hair_extra_2)), 
                                    max(c for r,c in hair_extra_2 if r==max(r for r,c in hair_extra_2)), 
                                    ln.gastrocnemius_end_r, slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
            hair_extra = [(r,c) for r,c in hair_extra_1+hair_extra_2+hair_extra_3 if (r,c) not in torso_side+upper_limb_side+bot_limb_side]
            hair+=hair_extra
            hair_acc = cgd.draw_slope_1p5_diagonal(head_center_row, head_center_col, head_center_row+1, 
                                                   left_down=True, right_down=False, left_up=False, right_up=False
                      )+cgd.draw_slope_0p5_diagonal(head_center_row, head_center_col, head_center_row+1, 
                                                   left_down=True, right_down=False, left_up=False, right_up=False
                      )
            skin+=[(r,c) for r,c in raw_body_side if (r,c) not in sk+sleeves+tops]
            keys = [# "torso_side", "upper_limb_side", "bot_limb_side", 
                   "face","hair","hair_acc",  "skin",
                "tops","neckless", "sk", "sk_white_acc", "ribbon",

                "sleeves"]#
            colors = [[cgc.hex_to_rgb("#ffe6b7")],
                      [cgc.hex_to_rgb("#66261a"), cgc.hex_to_rgb("#390c05")],[cgc.hex_to_rgb("#2b0c06")],
                      [cgc.hex_to_rgb("#ffe6b7")],

                      [cgc.hex_to_rgb("#5838c4")],#tops
                      [cgc.hex_to_rgb("#430196")],#tops
                      [[0,0,0], cgc.hex_to_rgb("#2d015d")],#sk

                      [cgc.hex_to_rgb("#eee9ff"), cgc.hex_to_rgb("#b5abd6")],
                      [cgc.hex_to_rgb("#6c6ddb")],
                      [[0,0,0], cgc.hex_to_rgb("#4d0092")],

                      [[0,0,0]],[[0.3,0.3,0.3]],[[0.6,0.6,0.6]],
                     [[0,1,0]], [cgc.hex_to_rgb("#ffe500")],[[0,1,1]],
                       ]#

    elif name == 'noise':
        ln = SimpleNamespace(**line_dict)

        cape_head = [(r,c) for r,c in head_raw if r<=head_center_row or c>head_center_col]
        face = [(r,c) for r,c in head+neck if (r,c) not in cape_head]
        hair = cgd.verticle_line(min(r for r,c in face), max(c for r,c in face if r==min(r for r,c in face)), 
                                 max(r for r,c in face), bend = 'right')
        chain = cgd.verticle_line(max(r for r,c in face), min(c for r,c in face if r==max(r for r,c in face))-1, 
                                 max(r for r,c in face), bend = 'left')
        cape_back= [(r,c) for r,c in head_raw if (r,c) not in cape_head+face]
        cape_upper = cgd.draw_trapezoid(max(r for r,c in head_raw)+1, 
                                          min(c for r,c in raw_body if r==max(r for r,c in head_raw)+1),  
                                          max(c for r,c in raw_body if r==max(r for r,c in head_raw)+1), ln.femoral_head_r,
                                          slope_left = '0.5', slope_right = '0.5', direction = 'lr',bend_left = 'left', bend_right = 'right'
                                    )[-2]
        cape_bottom = cgd.draw_trapezoid(max(r for r,c in cape_upper), 
                                          min(c for r,c in cape_upper if r==max(r for r,c in cape_upper))+3,  
                                          max(c for r,c in cape_upper if r==max(r for r,c in cape_upper))-3, max(r for r,c in raw_body),
                                          slope_left = '0.5', slope_right = '0.5', direction = 'lr',bend_left = 'left', bend_right = 'right'
                                    )[-2]
        cape_acc = [(r,c) for r,c in cape_head if r==head_center_row and c<=head_center_col
                   ]+[(r,c) for r,c in cape_upper if r==max(r for r,c in cape_upper)
                   ]+[(r,c) for r,c in cape_bottom if r==max(r for r,c in cape_bottom)
                   ]
        cape = cape_head+cape_upper+cape_bottom
        keys = ["face", 
                
                "cape_back" ,"cape_head","cape_upper", "cape_bottom",
                
                "cape_acc", "hair", "chain"]#  'raw_body' ,
        colors = [[cgc.hex_to_rgb("#ffe6b7")],
                  [cgc.hex_to_rgb("#390000")],
                  [cgc.hex_to_rgb("#ad0024"),cgc.hex_to_rgb("#390000") ], 
                  [cgc.hex_to_rgb("#ad0024"),cgc.hex_to_rgb("#800000") ], 
                  [cgc.hex_to_rgb("#000000"),cgc.hex_to_rgb("#cd0000") ], 
                  [cgc.hex_to_rgb("#dbb800")], 
                  [cgc.hex_to_rgb("#ded6ff")],[cgc.hex_to_rgb("#ffe500")]
                   ]#
    
    elif name == 'echo':
        hair = [(r,c) for r,c in cgd.draw_trapezoid(head_center_row, head_center_col-n_size, head_center_col+n_size, max(r for r,c in head_raw), 
                                  slope_left = 'inf', slope_right = 'inf', direction = 'lr',bend_left = 'left', bend_right = 'right'
                           )[-2]+cgd.draw_trapezoid(head_center_row-1, 
                                                    min(c for r,c in head_raw if r==head_center_row-1), 
                                                    max(c for r,c in head_raw if r==head_center_row-1), min(r for r,c in head_raw), 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                      bend_left = 'left', bend_right = 'right')[-2]
                if (r,c) not in raw_body]
        face = [(r,c) for r,c in head if (r,c) not in hair]

        hair+=[(r,c) for r,c in head if r<head_center_row]+[(head_center_row, head_center_col),
                                                            (head_center_row, head_center_col-n_size),
                                                            (head_center_row, head_center_col+n_size),
                                                           ]
        ln = SimpleNamespace(**line_dict)
        shoes = [(r,c) for r,c in raw_body if r >=ln.sartorius_turning_r and (r,c) not in hand_dict['hand']]
        leg = [(r,c) for r,c in thigh_dict["thigh"] if (r,c) not in shoes ]
        uplj = SimpleNamespace(**upperlimb_joint_dict)


        blue =  [(r,c) for r,c in torso_dict["torso"]if (r,c) not in neck
                ]+botlimb_joint_dict["left_thigh_joint"
                ]+botlimb_joint_dict["right_thigh_joint"
                ]+[(max(r for r,c in uplj.left_shoulder_joint), 
                    max(c for r,c in uplj.left_shoulder_joint if r==max(r for r,c in uplj.left_shoulder_joint)))
                ]+[(max(r for r,c in uplj.right_shoulder_joint), 
                    min(c for r,c in uplj.right_shoulder_joint if r==max(r for r,c in uplj.right_shoulder_joint)))
                ]
        #line =  [(r,c) for r,c in raw_body if r ==ln.femoral_head_r ] 
        blue_extra=cgd.draw_trapezoid(ln.femoral_head_r, 
                                      min(c for r,c in blue if r==ln.femoral_head_r),  
                                      max(c for r,c in blue if r==ln.femoral_head_r), ln.breastbone_end_r,
                                      slope_left = '0.5', slope_right = '0.5', direction = 'rl',bend_left = 'left', bend_right = 'right'
                                )[-2]+cgd.draw_trapezoid(ln.femoral_head_r, 
                                      min(c for r,c in blue if r==ln.femoral_head_r),  
                                      max(c for r,c in blue if r==ln.femoral_head_r), ln.femoral_head_r+1,
                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr',bend_left = 'left', bend_right = 'right'
                                )[-2]


        blue+=blue_extra
        remove = [(r, f(cs)) for r in {r for r, c in blue} for cs in ([c for rr, c in blue if rr == r],) 
            if cs for f  in (min, max)
                 ]
        blue_white = [(r,c) for r,c in blue if (r,c) not in remove]

        buttons = [(max(r for r,c in blue_white)-1, min(c for r,c in blue_white if r==max(r for r,c in blue_white)-1)),
                   (max(r for r,c in blue_white)-1, max(c for r,c in blue_white if r==max(r for r,c in blue_white)-1)),
                  (max(r for r,c in blue_white)-3, min(c for r,c in blue_white if r==max(r for r,c in blue_white)-3)),
                   (max(r for r,c in blue_white)-3, max(c for r,c in blue_white if r==max(r for r,c in blue_white)-3)),
                  (max(r for r,c in blue_white)-5, min(c for r,c in blue if r==max(r for r,c in blue_white)-5)),
                   (max(r for r,c in blue_white)-5, max(c for r,c in blue if r==max(r for r,c in blue_white)-5)),
                  (min(r for r,c in blue_white), min(c for r,c in blue_white if r==min(r for r,c in blue_white)))]


        tie = cgd.verticle_line(min(r for r,c in blue_white), min(c for r,c in blue_white if r==min(r for r,c in blue_white)), 
                          min(r for r,c in blue_white)+2, bend = 'left'
                         )+cgd.verticle_line(min(r for r,c in blue_white), min(c for r,c in blue_white if r==min(r for r,c in blue_white)), 
                          min(r for r,c in blue_white)+2, bend = 'right'
                         )
        blue_sleeves = [(r,c) for r,c in upperarm_dict["upperarm"]+forearm_dict["forearm"]+hand_dict["hand"] 
                        if r>=ln.deltoid_end_r+1 and (r,c) not in blue
                ]+[(ln.deltoid_end_r+1, min(c for r,c in upperarm_dict["upperarm"] if r==ln.deltoid_end_r+1)-1)
                ]+[(ln.deltoid_end_r+1, max(c for r,c in upperarm_dict["upperarm"] if r==ln.deltoid_end_r+1)+1)
                ]+[(ln.deltoid_end_r+1, min(c for r,c in upperarm_dict["upperarm"] if r==ln.deltoid_end_r+1)+1)
                ]+[(ln.deltoid_end_r+1, max(c for r,c in upperarm_dict["upperarm"] if r==ln.deltoid_end_r+1)-1)
                ]
        blue_sleeves_extra = cgd.draw_trapezoid(min(r for r,c in blue_sleeves)+2, 
                                      min(c for r,c in blue_sleeves if c<head_center_col and r==min(r for r,c in blue_sleeves)+2),  
                                      max(c for r,c in blue_sleeves if c<head_center_col and r==min(r for r,c in blue_sleeves)+2), 
                                      min(r for r,c in shoes),
                                      slope_left = '0.5', slope_right = 'inf', direction = 'll',bend_left = 'left', bend_right = 'right'
                                )[-2]+cgd.draw_trapezoid(min(r for r,c in blue_sleeves)+2, 
                                      min(c for r,c in blue_sleeves if c>head_center_col and r==min(r for r,c in blue_sleeves)+2),  
                                      max(c for r,c in blue_sleeves if c>head_center_col and r==min(r for r,c in blue_sleeves)+2), 
                                      min(r for r,c in shoes),
                                      slope_left = 'inf', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                                )[-2]
        blue_sleeves+=blue_sleeves_extra
        sleeves_acc_part1 = cgd.draw_slope_0p5_diagonal(max(r for r,c in blue_sleeves), 
                                                min(c for r,c in blue_sleeves if c<head_center_col and r==max(r for r,c in blue_sleeves))+2, 
                                                max(r for r,c in blue_sleeves)-2, 
                                                left_down=False, right_down=False, left_up=False, right_up=True
                  )+cgd.draw_slope_0p5_diagonal(max(r for r,c in blue_sleeves), 
                                                max(c for r,c in blue_sleeves if c>head_center_col and r==max(r for r,c in blue_sleeves))-2, 
                                                max(r for r,c in blue_sleeves)-2, 
                                                left_down=False, right_down=False, left_up=True, right_up=False
                  )+cgd.draw_slope_0p5_diagonal(max(r for r,c in blue_sleeves), 
                                                min(c for r,c in blue_sleeves if c<head_center_col and r==max(r for r,c in blue_sleeves))+2, 
                                                max(r for r,c in blue_sleeves)-1, 
                                                left_down=False, right_down=False, left_up=True, right_up=False
                  )+cgd.draw_slope_0p5_diagonal(max(r for r,c in blue_sleeves), 
                                                max(c for r,c in blue_sleeves if c>head_center_col and r==max(r for r,c in blue_sleeves))-2, 
                                                max(r for r,c in blue_sleeves)-1, 
                                                left_down=False, right_down=False, left_up=False, right_up=True
                  )
        sleeves_acc_part2 = cgd.draw_slope_0p5_diagonal(min(r for r,c in sleeves_acc_part1), 
                                           min(c for r,c in sleeves_acc_part1 if c<head_center_col and r==min(r for r,c in sleeves_acc_part1)), 
                                           min(r for r,c in sleeves_acc_part1)-1, 
                                           left_down=False, right_down=False, left_up=True, right_up=False
                  )+cgd.draw_slope_0p5_diagonal(min(r for r,c in sleeves_acc_part1), 
                                           max(c for r,c in sleeves_acc_part1 if c>head_center_col and r==min(r for r,c in sleeves_acc_part1)), 
                                           min(r for r,c in sleeves_acc_part1)-1, 
                                           left_down=False, right_down=False, left_up=False, right_up=True
                  )
        sleeves_acc = sleeves_acc_part1+sleeves_acc_part2
        sleeves_frill = [(r,c) for r,c in cgd.draw_trapezoid(max(r for r,c in blue_sleeves), 
                                      min(c for r,c in blue_sleeves if c<head_center_col and r==max(r for r,c in blue_sleeves)),  
                                      max(c for r,c in blue_sleeves if c<head_center_col and r==max(r for r,c in blue_sleeves)), 
                                      max(r for r,c in blue_sleeves)+1,
                                      slope_left = '0.5', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                                )[-2]+cgd.draw_trapezoid(max(r for r,c in blue_sleeves), 
                                      min(c for r,c in blue_sleeves if c>head_center_col and r==max(r for r,c in blue_sleeves)),  
                                      max(c for r,c in blue_sleeves if c>head_center_col and r==max(r for r,c in blue_sleeves)), 
                                      max(r for r,c in blue_sleeves)+1,
                                      slope_left = '0.5', slope_right = '0.5', direction = 'll',bend_left = 'left', bend_right = 'right'
                                )[-2]  if (r,c) not in blue_sleeves
                      ]+[(max(r for r,c in blue_sleeves), 
                          min(c for r,c in blue_sleeves if c<head_center_col and r==max(r for r,c in blue_sleeves)))
                        ]+[(max(r for r,c in blue_sleeves), 
                          max(c for r,c in blue_sleeves if c>head_center_col and r==max(r for r,c in blue_sleeves)))
                        ]
        white_raw = [(r,c) for r,c in raw_body  if (r,c) not in head+blue+blue_sleeves and r <ln.femoral_head_r]
        white_remove = [(max(r for r,c in white_raw), min(c for r,c in white_raw if r==max(r for r,c in white_raw))),
                 (max(r for r,c in white_raw), max(c for r,c in white_raw if r==max(r for r,c in white_raw))), ]
        white = [(r,c) for r,c in white_raw if (r,c) not in white_remove]
        keys = ["face", "leg", "shoes", 
                "blue_sleeves", "sleeves_frill","sleeves_acc", 
                "blue","blue_white", "white",
                "hair", "tie", "buttons"]#  'raw_body' 
        colors = [[cgc.hex_to_rgb("#ffe6b7")],[cgc.hex_to_rgb("#ffe6b7")],[[1,1,1], cgc.hex_to_rgb("#d2d8d8")],
                  [cgc.hex_to_rgb("#370ddd")],[cgc.hex_to_rgb("#d2d8d8")],[[1,1,1]],
                  [cgc.hex_to_rgb("#3400ff")],[[1,1,1]],[[1,1,1]],
                  [cgc.hex_to_rgb("#ded6ff")],[[0,0,0]], [cgc.hex_to_rgb("#0085ff")],

               ]#
    local_vars = locals()
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})
    #return {k: [local_vars[k], c] for k, c in zip(keys, colors)}

n_marker = 14
marker_center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
marker_center_col = max(c for r, c in hex_rc_arr) // 2

head_row = marker_center_row-n_marker//2-2
head_col_left = marker_center_col-n_marker//2+1
head_col_right =  marker_center_col+n_marker//2-2
print(head_row , head_col_left, head_col_right)
char_dict_ = {
        'marker':draw_marker((  marker_center_row, marker_center_col), n_marker, version = 'full'),
     
    'noise':draw_char(8, max(c for r,c in hex_rc_arr)-3, name='noise') ,  
    
     'black alice':draw_char(head_row-6, 9,  name='black alice') ,  
     'white alice':draw_char(head_row-6, max(c for r,c in hex_rc_arr)-10,  name='white alice')    ,
       'echo':draw_char(max(r for r, c in hex_rc_arr)-13, marker_center_col+12, name='echo')    ,
     
     'noise':draw_char(max(r for r, c in hex_rc_arr)-13, marker_center_col-12, name='noise') ,  
    
                }

#hex_colors = cgc.select_normal_color([True]*len(hex_rc_arr),cgc.hex_to_rgb("#1e0000"), np.ones(3)*sigma_color*3) 


hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#ffb5b5"), cgc.hex_to_rgb("#cb0000"), hex_rc_arr, hex_colors, 
                                    (marker_center_row, marker_center_col), n_marker+5,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='linear',
                        start_n=0, end_n=None)
hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#cb0000"), [0,0,0], hex_rc_arr, hex_colors, 
                                    (marker_center_row, marker_center_col), max(c for r, c in hex_rc_arr) ,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='sigmoid',
                        start_n=n_marker+5, end_n=None)

for char, dict_ in char_dict_.items(): 
    if "echo" in char or "noise" in char:
        alpha_part = 0.6
    elif "alice" in char:
        alpha_part =0.8
    else:
        alpha_part = 1
    for key in dict_.keys():
        prev_hex_colors = hex_colors.copy()
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
        hex_colors = (1 - alpha_part) * prev_hex_colors + alpha_part * hex_colors
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'PandoraHearts_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')