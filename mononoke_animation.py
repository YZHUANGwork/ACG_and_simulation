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
import TRAJECTORY_magneticfield_invertpendulum as TRAJ

rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W, IMG_H, HEX_R, dx_hex_center, dy_hex_center = detail_info


sigma_color = 0.03
#------------bkgd-----------------
sigma_color = 0.02


skyline = 10

ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
wall = [(r,c) for r,c in hex_rc_arr if r <skyline]
pillar = cgd.draw_block((0, (1, 3)), skyline)+cgd.draw_block((0, (11,13)), skyline)+cgd.draw_block((0, (21,23)), skyline)
hex_colors = cgc.color_row_gradient(wall, cgc.hex_to_rgb("#9b9a7f"), cgc.hex_to_rgb("#472008"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05)   
select_pillar= cg.select_mask(pillar,hex_rc_arr)
hex_colors[select_pillar] = cgc.select_normal_color(select_pillar,cgc.hex_to_rgb("#df302a"), np.ones(3)*sigma_color) 
    
hex_colors = cgc.color_row_gradient(ground, cgc.hex_to_rgb("#1c2e24"), cgc.hex_to_rgb("#206458"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05)    


animation_base_hex_colors = hex_colors.copy()


def draw_umbrella(center_r, center_c, n_center):
    n_part_1 = n_center*2+n_center
    n_part_2 = n_part_1+n_center
    n_part_3 = n_part_2+n_center*2
    n_part_4 = n_part_3+n_center*2
    
    part_0 = cg.hex_neighbours_n(center_r,center_c, n=n_center, keep_origin = True)   
    part_1 = cg.hex_neighbours_n(center_r,center_c, n=n_part_1, keep_origin = True)   
    part_2 = cg.hex_neighbours_n(center_r,center_c, n=n_part_2, keep_origin = True)   
    part_3 = cg.hex_neighbours_n(center_r,center_c, n=n_part_3, keep_origin = True)   
    part_4 = cg.hex_neighbours_n(center_r,center_c, n=n_part_4, keep_origin = True)   
    #print(part_0)
    return {
        '4': [part_4, [cgc.hex_to_rgb("#a83503")]],
        '3': [part_3, [cgc.hex_to_rgb("#bbd636")]],
        '2': [part_2, [cgc.hex_to_rgb("#fdfb24")]],
        '1': [part_1, [cgc.hex_to_rgb("#a83503")]],
        '0': [part_0, [cgc.hex_to_rgb("#442c7d")]],
        }


def draw_mouse(center_r, center_c, n_center, direction = 'hor', bend = 'left'):
    n_part_1 = n_center+math.ceil(n_center/2)
    n_part_2 = n_part_1+math.ceil(n_center/2)
    part_0 = cg.hex_neighbours_n(center_r,center_c, n=n_center, keep_origin = True)   
    part_1 = cg.hex_neighbours_n(center_r,center_c, n=n_part_1, keep_origin = True)   
    if direction == 'hor':
        part_2 = cg.hex_neighbours_n(center_r,center_c, n=n_part_2, keep_origin = True
                                ) + cgd.horizontal_lines([(center_r, (center_c-n_part_2-2, center_c+n_part_2+2) ) ])  
        n_part_3 = n_part_2+1
        temp = cg.hex_neighbours_n(center_r,center_c, n=n_part_3, keep_origin = True)
        body = temp+cgd.draw_slope_0p5_diagonal(center_r-n_part_3, 
                                                max(c for r,c in temp if r ==center_r-n_part_3), center_r-n_part_3-1, 
                                                left_down=False, right_down=False, left_up=False, right_up=True 
                                               )+cgd.draw_slope_0p5_diagonal(center_r+n_part_3, 
                                                max(c for r,c in temp if r ==center_r+n_part_3), center_r+n_part_3+1, 
                                                left_down=False, right_down=True , left_up=False, right_up=False
                                               )+cgd.draw_trapezoid(center_r-n_part_3, 
                                min(c for r,c in temp if r ==center_r-n_part_3), 
                                max(c for r,c in temp if r ==center_r-n_part_3), center_r, 
                                slope_left = '1.5', slope_right = '8/3', direction = 'lr'
                                   )[-2]+cgd.draw_trapezoid(center_r+n_part_3, 
                                                            min(c for r,c in temp if r ==center_r+n_part_3), 
                                                            max(c for r,c in temp if r ==center_r+n_part_3), center_r, 
                                                            slope_left = '1.5', slope_right = '8/3', direction = 'lr')[-2]
        
        part_3 = [(r,c) for r,c in body if c <=center_c+n_part_3]
        part_4 = [(r,c) for r,c in body if c >center_c+n_part_3 and c <=center_c+n_part_3+n_part_3]
        part_5 = [(r,c) for r,c in body if c >center_c+n_part_3+n_part_3 and c <=max(c for r,c in body if r ==center_r)]
        part_6 = cgd.horizontal_lines([(center_r, (max(c for r,c in body if r ==center_r),
                                                  max(c for r,c in body if r ==center_r)+n_part_3*2) ) ])  
    elif direction == 'ver':
        part_2 = cg.hex_neighbours_n(center_r,center_c, n=n_part_2, keep_origin = True
                                ) + cgd.verticle_line(center_r, center_c, center_r-n_part_2-2, bend = bend
                                                     ) + cgd.verticle_line(center_r, center_c, center_r+n_part_2+2, bend = bend)
        
    print(part_4)
    return {
        '6': [part_6, [cgc.hex_to_rgb("#d88eff")] ],
        '5': [part_5, [cgc.hex_to_rgb("#8effd0"), cgc.hex_to_rgb("#d88eff")] ],
        '4': [part_4, [cgc.hex_to_rgb("#d88eff"), cgc.hex_to_rgb("#8effd0")] ],
        '3': [part_3, [cgc.hex_to_rgb("#d88eff")]],
        '2': [part_2, [cgc.hex_to_rgb("#001186")]],
        '1': [part_1, [cgc.hex_to_rgb("#00ffff")]],
        '0': [part_0, [cgc.hex_to_rgb("#ff0800")]],
            }

def draw_scale(center_row, center_col, n = 2):
    center_part = cg.hex_neighbours_n(center_row,center_col, n=n, keep_origin = True)
    top_part = cgd.draw_trapezoid(center_row-n, 
                                  min(c for r,c in center_part if r == center_row-n), 
                                  max(c for r,c in center_part if r == center_row-n), center_row-1, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                                 )[-2]+cgd.draw_trapezoid(center_row-n, 
                                  min(c for r,c in center_part if r == center_row-n), 
                                  max(c for r,c in center_part if r == center_row-n), center_row-n-n, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                                 )[-2]
    top_tip = cgd.verticle_line(min(r for r,c in top_part), center_col, min(r for r,c in top_part)-2, bend = 'left'
                               )+cgd.verticle_line(min(r for r,c in top_part), center_col, min(r for r,c in top_part)-2, bend = 'right'
                               )
    
    top_part_acc = [(center_row-n-1, min(c for r,c in top_part if r == center_row-n-1)-1),
                   (center_row-n-1, max(c for r,c in top_part if r == center_row-n-1)+1)]
    other = cg.hex_neighbours_n(min(r for r,c in top_part),center_col, n=n*6, keep_origin = True) 
    
    remove_part_1 = cgd.draw_trapezoid(center_row, 
                                  min(c for r,c in center_part if r == center_row+n)-1, 
                                  max(c for r,c in center_part if r == center_row+n)+1, center_row+n, 
                                  slope_left = '1.5', slope_right = '1.5', direction = 'lr'
                                 )[-2]
    remove_part_2 = cgd.draw_trapezoid(max(r for r, c in remove_part_1),
                                  min(c for r,c in remove_part_1 if r == max(r for r, c in remove_part_1)), 
                                  max(c for r,c in remove_part_1 if r == max(r for r, c in remove_part_1)), 
                                      max(r for r, c in remove_part_1)+n+1, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                 )[-2]
    
    remove_part_3 = cgd.draw_block((max(r for r, c in remove_part_2),
                                    (min(c for r,c in remove_part_2 if r == max(r for r, c in remove_part_2)), 
                                     max(c for r,c in remove_part_2 if r == max(r for r, c in remove_part_2)))
                                   ), max(r for r, c in remove_part_2)+n+n)
    
    
    remove_part_4 = cgd.draw_trapezoid(min(r for r,c in top_part),
                                  min(c for r,c in other if r ==min(r for r,c in top_part))+1, 
                                  max(c for r,c in other if r ==min(r for r,c in top_part))-1, 
                                       min(r for r, c in top_part)+n, 
                                  slope_left = '8/3', slope_right = '8/3', direction = 'rl'
                                 )[-2]
    

    
    
    remove = remove_part_1+remove_part_2+remove_part_3+remove_part_4
    bottom_part_1 = cgd.draw_trapezoid(center_row+n, 
                                  min(c for r,c in center_part if r == center_row+n)+1, 
                                  max(c for r,c in center_part if r == center_row+n)-1, center_row+n+n, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                 )[-2]
    bottom_part_2 = cgd.draw_trapezoid(max(r for r, c in bottom_part_1), 
                                  min(c for r,c in bottom_part_1 if r == max(r for r, c in bottom_part_1)), 
                                  max(c for r,c in bottom_part_1 if r == max(r for r, c in bottom_part_1)), 
                                       max(r for r, c in bottom_part_1)+n, 
                                  slope_left = '1.5', slope_right = '1.5', direction = 'lr'
                                 )[-2]
    bottom_part_3 = cgd.draw_trapezoid(max(r for r, c in bottom_part_2), 
                                  min(c for r,c in bottom_part_2 if r == max(r for r, c in bottom_part_2)), 
                                  max(c for r,c in bottom_part_2 if r == max(r for r, c in bottom_part_2)), 
                                       max(r for r, c in bottom_part_2)+n-1, 
                                  slope_left = '8/3', slope_right = '8/3', direction = 'rl'
                                 )[-2]
    
    bottom_part = [(r,c) for r,c in other if r >=min(r for r,c in top_part) 
                   and (r,c ) not in remove]#+bottom_part_1+bottom_part_2+bottom_part_3
    
    plate_r = min(r for r,c in bottom_part)-1
    plate_l = 3
    top_plate = cgd.horizontal_lines([(plate_r, (min(c for r,c in other if r==plate_r)-plate_l,
                                                 min(c for r,c in other if r==plate_r)+plate_l)),
                                      (plate_r, (max(c for r,c in other if r==plate_r)-plate_l,
                                                max(c for r,c in other if r==plate_r)+plate_l))])
    
    strings = cgd.verticle_line(plate_r, min(c for r,c in top_plate), center_row+n, bend = 'left'
                              )+cgd.verticle_line(plate_r, max(c for r,c in top_plate), center_row+n, bend = 'right'
                              )
    n_bell= 1
    bells = cg.hex_neighbours_n(center_row+n+n_bell,min(c for r,c in strings if r == center_row+n), n=n_bell, keep_origin = True
                               )+cg.hex_neighbours_n(center_row+n+n_bell,max(c for r,c in strings if r == center_row+n), n=n_bell, 
                                                     keep_origin = True)
    dr = max(r for r,c in bottom_part)-min(r for r,c in bottom_part)
    dr_seg1 = dr//3+2
    dr_seg2 = dr//3
    add_part_1 = cgd.draw_triangle(min(r for r,c in bottom_part), 
                                   min(c for r,c in bottom_part if r==min(r for r,c in bottom_part)),  
                                   min(r for r,c in bottom_part)+dr_seg1, slope_left = 'inf', slope_right = '0.5', direction = 'lr',
                                   bend_left = 'left', bend_right = 'right'
                                  )[-2]+cgd.draw_triangle(min(r for r,c in bottom_part), 
                                                          max(c for r,c in bottom_part if r==min(r for r,c in bottom_part)),  
                                   min(r for r,c in bottom_part)+dr_seg1, slope_left = '0.5', slope_right = 'inf', direction = 'lr',
                                   bend_left = 'left', bend_right = 'right'
                                  )[-2]
    
    add_part_2 = cgd.draw_trapezoid(max(r for r,c in add_part_1), 
                                  min(c for r,c in add_part_1 if r == max(r for r, c in add_part_1)), 
                                  min(c for r,c in bottom_part if r == max(r for r, c in add_part_1)), 
                                       max(r for r, c in add_part_1)+dr_seg2, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'rr'
                                 )[-2]+cgd.draw_trapezoid(max(r for r,c in add_part_1), 
                                                          max(c for r,c in bottom_part if r == max(r for r, c in add_part_1)),
                                                          max(c for r,c in add_part_1 if r == max(r for r, c in add_part_1)),
                                                          max(r for r, c in add_part_1)+dr_seg2, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'll'
                                 )[-2]
    
    add_part_3 = cgd.draw_trapezoid(max(r for r,c in add_part_2), 
                                  min(c for r,c in add_part_2 if r == max(r for r, c in add_part_2)), 
                                  min(c for r,c in bottom_part if r == max(r for r, c in add_part_2)), 
                                       max(r for r, c in bottom_part_3), 
                                  slope_left = '1.5', slope_right = '0.5', direction = 'rr'
                                 )[-2]+cgd.draw_trapezoid(max(r for r,c in add_part_2), 
                                                          max(c for r,c in bottom_part if r == max(r for r, c in add_part_2)),
                                                          max(c for r,c in add_part_2 if r == max(r for r, c in add_part_2)),
                                                          max(r for r, c in bottom_part_3), 
                                  slope_left = '0.5', slope_right = '1.5', direction = 'll'
                                 )[-2]
    
    main_body_part = bottom_part+bottom_part_1+bottom_part_2+bottom_part_3+add_part_1+add_part_2+add_part_3
    
    boundary =  [(r, f(cs)) for r in {r for r, c in main_body_part} for cs in ([c for rr, c in main_body_part if rr == r],) 
                 if cs for f in (min, max)]+[(r, f(cs)) for r in {r for r, c in add_part_3} 
                                             for cs in ([c for rr, c in add_part_3 if rr == r],) 
                 if cs for f in (min, max, lambda cs: min(cs) + 1, lambda cs: max(cs) - 1)
                                            ]+[(r,c) for r, c in bottom_part_3 if r == max(r for r,c in bottom_part_3)]
    
    center_part_boundary_1 = [(row,col) for row,col in main_body_part if (row,col) in [(r, f(cs)) for r in {r for r, c in center_part} 
                                                                         for cs in ([c for rr, c in center_part if rr == r],) 
                                                                         if cs for f in (lambda cs: min(cs) - 1, lambda cs: max(cs) + 1)]]
    center_part_boundary_2 = [(row,col) for row,col in main_body_part if (row,col) in [(r, f(cs)) for r in {r for r, c in center_part} 
                                                                         for cs in ([c for rr, c in center_part if rr == r],) 
                                                                         if cs for f in (lambda cs: min(cs) - 2, lambda cs: max(cs) + 2,
                                                                                        lambda cs: min(cs) - 3, lambda cs: max(cs) + 3)]]
    
    acc_dots_centers = [(min(r for r,c in bottom_part)+dr_seg1, 
                         min(c for r,c in main_body_part if r == min(r for r,c in bottom_part)+dr_seg1)+3),
                        (center_row-n, 
                         min(c for r,c in main_body_part if r == center_row-n)+2),
                        (min(r for r,c in bottom_part)+dr_seg1+dr_seg2-1, 
                         min(c for r,c in main_body_part if r == min(r for r,c in bottom_part)+dr_seg1+dr_seg2-1)+4),
                       
                        (min(r for r,c in bottom_part)+dr_seg1, 
                         max(c for r,c in main_body_part if r == min(r for r,c in bottom_part)+dr_seg1)-3),
                        (center_row-n, 
                         max(c for r,c in main_body_part if r == center_row-n)-2),
                        (min(r for r,c in bottom_part)+dr_seg1+dr_seg2-1, 
                         max(c for r,c in main_body_part if r == min(r for r,c in bottom_part)+dr_seg1+dr_seg2-1)-4),
                       
                        (center_row, center_col),
                        
                        (max(r for r,c in bottom_part_2)-1, 
                         min(c for r,c in bottom_part_2 if r == max(r for r,c in bottom_part_2)-1)+1),
                        (max(r for r,c in bottom_part_2)-1, 
                         max(c for r,c in bottom_part_2 if r == max(r for r,c in bottom_part_2)-1)-1)
                       ]
    dots = []
    for acc_dots_center in acc_dots_centers:
        dots.extend(cg.hex_neighbours_n(acc_dots_center[0],acc_dots_center[1], n=1, keep_origin = True))
    valid_dots = [(r,c ) for r,c in dots if (r,c) in main_body_part+center_part]
    tail = cgd.draw_trapezoid(max(r for r,c in bottom_part_3), 
                                  min(c for r,c in bottom_part_3 if r == max(r for r, c in bottom_part_3)), 
                                  max(c for r,c in bottom_part_3 if r == max(r for r, c in bottom_part_3)), 
                                       max(r for r, c in bottom_part_3)+n, 
                                  slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                                 )[-2]
    tail_tip = cgd.draw_block((max(r for r, c in tail),
                                    (min(c for r,c in tail if r == max(r for r, c in tail)), 
                                     max(c for r,c in tail if r == max(r for r, c in tail)))
                                   ), max(r for r, c in tail)+1)
    
    return {
        '14': [tail_tip, [cgc.hex_to_rgb("#f66242")]],
        '13': [tail, [cgc.hex_to_rgb("#c2ffed")]],
        '12': [main_body_part, [cgc.hex_to_rgb("#4d947f")]],
        
        '11': [bells, [cgc.hex_to_rgb("#f1bb68")]],
        '10': [strings, [cgc.hex_to_rgb("#3f4447")]],
        '9': [top_plate, [cgc.hex_to_rgb("#cfdfdf")]],
        '8': [center_part, [cgc.hex_to_rgb("#7bd6bb")]],
        
        '7': [top_tip, [cgc.hex_to_rgb("#c2ffed")]],
        '6': [boundary, [cgc.hex_to_rgb("#f1bb68")]],
        '5': [center_part_boundary_1, [cgc.hex_to_rgb("#c2ffed")]],
        '4': [center_part_boundary_2, [cgc.hex_to_rgb("#f1bb68")]],
        
        
        '3': [valid_dots, [cgc.hex_to_rgb("#f18968")]],
        '2': [acc_dots_centers, [cgc.hex_to_rgb("#f4fffe")]],
        '1': [top_part, [cgc.hex_to_rgb("#c2ffed")]],
         '0': [top_part_acc, [cgc.hex_to_rgb("#e4547f")]],
        
            }
umbrella_locs = [(0, 34), (15, 26)]
for umbrella_loc in umbrella_locs:
    print(umbrella_loc)
    dict_ = draw_umbrella(umbrella_loc[0], umbrella_loc[1], 1)
    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01) 

mouse_locs = [(22, 3), (7,10)]
for mouse_loc in mouse_locs:
    dict_ = draw_mouse(mouse_loc[0], mouse_loc[1],1)
    for key in dict_.keys():
            part  = dict_.get(key)[0]
            colors = dict_.get(key)[1] 
            if len(colors) == 1:

                select_part= cg.select_mask(part,hex_rc_arr)
                hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
            else:
                hex_colors = cgc.color_row_gradient(part, 
                                            colors[0],colors[1],
                                            hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01) 

base_hex_colors = hex_colors.copy()



def translate_hex_shape_col(coords, pivot_rc, scale, detail_info):
    """Rigidly shift a hex shape's columns toward the pivot, without
    deforming its internal geometry.
 
    These parts (top_plate/strings/bells) are built from a LEFT half
    (bend='left', anchored at the min column) and a RIGHT half
    (bend='right', anchored at the max column), mirrored around the
    center. Shrinking toward the pivot means the two halves move in
    OPPOSITE directions -- left side's column increases toward the pivot,
    right side's decreases -- so each half needs its own anchor/delta,
    not one delta applied to the whole combined shape.
    """
    coords = np.asarray(coords).reshape(-1, 2)
    if coords.shape[0] == 0:
        return []
    rows, cols = coords[:, 0], coords[:, 1]
 
    IMG_W, IMG_H, HEX_R, dx_hex_center, dy_hex_center = detail_info
    apothem = np.sqrt(3) / 2 * HEX_R
    px0, _ = cg.hex_center_pixel(pivot_rc[0], pivot_rc[1], detail_info)
 
    mid_col = (cols.min() + cols.max()) / 2.0
    left_mask = cols <= mid_col
    right_mask = ~left_mask
 
    cols_out = cols.copy()
    for mask, pick_anchor in [(left_mask, np.min), (right_mask, np.max)]:
        if not np.any(mask):
            continue
        anchor_row = rows[mask][0]
        anchor_col = pick_anchor(cols[mask])
        x_anchor, _ = cg.hex_center_pixel(anchor_row, anchor_col, detail_info)
        # this naturally picks up the right sign: if the anchor sits left of
        # the pivot, scaling (x_anchor - px0) toward 0 moves it rightward
        # (positive delta); if it sits right of the pivot, the same formula
        # moves it leftward (negative delta) -- no separate +/- needed
        x_anchor_scaled = px0 + scale * (x_anchor - px0)
        stagger_anchor = apothem if anchor_row % 2 == 1 else 0.0
        anchor_col_new = round((x_anchor_scaled - stagger_anchor) / dx_hex_center)
        delta = anchor_col_new - anchor_col
        cols_out[mask] = cols[mask] + delta
 
    rows_int = rows.astype(int)
    return list(dict.fromkeys(zip(rows_int.tolist(), cols_out.astype(int).tolist())))
#------read solution-----------------------
pin = TRAJ.MagPin()
sol_pin = pin.solve()
theta_pin = sol_pin.y[0] 

kap = TRAJ.KapitzaPendulum()
sol_k = kap.solve()
# phi (rod angle) and pivot height, both as functions of the solved time grid
ph = sol_k.y[0]
zp = kap.z_pivot(sol_k.t)
 
# bob position in physical (world) metres: x = l*sin(phi), z = z_pivot + l*cos(phi)
x_bob = kap.l * np.sin(ph)
z_bob = zp + kap.l * np.cos(ph)
 
# the pivot's vertical motion carries z_bob's *mean* well away from 0 (it
# sits near +l, not near the origin), while x_bob is already ~symmetric
# about 0. Center both on their own mean so that adding a half-canvas
# offset (OFFSET_X/OFFSET_Y below) places the trajectory in the middle
# of whichever canvas it's projected onto, instead of drifting off-frame.
x_bob_c = x_bob - x_bob.mean()
z_bob_c = z_bob - z_bob.mean()
 
DOMAIN_W = x_bob_c.max() - x_bob_c.min()
DOMAIN_H = z_bob_c.max() - z_bob_c.min()
 
CANVAS_PHYSICAL_H = DOMAIN_H*5
CANVAS_PHYSICAL_W = DOMAIN_W*5
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
OFFSET_X = CANVAS_PHYSICAL_W /2
OFFSET_Y = CANVAS_PHYSICAL_H / 2+ (CANVAS_PHYSICAL_H /5)
 
 
# ── ANIMATION TIMING ────────────────────────────────────────────────────
Phase0_end = 10           # intro frames that just hold the base scene
total_seconds = 18.0
 
# one animation frame per solved time-step -- no resampling


Phase1_end = Phase0_end + len(sol_k.t)
Phase2_end = Phase1_end + len(sol_pin.t)

total_frames = Phase2_end

X = x_bob_c   # bob x-position, one value per sol_k.t sample
Y = z_bob_c   # bob z-position, one value per sol_k.t sample
 
 
# ── PROJECT ONTO THE HEX GRID: physical grid -> pixel -> hex index
rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}
grid_hex_rc = cg.world_metres_to_hex_index(
    X.ravel() + OFFSET_X, Y.ravel() + OFFSET_Y, detail_info,
    canvas_physical_x_range=canvas_physical_x_range,
    canvas_physical_y_range=canvas_physical_y_range,
)

frame_to_hex_idx = {
    frame_i: rc_to_idx[rc]
    for frame_i, rc in enumerate(grid_hex_rc)
    if rc in rc_to_idx
}

LAST_PHASE1_FRAME = len(sol_k.t) - 1     # last valid trajectory index (0-based)

hex_idx = frame_to_hex_idx[LAST_PHASE1_FRAME]
row, col = hex_rc_arr[hex_idx]
pin_row, pin_col = row, col

# ── UPDATE ──────────────────────────────────────────────────────────────
 
 
def update(snapshot):
    if snapshot < Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(base_hex_colors)
    elif snapshot >= Phase0_end and snapshot <= Phase1_end:
        current_snapshot = snapshot - Phase0_end
 
        current_hex_colors = animation_base_hex_colors.copy()

        
        hex_idx = frame_to_hex_idx.get(current_snapshot)
        if hex_idx is not None:
            row, col = hex_rc_arr[hex_idx]
            dict_ = draw_scale(row, col, n = 2)

            for key in dict_.keys():
                part  = dict_.get(key)[0]
                colors = dict_.get(key)[1] 
                if len(colors) == 1:

                    select_part= cg.select_mask(part,hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
                else:
                    current_hex_colors = cgc.color_row_gradient(part, 
                                                colors[0],colors[1],
                                                hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01) 

        pc.set_facecolor(current_hex_colors)
    elif snapshot >= Phase1_end:
        current_snapshot2 = snapshot - Phase1_end
 
        current_hex_colors = animation_base_hex_colors.copy()
 
        if 0 <= current_snapshot2 < len(theta_pin):
            theta_now = theta_pin[current_snapshot2]
 
            foreshorten = abs(np.cos(theta_now - pin.theta_B))
 
            dict_ = draw_scale(pin_row, pin_col, n=2)
 
            for key in dict_.keys():
                part = dict_.get(key)[0]
                colors = dict_.get(key)[1]
                if key in ('9', '10', '11'):
                    # strings, top_plate, bells: these are anchored off a
                    # reference column (from `other`), not the pivot -- they
                    # don't deform, they just start from a different (shrunk)
                    # column, so this is a rigid translation, not a rotation
                    part = translate_hex_shape_col(part, (pin_row, pin_col), foreshorten, detail_info)
                else:
                    part = cg.foreshorten_hex_shape(part, (pin_row, pin_col), foreshorten, detail_info)
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3) * sigma_color)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                                colors[0], colors[1],
                                                hex_rc_arr, hex_colors, sort='col', sigma_color=sigma_color, end_weight=0.01)
 
        pc.set_facecolor(current_hex_colors)
    return (pc,)

total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
 
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'mononoke_animation.mp4')
 
print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")
 