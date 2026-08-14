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
#------------bkgd-----------------

sky_line_color = "#9d6c4d"

sidewall_color = "#230137"
frontwall_color = "#230137"


start_row = 20
start_col = [14, 35]
end_row = 22
BLOCK_end_row = 5
left_STREET_LINE, right_STREET_LINE, start_LINE, end_LINE, STREET_LINE_bound_region, vertices = cgd.draw_trapezoid(start_row, 
                                                                                                                   start_col[0],
                                                                                                                   start_col[1],end_row  , 
                                                                                                                   slope_left = '0.5', slope_right = '0.5', direction = 'lr')

right_STREET_LINE_row_to_col = {r: c for r, c in right_STREET_LINE}
left_STREET_LINE_row_to_col = {r: c for r, c in left_STREET_LINE}

# ── SKY ─────────────────────────────────────────────────────────────    
SKY = [(r,c) for r, c in hex_rc_arr if r< start_row-1]
hex_colors = cgc.color_row_gradient(SKY, cgc.hex_to_rgb("#c1fff4"), cgc.hex_to_rgb("#ffffff"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01) 
hex_colors = cgc.color_row_gradient([(r,c) for r, c in hex_rc_arr if r>=BLOCK_end_row],
                                    cgc.hex_to_rgb("#e8e8da"), cgc.hex_to_rgb("#e8dcb1"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01)    

hex_colors = cgc.color_row_gradient(STREET_LINE_bound_region, cgc.hex_to_rgb("#e8dcb1"), cgc.hex_to_rgb("#cbb97a"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01)    

# ── SKYLINE ─────────────────────────────────────────────────────────────    
select_SKYLINE = [row== start_row-1 for row, col in hex_rc_arr]
SKYLINE_colors = cgc.select_normal_color(select_SKYLINE, cgc.hex_to_rgb(sky_line_color), np.ones(3)*sigma_color)


left_STREET_END = vertices[2]
right_STREET_END = vertices[3]


side_row_start = -1 
SIDEWALL = [(row, col) for row, col in hex_rc_arr if
                   (col > right_STREET_LINE_row_to_col[max(min(row, right_STREET_END[0]),start_row)] or 
                   col < left_STREET_LINE_row_to_col[max(min(row, left_STREET_END[0]),start_row)]) and
                   row >= side_row_start and row <= end_row]
 
FRONTWALL = [(row, col) for row, col in hex_rc_arr if (col <left_STREET_END[1] or col > right_STREET_END[1]) and 
                    row >= side_row_start  and row <= end_row ]
block_center_col = 22
BLOCK_part0 =  cgd.draw_block((0,  (block_center_col-1, block_center_col+1)),start_row)
BLOCK_part1 = cgd.draw_triangle(start_row, min(c for r,c in BLOCK_part0 if r == start_row), BLOCK_end_row, 
                         slope_left = '0.5', slope_right = 'inf', direction = 'lr'
                             )[-2]
BLOCK_part2 = cgd.draw_block((BLOCK_end_row,  (min(c for r,c in BLOCK_part1 if r == BLOCK_end_row), 
                                               max(c for r,c in BLOCK_part1 if r == BLOCK_end_row))),0)

hex_colors = cgc.color_row_gradient(BLOCK_part1+BLOCK_part2, cgc.hex_to_rgb("#d8cec4" ), cgc.hex_to_rgb("#c4a295"), 
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01)  

hex_colors = cgc.color_row_gradient(BLOCK_part0, cgc.hex_to_rgb("#e8dcb1" ), cgc.hex_to_rgb("#842648"), 
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01)  

n_head = 1
CHAR_head_locs = [(10, block_center_col+4), 
                (1, max(c for r,c in SIDEWALL)+1)]
for CHAR_head_loc in CHAR_head_locs:
    
    head, neck, body_temp, pelvis_temp, thigh_temp,details = cgd.draw_body(CHAR_head_loc[0],CHAR_head_loc[1], n = n_head, start = 'head')
    [center_col, head_center_row, body_center_row, thigh_center_row]  = details

    CHAR_body_part0 = cgd.draw_trapezoid(body_center_row-n_head, 
                                     min(c for r,c in body_temp if r == body_center_row-n_head)-1, 
                                     max(c for r,c in body_temp if r == body_center_row-n_head)+1, 
                                     body_center_row+n_head,slope_left = '0.5', slope_right = '0.5', direction = 'rl')[-2]   

    CHAR_body_part1 = cgd.draw_block((body_center_row+n_head, (
        min(c for r,c in CHAR_body_part0 if r == body_center_row+n_head), 
        max(c for r,c in CHAR_body_part0 if r == body_center_row+n_head)
                                                      )),thigh_center_row+n_head) 
    CHAR = head+neck+CHAR_body_part0+CHAR_body_part1
    select_CHAR = cg.select_mask(CHAR,hex_rc_arr)
    hex_colors[select_CHAR] = cgc.select_normal_color(select_CHAR,  [0.3, 0.3, 0.3], np.ones(3)*sigma_color)
hex_colors = cgc.color_row_gradient(SIDEWALL, cgc.hex_to_rgb("#e8dcb1"), cgc.hex_to_rgb("#cbb97a"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01)  

hex_colors = cgc.color_row_gradient(FRONTWALL, cgc.hex_to_rgb("#e8dcb1" ), cgc.hex_to_rgb("#842648"), 
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01)  


center_row = 5
center_col = 6
row_seg1 = 3
row_seg2 = 4
row_seg3 = 2
row_seg4 = 2
row_seg5 = 2
n_head = 3

body_part0 = cgd.draw_trapezoid(center_row, center_col-1, center_col+1, center_row+row_seg1, 
                          slope_left = '1.5', slope_right = '1.5', direction = 'lr'
                          )[-2]
body_part1 = cgd.draw_block((center_row+row_seg1, (min(c for r,c in body_part0 if r == center_row+row_seg1), 
                                                   max(c for r,c in body_part0 if r == center_row+row_seg1)
                                                  )),center_row+row_seg1+row_seg2)
body_part2 =  cgd.draw_trapezoid(center_row+row_seg1+row_seg2, 
                                 min(c for r,c in body_part1 if r == center_row+row_seg1+row_seg2), 
                                 max(c for r,c in body_part1 if r == center_row+row_seg1+row_seg2), 
                                 center_row+row_seg1+row_seg2+row_seg3,
                          slope_left = '8/3', slope_right = '8/3', direction = 'rl'
                          )[-2]                    
body_part3 =  cgd.draw_block((center_row+row_seg1+row_seg2+row_seg3-1, 
                              (min(c for r,c in body_part2 if r == center_row+row_seg1+row_seg2+row_seg3-1), 
                               max(c for r,c in body_part2 if r == center_row+row_seg1+row_seg2+row_seg3-1)
                                                  )),center_row+row_seg1+row_seg2+row_seg3+row_seg4)
body_part4 = cgd.draw_trapezoid(center_row+row_seg1+row_seg2+row_seg3+row_seg4, 
                                 min(c for r,c in body_part3 if r == center_row+row_seg1+row_seg2+row_seg3+row_seg4), 
                                 max(c for r,c in body_part3 if r == center_row+row_seg1+row_seg2+row_seg3+row_seg4), 
                                 center_row+row_seg1+row_seg2+row_seg3+row_seg4+row_seg5,
                          slope_left = '0.5', slope_right = '1.5', direction = 'rl'
                          )[-2]              

body = body_part0+body_part1+body_part2

rows = {r for r, c in body}
remove_part0 = [(r, f(cs)) for r in {r for r, c in body_part0} for cs in ([c for rr, c in body_part0 if rr == r],) 
    if cs for f in (min, max, lambda cs: min(cs) + 1, lambda cs: max(cs) - 1)]

remove_part1 = [(r, f(c for rr, c in body_part1 if rr == r)) 
                for r in {r for r, c in body_part1} for f in (min, max)]

remove_part2 = [(r, f(cs)) for r in {r for r, c in body_part2} for cs in ([c for rr, c in body_part2 if rr == r],) 
    if cs for f in (min, max, lambda cs: min(cs) + 1, lambda cs: max(cs) - 1, lambda cs: min(cs) + 2, lambda cs: max(cs) - 2)]

remove =[(r, c) for r, c in remove_part0+remove_part1+remove_part2]


face_v1 = [(r,c) for r,c in body if (r,c) not in remove]

add = cgd.draw_triangle(min(r for r,c in face_v1)+1, center_col, min(r for r,c in face_v1), 
                        slope_left = '0.5', slope_right = '0.5', direction = 'lr' )[-2]

face = [(r,c) for r,c in face_v1 if (r,c) not in add]

right_ear = cgd.draw_triangle(center_row-1, max(c for r,c in body)-1, center_row+row_seg1, 
                         slope_left = '1.5', slope_right = 'inf', direction = 'lr'
                             )[-2]+cgd.verticle_line(center_row-1, max(c for r,c in body)-1, center_row+row_seg1, bend = 'right')
left_ear = cgd.draw_triangle(center_row-1, min(c for r,c in body)+1, center_row+row_seg1, 
                         slope_left = 'inf', slope_right ='1.5' , direction = 'lr'
                             )[-2]+cgd.verticle_line(center_row-1, min(c for r,c in body)+1, center_row+row_seg1, bend = 'left')

eyes = cgd.verticle_line(min(r for r,c in face)+2, center_col, min(r for r,c in face)+4, bend = 'right'
                         )+cgd.verticle_line(min(r for r,c in face)+2, center_col, min(r for r,c in face)+4, bend = 'left'
                         )+cgd.verticle_line(min(r for r,c in face)+2, center_col-2, min(r for r,c in face)+4, bend = 'left'
                         )+cgd.verticle_line(min(r for r,c in face)+2, center_col+2, min(r for r,c in face)+4, bend = 'right'
                         )

row_seg6 = 4
torso_row = center_row+row_seg1+row_seg2+row_seg3+row_seg4-1
torso = cgd.draw_trapezoid(torso_row, 
                                 min(c for r,c in body_part3 if r == torso_row)+1, 
                                 max(c for r,c in body_part3 if r == torso_row)-1, 
                                 torso_row+row_seg6,
                          slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                          )[-2]      
select_torso = cg.select_mask(torso,hex_rc_arr)
hex_colors[select_torso] = cgc.select_normal_color(select_torso,  [0,0,0], np.ones(3)*sigma_color)
n_foot = 2
left_foot = cg.hex_neighbours_n(torso_row+row_seg6, min(c for r,c in torso if r == torso_row+row_seg6), n=n_foot, keep_origin = True
                               )+cg.hex_neighbours_n(torso_row+row_seg6-2, 
                                                     min(c for r,c in torso if r == torso_row+row_seg6), n=n_foot, keep_origin = True
                               )

right_foot = cg.hex_neighbours_n(torso_row+row_seg6, max(c for r,c in torso if r == torso_row+row_seg6), n=n_foot, keep_origin = True
                               )+cg.hex_neighbours_n(torso_row+row_seg6-2, 
                                                     max(c for r,c in torso if r == torso_row+row_seg6), n=n_foot, keep_origin = True
                               )
temp_r = torso_row+row_seg6-3
left_foot_detail = cgd.draw_slope_1p5_diagonal(temp_r, min(c for r,c in left_foot if r == temp_r)+1, 
                                               temp_r+1, left_down=True, right_down=False, left_up=False, right_up=False
                                              )+cgd.horizontal_lines([(temp_r, (min(c for r,c in left_foot if r == temp_r)+1,
                                                                                max(c for r,c in left_foot if r == temp_r)-1))])

right_foot_detail = cgd.draw_slope_1p5_diagonal(temp_r, max(c for r,c in right_foot if r == temp_r)-1, 
                                               temp_r+1, left_down=False, right_down=True, left_up=False, right_up=False
                                              )+cgd.horizontal_lines([(temp_r, (min(c for r,c in right_foot if r == temp_r)+1,
                                                                                max(c for r,c in right_foot if r == temp_r)-1))])

front_feet = cgd.draw_trapezoid(torso_row+row_seg6, max(c for r,c in left_foot if r == torso_row+row_seg6), center_col,
                                 torso_row+row_seg6+2,slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                          )[-2]+cgd.draw_trapezoid(torso_row+row_seg6,center_col, min(c for r,c in right_foot if r == torso_row+row_seg6), 
                                 torso_row+row_seg6+2,slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                          )[-2]   

front_feet_detail = [(r,c) for r,c in front_feet if r==max(r for r,c in front_feet)]
right_ear_boundary = [(r, f(cs)) for r in {r for r, c in right_ear} for cs in ([c for rr, c in right_ear if rr == r],) 
                                  if cs for f in (min, max, lambda cs: min(cs) + 1)
                                 ]

left_ear_boundary = [(r, f(cs)) for r in {r for r, c in left_ear} for cs in ([c for rr, c in left_ear if rr == r],) 
                                  if cs for f in (min, max, lambda cs: max(cs) - 1)
                                 ]

right_ear_detail = [(r,c) for r,c in right_ear if (r,c) not in right_ear_boundary]
left_ear_detail = [(r,c) for r,c in left_ear if (r,c) not in left_ear_boundary]

details = cg.hex_neighbours_n(torso_row+row_seg6, min(c for r,c in torso if r == torso_row+row_seg6), n=n_foot-1, 
                                   keep_origin = True
                               )+cg.hex_neighbours_n(torso_row+row_seg6, 
                                                     max(c for r,c in torso if r == torso_row+row_seg6), n=n_foot-1, keep_origin = True
                               )+right_ear_detail+left_ear_detail+left_foot_detail+right_foot_detail+front_feet_detail

hex_colors = cgc.color_row_gradient(body_part3+body_part4, cgc.hex_to_rgb("#251c3b"), cgc.hex_to_rgb("#8a68e8"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01)    

select_feet = cg.select_mask(left_foot+right_foot+front_feet,hex_rc_arr)
hex_colors[select_feet] = cgc.select_normal_color(select_feet,  [0,0,0], np.ones(3)*sigma_color)


select_ears = cg.select_mask(right_ear+left_ear,hex_rc_arr)
hex_colors[select_ears] = cgc.select_normal_color(select_ears,  [0,0,0], np.ones(3)*sigma_color)

select_details = cg.select_mask(details,hex_rc_arr)
hex_colors[select_details] = cgc.select_normal_color(select_details, cgc.hex_to_rgb("#3b96f4")  , np.ones(3)*sigma_color) 

select_body= cg.select_mask(body,hex_rc_arr)
hex_colors[select_body] = cgc.select_normal_color(select_body, cgc.hex_to_rgb("#8a68e8")  , np.ones(3)*sigma_color) 

select_face = cg.select_mask(face,hex_rc_arr)
hex_colors[select_face] = cgc.select_normal_color(select_face,[0,0,0], np.ones(3)*sigma_color)  

select_eyes = cg.select_mask(eyes,hex_rc_arr)
hex_colors[select_eyes] = cgc.select_normal_color(select_eyes,cgc.hex_to_rgb("#3b96f4") , np.ones(3)*sigma_color)    



pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'omen_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')