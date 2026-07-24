import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
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



sigma_color = 0.01

skyline = 10
select_skyline = [r == skyline for r,c in hex_rc_arr ]

ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
wall = [(r,c) for r,c in hex_rc_arr if r <skyline]
outside = cgd.draw_block((0,(2, 32)) , skyline)
hex_colors = cgc.color_row_gradient(wall, cgc.hex_to_rgb("#020c20"), cgc.hex_to_rgb("#747a89"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.1, period=5)    

hex_colors = cgc.color_row_gradient(ground, cgc.hex_to_rgb("#474746"), cgc.hex_to_rgb("#8d8d8b"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.05)    

head_row = 17
center_col = 17
n_head = 2

n_pickel = 2
pickel_start = 4
picke_end = 16

pickel = cg.hex_neighbours_n(pickel_start,center_col-3, n=n_pickel , keep_origin = True)+cgd.draw_block((pickel_start,(center_col-3-n_pickel, center_col-3+n_pickel)) , picke_end)+cg.hex_neighbours_n(picke_end,center_col-3, n=n_pickel , keep_origin = True)

pickel_front = cg.hex_neighbours_n(pickel_start,center_col-3, n=n_pickel-1 , keep_origin = True)+cgd.draw_block((pickel_start,(center_col-3-n_pickel+1, center_col-3+n_pickel-1)) , picke_end)+cg.hex_neighbours_n(picke_end,center_col-3, n=n_pickel-1 , keep_origin = True)

pickel_eye = [(pickel_start+n_pickel, min(c for r,c in pickel if r == pickel_start+n_pickel)+1),
              (pickel_start+n_pickel, max(c for r,c in pickel if r == pickel_start+n_pickel)-1)]
pickel_eyebrow = cgd.horizontal_lines([(pickel_start, (min(c for r,c in pickel if r == pickel_start)+1, 
                                                       max(c for r,c in pickel if r == pickel_start)-1))])

_, _, _,_, pickel_mouth, vertex = cgd.draw_trapezoid(pickel_start+n_pickel+2, 
                                                     min(c for r,c in pickel if r == pickel_start+n_pickel+2)+1, 
                                                     max(c for r,c in pickel if r == pickel_start+n_pickel+2)-1, 
                                                     pickel_start+n_pickel+3,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'rl')

pickel_tongue = [(max(r for r,c in pickel_mouth ),max(c for r,c in pickel_mouth if r == max(r for r,c in pickel_mouth )) ) ]

select_pickel = cg.select_mask(pickel  ,hex_rc_arr)
hex_colors[select_pickel] = cgc.select_normal_color(select_pickel,cgc.hex_to_rgb("#3b8d0e"), np.ones(3)*sigma_color*3) 

select_pickel_front = cg.select_mask(pickel_front  ,hex_rc_arr)
hex_colors[select_pickel_front] = cgc.select_normal_color(select_pickel_front,cgc.hex_to_rgb("#88d608"), np.ones(3)*sigma_color*3) 


select_pickel_eye = cg.select_mask(pickel_eye  ,hex_rc_arr)
hex_colors[select_pickel_eye] = cgc.select_normal_color(select_pickel_eye,[1,1,1], np.ones(3)*sigma_color) 

select_pickel_eyebrow = cg.select_mask(pickel_eyebrow  ,hex_rc_arr)
hex_colors[select_pickel_eyebrow] = cgc.select_normal_color(select_pickel_eyebrow, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color) 

select_pickel_mouth = cg.select_mask(pickel_mouth  ,hex_rc_arr)
hex_colors[select_pickel_mouth] = cgc.select_normal_color(select_pickel_mouth,cgc.hex_to_rgb("#42151c"), np.ones(3)*sigma_color) 

select_pickel_tongue = cg.select_mask(pickel_tongue  ,hex_rc_arr)
hex_colors[select_pickel_tongue] = cgc.select_normal_color(select_pickel_tongue,cgc.hex_to_rgb("#d44545"), np.ones(3)*sigma_color) 

n_leg = 2
leg_start = 5
leg_end = 20
leg_col = center_col+6


leg_head = cg.hex_neighbours_n(leg_start,leg_col, n=n_leg , keep_origin = True)
leg_hair = cgd.draw_triangle(leg_start-n_leg-n_leg, leg_col,  leg_start, 
                             slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                            )[-2]+cgd.draw_triangle(leg_start+n_leg, max(c for r,c in leg_head if r == leg_start+n_leg),  leg_start-n_leg, 
                             slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                                   )[-2]+cgd.draw_triangle(leg_start-n_leg, 
                                                                           max(c for r,c in leg_head if r == leg_start-n_leg),  
                                                                           leg_start+n_leg, 
                             slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                                   )[-2]
select_leg_hair = cg.select_mask(leg_hair ,hex_rc_arr)
hex_colors[select_leg_hair] = cgc.select_normal_color(select_leg_hair, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color*3) 


knee_r = leg_start+8
knee = cg.hex_neighbours_n(knee_r,leg_col+1, n=n_leg-1 , keep_origin = True)
thigh = cgd.draw_slope_0p5_diagonal(leg_start, min(c for r,c in leg_head if r == leg_start), knee_r,
                                   left_down=False, right_down=True, left_up=False, right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(leg_start, min(c for r,c in leg_head if r == leg_start)+1, leg_start+6,
                                   left_down=False, right_down=True, left_up=False, right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(knee_r, leg_col+1, leg_start+2,
                                   left_down=False, right_down=False, left_up=True , right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(leg_start, leg_col, leg_start+3,
                                   left_down=False, right_down=True , left_up= False, right_up=False
                                  )
calf = cgd.draw_slope_0p5_diagonal(knee_r, min(c for r,c in knee if r == knee_r), knee_r+5,
                                   left_down=False, right_down=True, left_up=False, right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(knee_r, min(c for r,c in knee if r == knee_r)+1, knee_r+3,
                                   left_down=False, right_down=True, left_up=False, right_up=False)
ankle_r = max(r for r, c in calf)
foot = cgd.draw_slope_0p5_diagonal(ankle_r, min(c for r,c in calf if r == ankle_r), ankle_r+2,
                                   left_down=True , right_down=False, left_up=False, right_up=False
                                  )
leg = list(set(leg_head+knee +thigh+calf+foot))
select_leg = cg.select_mask(leg,hex_rc_arr)
hex_colors[select_leg] = cgc.select_normal_color(select_leg,cgc.hex_to_rgb("#c6c0b5"), np.ones(3)*sigma_color*3) 

leg_eye = [(leg_start+1, min(c for r,c in leg_head if r == leg_start+1)),
              (leg_start+1, max(c for r,c in leg_head if r == leg_start+1)-1)]
leg_eyebrow = cgd.horizontal_lines([(leg_start-1, (min(c for r,c in leg_head if r == leg_start-1), 
                                                       max(c for r,c in leg_head if r == leg_start-1)-1))])




select_leg_eye = cg.select_mask(leg_eye  ,hex_rc_arr)
hex_colors[select_leg_eye] = cgc.select_normal_color(select_leg_eye,[1,1,1], np.ones(3)*sigma_color) 

select_leg_eyebrow = cg.select_mask(leg_eyebrow  ,hex_rc_arr)
hex_colors[select_leg_eyebrow] = cgc.select_normal_color(select_leg_eyebrow, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color) 

Rick_head_loc = (head_row-3, center_col-8)
Rick_hair =  cgd.draw_triangle(Rick_head_loc[0]-4, Rick_head_loc[1],  head_row, 
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                               )[-2]+cgd.draw_triangle(Rick_head_loc[0]+6, Rick_head_loc[1],  head_row-4, 
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                                                                      )[-2]+cgd.draw_triangle(Rick_head_loc[0], Rick_head_loc[1],  
                                                                                              head_row+2, 
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                                                      )[-2]

Rick_head = cg.hex_neighbours_n(Rick_head_loc[0], Rick_head_loc[1], n=1 , keep_origin = True)
Rick_neck_r = max(r for r,c in Rick_hair)
Rick_neck = cgd.horizontal_lines([(Rick_neck_r, (Rick_head_loc[1]-1, Rick_head_loc[1]+1))])

_, _, _,_, Rick_body, vertex = cgd.draw_trapezoid(Rick_neck_r+1, 
                                                     min(c for r,c in Rick_hair if r == Rick_neck_r-1), 
                                                     max(c for r,c in Rick_hair if r == Rick_neck_r-1), 
                                                     22,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr')



Morty_head = cg.hex_neighbours_n(head_row, center_col, n=n_head , keep_origin = True)


Morty_neck = [(r,c) for r,c in Morty_head if r == head_row+n_head]

_, _, _,_, Morty_body, vertex = cgd.draw_trapezoid(head_row+n_head, 
                                                     min(c for r,c in Morty_head if r == head_row+n_head), 
                                                     max(c for r,c in Morty_head if r == head_row+n_head), 
                                                     22,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr')


select_Morty_body = cg.select_mask(Morty_body  ,hex_rc_arr)
hex_colors[select_Morty_body] = cgc.select_normal_color(select_Morty_body,cgc.hex_to_rgb("#fffd3b"), np.ones(3)*sigma_color) 


select_Morty_head = cg.select_mask(Morty_head  ,hex_rc_arr)
#hex_colors[select_Morty_head] = cgc.select_normal_color(select_Morty_head, cgc.hex_to_rgb("#693f1d"), np.ones(3)*sigma_color) 
hex_colors = cgc.color_hex_gradient(Morty_head,  cgc.hex_to_rgb("#9f6230") , cgc.hex_to_rgb("#9f4830" ),
                                    hex_rc_arr, hex_colors, (head_row, center_col),
                                    n_head,  sigma_color=sigma_color, end_weight= 0.01)

select_Rick_neck = cg.select_mask(Rick_neck  ,hex_rc_arr)
hex_colors[select_Rick_neck] = cgc.select_normal_color(select_Rick_neck, cgc.hex_to_rgb("#c6c0b5"), np.ones(3)*sigma_color) 

select_Rick_hair = cg.select_mask(Rick_hair ,hex_rc_arr)
hex_colors[select_Rick_hair] = cgc.select_normal_color(select_Rick_hair, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color*3) 

select_Rick_head = cg.select_mask(Rick_head  ,hex_rc_arr)
hex_colors[select_Rick_head] = cgc.select_normal_color(select_Rick_head, cgc.hex_to_rgb("#c6c0b5"), np.ones(3)*sigma_color) 


select_Rick_body = cg.select_mask(Rick_body  ,hex_rc_arr)
hex_colors[select_Rick_body] = cgc.select_normal_color(select_Rick_body,[0.95, 0.95, 0.95], np.ones(3)*sigma_color) 



pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)


OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'RickMorty_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')