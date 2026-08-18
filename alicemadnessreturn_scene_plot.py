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
skyline = 10

ground_center_row = 35
ground_center_col = 20
n_ground = 25
ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
wall = [(r,c) for r,c in hex_rc_arr if r <skyline]
hex_colors = cgc.color_row_gradient(wall, cgc.hex_to_rgb("#cd874e"), cgc.hex_to_rgb("#cdc895"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05)   



hex_colors =cgc.color_row_gradient(ground, cgc.hex_to_rgb("#cdb895"), cgc.hex_to_rgb("#703600"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*2, end_weight= 0.05)    


def modify_pelvis(pelvis):
    pelvis_left, pelvis_right = min(pelvis, key=lambda x: x[1]), max(pelvis, key=lambda x: x[1])
    pelvis_tot = pelvis+[(pelvis_left[0], pelvis_left[1] - 1), (pelvis_right[0], pelvis_right[1] + 1)]
    return pelvis_tot
def modify_body(body, body_center_row, view = 'front'):
    body_left, body_right = min(c for r,c in body if r == body_center_row), max(c for r,c in body if r == body_center_row)
    remove = [(body_center_row, min(c for r,c in body if r == body_center_row)),
              (body_center_row, max(c for r,c in body if r == body_center_row)),
              (body_center_row+1, min(c for r,c in body if r == body_center_row+1)),
              (body_center_row+1, max(c for r,c in body if r == body_center_row+1)),
              (body_center_row-1, min(c for r,c in body if r == body_center_row-1)),
              (body_center_row-1, max(c for r,c in body if r == body_center_row-1))]
    body_final = [(r,c) for r, c in body if (r,c) not in remove]
    return body_final

def modify_thigh(thigh, thigh_center_row, view = 'front'):
    
    thigh_top_row = min(r for r,c in thigh)
    left, right = min(c for r,c in thigh if r == thigh_top_row)-1, max(c for r,c in thigh if r == thigh_top_row)+1
    add = cgd.draw_slope_0p5_diagonal(thigh_top_row, left, thigh_center_row-1, 
                                         left_down=True , right_down=False, left_up=False, right_up=False
                                        )+cgd.draw_slope_0p5_diagonal(thigh_top_row, right, thigh_center_row-1, 
                                         left_down=False , right_down=True , left_up=False, right_up=False
                                        )

 
    thigh_remove = [(thigh_center_row, min(c for r,c in thigh if r == thigh_center_row)),
                        (thigh_center_row, min(c for r,c in thigh if r == thigh_center_row)+1),
                        (thigh_center_row, max(c for r,c in thigh if r == thigh_center_row)),
                        (thigh_center_row+1, min(c for r,c in thigh if r == thigh_center_row+1)),
                        (thigh_center_row+1, max(c for r,c in thigh if r == thigh_center_row+1)),
                        (thigh_center_row+2, min(c for r,c in thigh if r == thigh_center_row+2)),
                        (thigh_center_row+2, max(c for r,c in thigh if r == thigh_center_row+2)),
                       ]
    thigh_final = [x for x in thigh if x not in thigh_remove]
    return thigh_final+add

def draw_shoulder(body, n):
    
    body_top_r = min(r for r, c in body)
    body_bottom_r = max(r for r, c in body)
    

    arm_basic = cgd.draw_slope_0p5_diagonal(body_top_r, min(c for r, c in body if r ==body_top_r), body_bottom_r,
                                        left_down=True , right_down= False, left_up=False, right_up=False)
    
    #verticle_line(body_top_r, min(c for r, c in body if r ==body_top_r), body_bottom_r)
                                              
    elbow_start =max(arm_basic, key=lambda x: x[0])

    return arm_basic,elbow_start
    
def draw_calf(thigh, thigh_center_row, n):   
    thigh_bottom_r =  max(r for r, c in thigh)
    ankle_row = thigh_bottom_r+n
    knee_extensions = math.ceil(n/2)

    calf_tot = []
    knee_c= cgd.draw_slope_0p5_diagonal(thigh_bottom_r, max(c for r, c in thigh if r ==thigh_bottom_r), thigh_bottom_r+1,
                                        left_down= False, right_down=True , left_up=False, right_up=False)
    
    
    knee = (thigh_bottom_r+1, max(c for r,c in knee_c if r ==thigh_bottom_r+1))
    start_ = knee
    calf_tot.extend([knee])
    #print(knee)
    for calf_thick in range(knee_extensions ):
        calf =cgd.draw_slope_0p5_diagonal(start_[0], start_[1]-calf_thick, ankle_row, left_down=True , 
                                         right_down=False, left_up=False, right_up=False)
        if calf_thick == 0:
            ankle = max(calf, key=lambda x: x[0])
        calf_tot += calf
        start_ = calf[math.floor(len(calf)/2)]
    #print(calf_tot)
    return calf_tot, knee, ankle

def draw_feet(ankle, n):   
    foot = cgd.verticle_line(ankle[0], ankle[1], ankle[0]+n)

    return foot


def draw_Alice(center_row, center_col, n_head):
    head, neck, body, pelvis, thigh, raw_body_detail= cgd.draw_body(center_row, center_col, n = n_head)
    select_head= cg.select_mask(head,hex_rc_arr)
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail

    pelvis_final = modify_pelvis(pelvis)
    body_final = modify_body(body, body_center_row )
    thigh_final = modify_thigh(thigh, thigh_center_row)

    arm,elbow_start = draw_shoulder(body, n_head)

    calf, knee, ankle = draw_calf(thigh_final, thigh_center_row, n_head)
    foot = draw_feet(ankle, n_head)



    remove_hair = [(head_center_row, min(c for r,c in head if r == head_center_row)),
                  (head_center_row, max(c for r,c in head if r == head_center_row))
                  ]
    add_hair = cgd.verticle_line(head_center_row, min(c for r,c in head if r == head_center_row)+1, 
                                 head_center_row+n_head+n_head, bend = 'right'
                                    )+cgd.verticle_line(head_center_row, min(c for r,c in head if r == head_center_row), 
                                                        head_center_row+n_head+n_head+n_head, bend = 'right'
                                    )
    hair = [(r,c)  for r,c in head if (r,c) not in remove_hair
           ] +neck

    face = cg.hex_neighbours_n(max(r for r,c in head)-1,
                               max(c for r,c in head if r ==max(r for r,c in head)-1)-1,
                                   n=n_head-1, keep_origin =True  , return_frontier=False )


    raw_body = head+ neck+ body_final+ pelvis_final+ thigh_final#+arm
    
    sk_row = thigh_center_row-1
    sk = [(r,c)  for r,c in thigh_final if r <=sk_row]
    leg =[(r,c)  for r,c in thigh_final if r >sk_row]
    full_leg = leg+calf+foot

    boots_row = knee[0]+1
    socks = [(r,c)  for r,c in full_leg if r <=boots_row]
    sock_white_rows = [(r,c)  for r,c in socks if  r == min(r for r,c in full_leg) or  
                       r == min(r for r,c in full_leg)+2 or r == min(r for r,c in full_leg)+4 ]

    neckless = [(min(r for r,c in neck), max(c for r,c in neck if r == min(r for r,c in neck)))]

    op = body_final+pelvis_final+sk

    sleeves = [(r,c)  for r,c in arm if r <=body_center_row]

    sleeves_add = cg.hex_neighbours_n(min(r for r,c in arm)+1, max(c for r,c in sleeves if r ==min(r for r,c in arm)+1),
                                      n=1, keep_origin =True )

    forearm = cgd.draw_slope_0p5_diagonal(elbow_start[0], elbow_start[1], elbow_start[0]+n_head, left_down=False, 
                                                     right_down=True, left_up=False, right_up=False
                                         )+cgd.horizontal_lines([(elbow_start[0], (max(c for r, c in op if r==elbow_start[0])+1,
                                                                                  max(c for r, c in op if r==elbow_start[0])+2))])

    full_arm =[(r,c)  for r,c in arm if r >body_center_row]+forearm


    waist_row = body_center_row+n_head//2
    top_body = [(r,c)  for r,c in op if r <=waist_row]
    waist= [(r,c)  for r,c in op if r ==waist_row]
    bot_body = [(r,c)  for r,c in op if r >waist_row and r <max(r for r,c in op)]


    add_apron = [(r, f(cs)) for r in {r for r, c in top_body} for cs in ([c for rr, c in top_body if rr == r],) 
        if cs for f in (min, max)
             ]

    remove = [(r, f(cs)) for r in {r for r, c in bot_body} for cs in ([c for rr, c in bot_body if rr == r],) 
        if cs for f in (min, lambda cs: min(cs) + 1)
             ]
    apron = [(r,c) for r,c in bot_body if (r,c) not in remove]+add_apron +waist


    return {
            'a': [hair, [[0,0,0]]],
            'b': [face, [cgc.hex_to_rgb("#fee9d2")]],
            'c': [add_hair, [[0,0,0]]],
            'd': [op, [cgc.hex_to_rgb("#1400ff")]],
            'e': [apron, [[0.9, 0.9, 0.9]]],
            'f': [sleeves+sleeves_add, [cgc.hex_to_rgb("#1400ff")]],
            'g': [neckless, [[0.6,0.6,0.6]]],
            'h': [full_leg, [[0,0,0]]],
            'i': [sock_white_rows, [[1,1,1]]],
            'j': [full_arm, [cgc.hex_to_rgb("#fee9d2")]],
            }

def draw_teapot(center_row, center_col, n_teapot):
    teapot_body_part1 = cg.hex_neighbours_n(center_row,center_col, n=n_teapot, keep_origin = True)
    teapot_body_part2 = cgd.draw_block((center_row-n_teapot, (min(c for r,c in teapot_body_part1 if r == center_row-n_teapot),
                                                            max(c for r,c in teapot_body_part1 if r == center_row-n_teapot)
                                                           )), center_row-n_teapot-n_teapot)
    teapot_body_part3 = cgd.draw_trapezoid(min(r for r,c in teapot_body_part2), 
                                           min(c for r,c in teapot_body_part2 if r == min(r for r,c in teapot_body_part2)), 
                                           max(c for r,c in teapot_body_part2 if r == min(r for r,c in teapot_body_part2)), 
                                           min(r for r,c in teapot_body_part2)-n_teapot, 
                                           slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                  bend_left = 'left', bend_right = 'right')[-2]
    
    teapot_body_part4 = cgd.draw_trapezoid(center_row+n_teapot, 
                                           min(c for r,c in teapot_body_part1 if r == center_row+n_teapot), 
                                           max(c for r,c in teapot_body_part1 if r == center_row+n_teapot), 
                                           center_row+n_teapot+1, 
                                           slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                  bend_left = 'left', bend_right = 'right')[-2]
    teapot_body_part5 = cgd.draw_trapezoid(max(r for r,c in teapot_body_part4), 
                                           min(c for r,c in teapot_body_part4 if r == max(r for r,c in teapot_body_part4)), 
                                           max(c for r,c in teapot_body_part4 if r == max(r for r,c in teapot_body_part4)), 
                                           max(r for r,c in teapot_body_part4)+1, 
                                           slope_left = '1.5', slope_right = '1.5', direction = 'lr',
                  bend_left = 'left', bend_right = 'right')[-2]
    teapot_body_part6 = cg.hex_neighbours_n(center_row-1,min(c for r,c in teapot_body_part1 if r == center_row-1)-1, 
                                            n=n_teapot-1, keep_origin = False, return_frontier=True)[-1]
    teapot_body_part7 = cgd.draw_slope_0p5_diagonal(center_row, center_col+n_teapot+1, center_row-n_teapot, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True )
    teapot_body_part8 = cgd.draw_slope_0p5_diagonal(min(r for r,c in teapot_body_part7), 
                                                    max(c for r,c in teapot_body_part7), min(r for r,c in teapot_body_part7)+1, 
                                                    left_down=False, right_down=True , left_up=False , right_up= False
                                                   )
    teapot_body_part9 = cgd.draw_slope_0p5_diagonal(center_row+1, 
                                                    min(c for r,c in teapot_body_part1 if r == center_row+1), center_row+2,
                                                    left_down=True , right_down= False, left_up=False , right_up= False
                                                   )
    teapot_body = teapot_body_part1+teapot_body_part2+teapot_body_part3+teapot_body_part4+teapot_body_part5+teapot_body_part6+teapot_body_part7+teapot_body_part8+teapot_body_part9
    
    teapot_eye = cg.hex_neighbours_n(center_row,center_col, n=n_teapot-1, keep_origin = True)
    return {
            'a': [teapot_body, [cgc.hex_to_rgb("#442209")]],
        'b': [teapot_eye, [cgc.hex_to_rgb("#bd0101")]],
           
            }


def draw_ruin(center_row, center_col, n):
    head, neck, body, pelvis, thigh, raw_body_detail= cgd.draw_body(center_row, center_col, n = n, start = 'body')
    raw_body = [(r,c ) for (r,c) in head+neck+body+pelvis if c >= center_col - n+1 ] 
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail

    face = [(r,c ) for (r,c) in raw_body if c ==center_col - n+1 ]
    
    hand = cgd.draw_slope_0p5_diagonal(head_center_row, center_col, head_center_row+2, 
                                                    left_down=False, right_down=True , left_up=False, right_up=False
                                      )+cgd.draw_slope_0p5_diagonal(body_center_row, center_col, body_center_row+2, 
                                                    left_down=False, right_down=True , left_up=False, right_up=False
                                      )
    leg = cgd.draw_slope_0p5_diagonal(max(r for r,c in pelvis), min(c for r,c in pelvis if r == max(r for r,c in pelvis)),
                                      max(r for r,c in pelvis)+2,  left_down=True, right_down= False, left_up=False, right_up=False
                                      )+cgd.draw_slope_0p5_diagonal(max(r for r,c in pelvis), 
                                                                    max(c for r,c in pelvis if r == max(r for r,c in pelvis)),
                                      max(r for r,c in pelvis)+2,  left_down=False, right_down=True , left_up=False, right_up=False
                                      )
    return {
            'a': [raw_body, [[0,0,0]]],
           'b': [face, [cgc.hex_to_rgb("#aa9a88")]],
        'c': [hand, [cgc.hex_to_rgb("#aa9a88")]],
        'd': [leg, [[0,0,0]]],
            }
    
n = 4
center_row = 13
center_col = 30

n_teapot = 2
teapot_center_row = 16
teapot_center_col = 17

n_head = 2
Alice_center_row = 3
Alice_center_col = 5

teapot_center_row = Alice_center_row+n_head+n_head+n_head+n_head+n_head
teapot_center_col = Alice_center_col+2
    

  
char_dict_ = {'ruin': draw_ruin(center_row, center_col, n),
              'Alice': draw_Alice(Alice_center_row, Alice_center_col, n_head),
              'teapot': draw_teapot(teapot_center_row, teapot_center_col, n_teapot)
             }
for dict_ in char_dict_.values(): 
    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*3) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color*3, end_weight= 0.01) 

pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'cg'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'alicemadnessreturn_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')