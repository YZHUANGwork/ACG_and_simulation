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
sigma_color = 0.03

HAIR_COLOR = [0.9, 0.9, 0.9]
SKIN_COLOR = cgc.hex_to_rgb("#e8d1be")
def draw_A2(head_center_row, head_center_col):
    head_raw, raw_body,  line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict,upperarm_dict, forearm_dict, hand_dict,  plot_temp = cgf.draw_human_body(head_center_row, head_center_col)
    ln = SimpleNamespace(**line_dict)
    btlj = SimpleNamespace(**botlimb_joint_dict)
    uplj = SimpleNamespace(**upperlimb_joint_dict)
    tr = SimpleNamespace(**torso_dict) 
    
    n_head = ln.n
    hair_part1 = [(r,c) for r,c in cg.hex_neighbours_n(head_center_row,head_center_col, n=n_head, keep_origin = True) 
                  if r < head_center_row]
    hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                   min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                   ln.shoulder_joint_r, bend = 'right'
                                  )+cgd.verticle_line(max(r for r,c in hair_part1), 
                                   max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                   ln.shoulder_joint_r, bend = 'left'
                                  )
    hair_back = cgd.draw_block((max(r for r,c in hair_part1), 
                                (min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)), 
                                 max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)))
                               ), ln.bellybutton_r)
    hair_part4 = cgd.verticle_line(head_center_row, head_center_col,
                                   head_center_row+1, bend = 'right'
                                  )
    hair_part3 = [(r,c) for r,c in hair_back if (r,c) not in raw_body]
    hair =hair_part1 +hair_part2+hair_part3+hair_part4
    
    black_skin_part1_raw = hand_dict["left_hand"]+hand_dict["right_hand"]+forearm_dict["left_forearm"]+forearm_dict["right_forearm"]+uplj.left_elbow_joint+uplj.right_elbow_joint+uplj.left_wrist_joint+uplj.right_wrist_joint+tr.pelvis_part3+tr.pelvis_part4+btlj.left_thigh_joint+btlj.right_thigh_joint
    
    black_skin_part1_remove = [(min(r for r,c in tr.pelvis_part3),  
                               math.floor((min(c for r,c in tr.pelvis_part3 if r == min(r for r,c in tr.pelvis_part3))+
                                           max(c for r,c in tr.pelvis_part3 if r == min(r for r,c in tr.pelvis_part3)))/2)),
                              (min(r for r,c in tr.pelvis_part3),  
                               math.ceil((min(c for r,c in tr.pelvis_part3 if r == min(r for r,c in tr.pelvis_part3))+
                                           max(c for r,c in tr.pelvis_part3 if r == min(r for r,c in tr.pelvis_part3)))/2))]
    black_skin_part1 = [x for x in black_skin_part1_raw if x not in black_skin_part1_remove]
    
   
    left_shoulder_joint = uplj.left_shoulder_joint
    black_skin_part2 = cgd.verticle_line(ln.shoulder_joint_r, min(c for r,c in tr.torso if r == ln.shoulder_joint_r),
                                    ln.breastbone_end_r-1, bend = 'right'
                                  )+cgd.verticle_line(ln.shoulder_joint_r, max(c for r,c in tr.torso if r == ln.shoulder_joint_r),
                                    ln.breastbone_end_r-1, bend = 'left'
                                  )
    black_skin_part3_raw = cgd.draw_trapezoid(max(r for r,c in black_skin_part2), 
                                          min(c for r,c in black_skin_part2 if r == max(r for r,c in black_skin_part2)), 
                                          max(c for r,c in black_skin_part2 if r == max(r for r,c in black_skin_part2)),
                                          max(r for r,c in black_skin_part2)+3, slope_left = '0.5', slope_right = '0.5', direction = 'lr',
                                              bend_left = 'left', bend_right = 'right')[-2]
    black_skin_part3 = [(r,c ) for r,c in black_skin_part3_raw if (r,c) in tr.torso
                       ]
    black_skin_part4 = [(r,c) for r,c in upperarm_dict["left_upperarm"] if r == min(r for r,c in uplj.left_elbow_joint)-1 
                       ]+[(r,c) for r,c in upperarm_dict["right_upperarm"] if r == min(r for r,c in uplj.right_elbow_joint)-1 
                       ]+[(r,c) for r,c in upperarm_dict["left_upperarm"] if r == min(r for r,c in uplj.left_elbow_joint)-4 
                       ]+[(r,c) for r,c in upperarm_dict["right_upperarm"] if r == min(r for r,c in uplj.right_elbow_joint)-4 
                       ]
    
    black_skin_part5 = cgd.verticle_line(max(r for r,c in btlj.left_thigh_joint), 
                                                 max(c for r,c in btlj.left_thigh_joint if r == max(r for r,c in btlj.left_thigh_joint)),
                                                 max(r for r,c in btlj.left_thigh_joint)+3, bend = 'right'
                                        )+cgd.verticle_line(max(r for r,c in btlj.right_thigh_joint), 
                                                 min(c for r,c in btlj.right_thigh_joint if r == max(r for r,c in btlj.right_thigh_joint)),
                                                 max(r for r,c in btlj.right_thigh_joint)+3, bend = 'left'
                                        )
    black_skin_part6 = [(r,c) for r,c in thigh_dict["left_thigh"] if r > max(r for r,c in black_skin_part5)
                       ]+btlj.left_knee_joint+calf_dict["left_calf"
                                                       ]+[(r,c) for r,c in feet_dict["left_foot"] 
                                                          if r < max(r for r,c in feet_dict["left_foot"])]
    black_skin_part7_raw = [(r,c) for r,c in thigh_dict["right_thigh"] if r > max(r for r,c in black_skin_part5) 
                          and r < min(r for r,c in btlj.right_knee_joint)-2]
    black_skin_part7_remove = [(max(r for r,c in black_skin_part7_raw)-1, 
                                (max(c for r,c in black_skin_part7_raw if r == max(r for r,c in black_skin_part7_raw)-1) + 
                                 min(c for r,c in black_skin_part7_raw if r == max(r for r,c in black_skin_part7_raw)-1)
                                )//2)
                              ]
    
    black_skin_part7 = [x for x in black_skin_part7_raw if x not in black_skin_part7_remove]
    
    black_skin_part8_raw = cgd.verticle_line(max(r for r,c in btlj.right_knee_joint), 
                                                 max(c for r,c in btlj.right_knee_joint if r == max(r for r,c in btlj.right_knee_joint)),
                                                 max(r for r,c in btlj.right_ankle_joint), bend = 'right'
                                        )
    black_skin_part8 = [x for x in black_skin_part8_raw if x in raw_body
                       ]+[(r,c) for r,c in feet_dict["right_foot"] if r < max(r for r,c in feet_dict["right_foot"])
                         ] 
    black_skin_part9_raw = [(r,c) for r,c in btlj.right_knee_joint if r>min(r for r,c in btlj.right_knee_joint)]
    black_skin_part9_remove = [((min(r for r,c in btlj.right_knee_joint)+max(r for r,c in btlj.right_knee_joint))//2, 
                                (min(c for r,c in btlj.right_knee_joint if r == (min(r for r,c in btlj.right_knee_joint)+max(r for r,c in btlj.right_knee_joint))//2)+max(c for r,c in btlj.right_knee_joint if r == (min(r for r,c in btlj.right_knee_joint)+max(r for r,c in btlj.right_knee_joint))//2))//2)]
    black_skin_part9 = [x for x in black_skin_part9_raw if x not in black_skin_part9_remove]
    
    
    black_skin = black_skin_part1+black_skin_part2+black_skin_part3+black_skin_part4+black_skin_part5+black_skin_part6+black_skin_part7+black_skin_part8+black_skin_part9
    
    grey_skin = [(r,c) for r,c in raw_body if r>= max(r for r,c in black_skin_part7) and (r,c) not in black_skin]
    
    lance_loc = (max(r for r,c in hand_dict["left_hand"]), 
                 min(c for r,c in hand_dict["left_hand"] if r == max(r for r,c in hand_dict["left_hand"]))
                )
    lance_raw  = cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1], ln.foot_r , 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                                            )+cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1]+1, ln.foot_r, 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                                            )+cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1], 3, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )+cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1]+1,3, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )
    
    lance_rod_raw = [(r,c) for r,c in lance_raw if (r,c) not in raw_body and r <ln.gastrocnemius_end_r-n_head]
    lance_tip_raw = [(r,c) for r,c in lance_raw if (r,c) not in raw_body 
                       and r >=ln.gastrocnemius_end_r]
    lance_tip = []
    for loc in lance_tip_raw:
        if loc[0] <min(r for r, c in lance_tip_raw)+2:
            n_lance = 2 
        elif loc[0] <max(r for r, c in lance_tip_raw)-2:
            n_lance = 1
        else:
            n_lance = 0
        lance_tip.extend(cg.hex_neighbours_n(loc[0], loc[1], n=n_lance, keep_origin = True, return_frontier=False))
    lance_rod = []
    for loc in lance_rod_raw:
        if loc[0] >max(r for r, c in lance_rod_raw)-2:
            n_rod = 2 
        elif loc[0] >max(r for r, c in lance_rod_raw)-5:
            n_rod = 1
        else:
            n_rod = 0
        lance_rod.extend(cg.hex_neighbours_n(loc[0], loc[1], n=n_rod, keep_origin = True, return_frontier=False))
        
    lance_blue_raw = cgd.draw_slope_1p5_diagonal(min(r for r,c in lance_tip_raw)+1, 
                                             min(c for r,c in lance_raw if r==min(r for r,c in lance_tip_raw)+1), 
                                             max(r for r,c in lance_rod_raw)-2, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )+cgd.draw_slope_1p5_diagonal(min(r for r,c in lance_tip_raw)+1, 
                                             min(c for r,c in lance_raw if r==min(r for r,c in lance_tip_raw)+1)+1, 
                                             max(r for r,c in lance_rod_raw)-2, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )+cgd.draw_slope_1p5_diagonal(max(r for r,c in lance_rod_raw)-4, 
                                             min(c for r,c in lance_raw if r==max(r for r,c in lance_rod_raw)-4), 
                                             max(r for r,c in lance_rod_raw)-4, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )+cgd.draw_slope_1p5_diagonal(max(r for r,c in lance_rod_raw)-4, 
                                             min(c for r,c in lance_raw if r==max(r for r,c in lance_rod_raw)-4)+1, 
                                             max(r for r,c in lance_rod_raw)-4, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )
    lance_blue = [(r,c) for r,c in lance_blue_raw if (r,c)  in lance_rod or (r,c)  in lance_tip]
    keys = [ "raw_body", "hair", "black_skin", "grey_skin", "lance_rod" , "lance_tip", "lance_blue"]
    colors = [[SKIN_COLOR],
              
              [HAIR_COLOR],[[0., 0., 0.]],
              [cgc.hex_to_rgb("#9b8d7e"), cgc.hex_to_rgb("#52493e")], 
              [[0.3, 0.3, 0.3]],[cgc.hex_to_rgb("#ffe500"), [1,1,1]],[cgc.hex_to_rgb("#00ffff")] 
             ]
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}

#def draw_type4O(handle_center_row, handle_center_col, weapontype = 'lance'):
    #if weapontype == 'sword':
    #elif weapontype == 'blade':
    #elif weapontype == 'lance':

def draw_char_tiny(weapon_center_row, weapon_center_col, name, n=2 ):
    
    if 'A2' in name and 'type 4O lance' in name:
        lance_loc = (weapon_center_row, weapon_center_col)
        #lance_loc = (head_center_row+n_head, head_center_col+n_head+n_head+n_head+n_head)
        
        
        head_center_row = weapon_center_row-n
        head_center_col = weapon_center_col - (n+n+n+n)
        body_dict, plot_temp = cgf.draw_tiny_body_(head_center_row, head_center_col, n)
        bd = SimpleNamespace(**body_dict)
        n_head = bd.n
        
        
        remove = [(bd.body_top_r, min(c for r,c in bd.raw_body)), 
                  (bd.body_top_r, max(c for r,c in bd.raw_body)), 
                  (bd.head_center_row, head_center_col+n_head)
                 ]
        raw_body = [(r,c) for r,c in bd.raw_body if (r,c) not in remove]

        
        hair_part1 = [(r,c) for r,c in bd.head_raw
                      if r <= head_center_row or c<head_center_col]
        
    
        hair_part2_raw = cgd.verticle_line(head_center_row, 
                                       min(c for r,c in hair_part1 if r==head_center_row),
                                       bd.body_bottom_r, bend = 'left'
                                      )+cgd.verticle_line(head_center_row, 
                                       min(c for r,c in hair_part1 if r==head_center_row),
                                       bd.body_bottom_r, bend = 'right'
                                      )+cgd.verticle_line(head_center_row-1, 
                                       max(c for r,c in hair_part1 if r==head_center_row-1),
                                       bd.body_bottom_r, bend = 'left'
                                      )

        hair_part2 = [(r,c) for r,c in hair_part2_raw if (r,c) not in raw_body]
        hair_part3_raw = cgd.draw_trapezoid(head_center_row, 
                                          min(c for r,c in hair_part1 if r == head_center_row), 
                                          max(c for r,c in hair_part1 if r == head_center_row), 
                                          bd.body_bottom_r, slope_left = '0.5', slope_right = '0.5', direction = 'll',
                                          bend_left = 'left', bend_right = 'right')[-2]
        hair_part3 = [(r,c) for r,c in hair_part3_raw if (r,c) not in raw_body]
        
        
        hair =hair_part1+hair_part2+hair_part3

        black_skin = [bd.left_elbow]+[bd.right_elbow]+[(r,c) for r,c in bd.upper_arm if r >=bd.body_center_row
                                                      ]+[(r,c) for r,c in bd.body if r <=bd.body_center_row and r >bd.body_top_r
                                                      ]+bd.pelvis+[(r,c) for r,c in bd.left_leg if r >bd.bottom_top_r+1
                                                                  ]+[(r,c) for r,c in bd.right_leg if r ==bd.bottom_top_r+2 
                                                                     or r == max(r for r,c in bd.right_leg)
                                                                  ]+cgd.verticle_line(bd.body_top_r, 
                                                                           min(c for r,c in bd.body_part1 if r == bd.body_top_r), 
                                                                           bd.body_center_row, bend = 'left'
                                                                  )+cgd.verticle_line(bd.body_top_r, 
                                                                           max(c for r,c in bd.body_part1 if r == bd.body_top_r), 
                                                                           bd.body_center_row, bend = 'right'
                                                                  )
        grey_skin = [(r,c) for r,c in bd.right_leg if r >bd.bottom_top_r+1 and (r,c) not in black_skin ]

        
        
        
        lance_part1_raw  = cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1], bd.thigh_center_row, 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                                            )+cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1]+1, bd.thigh_center_row, 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                                            )
        lance_part1 = [(r,c) for r,c in lance_part1_raw if (r,c) not in bd.left_upper_arm]
        lance_part2 =cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1], lance_loc[0]-2, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )+cgd.draw_slope_1p5_diagonal(lance_loc[0], lance_loc[1]+1, lance_loc[0]-2, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )
        lance_tip = [lance_loc]
        keys = [ "raw_body", "hair", "black_skin", "grey_skin", "lance_part1", "lance_part2", "lance_tip"]
        colors = [[SKIN_COLOR],

                  [HAIR_COLOR],[[0, 0, 0]],[cgc.hex_to_rgb("#9b8d7e"), cgc.hex_to_rgb("#52493e")], 
                  [[0.25, 0.25,0.25]],
                  [[0.7, 0.7, 0.7], cgc.hex_to_rgb("#00ffff")], [cgc.hex_to_rgb("#00ffff")] 
                 ]
        
    elif 'A2' in name and 'type 4O sword' in name:
        sword_loc = (weapon_center_row, weapon_center_col)
        #lance_loc = (head_center_row+n_head, head_center_col+n_head+n_head+n_head+n_head)
        
        
        head_center_row = weapon_center_row-n-n-n-n-n
        head_center_col = weapon_center_col 
        body_dict, plot_temp = cgf.draw_tiny_body_(head_center_row, head_center_col, n)
        bd = SimpleNamespace(**body_dict)
        n_head = bd.n
        
        
        remove = [(bd.body_top_r, min(c for r,c in bd.raw_body)), 
                  (bd.body_top_r, max(c for r,c in bd.raw_body)), 
                  (bd.head_center_row, head_center_col+n_head),
                  bd.left_elbow, bd.right_elbow
                 ]
        raw_body = [(r,c) for r,c in bd.raw_body if (r,c) not in remove]

        
        hair_part1 = [(r,c) for r,c in bd.head_raw
                      if r <= head_center_row or c<head_center_col or c>head_center_col]
        
    
        hair_part2_raw = cgd.verticle_line(head_center_row, 
                                       min(c for r,c in hair_part1 if r==head_center_row),
                                       bd.head_bottom_r, bend = 'left'
                                      )+cgd.verticle_line(head_center_row, 
                                       min(c for r,c in hair_part1 if r==head_center_row),
                                       bd.head_bottom_r, bend = 'right'
                                      )+cgd.verticle_line(head_center_row, 
                                       max(c for r,c in hair_part1 if r==head_center_row),
                                       bd.head_bottom_r, bend = 'left'
                                      )+cgd.verticle_line(head_center_row, 
                                       max(c for r,c in hair_part1 if r==head_center_row),
                                       bd.head_bottom_r, bend = 'right'
                                      )

        hair_part2 = [(r,c) for r,c in hair_part2_raw if (r,c) not in raw_body]
        hair_back_raw = cgd.draw_trapezoid(max(r for r,c in hair_part2), 
                                          min(c for r,c in hair_part2 if r == max(r for r,c in hair_part2)), 
                                          max(c for r,c in hair_part2 if r == max(r for r,c in hair_part2)), 
                                          max(r for r,c in hair_part2)-3, slope_left = '1.5', slope_right = '1.5', direction = 'lr',
                                          bend_left = 'left', bend_right = 'right')[-2]
        hair_back = [(r,c) for r,c in hair_back_raw if (r,c) not in raw_body]
        hair =hair_part1+hair_part2
        
        black_skin_arm = cgd.horizontal_lines([
            (bd.left_elbow[0], (bd.left_elbow[1]+1, bd.left_elbow[1]+3)),
            (bd.right_elbow[0], (bd.right_elbow[1]-1, bd.right_elbow[1]-3))]
        )
        black_skin = [(r,c) for r,c in bd.upper_arm if r >=bd.body_center_row and (r,c) in raw_body
                                                      ]+[(r,c) for r,c in bd.body if r <=bd.body_center_row and r >bd.body_top_r
                                                      ]+bd.pelvis+[(r,c) for r,c in bd.left_leg if r >bd.bottom_top_r+1
                                                                  ]+[(r,c) for r,c in bd.right_leg if r ==bd.bottom_top_r+2 
                                                                     or r == max(r for r,c in bd.right_leg)
                                                                  ]+cgd.verticle_line(bd.body_top_r, 
                                                                           min(c for r,c in bd.body_part1 if r == bd.body_top_r), 
                                                                           bd.body_center_row, bend = 'left'
                                                                  )+cgd.verticle_line(bd.body_top_r, 
                                                                           max(c for r,c in bd.body_part1 if r == bd.body_top_r), 
                                                                           bd.body_center_row, bend = 'right'
                                                                  )+black_skin_arm
        grey_skin = [(r,c) for r,c in bd.right_leg if r >bd.bottom_top_r+1 and (r,c) not in black_skin ]
        sword_raw = cgd.verticle_line(sword_loc[0], sword_loc[1],
                                       max(r for r,c in raw_body)+n+n, bend = 'left'
                                      )+cgd.verticle_line(sword_loc[0], sword_loc[1],
                                       max(r for r,c in raw_body)+n+n, bend = 'right'
                                      )
        sword = [(r,c) for r,c in sword_raw if (r,c) not  in black_skin_arm]
        keys = [ "raw_body","hair_back",  "hair", "black_skin", "grey_skin", "sword_raw"]#"sword_handle", "sword"]
        colors = [[SKIN_COLOR],[HAIR_COLOR,[0.55, 0.55,0.55]],

                  [HAIR_COLOR],[[0, 0, 0]],[cgc.hex_to_rgb("#9b8d7e"), cgc.hex_to_rgb("#52493e")], 
                  [[0.25, 0.25,0.25], [0.85, 0.85, 0.85]],
                  [[0.85, 0.85, 0.85]],
                 ]
    elif 'A2' in name and 'Virtuous Contract' in name:
        sword_loc = (weapon_center_row, weapon_center_col)
        #lance_loc = (head_center_row+n_head, head_center_col+n_head+n_head+n_head+n_head)
        
        
        head_center_row = weapon_center_row-n
        head_center_col = weapon_center_col - (n+n+n)
        
        body_dict, plot_temp = cgf.draw_tiny_body_(head_center_row, head_center_col, n)
        bd = SimpleNamespace(**body_dict)
        n_head = bd.n
        
        
        remove = [(bd.body_top_r, min(c for r,c in bd.raw_body)), 
                  (bd.body_top_r, max(c for r,c in bd.raw_body)), 
                  (bd.head_center_row, head_center_col+n_head)
                 ]
        raw_body = [(r,c) for r,c in bd.raw_body if (r,c) not in remove]

        
        hair_part1 = [(r,c) for r,c in bd.head_raw
                      if r <= head_center_row or c<head_center_col]
        
    
        hair_part2_raw = cgd.verticle_line(head_center_row, 
                                       min(c for r,c in hair_part1 if r==head_center_row),
                                       bd.neck_row, bend = 'left'
                                      )+cgd.verticle_line(head_center_row, 
                                       min(c for r,c in hair_part1 if r==head_center_row),
                                       bd.neck_row, bend = 'right'
                                      )+cgd.verticle_line(head_center_row-1, 
                                       max(c for r,c in hair_part1 if r==head_center_row-1),
                                       bd.neck_row, bend = 'left'
                                      )
        
      
        hair_part2 = [(r,c) for r,c in hair_part2_raw if (r,c) not in raw_body]
        hair =hair_part1+hair_part2
        
 
        black_skin = [bd.left_elbow]+[bd.right_elbow]+[(r,c) for r,c in bd.upper_arm if r >=bd.body_center_row and (r,c) in raw_body
                                                      ]+[(r,c) for r,c in bd.body if r <=bd.body_center_row and r >bd.body_top_r
                                                      ]+bd.pelvis+[(r,c) for r,c in bd.left_leg if r >bd.bottom_top_r+1
                                                                  ]+[(r,c) for r,c in bd.right_leg if r ==bd.bottom_top_r+2 
                                                                     or r == max(r for r,c in bd.right_leg)
                                                                  ]+cgd.verticle_line(bd.body_top_r, 
                                                                           min(c for r,c in bd.body_part1 if r == bd.body_top_r), 
                                                                           bd.body_center_row, bend = 'left'
                                                                  )+cgd.verticle_line(bd.body_top_r, 
                                                                           max(c for r,c in bd.body_part1 if r == bd.body_top_r), 
                                                                           bd.body_center_row, bend = 'right'
                                                                  )
        grey_skin = [(r,c) for r,c in bd.right_leg if r >bd.bottom_top_r+1 and (r,c) not in black_skin ]
        sword_raw  = cgd.draw_slope_0p5_diagonal(sword_loc[0], sword_loc[1], bd.pelvis_row, 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                                            )+cgd.draw_slope_0p5_diagonal(sword_loc[0], sword_loc[1], sword_loc[0]-2, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )
        sword = [(r,c) for r,c in sword_raw if (r,c) not  in raw_body]
        
        keys = [ "raw_body", "hair", "black_skin", "grey_skin", "sword"]
        colors = [[SKIN_COLOR],

                  [HAIR_COLOR],[[0, 0, 0]],[cgc.hex_to_rgb("#9b8d7e"), cgc.hex_to_rgb("#52493e")], 
                  [[0.95, 0.95,0.95], [0.7, 0.7,0.7]],
                 ]
        
    elif name == '2B':
        sword_loc = (weapon_center_row, weapon_center_col)
        #sword_loc = (head_center_row+n_head, head_center_col-n_head-n_head-n_head)
        
        
        head_center_row = weapon_center_row-n
        head_center_col = weapon_center_col + (n+n+n)
        
        body_dict, plot_temp = cgf.draw_tiny_body_(head_center_row, head_center_col, n)
        bd = SimpleNamespace(**body_dict)
        n_head = bd.n
        
        
        
        hair_part1 = [(r,c) for r,c in bd.head_raw
                      if r <= head_center_row or c>head_center_col]
        hair_part2_raw = cgd.verticle_line(head_center_row, 
                                       max(c for r,c in hair_part1 if r==head_center_row),
                                       bd.head_bottom_r, bend = 'left'
                                      )+cgd.verticle_line(head_center_row, 
                                       max(c for r,c in hair_part1 if r==head_center_row),
                                       bd.head_bottom_r, bend = 'right'
                                      )+cgd.verticle_line(head_center_row-1, 
                                       min(c for r,c in hair_part1 if r==head_center_row-1),
                                       bd.head_bottom_r, bend = 'left'
                                      )

        hair_part2 = [(r,c) for r,c in hair_part2_raw if (r,c) not in bd.raw_body]
        
        hair =hair_part1+hair_part2
        
        black_part_1_raw = bd.neck+bd.body+bd.upper_arm+[(r,c) for r,c in bd.head_raw if r == head_center_row+1 and (r,c) not in hair
                                                  ]+[(r,c) for r,c in bd.leg if r >bd.body_bottom_r+2]
        black_part_1_remove =cgd.horizontal_lines([(bd.neck_row+2, (min(c for r,c in bd.neck), max(c for r,c in bd.neck)))]
                                                 )
        black_part_1 = [(r,c) for r,c in black_part_1_raw if (r,c) not in black_part_1_remove]
        
        black_part_2_raw = cgd.draw_trapezoid(bd.body_bottom_r, 
                                          min(c for r,c in bd.body if r == bd.body_bottom_r), 
                                          max(c for r,c in bd.body if r == bd.body_bottom_r), 
                                          bd.body_bottom_r+3, slope_left = '0.5', slope_right = '0.5', direction = 'lr',
                                          bend_left = 'left', bend_right = 'right')[-2]
        black_part_2_remove =[(r,c) for r,c in cgd.verticle_line(bd.body_bottom_r+1, 
                                                                 min(c for r,c in black_part_2_raw if r == bd.body_bottom_r+1)+1,
                                                max(r for r,c in black_part_2_raw), bend = 'left') if (r,c) in bd.left_leg]
        black_part_2 = [(r,c) for r,c in black_part_2_raw if (r,c) not in black_part_2_remove]
        black_part = black_part_1+black_part_2
        
        sleeves = [(r,c) for r,c in bd.upper_arm if r <max(r for r,c in bd.upper_arm ) and r >= max(r for r,c in bd.upper_arm )-2 ]
        
        remove = [(bd.body_top_r, min(c for r,c in bd.raw_body)), 
                  (bd.body_top_r, max(c for r,c in bd.raw_body)), 
                  (bd.head_center_row, head_center_col+n_head)
                 ]
        raw_body = [(r,c) for r,c in bd.raw_body if (r,c) not in remove]
        
        sword_raw  = cgd.draw_slope_0p5_diagonal(sword_loc[0], sword_loc[1], bd.pelvis_row, 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                                            )+cgd.draw_slope_0p5_diagonal(sword_loc[0], sword_loc[1], sword_loc[0]-2, 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                                            )
        sword = [(r,c) for r,c in sword_raw if (r,c) not in raw_body]
        
        keys = [ "raw_body", "hair" , "black_part","sleeves", "sword"]
        colors = [[SKIN_COLOR],

                  [HAIR_COLOR],[[0, 0, 0]],[[1,1,1]],
                  [[0.95, 0.95,0.95], [0.7, 0.7,0.7]]
                 ]
    elif name == '9S':
        sword_loc = (weapon_center_row, weapon_center_col)
        #sword_loc = (head_center_row+n_head, head_center_col-n_head-n_head-n_head)
        
        
        head_center_row = weapon_center_row-n
        head_center_col = weapon_center_col + (n+n+n)
        
        body_dict, plot_temp = cgf.draw_tiny_body_(head_center_row, head_center_col, n)
        bd = SimpleNamespace(**body_dict)
        n_head = bd.n
        
        hair = [(r,c) for r,c in bd.head_raw
                      if r <= head_center_row or c>head_center_col]
        black_part_raw = bd.body+bd.upper_arm+bd.pelvis+[(r,c) for r,c in bd.head_raw if r == head_center_row+1 and (r,c) not in hair
                                                  ]+[(r,c) for r,c in bd.leg if r != max(r for r, c in bd.leg)-3]
        
        remove = [(bd.body_top_r, min(c for r,c in bd.raw_body)), 
                  (bd.body_top_r, max(c for r,c in bd.raw_body)), 
                  (bd.head_center_row, head_center_col+n_head),
                  (bd.body_top_r, min(c for r,c in bd.raw_body)), 
                  (bd.body_top_r, max(c for r,c in bd.raw_body))
                 ]
        raw_body = [(r,c) for r,c in bd.raw_body if (r,c) not in remove]
        black_part = [(r,c) for r,c in black_part_raw if (r,c) not in remove]
        #sword_loc = (head_center_row+n_head, head_center_col-n_head-n_head-n_head)
        sword_raw  = cgd.draw_slope_0p5_diagonal(sword_loc[0], sword_loc[1], bd.pelvis_row, 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                                            )+cgd.draw_slope_0p5_diagonal(sword_loc[0], sword_loc[1], sword_loc[0]-2, 
                                             left_down=False, right_down=False, left_up=True, right_up=False
                                            )
        sword = [(r,c) for r,c in sword_raw if (r,c) not in raw_body]
        
        
        keys = [ "raw_body", "hair", "black_part", "sword"]
        colors = [[SKIN_COLOR],

                  [HAIR_COLOR],[[0, 0, 0]], 
                  [[0.35, 0.35,0.35], [0.7, 0.7,0.7]]
                 ]
    
    elif name == 'king':
        head_loc = (weapon_center_row, weapon_center_col)
        body = cg.hex_neighbours_n(head_loc[0], head_loc[1], n=3, keep_origin = True, return_frontier=False
                                  )
        eyemouth = [(min(r for r,c in body)+1, min(c for r,c in body if r == min(r for r,c in body)+1)+1),
                   (min(r for r,c in body)+1, max(c for r,c in body if r == min(r for r,c in body)+1)-1), head_loc]
        metal = cg.hex_neighbours_n(max(r for r,c in body), 
                                    min(c for r,c in body if r == max(r for r,c in body)), n=1, keep_origin = True, 
                                    return_frontier=False
                                  )+cg.hex_neighbours_n(max(r for r,c in body)-1, 
                                    max(c for r,c in body if r == max(r for r,c in body)-1), n=1, keep_origin = True, 
                                    return_frontier=False
                                  )+cg.hex_neighbours_n(max(r for r,c in body), 
                                    max(c for r,c in body if r == max(r for r,c in body))-1, n=1, keep_origin = True, 
                                    return_frontier=False
                                  )+cg.hex_neighbours_n(head_loc[0]+1, 
                                    min(c for r,c in body if r == head_loc[0]+1)+1, n=1, keep_origin = True, 
                                    return_frontier=False
                                  )+cg.hex_neighbours_n(head_loc[0]+5, 
                                    head_loc[1], n=1, keep_origin = True, 
                                    return_frontier=False
                                  )+cg.hex_neighbours_n(head_loc[0]+4, 
                                    head_loc[1]+3, n=1, keep_origin = True, 
                                    return_frontier=False
                                  )
        cradle = cgd.draw_block((min(r for r,c in body)-2, (head_loc[1]-5, head_loc[1]+5)), max(r for r,c in metal)+2)
        keys = ["cradle",  "body", "eyemouth", "metal", ]
        colors = [[cgc.hex_to_rgb("#beb7a9")], [cgc.hex_to_rgb("#8e8677")],[[0.7, 0.7,0.7]],

                  [cgc.hex_to_rgb("#65512a"), cgc.hex_to_rgb("#3e382b")],
                  [[0.35, 0.35,0.35], [0.7, 0.7,0.7]]
                 ]
        
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}

head_center_row = 5
head_center_col = max(c for r,c in hex_rc_arr)//2
    
_, _,  _, _, _,_, _, _, _, _, _,_, _, hand_dict,  _ = cgf.draw_human_body(head_center_row, head_center_col)
skyline_loc = (max(r for r,c in hand_dict["left_hand"]), 
                 min(c for r,c in hand_dict["left_hand"] if r == max(r for r,c in hand_dict["left_hand"]))
                )
skyline  = cgd.draw_slope_1p5_diagonal(skyline_loc[0], skyline_loc[1], max(r for r,c in hex_rc_arr), 
                                             left_down=True, right_down=False, left_up=False, right_up=False
                                            )+cgd.draw_slope_1p5_diagonal(skyline_loc[0], skyline_loc[1], 0, 
                                             left_down=False, right_down=False, left_up=False, right_up=True
                                            )
sorted_skyline  = sorted(skyline,  key=lambda x: x[0])
left_region = cgd.horizontal_lines([( r, (0, c)) for (r, c) in sorted_skyline])
right_region = cgd.horizontal_lines([( r, (c, max(col for row,col in hex_rc_arr))) for (r, c) in sorted_skyline])

select_left_region= cg.select_mask(left_region,hex_rc_arr)
#hex_colors[select_left_region] = cgc.select_normal_color(select_left_region, colors[0], np.ones(3)*sigma_color) 

hex_colors = cgc.color_row_gradient(left_region, cgc.hex_to_rgb("#ff007a"), cgc.hex_to_rgb("#2d6d3b"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear')   
hex_colors =cgc.color_row_gradient(right_region, cgc.hex_to_rgb("#2d6d3b"), cgc.hex_to_rgb("#dcffa3"),
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear')          
char_dict_ = {'A2': draw_A2(head_center_row, head_center_col),
              'A2 type 4O lance': draw_char_tiny(max(r for r,c in hex_rc_arr)//3, max(c for r,c in hex_rc_arr)//5, 'A2 type 4O lance'),
              'A2 type 4O sword': draw_char_tiny(max(r for r,c in hex_rc_arr)//4, max(c for r,c in hex_rc_arr)//3, 'A2 type 4O sword'),
              'king': draw_char_tiny(max(r for r,c in hex_rc_arr)//6*5, max(c for r,c in hex_rc_arr)//3, 'king'),
              
              '2B': draw_char_tiny(max(r for r,c in hex_rc_arr)//3, max(c for r,c in hex_rc_arr)//5*4, '2B'),
              
              'A2 Virtuous Contract': draw_char_tiny(max(r for r,c in hex_rc_arr)//5*3, max(c for r,c in hex_rc_arr)//5*4, 
                                                     'A2 Virtuous Contract'),
              '9S': draw_char_tiny(max(r for r,c in hex_rc_arr)//4*3, max(c for r,c in hex_rc_arr)//7*6, '9S'),
             }
for dict_ in char_dict_.values(): 
    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear') 

pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'Nier_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')