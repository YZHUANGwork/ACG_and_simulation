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

import re
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

def flip_wrtcol(flip_col, part):
    return [(r, 2*flip_col - c - (1 if r % 2 == 1 else 0)) for r, c in part]

STICK_LOC_joker = None
STICK_ROW_ciel = None

def draw_char(head_center_row, head_center_col, version):
    if 'joker' in version:
        head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(head_center_row, head_center_col, version='7')
        ln = SimpleNamespace(**line_dict)
        head = cg.hex_neighbours_n(head_center_row, head_center_col, n=2)
        hair_part1 = [(r,c) for r,c in head_raw if r < head_center_row-1]
        hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                       max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head), bend = 'left'
                                      )
        hair_part3 = cgd.draw_trapezoid(min(r for r,c in hair_part1), 
                                    min(c for r,c in hair_part1 if r==min(r for r,c in hair_part1)), 
                                    max(c for r,c in hair_part1 if r==min(r for r,c in hair_part1)), 
                                    min(r for r,c in hair_part1)-1, slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                                    bend_left = 'left', bend_right = 'right')[-2]
        hair = [(r,c) for r,c in hair_part1+hair_part2+hair_part3 if (r,c) not in neck]
        face = [(r,c) for r,c in head if (r,c) not in hair_part1+hair_part2]+[(r,c) for r,c in neck if r==max(r_ for r_,c_ in head)+1]
        elbow = [(r,c) for r,c in raw_body if r==ln.elbow_r]
        collar_raw_part1 = cgd.draw_trapezoid(max(r for r,c in face)+1, 
                                    min(c for r,c in neck if r==max(r for r,c in face)+1), 
                                    max(c for r,c in neck if r==max(r for r,c in face)+1), 
                                    max(r for r,c in face), slope_left = '1.5', slope_right = '1.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right')[-2]
        collar_raw_part2 = cgd.draw_trapezoid(min(r for r,c in collar_raw_part1), 
                                    min(c for r,c in collar_raw_part1 if r==min(r for r,c in collar_raw_part1)), 
                                    max(c for r,c in collar_raw_part1 if r==min(r for r,c in collar_raw_part1)), 
                                    min(r for r,c in collar_raw_part1)-1, slope_left = '0.5', slope_right = '0.5', direction = 'lr',
                                    bend_left = 'left', bend_right = 'right'
                 )[-2]
        collar = [(r,c) for r,c in collar_raw_part1+collar_raw_part2 if (r,c) not in face+hair]

        vest_raw = [(r,c) for r,c in torso_dict["torso"] if r<=ln.bellybutton_r and (r,c) not in collar]
        remove_vest = cgd.draw_triangle(ln.breastbone_end_r-1, head_center_col,  min(r for r,c in vest_raw), 
                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr', 
                     bend_left = 'left', bend_right = 'right')[-2]
        vest = [(r,c) for r,c in vest_raw if (r,c) not in remove_vest
               ]+[(min(r for r,c in vest_raw), min(c for r,c in vest_raw if r==min(r for r,c in vest_raw))-1),
                 (min(r for r,c in vest_raw), max(c for r,c in vest_raw if r==min(r for r,c in vest_raw))+1),
                 ]
        dark_argyle_raw = cgd.verticle_line(min(r for r,c in vest), 
                                       min(c for r,c in vest if r==min(r for r,c in vest))+1,
                                       max(r for r,c in vest), bend = 'left'
                         )+cgd.verticle_line(min(r for r,c in vest), 
                                       min(c for r,c in vest if r==min(r for r,c in vest))+1,
                                       max(r for r,c in vest), bend = 'right'
                         )+cgd.verticle_line(min(r for r,c in vest_raw), 
                                       max(c for r,c in vest_raw if r==min(r for r,c in vest_raw))-1,
                                       max(r for r,c in vest_raw), bend = 'left'
                         )+cgd.verticle_line(min(r for r,c in vest_raw), 
                                       max(c for r,c in vest_raw if r==min(r for r,c in vest_raw))-1,
                                       max(r for r,c in vest_raw), bend = 'right'
                         )+cgd.verticle_line(min(r for r,c in vest), 
                                       max(c for r,c in vest if r==min(r for r,c in vest)),
                                       max(r for r,c in vest), bend = 'left'
                         )+cgd.verticle_line(min(r for r,c in vest), 
                                       max(c for r,c in vest if r==min(r for r,c in vest)),
                                       max(r for r,c in vest), bend = 'right'
                         )
        buttons = [(ln.breastbone_end_r+1, head_center_col-1), 
                  (ln.breastbone_end_r+1, head_center_col+1),
                  (ln.breastbone_end_r+3, head_center_col-1), 
                  (ln.breastbone_end_r+3, head_center_col+1),
                  (ln.breastbone_end_r+5, head_center_col-1), 
                  (ln.breastbone_end_r+5, head_center_col+1),]
        dark_argyle = [(r,c) for r,c in dark_argyle_raw if (r,c) in vest]

        removed_left_arm = [(r,c) for r,c in upperarm_dict["left_upperarm"]+forearm_dict["left_forearm"]+hand_dict["left_hand"]
                            if r>max(r_ for r_, c_ in upperlimb_joint_dict["left_elbow_joint"])]
        raw_body_modified = [(r,c) for r,c in raw_body if (r,c) not in removed_left_arm]

        shirt  = [(r,c) for r,c in raw_body_modified if (r,c) not in head_raw+face+neck+vest and r<=max(r_ for r_,c_ in vest)
                 ]+[(r,c) for r,c in cgd.draw_trapezoid(max(r_ for r_,c_ in vest)+1, head_center_col, 
                                      min(c for r,c in torso_dict["torso"] if r==max(r_ for r_,c_ in vest)+1), 
                                      max(r_ for r_,c_ in vest)+5, slope_left = '0.5', slope_right = '0.5', direction = 'll',
                      bend_left = 'left', bend_right = 'right'
                   )[-2]+cgd.draw_trapezoid(max(r_ for r_,c_ in vest)+1, head_center_col, 
                                      max(c for r,c in torso_dict["torso"] if r==max(r_ for r_,c_ in vest)+1), 
                                      max(r_ for r_,c_ in vest)+5, slope_left = '0.5', slope_right = '0.5', direction = 'rr',
                      bend_left = 'left', bend_right = 'right')[-2] if (r,c) in raw_body_modified
                   ]+upperarm_dict["right_upperarm"]+forearm_dict["right_forearm"]+hand_dict["right_hand"]

        pants = [(r,c) for r,c in raw_body_modified if r>max(r_ for r_,c_ in vest) and (r,c) not in shirt]
        bowtie_part1 = [(r,c) for r,c in neck if r== max(r_  for r_,c_ in neck)]
        bowtie_part2 = cgd.draw_slope_0p5_diagonal(max(r for r,c in bowtie_part1), head_center_col, max(r for r,c in bowtie_part1)+2, 
                                                   left_down=True, right_down=True, left_up=True, right_up=True)
        bowtie = bowtie_part1+bowtie_part2

        start_loc = (max(r for r, c in upperlimb_joint_dict["left_elbow_joint"]), 
                     min(c for r,c in upperlimb_joint_dict["left_elbow_joint"]if r==max(r for r, c in upperlimb_joint_dict["left_elbow_joint"])
                        ))
        modified_left_arm = cgd.horizontal_lines([(start_loc[0], (start_loc[1]-5, start_loc[1]))])
        modified_left_hand_part1 = cgd.draw_slope_0p5_diagonal(start_loc[0], min(c for r,c in modified_left_arm)-1, start_loc[0]-2, 
                                                   left_down=True, right_down=True, left_up=True, right_up=False
                             )
        modified_left_hand_part2 = cgd.verticle_line(start_loc[0], min(c for r,c in modified_left_arm)-1,start_loc[0]-1, bend = 'right'
                                  )+cgd.verticle_line(min(r for r,c in modified_left_hand_part1), 
                                                      min(c for r,c in modified_left_hand_part1 if r==min(r for r,c in modified_left_hand_part1)),min(r for r,c in modified_left_hand_part1)-1, bend = 'right'
                                  )
        modified_left_hand = modified_left_hand_part1+modified_left_hand_part2
        blazer_out = cgd.draw_block((ln.shoulder_joint_r+1, 
                                    (min(c for r,c in upperlimb_joint_dict["right_shoulder_joint"] if r==ln.shoulder_joint_r+1)+1,
                                     max(c for r,c in upperlimb_joint_dict["right_shoulder_joint"] if r==ln.shoulder_joint_r+1)),
                                   ), ln.wrist_r
                                   )+[(r,c) for r,c in cgd.draw_block((ln.elbow_r, 
                                    (min(c for r,c in upperlimb_joint_dict["left_elbow_joint"] if r==ln.elbow_r),
                                     max(c for r,c in upperlimb_joint_dict["left_elbow_joint"] if r==ln.elbow_r)),
                                   ), ln.wrist_r
                                   ) if (r,c) not in raw_body_modified]+upperarm_dict["right_upperarm"]+forearm_dict["right_forearm"]+upperlimb_joint_dict["right_shoulder_joint"]+upperlimb_joint_dict["right_elbow_joint"]+upperlimb_joint_dict["left_shoulder_joint"]+[(r,c) for r,c in upperarm_dict["left_upperarm"] if r<=ln.deltoid_end_r+1]

        blazer_in = [(r,c) for r,c in cgd.draw_block((ln.shoulder_joint_r, 
                                    (min(c for r,c in raw_body_modified if r==ln.shoulder_joint_r),
                                    max(c for r,c in raw_body_modified if r==ln.shoulder_joint_r)),
                                   ), ln.wrist_r)
                     if (r,c) not in raw_body_modified+blazer_out]
        
        global STICK_LOC_joker
        STICK_LOC_joker = (min(r for r,c in modified_left_hand)+1,
                     min(c for r,c in modified_left_hand if r==min(r for r,c in modified_left_hand)+1)+1)
        
        
        keys = ["raw_body_modified", "hair", "face", 

                "collar", "vest", "dark_argyle", "buttons" ,


                "shirt", "bowtie", "pants",

                "blazer_out", 
                "modified_left_arm", "modified_left_hand", "blazer_in"]#
        colors = [[[0,0,0]],  [cgc.hex_to_rgb("#ea9560")],[[0,0,0], cgc.hex_to_rgb("#4a4036")],

                  [[0.6, 0.6, 0.6]],[cgc.hex_to_rgb("#b3979f")],[cgc.hex_to_rgb("#6d4e33")],[cgc.hex_to_rgb("#e1b826")], 


                  [[1,1,1]],[cgc.hex_to_rgb("#ffd233")],[[0,0,0]],
                 [cgc.hex_to_rgb("#351b66")],[[1,1,1]],[[0.8, 0.8, 0.8]], [[0,0,0], cgc.hex_to_rgb("#8f0b52")],]#


    elif 'joker' not in version:
        head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(head_center_row, head_center_col, version='5')
        ln = SimpleNamespace(**line_dict)
        head = cg.hex_neighbours_n(head_center_row, head_center_col, n=2)

        feet = feet_dict["feet"]
        left_foot = feet_dict["left_foot"]
        left_foot_modified = [(r,c+2) for r,c in left_foot]

        right_foot = feet_dict["right_foot"]
        right_foot_modified = [(r+2,c-2) for r,c in right_foot]

        remove_body = [(r,c) for r,c in raw_body if r>=ln.knee_r and (r,c) not in feet_dict["feet"]]
        knee_joint = botlimb_joint_dict["left_knee_joint"]
        left_modified_calf = cgd.draw_slope_0p5_diagonal(min(r for r,c in left_foot_modified), 
                                           max(c for r,c in left_foot_modified if r==min(r for r,c in left_foot_modified)), ln.knee_r, 
                                           left_down=False, right_down=False, left_up=True, right_up=False
                                   )+cgd.verticle_line(min(r for r,c in left_foot_modified), 
                                       max(c for r,c in left_foot_modified if r==min(r for r,c in left_foot_modified)),
                                       min(r for r,c in left_foot_modified)-3, bend = 'left'
                                   )+cgd.verticle_line(min(r for r,c in left_foot_modified), 
                                       max(c for r,c in left_foot_modified if r==min(r for r,c in left_foot_modified)),
                                       min(r for r,c in left_foot_modified)-1, bend = 'right'
                                   )+knee_joint



        right_modified_calf_part1 = cgd.draw_slope_0p5_diagonal(min(r for r,c in right_foot_modified), 
                                           max(c for r,c in right_foot_modified if r==min(r for r,c in right_foot_modified)), 
                                           min(r for r,c in right_foot_modified)-2, 
                                           left_down=False, right_down=False, left_up=False, right_up=True)
        right_modified_calf_part2 = cgd.verticle_line(min(r for r,c in right_modified_calf_part1), 
                                       max(c for r,c in right_modified_calf_part1 if r==min(r for r,c in right_modified_calf_part1)),
                                       ln.knee_r,  bend = 'right'
                                   )+cgd.verticle_line(min(r for r,c in right_modified_calf_part1), 
                                       max(c for r,c in right_modified_calf_part1 if r==min(r for r,c in right_modified_calf_part1)),
                                       ln.knee_r,  bend = 'left'
                                   )
        right_modified_calf = right_modified_calf_part1+right_modified_calf_part2


        if 'back' in version:   

            left_foot_modified_temp = flip_wrtcol(head_center_col, left_foot_modified)
            left_modified_calf = flip_wrtcol(head_center_col, left_modified_calf)
            right_foot_modified_temp = flip_wrtcol(head_center_col, right_foot_modified)
            right_modified_calf = flip_wrtcol(head_center_col, right_modified_calf)
            knee_joint  = flip_wrtcol(head_center_col, knee_joint)

            left_foot_flip_row = max(r for r,c in left_foot_modified_temp)-1
            right_foot_flip_row = max(r for r,c in right_foot_modified_temp)-1

            left_foot_modified =[(2*left_foot_flip_row-r, c) if r>=left_foot_flip_row 
                                  else (r,c) for r,c in left_foot_modified_temp]
            right_foot_modified =[(2*right_foot_flip_row-r, c) if r>=right_foot_flip_row 
                                  else (r,c) for r,c in right_foot_modified_temp]

        raw_body_modified = [(r,c) for r,c in raw_body if (r,c) not in remove_body+left_foot+right_foot
                            ]+knee_joint+left_foot_modified+left_modified_calf+right_foot_modified+right_modified_calf#

    if 'doll' in version:
        hair_part1 = [(r,c) for r,c in head_raw if r < head_center_row]
        hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                       min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head)-1, bend = 'left'
                                      )+cgd.verticle_line(max(r for r,c in hair_part1), 
                                       max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head)-1, bend = 'right'
                                      )
        
        hair_front = cg.hex_neighbours_n(head_center_row, head_center_col+1, n=1)
        
        hair = [(r,c) for r,c in hair_part1+hair_part2 if (r,c) not in neck]
        face = [(r,c) for r,c in head if (r,c) not in hair+hair_front]
        chocker = [(r,c) for r,c in  neck if (r,c) not in face]+[(max(r for r,c in neck)+1, head_center_col)]
        if 'back' in version:
            hair+=face+head+neck+chocker
            face, hair_front = [],[]
            chocker = []
        sk_row = ln.femoral_head_r
        top_part1 = [(r,c) for r,c in raw_body_modified if r>ln.breastbone_end_r and r<sk_row and (r,c) not in 
                 upperarm_dict["upperarm"]+forearm_dict["forearm"]+hand_dict["hand"]
                 ]
        if 'front' in version:
            if 'v1' in version:
                top_part2 = cgd.horizontal_lines([(ln.breastbone_end_r , (min(c for r,c in raw_body_modified if r== ln.breastbone_end_r),
                                                                 min(c for r,c in torso_dict["torso"] if r== ln.breastbone_end_r) )),
                                         (ln.breastbone_end_r , (max(c for r,c in torso_dict["torso"] if r== ln.breastbone_end_r),
                                                                 max(c for r,c in raw_body_modified if r== ln.breastbone_end_r) ))])
            elif 'v2' in version:
                top_part2 = cgd.horizontal_lines([(ln.breastbone_end_r , 
                                                   (min(c for r,c in torso_dict["torso"] if r== ln.breastbone_end_r)-1,
                                                    head_center_col-1)),
                                                  (ln.breastbone_end_r , 
                                                   (head_center_col+1,
                                                    max(c for r,c in torso_dict["torso"] if r== ln.breastbone_end_r)+1))])
        elif 'back' in version:
            if 'v1' in version:
                top_part2 = cgd.horizontal_lines([(ln.breastbone_end_r , 
                                               (min(c for r,c in raw_body_modified if r== ln.breastbone_end_r),
                                                 min(c for r,c in upperarm_dict["left_upperarm"] if r== ln.breastbone_end_r) )),
                                         (ln.breastbone_end_r , 
                                          (max(c for r,c in upperarm_dict["right_upperarm"] if r== ln.breastbone_end_r),
                                                                 max(c for r,c in raw_body_modified if r== ln.breastbone_end_r) ))])
            elif 'v2' in version:
                top_part2 = [(r,c) for r,c in raw_body_modified  if r== ln.breastbone_end_r+1
                             and (r,c) not in upperarm_dict["upperarm"]]
                                      
        tops = top_part1+top_part2
        chest_skin =[(r,c) for r,c in raw_body_modified if (r,c) not in tops+chocker and r<=ln.breastbone_end_r and r>=ln.shoulder_joint_r]
      
        sk_part1 = cgd.draw_trapezoid(max(r for r,c in tops), 
                                      min(c for r,c in tops if r==max(r for r,c in tops)), 
                                      max(c for r,c in tops if r==max(r for r,c in tops)), 
                                max(r for r,c in tops)+1, slope_left = '1.5', slope_right = '1.5', direction = 'lr',
                  bend_left = 'left', bend_right = 'right')[-2]
        sk_part2 = cgd.draw_trapezoid(max(r for r,c in sk_part1), 
                                      min(c for r,c in sk_part1 if r==max(r for r,c in sk_part1)), 
                                      max(c for r,c in sk_part1 if r==max(r for r,c in sk_part1)), 
                                max(r for r,c in sk_part1)+2, slope_left = '1.5', slope_right = '1.5', direction = 'rl',
                  bend_left = 'left', bend_right = 'right')[-2]
        sk = sk_part1+sk_part2
        sk_flower_locs = [(max(r for r,c in sk_part2), min(c for r,c in sk_part2 if r==max(r for r,c in sk_part2))-1),
                          (max(r for r,c in sk_part2)-1, min(c for r,c in sk_part2 if r==max(r for r,c in sk_part2)-1)-1),
                          (max(r for r,c in sk_part2), head_center_col),
                          
                          (max(r for r,c in sk_part2), max(c for r,c in sk_part2 if r==max(r for r,c in sk_part2))+1),
                          (max(r for r,c in sk_part2)-1, max(c for r,c in sk_part2 if r==max(r for r,c in sk_part2)-1)+1),
                          
                         ]
        sk_flowers = []
        for sk_flower_loc in sk_flower_locs:
            sk_flower = cg.hex_neighbours_n(sk_flower_loc[0], sk_flower_loc[1], n=1)
            if sk_flower_loc[1]<=head_center_col:
                sk_flowers+=[(r,c) for r,c in sk_flower if (r,c) not in sk]
            else:
                sk_flowers+=sk_flower
        if 'back' in version:
            sk_flowers = flip_wrtcol(head_center_col, sk_flowers)
        sk_ribbon = cgd.draw_slope_1p5_diagonal(ln.bellybutton_r+1, 
                                                max(c for r,c in tops+sk if r==ln.bellybutton_r+1), ln.bellybutton_r+2, 
                                                left_down=True, right_down=False, left_up=False, right_up=False)
        if 'v1' in version:
            modified_left_arm = cgd.draw_slope_1p5_diagonal(ln.elbow_r+1, 
                                                    min(c for r,c in raw_body_modified if r==ln.elbow_r+1), ln.bellybutton_r+3, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                              )+cgd.draw_slope_1p5_diagonal(ln.elbow_r+2, 
                                                    min(c for r,c in raw_body_modified if r==ln.elbow_r+2)+1, ln.bellybutton_r+3, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                              )
            modified_right_arm = cgd.draw_slope_1p5_diagonal(ln.elbow_r+1, 
                                                    max(c for r,c in raw_body_modified if r==ln.elbow_r+1), ln.bellybutton_r+3, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                                )+cgd.draw_slope_1p5_diagonal(ln.elbow_r+2, 
                                                    max(c for r,c in raw_body_modified if r==ln.elbow_r+2)-1, ln.bellybutton_r+3, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                                )
        elif 'v2' in version:
            modified_left_arm_part1 = cgd.horizontal_lines([(ln.shoulder_joint_r+1, 
                                                        (min(c for r,c in torso_dict["torso"] if r==ln.shoulder_joint_r+1)-8, 
                                                         min(c for r,c in torso_dict["torso"] if r==ln.shoulder_joint_r+1)))
                                                      ])
            modified_left_arm_part2 = cgd.draw_slope_0p5_diagonal(ln.shoulder_joint_r+1, 
                                                    min(c for r,c in modified_left_arm_part1), ln.shoulder_joint_r-1, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                                )
            modified_left_arm = modified_left_arm_part1+modified_left_arm_part2
            
            modified_right_arm_part1 = cgd.horizontal_lines([(ln.shoulder_joint_r+1, 
                                                        (max(c for r,c in torso_dict["torso"] if r==ln.shoulder_joint_r+1), 
                                                         max(c for r,c in torso_dict["torso"] if r==ln.shoulder_joint_r+1)+8)),
                                                      
                                                      ])
            modified_right_arm_part2 = cgd.draw_slope_0p5_diagonal(ln.shoulder_joint_r+1, 
                                                    max(c for r,c in modified_right_arm_part1), ln.shoulder_joint_r-1, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True
                                )
            modified_right_arm = modified_right_arm_part1+modified_right_arm_part2
            
            
            
          
        if 'back' in version:
            modified_left_arm = [(r,c) for r,c in modified_left_arm if (r,c) not in sk+tops]
            modified_right_arm = [(r,c) for r,c in modified_right_arm if (r,c) not in sk+tops]
            
        if 'v1' in version:
            sleeves = [(r,c) for r,c in upperarm_dict["upperarm"]+forearm_dict["forearm"] if r>ln.breastbone_end_r and r<=ln.elbow_r
                      ]+cgd.verticle_line(ln.breastbone_end_r, min(c for r,c in tops if r== ln.breastbone_end_r),ln.elbow_r, bend = 'left'
                      )+cgd.verticle_line(ln.breastbone_end_r, max(c for r,c in tops if r== ln.breastbone_end_r),ln.elbow_r, bend = 'right'
                      )+[(r,c) for r,c in modified_left_arm+modified_right_arm if  r==ln.elbow_r+1
                      ]
        elif 'v2' in version:
            sleeves_part1 = cg.hex_neighbours_n(ln.shoulder_joint_r+1, min(c for r,c in modified_right_arm)+4, n=1
                      )+cg.hex_neighbours_n(ln.shoulder_joint_r+1, max(c for r,c in modified_left_arm)-4, n=1
                      )
            sleeves_part2 = cgd.horizontal_lines([(ln.shoulder_joint_r+1, 
                                                   (min(c for r,c in sleeves_part1)-2, min(c for r,c in sleeves_part1))),
                                                  (ln.shoulder_joint_r+1, 
                                                   (max(c for r,c in sleeves_part1), max(c for r,c in sleeves_part1)+2)),
                                                  ])
            sleeves = sleeves_part1+sleeves_part2
        skin = [(r,c) for r,c in raw_body_modified if r>ln.sartorius_turning_r and r<ln.knee_r-1
               and (r,c) not in upperarm_dict["upperarm"]+forearm_dict["forearm"]+sk_flowers+sk
               ]
        
        arm = [(r,c) for r,c in  modified_right_arm+modified_left_arm if (r,c) not in sleeves]
        
        if 'front' in version:   
            right_leg = right_foot_modified+[(r,c) for r,c in right_modified_calf+thigh_dict["right_thigh"]  
                                             if (r,c) not in skin+sk_flowers+sk]
        
            left_leg = left_foot_modified+[(r,c) for r,c in left_modified_calf+thigh_dict["left_thigh"] 
                                           if (r,c) not in skin+sk_flowers+sk]
            
            keys_extra = [ "left_leg","right_leg",  ]
            keys_extra_colors = [
                [[0.8,0.8,0.8],[0.5, 0.5, 0.5], 'stripe'],[[0.9,0.9,0.9], [1,1,1],'row'] ]
            
        elif 'back' in version:
            right_leg = right_foot_modified+[(r,c) for r,c in right_modified_calf+thigh_dict["left_thigh"]  
                                             if (r,c) not in skin+sk_flowers+sk]
        
            left_leg = left_foot_modified+[(r,c) for r,c in left_modified_calf+thigh_dict["right_thigh"] 
                                           if (r,c) not in skin+sk_flowers+sk]
            
            keys_extra = ["right_leg", "left_leg",]
            keys_extra_colors = [
               [[0.9,0.9,0.9], [1,1,1],'row'],[[0.8,0.8,0.8],[0.5, 0.5, 0.5], 'stripe']
            ]
                
        keys = [ "hair", "hair_front", "face", "chocker", 
                
                "tops", "sleeves", "sk", "sk_ribbon", "sk_flowers", 
                
                "skin",  "chest_skin", "arm",
               
                 ]+keys_extra#
        colors = [ [[0.9,0.9,0.9]], [[1,1,1]], [cgc.hex_to_rgb("#ffe8d0")], [[1,1,1]], 
                  
                  [[1,1,1]], [[1,1,1]], [[1,1,1]],[[0,0,0]],[[0.85,0.85,0.85]], 
                  
                  [cgc.hex_to_rgb("#ffe8d0")],[cgc.hex_to_rgb("#ffe8d0")],[cgc.hex_to_rgb("#ffe8d0")], 
                  
                  
                 ]+keys_extra_colors

        
    if 'ciel' in version:
        global STICK_ROW_ciel
        STICK_ROW_ciel = max(r for r,c in hand_dict["hand"])
        
        
        hair_part1 = [(r,c) for r,c in head_raw if r < head_center_row]
        hair_part2 = cgd.verticle_line(max(r for r,c in hair_part1), 
                                       min(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head), bend = 'right'
                                      )+cgd.verticle_line(max(r for r,c in hair_part1), 
                                       max(c for r,c in hair_part1 if r==max(r for r,c in hair_part1)),
                                       max(r for r,c in head), bend = 'left'
                                      )+[(head_center_row, head_center_col),]
        eyepatch = cgd.verticle_line(head_center_row, head_center_col-1,head_center_row+1, bend = 'right'
                         )+cgd.verticle_line(head_center_row, head_center_col-1,head_center_row+1, bend = 'left'
                         )

        hair = [(r,c) for r,c in hair_part1+hair_part2 if (r,c) not in neck]
        face = [(r,c) for r,c in head if (r,c) not in hair_part1+hair_part2+eyepatch]
        hat_part1 = cgd.horizontal_lines([(min(r for r,c in hair)-1, (head_center_col-2, head_center_col))]
                    )+cgd.draw_slope_0p5_diagonal(min(r for r,c in hair)-1, head_center_col-2, head_center_row-2, 
                                                  left_down=True, right_down=False, left_up=False, right_up=False
                    )+cg.hex_neighbours_n(min(r for r,c in hair)-1, head_center_col-2, n=1, keep_origin = True, return_frontier=False)
        hat_part2 = cgd.verticle_line(min(r for r,c in hair)-1, head_center_col,
                                       min(r for r,c in hair)-2, bend = 'right'
                         )+[(max(r for r,c in hat_part1), min(c for r,c in hat_part1 if r==max(r for r,c in hat_part1))-1 )]
        hat = hat_part1+hat_part2
            
        hat_acc = cgd.verticle_line(min(r for r,c in hat), min(c for r,c in hat if r==min(r for r,c in hat)),
                                    min(r for r,c in hat)+1, bend = 'left'
                         )+[(min(r for r,c in hat), min(c for r,c in hat if r==min(r for r,c in hat))+1)]

        if 'back' in version:
            hat_acc = flip_wrtcol(head_center_col, hat_acc)
            hat= flip_wrtcol(head_center_col, hat)
            hair+=face+eyepatch+head
            face, eyepatch = [], []
        tops = [(r,c) for r,c in raw_body if r>max(r_ for r_,c_ in neck) and r <ln.femoral_head_r]
        if 'front' in version:
            collar_part1 =[(r,c) for r,c in  neck if (r,c) not in face]
        elif 'back' in version:
            collar_part1 =[(r,c) for r,c in  neck if (r,c) not in hair]
        collar_part2 = cgd.horizontal_lines([(min(r for r,c in collar_part1), 
                                              (min(c for r,c in collar_part1 if r==min(r for r,c in collar_part1))-1, 
                                               max(c for r,c in collar_part1 if r==min(r for r,c in collar_part1))+1))]
                    )
        collar = collar_part1+collar_part2
        if 'front' in version:
            ribbon = [(max(r for r,c in collar_part1)+1, head_center_col)]
        elif 'back' in version:
            ribbon = []
        
        
        if 'front' in version:    
            vest_part1 = [(r,c) for r,c in torso_dict["torso"] if r<ln.bellybutton_r and r>ln.breastbone_end_r and 
                    (r,c) not in [(ln.bellybutton_r-1,head_center_col )]]
            vest_part2 = cgd.verticle_line(min(r for r,c in vest_part1), 
                                           min(c for r,c in vest_part1 if r==min(r for r,c in vest_part1)),
                                           min(r for r,c in tops), bend = 'left'
                             )+cgd.verticle_line(min(r for r,c in vest_part1), 
                                           max(c for r,c in vest_part1 if r==min(r for r,c in vest_part1)),
                                           min(r for r,c in tops), bend = 'right'
                             )+cgd.verticle_line(min(r for r,c in vest_part1), 
                                           min(c for r,c in vest_part1 if r==min(r for r,c in vest_part1)),
                                           min(r for r,c in tops)+1, bend = 'right'
                             )+cgd.verticle_line(min(r for r,c in vest_part1), 
                                           max(c for r,c in vest_part1 if r==min(r for r,c in vest_part1)),
                                           min(r for r,c in tops)+1, bend = 'left'
                             )
            vest = vest_part1+vest_part2
        elif 'back' in version:
            vest = [(r,c) for r,c in torso_dict["torso"] if r<ln.bellybutton_r and r>=ln.shoulder_joint_r]
            
        pants_part1 = [(r,c) for r,c in raw_body_modified if r >=ln.femoral_head_r-1 and r<ln.sartorius_turning_r
                      and (r,c) not in upperarm_dict["upperarm"]+forearm_dict["forearm"]+hand_dict["hand"]
                     ]
        
        sk = cgd.verticle_line(max(r for r,c in pants_part1)-1, 
                                       max(c for r,c in pants_part1 if r==max(r for r,c in pants_part1)-1)+1,
                               ln.gastrocnemius_end_r,  bend = 'right')+cgd.verticle_line(max(r for r,c in pants_part1)-1, 
                                       max(c for r,c in pants_part1 if r==max(r for r,c in pants_part1)-1)+1,ln.foot_r,  
                                   bend = 'left')
        if 'back' in version:
            sk= flip_wrtcol(head_center_col, sk)
        pants_part2 = cgd.verticle_line(max(r for r,c in pants_part1)-1, 
                                       min(c for r,c in pants_part1 if r==max(r for r,c in pants_part1)-1)-1,
                                       max(r for r,c in pants_part1)+1,  bend = 'left'
                                   )+cgd.verticle_line(max(r for r,c in pants_part1)-1, 
                                       max(c for r,c in pants_part1 if r==max(r for r,c in pants_part1)-1)+1,
                                       max(r for r,c in pants_part1)+1,  bend = 'right'
                                   )+cgd.horizontal_lines([(max(r for r,c in pants_part1)-1,
                                              (min(c for r,c in pants_part1 if r==max(r for r,c in pants_part1)-1),
                                               max(c for r,c in pants_part1 if r==max(r for r,c in pants_part1)-1)))])

        pants = pants_part1+pants_part2
        belt = cgd.verticle_line(min(r for r,c in pants), min(c for r,c in pants if r==min(r for r,c in pants)),min(r for r,c in vest),  
                      bend = 'right')+cgd.verticle_line(min(r for r,c in pants), 
                                                       max(c for r,c in pants if r==min(r for r,c in pants)),min(r for r,c in vest),  
                      bend = 'left')
        skin = [(r,c) for r,c in raw_body_modified if r>=ln.sartorius_turning_r and r<ln.sartorius_turning_r+2
               and (r,c) not in upperarm_dict["upperarm"]+forearm_dict["forearm"]]+hand_dict["hand"]
        
        right_shoe = right_foot_modified+[(r,c) for r,c in right_modified_calf if r>min(r_ for r_,c_ in right_foot_modified)-3]
        left_shoe = left_foot_modified
        
        if 'front' in version:   
            right_sock = [(r,c) for r,c in right_modified_calf+thigh_dict["right_thigh"] if (r,c) not in skin+right_shoe+pants]
            left_sock = [(r,c) for r,c in left_modified_calf+thigh_dict["left_thigh"] if (r,c) not in skin+left_shoe+pants]
        elif 'back' in version:
            right_sock = [(r,c) for r,c in right_modified_calf+thigh_dict["left_thigh"] if (r,c) not in skin+right_shoe+pants]
            left_sock = [(r,c) for r,c in left_modified_calf+thigh_dict["right_thigh"] if (r,c) not in skin+left_shoe+pants]
        garter = [(min(r for r,c in right_sock)-2, min(c for r,c in right_sock if r==min(r for r,c in right_sock))),
                  (min(r for r,c in left_sock)-2, min(c for r,c in left_sock if r==min(r for r,c in left_sock))),

                   ]
        if 'front' in version:   
            keys_extra = [ "left_shoe", "left_sock", "right_shoe", "right_sock", ]
            keys_extra_colors = [
                
                [cgc.hex_to_rgb("#c3becb")], 
                [[0,0,0],[1,1,1], 'stripe'],
                 [cgc.hex_to_rgb("#f0eafa")],[[0.1,0.1,0.1]], ]
            
        elif 'back' in version:
            keys_extra = ["right_shoe", "right_sock", "left_shoe", "left_sock",  ]
            keys_extra_colors = [
                [cgc.hex_to_rgb("#f0eafa")],[[0.1,0.1,0.1]],
                [cgc.hex_to_rgb("#c3becb")], 
                [[0,0,0],[1,1,1], 'stripe'],
                  ]
            
        keys = ["hair", "hat", "hat_acc",

                "face", "eyepatch", "tops", "collar", "ribbon", "vest", 

               "sk", "pants", "belt", "skin", "garter",]+keys_extra#
        colors = [ [cgc.hex_to_rgb("#545964")], [cgc.hex_to_rgb("#0400d0")],[[0.9,0.9,0.9]], 

                  [cgc.hex_to_rgb("#f8d6b2")], [[0,0,0]],[[0.8,0.8,0.8],[1,1,1], 'stripe'],  [[1,1,1]],[cgc.hex_to_rgb("#7333ac")], 
                  [[0,0,0]],

                  [[0,0,1], cgc.hex_to_rgb("#1d1b79"), 'row' ], 
                  [cgc.hex_to_rgb("#3533ac")],[[0,0,1]],[cgc.hex_to_rgb("#f8d6b2")],[[0,0,0]],

                 ]+keys_extra_colors


    local_vars = locals()
                
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})


def draw_bkgd(ciel_loc, doll_loc,joker_loc, hex_rc_arr):
    
    joker_spot_loc = (joker_loc[0]+5, joker_loc[1])
    joker_spot_in = cg.hex_neighbours_n(joker_spot_loc[0], joker_spot_loc[1], n=7)
    joker_spot_out = cg.hex_neighbours_n(joker_spot_loc[0], joker_spot_loc[1], n=15)
    joker_spot_in_color = [cgc.hex_to_rgb("#bffff6")]#172421
    joker_spot_out_COLOR = [cgc.hex_to_rgb("#bffff6"), cgc.hex_to_rgb("#2f0919"), 'sigmafactor=1.2',
                   [joker_spot_loc, 8, 7, 15, 'hex', ]]
    joker_region = cgd.draw_trapezoid(min(r for r,c in joker_spot_out), 
                                    min(c for r,c in joker_spot_out if r==min(r for r,c in joker_spot_out)), 
                                    max(c for r,c in hex_rc_arr), min(r for r,c in hex_rc_arr), 
                                    slope_left = '0.5', slope_right = 'inf', direction = 'rl',bend_left = 'left', bend_right = 'right'
               )[-2]+ cgd.draw_trapezoid(min(r for r,c in joker_spot_out), 
                                    min(c for r,c in joker_spot_out if r==min(r for r,c in joker_spot_out)), 
                                    max(c for r,c in hex_rc_arr), max(r for r,c in hex_rc_arr), 
                                    slope_left = '0.5', slope_right = 'inf', direction = 'lr',bend_left = 'left', bend_right = 'right'
               )[-2]      
    
    ciel_region_raw = cgd.draw_trapezoid(max(r for r,c in hex_rc_arr), 
                                    min(c for r,c in joker_region if r==max(r for r,c in hex_rc_arr))+5, 
                                    max(c for r,c in hex_rc_arr), min(r for r,c in hex_rc_arr), 
                                    slope_left = '1.5', slope_right = 'inf', direction = 'lr',bend_left = 'left', bend_right = 'right'
               )[-2]
    ciel_region = [(r,c) for r,c in ciel_region_raw if (r,c) not in joker_region]
    
    stage_region = [(r,c) for r,c in hex_rc_arr if r>=max(r for r,c in hex_rc_arr)-13 and (r,c) not in joker_region]
    bottom_region = [(r,c) for r,c in hex_rc_arr if (r,c) not in joker_region+ciel_region+stage_region]
    doll_region = [(r,c) for r,c in hex_rc_arr if c<doll_loc[1]+3 and (r,c) not in stage_region+bottom_region]
    

    joker_region_color = [cgc.hex_to_rgb("#2f0919")]
    ciel_region_color = [cgc.hex_to_rgb("#ffc647"),'sigmafactor=0.5'] 
    doll_region_color = [[0,0,0]]
    stage_region_color = [cgc.hex_to_rgb("#f63b2f"),'sigmafactor=3']  
    bottom_region_color = [cgc.hex_to_rgb("#1e3226")]#    172421
    
    keys = ["joker_region", "ciel_region","stage_region", "bottom_region", "doll_region",  "joker_spot_out", "joker_spot_in",]#
    colors = [joker_region_color, ciel_region_color, stage_region_color, bottom_region_color, doll_region_color, joker_spot_out_COLOR,joker_spot_in_color, ]#
    local_vars = locals()
                
    return cgc.keep_top_color({k: [local_vars[k], c] for k, c in zip(keys, colors)})


center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
center_col = max(c for r, c in hex_rc_arr) // 2

ciel_loc = (center_row-8, center_col)
doll_loc = (center_row-20, center_col-28)
joker_loc = (max(r for r, c in hex_rc_arr)-25, center_col+30)

char_dict_ = {
    'bkdg': draw_bkgd(ciel_loc, doll_loc,joker_loc, hex_rc_arr),
    'ciel front': draw_char(ciel_loc[0], ciel_loc[1], 'ciel front'),
              'doll front': draw_char(doll_loc[0], doll_loc[1], 'doll front v1'),
              'joker':  draw_char(joker_loc[0], joker_loc[1], 'joker')
    
}

for char, dict_ in char_dict_.items(): 
    for key in dict_.keys():
        
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        
        m = re.search(r"sigmafactor\s*=\s*([\d.]+)", str(colors))
        sigmafactor = float(m.group(1)) if m else 1.

        is_single_with_modifier = len(colors) == 2 and isinstance(colors[1], str) and 'sigmafactor' in colors[1]
         
        
        is_striped = len(colors) == 3 and isinstance(colors[-1], str) and 'stripe' in colors[-1]
        if is_striped:
            select_part= cg.select_mask(part,hex_rc_arr)
            stripe_target_colors = cgc.color_stripe(hex_rc_arr, STRIPE_WIDTH=1, STRIPE_PALETTE=[np.array(colors[0]), np.array(colors[1])])
            hex_colors[select_part] = np.array([cgc.select_normal_color([True], c, np.ones(3)*sigma_color*2)[0]
                                                    for c in stripe_target_colors[select_part] 
                                                   ])
            
            continue
        if len(colors) == 1 or is_single_with_modifier:

            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*sigmafactor) 
            
        else:
            if len(colors) == len(hex_rc_arr):
                print('FULLY DEFINED COLORS')
                select_part = cg.select_mask(part, hex_rc_arr)
                
                for idx in np.where(select_part)[0]:
                    hex_colors[idx] = cgc.select_normal_color([True], colors[idx], np.ones(3) * sigma_color*sigmafactor)[0]
            else:
                sort_info = colors[-1]
                if 'hex' in str(sort_info):
                    print('sored by hex')
                    
                    
                    center =sort_info[0]
                    n_layers = sort_info[1]
                    start_n = sort_info[2]
                    end_n=sort_info[3]
                    if 'sigmoid' in str(sort_info):
                        mode = 'sigmoid'
                    else:
                        mode = 'linear'
                        
                    print(mode)
                    hex_colors = cgc.color_hex_gradient(part,colors[0],colors[1], hex_rc_arr, 
                            hex_colors, center, n_layers,
                            sigma_color=sigma_color*sigmafactor, end_weight=0.01, period=None, mode=mode,
                            start_n=start_n, end_n=end_n)
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
                            hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color*sigmafactor, 
                                                        end_weight= 0.01, mode = mode, pivot=pivot ) 

                else:
                    print(key)
                    print('row gradient')
                    
                    if 'sigmoid' in sort_info:
                        mode = 'sigmoid'
                    else:
                        mode = 'linear'
                        
                    if 'max' in sort_info:
                        pivot = 'max'
                    elif 'min' in sort_info:
                        pivot = 'min'
                    else:
                        pivot = None
                        
                    print('mode', mode, 'pivot', pivot)

                    hex_colors = cgc.color_row_gradient(part, 
                            colors[0],colors[1],
                            hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*sigmafactor, 
                                                        end_weight= 0.01, mode = mode, pivot=pivot )

pc.set_facecolor(hex_colors)
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'BlackButler_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')