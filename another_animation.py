import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import astropy.units as u
import os
from collections import defaultdict
import math
import cg_plot_fn as cg

import cg_draw_fn as cgd
import cg_color_fn as cgc
from TRAJECTORY_pendulum_drip import Pendulum, Drip

def draw_body(start_center_row, start_center_col, n = 2):
    center_col = start_center_col
    head_center_row = start_center_row 
    body_center_row = head_center_row+n+1+n+1
    thigh_center_row = body_center_row+n+1+n+1
    
    head = cg.hex_neighbours_n(head_center_row,center_col, n=n, keep_origin = True)
    head_bottom_r = head_center_row+n
    head_bottom_line_info = (max(r for r, c in head), 
                             (min(c for r, c in head if r ==head_bottom_r), max(c for r, c in head if r ==head_bottom_r))
                            )
    neck = cgd.draw_block(head_bottom_line_info, head_bottom_line_info[0]+1)#+n-1)
    neck_valid = [x for x in neck if x not in head]
    
    
    body = cg.hex_neighbours_n(body_center_row,center_col, n=n, keep_origin = True)    
    body_bottom_r = body_center_row+n
    body_bottom_line_info = (max(r for r, c in body), 
                             (min(c for r, c in body if r ==body_bottom_r), max(c for r, c in body if r ==body_bottom_r))
                            )
    pelvis = cgd.draw_block(body_bottom_line_info, body_bottom_line_info[0]+1)
    pelvis_valid = [x for x in pelvis if x not in body]
    
    thigh = cg.hex_neighbours_n(thigh_center_row,center_col, n=n, keep_origin = True)    
    
    return head, neck_valid, body, pelvis_valid, thigh, [center_col, head_center_row, body_center_row, thigh_center_row]

def modify_pelvis(pelvis):
    pelvis_left, pelvis_right = min(pelvis, key=lambda x: x[1]), max(pelvis, key=lambda x: x[1])
    pelvis_tot = pelvis+[(pelvis_left[0], pelvis_left[1] - 1), (pelvis_right[0], pelvis_right[1] + 1)]
    return pelvis_tot
def modify_body(body, body_center_row, view = 'front'):
    body_left, body_right = min(c for r,c in body if r == body_center_row), max(c for r,c in body if r == body_center_row)
    if view == 'front':
        
        remove = cgd.draw_slope_0p5_diagonal(body_center_row, body_left, body_center_row+1, 
                                             left_down=False, right_down=True, left_up=False, right_up=False
                                            )+cgd.draw_slope_0p5_diagonal(body_center_row, body_right, body_center_row+1, 
                                             left_down=True , right_down=False, left_up=False, right_up=False
                                            )
        
    if view == 'side':
        remove = [(body_center_row, body_left), (body_center_row, body_right)]
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
    if view == 'front':
        thigh_remove = remove_inner_coords(thigh)
        thigh_remove+=[(thigh_center_row, 
                        int(min(c for r,c in thigh if r == thigh_center_row)+max(c for r,c in thigh if r == thigh_center_row))/2)]
        
    if view == 'side':
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

def draw_shoulder(body, n, view = 'front'):
    
    body_top_r = min(r for r, c in body)
    body_bottom_r = max(r for r, c in body)
    shoulder_extensions = math.ceil(n/2)
    if view == 'front':
        left_shoulder  = (body_top_r, min(c for r, c in body if r ==body_top_r))                 
        right_shoulder = (body_top_r, max(c for r, c in body if r ==body_top_r))
        left_arm_tot = []
        right_arm_tot = []

        for shoulder_extension in range(shoulder_extensions):
            left_arm = cgd.draw_slope_0p5_diagonal(left_shoulder[0], left_shoulder[1]-shoulder_extension-1, 
                                                 body_bottom_r-shoulder_extension, 
                                                left_down=True, right_down=False, left_up=False, right_up=False)
            right_arm = cgd.draw_slope_0p5_diagonal(right_shoulder[0], right_shoulder[1]+shoulder_extension+1, 
                                                  body_bottom_r-shoulder_extension, 
                                                 left_down=False, right_down=True, left_up=False, right_up=False)
            left_arm_tot+=left_arm
            right_arm_tot+=right_arm
            if shoulder_extension == 0:
                left_elbow_start = max(left_arm, key=lambda x: x[0])
                right_elbow_start = max(right_arm, key=lambda x: x[0])
        return left_arm_tot, right_arm_tot, left_elbow_start, right_elbow_start
    if view == 'side':
        arm_tot = []
        shoulder_c= int((min(c for r, c in body if r ==body_top_r)+max(c for r, c in body if r ==body_top_r))/2)
        shoulder = (body_top_r, shoulder_c)
        arm_basic = cgd.verticle_line(body_top_r, shoulder_c, body_bottom_r)
        elbow_start =max(arm_basic, key=lambda x: x[0])
        
        arm_tot+=arm_basic
        return arm_tot,elbow_start
    
def draw_calf(thigh, thigh_center_row, n, view = 'front'):   
    thigh_bottom_r =  max(r for r, c in thigh)
    ankle_row = thigh_bottom_r+n
    knee_extensions = math.ceil(n/2)

    if view == 'front':
        left_knee  = (thigh_bottom_r, min(c for r, c in thigh if r ==thigh_bottom_r))                      
        right_knee = (thigh_bottom_r, max(c for r, c in thigh if r ==thigh_bottom_r))    
        
        left_calf_tot = []
        right_calf_tot = []

        left_start_ = left_knee#(thigh_bottom_r, min(c for r, c in thigh if r ==thigh_bottom_r))
        right_start_= right_knee#(thigh_bottom_r, max(c for r, c in thigh if r ==thigh_bottom_r)) 
        for calf_thick in range(knee_extensions ):
            left_calf = cgd.draw_slope_0p5_diagonal(left_start_[0], left_start_[1]-1, ankle_row, left_down=False, 
                                             right_down=True, left_up=False, right_up=False)
            right_calf = cgd.draw_slope_0p5_diagonal(right_start_[0], right_start_[1]+1, ankle_row, left_down=True , 
                                         right_down=False, left_up=False, right_up=False)

            if calf_thick == 0:
                left_ankle = max(left_calf, key=lambda x: x[0])
                right_ankle = max(right_calf, key=lambda x: x[0])
            left_calf_tot += left_calf
            left_start_ = left_calf[math.floor(len(left_calf)/2)]

            right_calf_tot += right_calf
            right_start_ = right_calf[math.floor(len(right_calf)/2)]


        return left_calf_tot, right_calf_tot, left_knee, right_knee, left_ankle, right_ankle
    if view == 'side':
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

def draw_feet(left_ankle, right_ankle, n, view = 'front'):   
    if view == 'front':
        left_foot = cgd.draw_slope_0p5_diagonal(left_ankle[0], left_ankle[1], left_ankle[0]+n, left_down=True , 
                                         right_down=False, left_up=False, right_up=False)
        right_foot = cgd.draw_slope_0p5_diagonal(right_ankle[0], right_ankle[1], right_ankle[0]+n, left_down=False, 
                                         right_down=True , left_up=False, right_up=False)

        left_paw_start= max(left_foot, key=lambda x: x[0])
        right_paw_start= max(right_foot, key=lambda x: x[0])
        paw_row = left_paw_start[0]

        foot_extensions = n-2
        left_foot_ext = cgd.horizontal_lines([(left_paw_start[0], (left_paw_start[1], left_paw_start[1]-foot_extensions))])
        right_foot_ext = cgd.horizontal_lines([(right_paw_start[0], (right_paw_start[1], right_paw_start[1]+foot_extensions))])

        left_foot_tot = left_foot+left_foot_ext
        right_foot_tot = right_foot+right_foot_ext
        return left_foot_tot, right_foot_tot, paw_row
    
    if view == 'side':
        ankle = left_ankle
        foot = cgd.verticle_line(ankle[0], ankle[1], ankle[0]+n)

        return foot

def remove_inner_coords(arr):
    rows = defaultdict(list)
    for r, c in arr:
        rows[r].append(c)
    
    removed = []
    n_to_remove = None

    for row in sorted(rows.keys(), reverse=True):
        cols = sorted(rows[row])
        inner_cols = cols[1:-1]  # strip the two edge columns

        if n_to_remove is None:
            n_to_remove = len(inner_cols)  # bottom row: remove all inner

        if n_to_remove < 1:
            break

        n = min(n_to_remove, len(inner_cols))
        start = (len(inner_cols) - n) // 2   # center the slice
        to_remove = inner_cols[start : start + n]

        for c in to_remove:
            removed.append((row, c))

        n_to_remove -= 1

    return removed

center_row = 12
center_col = 17
# ── hex scene ────────────────────────────────────────────────────────────

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W, IMG_H, HEX_R, _dx_hex_center, _dy_hex_center = detail_info
sigma_color = 0.02
hex_colors = cgc.color_row_gradient([(r,c) for r,c in hex_rc_arr if r<=center_row], 
                                    cgc.hex_to_rgb("#fffc00"), cgc.hex_to_rgb("#ff0000"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.1)    
hex_colors = cgc.color_row_gradient([(r,c) for r,c in hex_rc_arr if r>center_row], 
                                    cgc.hex_to_rgb("#ff0000"), cgc.hex_to_rgb("#543300"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.1)    


n_eyeball = 9
n_head = 2
right_eyeball_r = center_row
right_eyeball_c = center_col

right_eye_ball = cg.hex_neighbours_n(right_eyeball_r, right_eyeball_c, n=n_eyeball, keep_origin = True, return_frontier=False)
right_eye_ball_top_r = right_eyeball_r-n_eyeball
right_eyesocket_r = right_eye_ball_top_r+1

right_eyesocket = cg.hex_neighbours_n(right_eyeball_r, right_eyeball_c, n=n_eyeball+1, keep_origin = False , return_frontier=True
                                         )[-1]+cg.hex_neighbours_n(right_eyeball_r, right_eyeball_c, n=n_eyeball+2, 
                                                                   keep_origin = False , return_frontier=True)[-1]


#++
remove = [(r,c) for r,c in right_eyesocket if c <=min(c for r,c in right_eyesocket if r ==right_eyeball_r+n_eyeball+2) or
         (r >right_eyeball_r and 
          c <=max(c for r,c in right_eyesocket if r ==right_eyeball_r)-1 and 
          c >=max(c for r,c in right_eyesocket if r ==right_eyeball_r+n_eyeball+2) )]
valid_right_eyesocket = [(r,c) for r,c in right_eyesocket if (r,c) not in remove]
eyesocket_r = right_eyeball_r-n_eyeball
right_eyesocket_full = valid_right_eyesocket+cgd.draw_slope_8p3_diagonal(eyesocket_r-1, 
                                          min(c for r,c in valid_right_eyesocket if r ==eyesocket_r-1), 
                                                                         eyesocket_r, 
                                          left_down=True , right_down=False, left_up=False, right_up=False
                                         )+cgd.draw_slope_8p3_diagonal(eyesocket_r-2, 
                                          min(c for r,c in valid_right_eyesocket if r ==eyesocket_r-2), 
                                                                       eyesocket_r, 
                                          left_down=True , right_down=False, left_up=False, right_up=False
                                         )+cgd.draw_slope_8p3_diagonal(eyesocket_r-2, 
                                          min(c for r,c in valid_right_eyesocket if r ==eyesocket_r-2)+1, 
                                                                       eyesocket_r, 
                                          left_down=True , right_down=False, left_up=False, right_up=False
                                         )+cgd.draw_slope_0p5_diagonal(right_eyeball_r, 
                                          max(c for r,c in valid_right_eyesocket if r ==right_eyeball_r), 
                                                                         right_eyeball_r+3, 
                                          left_down=True , right_down=False, left_up=False, right_up=False
                                         )+cgd.horizontal_lines([(right_eyeball_r, 
                                                                  (max(c for r,c in valid_right_eyesocket if r ==right_eyeball_r), 
                                                                   max(c for r,c in valid_right_eyesocket if r ==right_eyeball_r)+2))]
                                                               )+cgd.draw_slope_1p5_diagonal(right_eyeball_r+n_eyeball+1, 
                                          max(c for r,c in valid_right_eyesocket if r ==right_eyeball_r+n_eyeball+1), 
                                                                         right_eyeball_r+n_eyeball+1-2, 
                                          left_down= False, right_down=False, left_up=False, right_up=True 
                                         )+cgd.draw_slope_1p5_diagonal(right_eyeball_r+n_eyeball+1, 
                                          max(c for r,c in valid_right_eyesocket if r ==right_eyeball_r+n_eyeball+1)+1, 
                                                                         right_eyeball_r+n_eyeball+1-1, 
                                          left_down= False, right_down=False, left_up=False, right_up=True 
                                         )+cgd.draw_slope_1p5_diagonal(right_eyeball_r+n_eyeball+1, 
                                          min(c for r,c in valid_right_eyesocket if r ==right_eyeball_r+n_eyeball+1), 
                                                                         right_eyeball_r+n_eyeball+1-2, 
                                          left_down= False, right_down=False, left_up=True , right_up= False
                                         )+cgd.draw_slope_1p5_diagonal(right_eyeball_r+n_eyeball+1, 
                                          min(c for r,c in valid_right_eyesocket if r ==right_eyeball_r+n_eyeball+1)-1, 
                                                                         right_eyeball_r+n_eyeball+1-1, 
                                          left_down= False, right_down=False, left_up=True , right_up= False
                                         )
valid_right_eye_ball = [(r,c) for r,c in right_eye_ball if
                        c <=right_eyeball_c+n_eyeball-2 and
                        c >=right_eyeball_c-n_eyeball+2]

select_eyesocket = cg.select_mask(right_eyesocket_full,hex_rc_arr)

select_eyeball_regions = cg.select_mask(valid_right_eye_ball, hex_rc_arr)


#eyeball color
hex_colors = cgc.color_hex_gradient(valid_right_eye_ball,  cgc.hex_to_rgb("#ff0000") , cgc.hex_to_rgb("#00ff85" ),
                                    hex_rc_arr, hex_colors, (right_eyeball_r, right_eyeball_c),
                                    n_eyeball,  sigma_color=0.03, end_weight= 0.1, period = 5)
hex_colors[select_eyesocket] = cgc.select_normal_color(select_eyesocket, 
                                                               cgc.hex_to_rgb("#151b21"), np.ones(3)*sigma_color) 


char_dict = {"Sakuragi":[(right_eyeball_r-n_head, right_eyeball_c),"front"],# 
        "Akazawa": [(right_eyeball_r-n_head-n_head-n_head-1, min(c for r,c in valid_right_eyesocket)-n_head-n_head-n_head-n_head),"side"],
         "Sakakibara": [(right_eyeball_r-n_head-1, 
                         max(c for r,c in valid_right_eyesocket if r ==right_eyeball_r)+n_head+n_head),"side"],#
        }
umbrella_frames = []
n_umbrella = n_head+2
for key in char_dict.keys():
    head_center = char_dict.get(key)[0]
    umbrella_frame = cgd.draw_slope_0p5_diagonal(head_center[0], head_center[1], head_center[0]+n_umbrella, 
                                          left_down=True , right_down=True, left_up=True, right_up=True
                                         )+cgd.draw_slope_0p5_diagonal(head_center[0], head_center[1], head_center[0]-n_umbrella, 
                                          left_down=True , right_down=True, left_up=True, right_up=True
                                         )
    umbrella_frames.extend(umbrella_frame)
    
    print(key, head_center)
    head, neck, body, pelvis, thigh, raw_body_detail= draw_body(head_center[0], head_center[1], n = n_head)
    select_head= cg.select_mask(head,hex_rc_arr)
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail
    
    pelvis_final = modify_pelvis(pelvis)
    body_final = modify_body(body, body_center_row, view =char_dict.get(key)[1] )
    thigh_final = modify_thigh(thigh, thigh_center_row, view =char_dict.get(key)[1] )
    
    if key in ["Sakuragi", "Akazawa"]:
        if char_dict.get(key)[1] == "front":
            left_arm, right_arm, left_albow_start, right_albow_start = draw_shoulder(body, n_head, view =char_dict.get(key)[1] )

            left_calf, right_calf, left_knee, right_knee, left_ankle, right_ankle = draw_calf(thigh, thigh_center_row,n_head, 
                                                                                              view =char_dict.get(key)[1] )
            left_foot, right_foot, paw_row = draw_feet(left_ankle, right_ankle, n_head, view =char_dict.get(key)[1] )
            left_forearm = cgd.horizontal_lines([(left_albow_start[0], (left_albow_start[1], left_albow_start[1]-n_head))])
            right_forearm = cgd.horizontal_lines([(right_albow_start[0], (right_albow_start[1], right_albow_start[1]+n_head))])
            left_hand = [(max(r for r,c in left_forearm), min(c for r,c in left_forearm if r== max(r for r,c in left_forearm)) )]

            knee_r = left_knee[0]
            calf_tot = left_calf+right_calf
            shoes_tot = left_foot+right_foot
            right_hand = [(max(r for r,c in right_forearm), max(c for r,c in right_forearm if r== max(r for r,c in right_forearm)) )]
        elif char_dict.get(key)[1]  == "side":
            arm,elbow_start = draw_shoulder(body, n_head, view =char_dict.get(key)[1] )

            calf, knee, ankle = draw_calf(thigh_final, thigh_center_row, n_head,  view =char_dict.get(key)[1] )
            foot = draw_feet(ankle, ankle, n_head, view =char_dict.get(key)[1] )


            knee_r = knee[0]
            calf_tot = calf

            shoes_tot = foot
            forearm = cgd.draw_slope_0p5_diagonal(elbow_start[0], elbow_start[1], elbow_start[0]+n_head, left_down=False, 
                                                 right_down=True, left_up=False, right_up=False)
        sk_row = thigh_center_row-1
        sk = [(r,c)  for r,c in thigh_final if r <=sk_row]
        leg =[(r,c)  for r,c in thigh_final if r >sk_row]

        full_leg = leg+calf_tot+shoes_tot

        if key == "Sakuragi":
            sock_row = knee_r
            socks = [(r,c)  for r,c in calf_tot if r >sock_row]
            shirt = neck
            tie = [(min(r for r,c in body), center_col)]
            hair = cgd.draw_trapezoid(head_center[0]-n_head, 
                                      min(c for r,c in head if r == head_center[0]-n_head),
                                      max(c for r,c in head if r == head_center[0]-n_head), head_center[0]+n_head, 
                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr')[-2]
            select_hair= cg.select_mask(hair,hex_rc_arr)

            remove_head = [(r,c) for (r,c) in
                           cg.hex_neighbours_n(head_center[0], head_center[1],n=n_head, keep_origin =  False, return_frontier=True)[-1]
                          if r <=head_center[0] ] +[(r,c) for (r,c) in head
                          if r == head_center[0]-n_head+1] 

            face = [(r,c) for r,c in head if (r,c) not in remove_head]
            select_face = cg.select_mask(face,hex_rc_arr)
            hex_colors[select_hair] = cgc.select_normal_color(select_hair, cgc.hex_to_rgb("#c8823a") , np.ones(3)*sigma_color) 
            hex_colors[select_face] = cgc.select_normal_color(select_face, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 



            select_blazer= cg.select_mask(body_final+pelvis_final+left_arm+ right_arm+left_forearm+right_forearm,hex_rc_arr)
            select_skin= cg.select_mask(leg+left_hand+right_hand,hex_rc_arr)
        elif key == "Akazawa":
            sock_row = min(r for r,c in full_leg)
            arm+=cgd.draw_slope_0p5_diagonal(body_center_row+n_head, 
                                                     max(c for r,c in body_final if r == body_center_row+n_head)+1,
                                                     elbow_start[0]+n_head-1, left_down= False, 
                                             right_down=True , left_up=False, right_up=False)
            shirt = neck+cgd.draw_slope_0p5_diagonal(min(r for r,c in body), 
                                                     max(c for r,c in body if r == min(r for r,c in body)),
                                                     min(r for r,c in body)+1, left_down= False, 
                                             right_down=True , left_up=False, right_up=False)+[(r,c) for (r,c) in arm if 
                                                                                               r <=elbow_start[0]-2]
            tie = [(min(r for r,c in body), max(c for r,c in body if r == min(r for r,c in body)))]

            remove_hair = [(head_center_row, center_col+n_head)] 

            hair = [(r,c) for r,c in head if (r,c) not in remove_hair]

            pony_tail_left = cgd.verticle_line(min(r for r,c in hair)+1, min(c for r,c in hair if r == min(r for r,c in hair))-1, 
                                                                             body_center_row)


            pony_tail_right = cgd.verticle_line(min(r for r,c in hair)+1, max(c for r,c in hair if r == min(r for r,c in hair)+1), 
                                                                             body_center_row, bend = 'right')

            left_hair_acc = [(min(r for r,c in hair)+1, min(c for r,c in hair if r == min(r for r,c in hair)+1))]
            select_hair_acc =  cg.select_mask(left_hair_acc,hex_rc_arr)
            select_hair= cg.select_mask(hair+pony_tail_left+pony_tail_right,hex_rc_arr)

            face = cg.hex_neighbours_n(max(r for r,c in head)-1,
                                       max(c for r,c in head if r ==max(r for r,c in head)-1)-1,
                                           n=n_head-1, keep_origin =True  , return_frontier=False )

            select_face = cg.select_mask(face,hex_rc_arr)
            hex_colors[select_hair] = cgc.select_normal_color(select_hair, cgc.hex_to_rgb("#7b3728") , np.ones(3)*sigma_color) 
            hex_colors[select_face] = cgc.select_normal_color(select_face, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 
            hex_colors[select_hair_acc] = cgc.select_normal_color(select_hair_acc, cgc.hex_to_rgb("#5b85f1") , np.ones(3)*sigma_color) 

            select_blazer= cg.select_mask(body_final+pelvis_final,hex_rc_arr)
            select_skin= cg.select_mask(leg+forearm+[(r,c) for (r,c) in arm if r >elbow_start[0]-2],hex_rc_arr)


        select_shirt= cg.select_mask(shirt,hex_rc_arr)
        select_tie = cg.select_mask(tie,hex_rc_arr)
        select_sk= cg.select_mask(sk,hex_rc_arr)

        socks = [(r,c)  for r,c in full_leg if r >sock_row]
        select_socks= cg.select_mask(socks,hex_rc_arr)
        select_shoes= cg.select_mask(shoes_tot,hex_rc_arr)

        hex_colors[select_sk] = cgc.select_normal_color(select_sk, cgc.hex_to_rgb("#2e27aa") , np.ones(3)*sigma_color) 
        hex_colors[select_blazer] = cgc.select_normal_color(select_blazer, [0.1, 0.1, 0.1] , np.ones(3)*sigma_color) 
        hex_colors[select_shirt] = cgc.select_normal_color(select_shirt, [1,1,1] , np.ones(3)*sigma_color) 
        hex_colors[select_tie] = cgc.select_normal_color(select_tie, [1,0,0] , np.ones(3)*sigma_color) 
        hex_colors[select_skin] = cgc.select_normal_color(select_skin, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 
        hex_colors[select_socks] = cgc.select_normal_color(select_socks, [0.1, 0.1, 0.1] , np.ones(3)*sigma_color) 
        hex_colors[select_shoes] = cgc.select_normal_color(select_shoes, [0,0,0] , np.ones(3)*sigma_color) 


    else:
        
        arm,elbow_start = draw_shoulder(body, n_head, view =char_dict.get(key)[1] )
        pants_top_r = cgd.draw_trapezoid(pelvis_final[0][0], 
                                      min(c for r,c in pelvis_final if r == pelvis_final[0][0]),
                                      max(c for r,c in pelvis_final if r == pelvis_final[0][0]), pelvis_final[0][0]+1, 
                                      slope_left = '0.5', slope_right = '0.5', direction = 'rl')[-2]
        full_leg = cgd.draw_block((max(r for r,c in pants_top_r), (min(c for r,c in pants_top_r if r == max(r for r,c  in pants_top_r)),
                                                                 max(c for r,c in pants_top_r if r == max(r for r,c  in pants_top_r))
                                                                )), 
                                   thigh_center_row+n_head+n_head+n_head)
        forearm =    cgd.verticle_line(elbow_start[0], elbow_start[1], body_center_row+n_head+n_head, bend = 'right')
        hand = [forearm[-1]]
        select_skin= cg.select_mask(hand,hex_rc_arr)
        
        
        shirt = neck
        remove_hair = [(head_center_row, center_col-n_head)] 
        
        hair = [(r,c) for r,c in head if (r,c) not in remove_hair]
        face = cg.hex_neighbours_n(max(r for r,c in head)-1,
                                       min(c for r,c in head if r ==max(r for r,c in head)-1)+1,
                                           n=n_head-1, keep_origin =True  , return_frontier=False )
        select_hair= cg.select_mask(hair,hex_rc_arr)
        select_face = cg.select_mask(face,hex_rc_arr)
        hex_colors[select_hair] = cgc.select_normal_color(select_hair, cgc.hex_to_rgb("#8f6e4c") , np.ones(3)*sigma_color) 
        hex_colors[select_face] = cgc.select_normal_color(select_face, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 
            

        select_shirt= cg.select_mask(shirt,hex_rc_arr)    
        select_Gakuran= cg.select_mask(body_final+pelvis_final+full_leg,hex_rc_arr)
        hex_colors[select_Gakuran] = cgc.select_normal_color(select_Gakuran, [0.1, 0.1, 0.1], np.ones(3)*sigma_color) 
        hex_colors[select_shirt] = cgc.select_normal_color(select_shirt, [1,1,1] , np.ones(3)*sigma_color) 
        
        hex_colors[select_skin] = cgc.select_normal_color(select_skin, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 
        
base_hex_colors = hex_colors.copy()

hex_colors = cgc.desaturate_matrix(hex_colors, amount=0.85)        

animation_base_hex_colors = hex_colors.copy()
current_hex_colors = animation_base_hex_colors.copy()
                                                            
def draw_char(jaw_center_r, jaw_center_c, n, view = 'front'): 
    """Character geometry (same as draw_char in test_pendulum.py), returning
    [(coords, rgb), ...] instead of painting into a scene-specific
    hex_colors -- so it can be rotated before painting."""
    head_center = (jaw_center_r - n, jaw_center_c)
    head, neck, body, pelvis, thigh, raw = draw_body(head_center[0], head_center[1], n=n)
    center_col, head_center_row, body_center_row, thigh_center_row = raw

    pelvis_final = modify_pelvis(pelvis)
    body_final = modify_body(body, body_center_row, view=view)
    thigh_final = modify_thigh(thigh, thigh_center_row, view=view)

    neck_left, neck_right = min(neck, key=lambda x: x[1]), max(neck, key=lambda x: x[1])
    body_top_r = min(r for r, c in body)
    body_bottom_r = max(r for r, c in body)
    left_shoulder  = (body_top_r, min(c for r, c in body if r == body_top_r) - 1)
    right_shoulder = (body_top_r, max(c for r, c in body if r == body_top_r) + 1)
    left_arm  = cgd.verticle_line(left_shoulder[0], left_shoulder[1], body_bottom_r, bend='left')
    right_arm = cgd.verticle_line(right_shoulder[0], right_shoulder[1], body_bottom_r, bend='right')
    left_elbow_start  = max(left_arm, key=lambda x: x[0])
    right_elbow_start = max(right_arm, key=lambda x: x[0])

    left_calf, right_calf, left_knee, right_knee, left_ankle, right_ankle = draw_calf(thigh, thigh_center_row, n_head, view=view)
    left_foot, right_foot, paw_row = draw_feet(left_ankle, right_ankle, n_head, view=view)

    left_forearm = cgd.draw_slope_0p5_diagonal(left_elbow_start[0], left_elbow_start[1],
                                                left_elbow_start[0] + n_head,
                                                left_down=True, right_down=False, left_up=False, right_up=False)
    right_forearm = cgd.draw_slope_0p5_diagonal(right_elbow_start[0], right_elbow_start[1],
                                                 right_elbow_start[0] + n_head,
                                                 left_down=False, right_down=True, left_up=False, right_up=False)

    sk_row = thigh_center_row - 1
    sk = [(r, c) for r, c in thigh_final if r <= sk_row]
    leg = [(r, c) for r, c in thigh_final if r > sk_row]
    calf_tot = left_calf + right_calf
    shoes_tot = left_foot + right_foot
    sock_row = left_knee[0]
    socks = [(r, c) for r, c in calf_tot if r > sock_row]
    shirt = neck + left_arm + right_arm
    tie = [(min(r for r, c in body), center_col)]

    hair = cgd.draw_trapezoid(head_center[0] - n_head,
                               min(c for r, c in head if r == head_center[0] - n_head),
                               max(c for r, c in head if r == head_center[0] - n_head),
                               head_center[0] + n_head, slope_left='0.5', slope_right='0.5', direction='lr')[-2]
    remove_hair = [(max(r for r, c in hair), min(c for r, c in hair if r == max(r for r, c in hair))),
                   (max(r for r, c in hair), max(c for r, c in hair if r == max(r for r, c in hair)))]
    valid_hair = [(r, c) for r, c in hair if (r, c) not in remove_hair]

    face = cg.hex_neighbours_n(max(r for r, c in head) - 1,
                                min(c for r, c in head if r == max(r for r, c in head) - 1) + 1,
                                n=n_head - 1, keep_origin=True, return_frontier=False)
    remove_face = [(r, c) for (r, c) in face if r == min(r for r, c in face)]
    face_final = [(r, c) for r, c in face if (r, c) not in remove_face]

    lasso = neck + [(neck_left[0], neck_left[1] - 1), (neck_right[0], neck_right[1] + 1)] + \
            [(max(r for r, c in valid_hair), max(c for r, c in valid_hair if r == max(r for r, c in valid_hair)))]

    return [
        (valid_hair,                          cgc.hex_to_rgb("#969539")),
        (face_final,                          cgc.hex_to_rgb("#fee9d2")),
        (sk,                                  cgc.hex_to_rgb("#2e27aa")),
        (body_final + pelvis_final,           [0.7, 0.7, 0.7]),
        (shirt,                               [1, 1, 1]),
        (tie,                                 [1, 0, 0]),
        (leg + left_forearm + right_forearm,  cgc.hex_to_rgb("#fee9d2")),
        (socks,                               [0.1, 0.1, 0.1]),
        (shoes_tot,                           [0, 0, 0]),
        (lasso,                               [0, 0, 0]),
    ]

# ── physics ──────────────────────────────────────────────────────────────
pend = Pendulum()
sol = pend.solve()
t, theta, omega, phi, phidot = sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

l = pend.length(t)
z_pivot = pend.z_pivot
r, far = pend.r, pend.l_cyl - pend.r

x_tie = l * np.sin(theta)
z_tie = z_pivot - l * np.cos(theta)
x_near = x_tie - r * np.sin(phi)
z_near = z_tie + r * np.cos(phi)
x_far  = x_tie + far * np.sin(phi)
z_far  = z_tie - far * np.cos(phi)

vx_far = pend.k * np.sin(theta) + l * omega * np.cos(theta) + far * phidot * np.cos(phi)
vz_far = -pend.k * np.cos(theta) + l * omega * np.sin(theta) + far * phidot * np.sin(phi)

drip = Drip()
t_end = t[-1]
spawn_idx = np.searchsorted(t, np.arange(0, t_end, drip.dt_spawn))
drips = []
for j in spawn_idx:
    if j >= len(t):
        continue
    drips.append(drip.solve(t[j], x_far[j], z_far[j], vx_far[j], vz_far[j], t_end))





# CANVAS_PHYSICAL_H is NOT a free choice: given fixed IMG_H/HEX_R and the
# character's known row-span (8*n_head+5), matching the character's
# rendered height to the stick's real length (pend.l_cyl) DETERMINES this
# value exactly. Since this makes the visible window small (~1.5m) and L
# grows to 5m, the character is naturally off-canvas well before the end.

_span_rows = 8 * n_head + 5
_hex_row_size_m = pend.l_cyl / _span_rows
_px_per_metre = _dy_hex_center / _hex_row_size_m
 
DOMAIN_H = IMG_H / _px_per_metre
DOMAIN_W = IMG_W / _px_per_metre     # same px_per_metre in both axes -> hex cells stay regular
CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H = DOMAIN_W, DOMAIN_H
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
 
# character built once at reference pivot (0,0); rotated per-frame about
# this same pivot by phi(t), then translated to the tie point's current
# hex cell
char_parts = draw_char(0, 0, n=n_head, view='front')
 
# phi[0] is a known constant (0, the initial condition) -- no rotation
# needed to find the character's bottom row at t=0, just read it directly
_all_coords = np.concatenate([np.asarray(coords).reshape(-1, 2)
                               for coords, color in char_parts if len(coords) > 0])
max_r_char = int(_all_coords[:, 0].max())
 
OFFSET_X = DOMAIN_W / 2   # pivot (x=0) sits at the middle column
 
# solve OFFSET_Y so the tie point's mapped hex row lands at -max_r_char,
# which after translation (+max_r_char) puts the character's actual
# bottom row exactly at canvas row 0
dy_hex_center = detail_info[4]
OFFSET_Y = CANVAS_PHYSICAL_H * (1 + max_r_char * dy_hex_center / IMG_H) - z_tie[0]
 

px_per_metre = IMG_W / CANVAS_PHYSICAL_W
hex_row_size_m = detail_info[4] / px_per_metre     # same factor used to size the character (dy_hex_center)
n_string_samples = max(2, round(3 * l.max() / hex_row_size_m))
 
drip_base_color = np.array(cgc.hex_to_rgb("#ff002e"))
# later drips get darker (same hue, scaled brightness) -- 1.0 for the first, down to 0.4 for the last
drip_darken = [1.0 - 1. * k / max(1, len(drips) - 1) for k in range(len(drips))]
drip_base_colors = [drip_base_color * d for d in drip_darken]

pc.set_facecolor(current_hex_colors)
 

Phase0_end = 30          # frames before the portal appears (doorway still just red)
n_snapshots = len(t) + Phase0_end
def update(snapshot):
    if snapshot < Phase0_end:
        #current_hex_colors = 
        pc.set_facecolor(base_hex_colors.copy())
    else:
        current_snapshot = snapshot - Phase0_end    

        t_current = t[current_snapshot]

        # drip trail: paint the hex cell at the drop's current position and
        # LEAVE IT painted (accumulate on current_hex_colors, never reset) so
        # the trail builds up on the wall like a real stain
        for k, drip_sol in enumerate(drips):
            if t_current < drip_sol.t[0] or t_current > drip_sol.t[-1]:
                continue
            ti = min(np.searchsorted(drip_sol.t, t_current), len(drip_sol.t) - 1)
            xk, zk = drip_sol.y[0][ti], drip_sol.y[2][ti]
            if zk <= 0:
                continue
            rc_list = cg.world_metres_to_hex_index(np.array([xk + OFFSET_X]), np.array([zk + OFFSET_Y]), detail_info,
                                                    canvas_physical_x_range=canvas_physical_x_range,
                                                    canvas_physical_y_range=canvas_physical_y_range)
            row, col = rc_list[0]

            # drip footprint: shrinks with time since release, radius -> pixels -> hex rings
            elapsed = t_current - drip_sol.t[0]
            px_radius = drip.radius_at(elapsed) / CANVAS_PHYSICAL_W * IMG_W
            n_rings = max(0, round(px_radius / (2 * HEX_R)))
            region = cg.hex_neighbours_n(row, col, n=n_rings)
            select_drip = cg.select_mask(region, hex_rc_arr)
            current_hex_colors[select_drip] = cgc.select_normal_color(select_drip, drip_base_colors[k] , np.ones(3) * sigma_color*5)

        # per-frame render buffer: stains (current_hex_colors) stay permanent,
        # string + stick are redrawn fresh each frame on top (moving solids,
        # not stains) -- sampling them onto hex cells is what makes them
        # distort as they swing, since they're now subject to the grid's
        # own resolution
        frame_colors = current_hex_colors.copy()

        frac = np.linspace(0.0, 1.0, n_string_samples)
        xs_string = 0.0 + frac * (x_tie[current_snapshot] - 0.0)
        zs_string = z_pivot + frac * (z_tie[current_snapshot] - z_pivot)
        rc_list_string = cg.world_metres_to_hex_index(xs_string + OFFSET_X, zs_string + OFFSET_Y, detail_info,
                                                canvas_physical_x_range=canvas_physical_x_range,
                                                canvas_physical_y_range=canvas_physical_y_range)
        select_string = cg.select_mask(rc_list_string, hex_rc_arr)
        frame_colors[select_string] = [0,0,0]

        # stick -> character: rotate the pre-built parts about the reference
        # pivot (0,0) by phi(t) (the stick's own tilt), then shift every
        # resulting hex cell to the tie point's actual hex cell this frame
        tie_rc = cg.world_metres_to_hex_index(np.array([x_tie[current_snapshot] + OFFSET_X]), 
                                              np.array([z_tie[current_snapshot] + OFFSET_Y]), detail_info,
                                               canvas_physical_x_range=canvas_physical_x_range,
                                               canvas_physical_y_range=canvas_physical_y_range)[0]
        d_row, d_col = tie_rc[0] - 0, tie_rc[1] - 0

        for coords, color in char_parts:
            rotated = cg.rotate_hex_shape(coords, (0, 0), phi[current_snapshot], detail_info)
            shifted = [(row + d_row, col + d_col) for row, col in rotated]
            select_char = cg.select_mask(shifted, hex_rc_arr)
            frame_colors[select_char] = color#cgc.select_normal_color(select_char, color, np.ones(3) * sigma_color)

        pc.set_facecolor(frame_colors)
    return pc, 
 
total_seconds = 18.0
# FPS [1/time] = total_frames / total_seconds
FPS = (n_snapshots / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=int(n_snapshots),
                               interval=time_gap, blit=False)

OUTPUT_FOLDER = 'RESULT'
FILE_NAME = 'another_animation.mp4'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, FILE_NAME)



print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved → {OUTPUT_FILE}")