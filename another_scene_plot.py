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

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,_ = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)



sigma_color = 0.02


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

def draw_char(jaw_center_r, jaw_center_c, n_head, hex_colors, view = 'front'):
    head_center = (jaw_center_r-n_head, jaw_center_c)
    head, neck, body, pelvis, thigh, raw_body_detail= draw_body(head_center[0], head_center[1], n = n_head)
    select_head= cg.select_mask(head,hex_rc_arr)
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail
    
    pelvis_final = modify_pelvis(pelvis)
    body_final = modify_body(body, body_center_row, view =view)
    thigh_final = modify_thigh(thigh, thigh_center_row, view =view)
    
    neck_left, neck_right = min(neck, key=lambda x: x[1]), max(neck, key=lambda x: x[1])
    
    #left_arm, right_arm, left_albow_start, right_albow_start = draw_shoulder(body, n_head, view = view)
    body_top_r = min(r for r, c in body)
    body_bottom_r = max(r for r, c in body)
    left_shoulder  = (body_top_r, min(c for r, c in body if r ==body_top_r)-1)                 
    right_shoulder = (body_top_r, max(c for r, c in body if r ==body_top_r)+1)
    left_arm = cgd.verticle_line(left_shoulder[0], left_shoulder[1], body_bottom_r, bend = 'left')
    right_arm = cgd.verticle_line(right_shoulder[0], right_shoulder[1], body_bottom_r, bend = 'right')
    left_elbow_start, right_elbow_start = max(left_arm, key=lambda x: x[0]), max(right_arm, key=lambda x: x[0])
    
    left_calf, right_calf, left_knee, right_knee, left_ankle, right_ankle = draw_calf(thigh, thigh_center_row,n_head, 
                                                                                      view =view)
    left_foot, right_foot, paw_row = draw_feet(left_ankle, right_ankle, n_head, view =view)
    left_forearm = cgd.draw_slope_0p5_diagonal(left_elbow_start[0], left_elbow_start[1], left_elbow_start[0]+n_head, left_down=True , 
                                         right_down= False, left_up=False, right_up=False)
    
    #cgd.verticle_line(left_elbow_start[0], left_elbow_start[1], left_elbow_start[0]+n_head, bend = 'left')
    
    
    right_forearm = cgd.draw_slope_0p5_diagonal(right_elbow_start[0], right_elbow_start[1], right_elbow_start[0]+n_head, left_down=False, 
                                         right_down= True, left_up=False, right_up=False)
    
    #cgd.verticle_line(right_elbow_start[0], right_elbow_start[1], right_elbow_start[0]+n_head, bend = 'right')
    
    
    left_hand = [(max(r for r,c in left_forearm), min(c for r,c in left_forearm if r== max(r for r,c in left_forearm)) )]
    right_hand = [(max(r for r,c in right_forearm), min(c for r,c in right_forearm if r== max(r for r,c in right_forearm)) )]

    knee_r = left_knee[0]
    calf_tot = left_calf+right_calf
    shoes_tot = left_foot+right_foot
    
    sk_row = thigh_center_row-1
    sk = [(r,c)  for r,c in thigh_final if r <=sk_row]
    leg =[(r,c)  for r,c in thigh_final if r >sk_row]

    full_leg = leg+calf_tot+shoes_tot
    sock_row = knee_r
    socks = [(r,c)  for r,c in calf_tot if r >sock_row]
    shirt = neck+left_arm+right_arm
    tie = [(min(r for r,c in body), center_col)]
    hair = cgd.draw_trapezoid(head_center[0]-n_head, 
                              min(c for r,c in head if r == head_center[0]-n_head),
                              max(c for r,c in head if r == head_center[0]-n_head), head_center[0]+n_head, 
                              slope_left = '0.5', slope_right = '0.5', direction = 'lr')[-2]
    remove_hair = [(max(r for r,c in hair), min(c for r, c in hair if r ==max(r for r,c in hair))),
                   (max(r for r,c in hair), max(c for r, c in hair if r ==max(r for r,c in hair)))]
    valid_hair = [(r,c) for r,c in hair if (r,c) not in remove_hair]
    select_hair= cg.select_mask(valid_hair,hex_rc_arr)
    
    face = cg.hex_neighbours_n(max(r for r,c in head)-1,
                                       min(c for r,c in head if r ==max(r for r,c in head)-1)+1,
                                           n=n_head-1, keep_origin =True  , return_frontier=False )
    remove_face = [(r,c) for (r,c) in face  if r ==min(r for r,c in face)]
    
    select_face = cg.select_mask([(r,c) for r,c in face if (r,c) not in remove_face],hex_rc_arr)
    hex_colors[select_hair] = cgc.select_normal_color(select_hair, cgc.hex_to_rgb("#969539") , np.ones(3)*sigma_color) 
    hex_colors[select_face] = cgc.select_normal_color(select_face, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 


    lasso = neck+[(neck_left[0], neck_left[1] - 1), (neck_right[0], neck_right[1] + 1)
                 ]+[(max(r for r,c in valid_hair), max(c for r,c in valid_hair if r==max(r for r,c in valid_hair)))]
    
    select_vest= cg.select_mask(body_final+pelvis_final,hex_rc_arr)
    select_skin= cg.select_mask(leg+left_forearm+right_forearm,hex_rc_arr)
    

    select_lasso= cg.select_mask(lasso,hex_rc_arr)
    select_shirt= cg.select_mask(shirt,hex_rc_arr)
    select_tie = cg.select_mask(tie,hex_rc_arr)
    select_sk= cg.select_mask(sk,hex_rc_arr)

    socks = [(r,c)  for r,c in full_leg if r >sock_row]
    select_socks= cg.select_mask(socks,hex_rc_arr)
    select_shoes= cg.select_mask(shoes_tot,hex_rc_arr)

    hex_colors[select_sk] = cgc.select_normal_color(select_sk, cgc.hex_to_rgb("#2e27aa") , np.ones(3)*sigma_color) 
    hex_colors[select_vest] = cgc.select_normal_color(select_vest, [0.7, 0.7, 0.7] , np.ones(3)*sigma_color) 
    hex_colors[select_shirt] = cgc.select_normal_color(select_shirt, [1,1,1] , np.ones(3)*sigma_color) 
    hex_colors[select_tie] = cgc.select_normal_color(select_tie, [1,0,0] , np.ones(3)*sigma_color) 
    hex_colors[select_skin] = cgc.select_normal_color(select_skin, cgc.hex_to_rgb("#fee9d2") , np.ones(3)*sigma_color) 
    hex_colors[select_socks] = cgc.select_normal_color(select_socks, [0.1, 0.1, 0.1] , np.ones(3)*sigma_color) 
    hex_colors[select_shoes] = cgc.select_normal_color(select_shoes, [0,0,0] , np.ones(3)*sigma_color) 
    
    hex_colors[select_lasso] = cgc.select_normal_color(select_lasso, [0,0,0] , np.ones(3)*sigma_color) 
    return hex_colors
n_head = 2


center_row = 12
center_col = 17

hex_colors = cgc.color_row_gradient([(r,c) for r,c in hex_rc_arr if r<=center_row], 
                                    cgc.hex_to_rgb("#fffc00"), cgc.hex_to_rgb("#ff0000"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.1)    
hex_colors = cgc.color_row_gradient([(r,c) for r,c in hex_rc_arr if r>center_row], 
                                    cgc.hex_to_rgb("#ff0000"), cgc.hex_to_rgb("#543300"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.1)    


n_eyeball = 9
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

'''
boundary_left1 = cgd.verticle_line(
    min(r for r,c in right_eyesocket_full if c ==min(c for r,c in right_eyesocket_full if r <center_row and r >0) and r <center_row ), 
    min(c for r,c in right_eyesocket_full if r <center_row and r >0), 22, bend = 'right')
boundary_left2 = cgd.draw_slope_0p5_diagonal(
    min(r for r,c in right_eyesocket_full if  r >center_row  and c == min(c for r,c in right_eyesocket_full if r >center_row )), 
    min(c for r,c in right_eyesocket_full if r >center_row and r <=22), 0, left_down=False, right_down=False, left_up=True , right_up=False)+cgd.draw_slope_0p5_diagonal(
    min(r for r,c in right_eyesocket_full if  r >center_row  and c == min(c for r,c in right_eyesocket_full if r >center_row )), 
    min(c for r,c in right_eyesocket_full if r >center_row and r <=22), 22, left_down=False, right_down=True,left_up=False, right_up=False)
boundary_left = boundary_left1+boundary_left2
boundary_right = cg.hex_neighbours_n(right_eyeball_r, right_eyeball_c, n=n_eyeball+1, keep_origin = False , return_frontier=True
                                         )[-1]
sclera_boundary_left = {r: max(c for r2, c in boundary_left if r2 == r) for r, _ in boundary_left}
sclera_boundary_right = {r: max(c for r2, c in boundary_right if r2 == r) for r, _ in boundary_right}
print(sclera_boundary_left, sclera_boundary_right)
sclera = cgd.horizontal_lines([(r, (sclera_boundary_left[r], sclera_boundary_right[r])) 
                               for r in sclera_boundary_left if r in sclera_boundary_right])
select_sclera = cg.select_mask(sclera,hex_rc_arr)
hex_colors[select_sclera] = cgc.select_normal_color(select_sclera, 
                                                               [1,1,1], np.ones(3)*sigma_color) 
'''

#eyeball color color_hex_gradient(arr, color_i, color_f, hex_rc_arr, hex_colors, center, n_layers, sigma_color=0.03, end_weight=0.2, period=None)
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
for key in char_dict.keys():
    head_center = char_dict.get(key)[0]
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

hanging_center = (7, 26)
string_ = cgd.verticle_line(hanging_center[0], hanging_center[1], 0, bend='left')
select_string_= cg.select_mask(string_,hex_rc_arr)    
hex_colors[select_string_] = [0,0,0]
hex_colors = draw_char(hanging_center[0], hanging_center[1], n_head, hex_colors)  
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'another_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')