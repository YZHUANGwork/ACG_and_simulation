import numpy as np
import matplotlib
import math
from scipy.interpolate import interp1d
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
import SIM_montecarlo as SIM
from scipy.stats import maxwell, norm
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info


sigma_color = 0.03

def draw_bkgd(center_r,center_c, n_shell):
    #odd_n_shells = np.arange(1,n_shell,2)
    #even_n_shells = np.arange(0,n_shell,2)
    shells = np.arange(n_shell)
    group = (shells - 1) // 2
    odd_n_shells = shells[group % 2 == 0]
    even_n_shells = shells[group % 2 == 1]

    shells_odd = []
    shells_even = []
    for odd_n_shell in odd_n_shells:
        shells_odd+=cg.hex_neighbours_n(center_r,center_c, n=odd_n_shell, keep_origin = False, return_frontier=True)[-1]
    for even_n_shell in even_n_shells:
        shells_even+=cg.hex_neighbours_n(center_r,center_c, n=even_n_shell, keep_origin = False, return_frontier=True)[-1]
    
    return {'shells_odd': [shells_odd, [[0,0,0]]],
            'shells_even': [shells_even, [cgc.hex_to_rgb("#470611") ]],##640012
         }
    

    
def get_function(name):
    if name == 'maxwell':
        rv = maxwell()    

        x = np.linspace(maxwell.ppf(0.01),
                    maxwell.ppf(0.99), 100)
        pdf =rv.pdf(x)
    elif name == 'exp':
        A = 17.106
        B = 1.8223
        C = 0.65911
        D = 18.292
        E = 20869
        F = -2.35
        x = np.linspace(2, 50, 100)
        pdf = (10 ** ( 2 - A * np.exp ( -B * x ** C) - D * np.exp( -E * x ** F ) ) /100)
    elif name == 'flat normal':
        x = np.linspace(-5,5, 200)
        x0, sigma, n = 0.0, 4, 2  # n=1 -> ordinary gaussian, n>1 -> flatter top
        pdf = np.exp(-np.abs((x - x0) / sigma) ** (2 * n))

    elif name == 'normal':
        x = np.linspace(norm.ppf(0.1), norm.ppf(0.9), 100)
        pdf = norm.pdf(x)
         
    elif name == 'uniform':
        x = np.linspace(-1,1, 100)
        pdf = np.ones(x.shape)

    return x, pdf
def ellipse_pdf(x, y, a = 1., b = 1.):
            return np.where((x / a) ** 2 + (y / b) ** 2 <= 1, 1.0, 0.0)

def draw_urahara(head_center_row, head_center_col, n_head):
    body_info_dict = dict()
    head, neck, body, pelvis, thigh, raw_body_detail= cgd.draw_body(head_center_row, head_center_col, n = n_head)
    raw_body = head+ neck+ body+ pelvis+ thigh#+arm
    center_col, head_center_row, body_center_row, thigh_center_row = raw_body_detail
        
    body_top_r = min(r for r, c in body)
    body_bot_r = max(r for r, c in body)
    
    chest =cgd.draw_trapezoid(body_top_r, 
                               min(c for r,c in body if r ==body_top_r), max(c for r,c in body if r ==body_top_r), 
                               body_center_row, slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                  bend_left = 'right', bend_right = 'left'
                                )[-2]
    
    
    body_info_dict['left clothing'] = (min(r for r,c in neck),  min(c for r,c in neck if r ==min(r for r,c in neck))-2)
    body_info_dict['right clothing'] = (min(r for r,c in neck),  max(c for r,c in neck if r ==min(r for r,c in neck))+2)
    
    
    face_raw =cgd.draw_trapezoid(head_center_row+n_head, 
                               min(c for r,c in head if r ==head_center_row+n_head)+1, 
                               max(c for r,c in head if r ==head_center_row+n_head)-1, 
                               head_center_row+1, slope_left = '0.5', slope_right = '0.5', direction = 'lr',
                  bend_left = 'right', bend_right = 'left'
                                )[-2]+cgd.horizontal_lines([(head_center_row, (center_col-n_head+2, center_col+n_head-2))])
    
    hair_raw = [(r,c) for r,c in head if (r,c) not in face_raw]
    hair_add = cgd.draw_slope_0p5_diagonal(head_center_row, center_col, head_center_row+1, 
                                           left_down=False, right_down=True, left_up=False, right_up=False
                                          )+cgd.draw_slope_0p5_diagonal(head_center_row+1, 
                                                                        min(c for r,c in head if r ==head_center_row+1), 
                                                                        head_center_row+n_head, 
                                           left_down=True, right_down=False, left_up=False, right_up=False
                                          )+cgd.draw_slope_0p5_diagonal(head_center_row+1, 
                                                                        max(c for r,c in head if r ==head_center_row+1),  
                                                                        head_center_row+n_head, 
                                           left_down=False, right_down=True, left_up=False, right_up=False
                                          )+cgd.draw_slope_0p5_diagonal(head_center_row+n_head, 
                                                                        min(c for r,c in head if r ==head_center_row+n_head), 
                                                                        head_center_row+n_head+1, 
                                           left_down=True, right_down=False, left_up=False, right_up=False
                                          )+cgd.draw_slope_0p5_diagonal(head_center_row+n_head, 
                                                                        max(c for r,c in head if r ==head_center_row+n_head),  
                                                                        head_center_row+n_head+1, 
                                           left_down=False, right_down=True, left_up=False, right_up=False
                                          )
    skin = face_raw+neck+chest
    hair = hair_raw+hair_add
    
    body_info_dict['head top']  = (head_center_row-n_head-1, center_col)
    return {
            'skin': [list(set(skin)), [[1,1,1]] ],#[cgc.hex_to_rgb("#84807b"),  cgc.hex_to_rgb("#fee9d2")]
            'hair': [list(set(hair)), [[0,0,0], cgc.hex_to_rgb("#faff00")]],
            
        }, body_info_dict


def draw_sym(x, pdf, loc,DOMAIN_W_SCALE ,DOMAIN_H_SCALE, detail_info, N = 500 ):
       
    val = np.array([x, pdf]).T
    sorted_val = np.array(sorted(val, key = lambda val: val[1], reverse = True ))
    sim = SIM.PDF_montecarlo((x, pdf), N=N, simulate_type='cdf', seed=1, spacing='linear')
    x_sim_pdf, y_sim_pdf = sim.pdf_sim()
    
    grid_hex_rc = cg.fn2grid(x_sim_pdf, y_sim_pdf,DOMAIN_W_SCALE ,DOMAIN_H_SCALE ,loc[0],  loc[1], sorted_val[0][0], sorted_val[0][1], detail_info)
    return grid_hex_rc, [x_sim_pdf, y_sim_pdf]

def draw_urahara_clothing(body_info_dict, N, detail_info, ver):
    grid_info_dict = dict()
    if ver in ['mode1', 'mode0']:
        x,pdf =get_function('maxwell')
        grid_hex_rc_left_black , _ = draw_sym(x, pdf[::-1], body_info_dict['left clothing'], 4,3, detail_info, N = N )
        grid_info_dict['left black'] = [grid_hex_rc_left_black, [[0,0,0]]]

        grid_hex_rc_right_black , _  = draw_sym(x, pdf, body_info_dict['right clothing'], 4,3, detail_info, N = N )
        grid_info_dict['right black'] = [grid_hex_rc_right_black, [[0,0,0]]]
        
        if ver == 'mode1':
            x,pdf =get_function('exp')
            grid_hex_rc_left_white , _  = draw_sym(x, pdf[::-1], 
                                                   (body_info_dict['left clothing'][0], body_info_dict['left clothing'][1]),
                                                   10, 3, detail_info, N = N )
            grid_info_dict['left white'] = [grid_hex_rc_left_white, [[0.95,0.95,0.95]]]

            grid_hex_rc_right_white , _  = draw_sym(x, pdf,
                                                    (body_info_dict['right clothing'][0],body_info_dict['right clothing'][1]),
                                                    10, 3, detail_info, N = N )
            grid_info_dict['right white'] = [grid_hex_rc_right_white, [[0.95,0.95,0.95]]]

    elif ver in ['mode2', 'mode3']:

        x,pdf =get_function('maxwell')
        grid_hex_rc_left_green , _  = draw_sym(x, pdf[::-1], body_info_dict['left clothing'], 3,3, detail_info, N = N )
        grid_info_dict['left green'] = [grid_hex_rc_left_green,[[0,0,0]]]#cgc.hex_to_rgb("#3b6632")

        grid_hex_rc_right_green , _  = draw_sym(x, pdf, body_info_dict['right clothing'], 3,3, detail_info, N = N )
        grid_info_dict['right green'] = [grid_hex_rc_right_green,[cgc.hex_to_rgb("#87ff00")]]
        
        if ver == 'mode2':

            x,pdf =get_function('flat normal')
            sim = SIM.PDF_montecarlo((x, pdf), N=N, simulate_type='pdf', seed=1, spacing='linear')
            x_sim_pdf, y_sim_pdf = sim.pdf_sim()
            grid_hex_rc_hat = cg.fn2grid(x_sim_pdf, y_sim_pdf,4 ,4,
                                               body_info_dict['head top'][0]-2,  body_info_dict['head top'][1],
                                  np.mean(x), interp1d(x,pdf)(np.mean(x)), detail_info)
            
            
            grid_info_dict['hat'] = [grid_hex_rc_hat, [cgc.hex_to_rgb("#3b6632")]]
            
    
    return grid_info_dict

def hair_region(r, c):
    if r < 2:
        return cg.hex_neighbours_n(r, c, n=3, keep_origin=True, return_frontier=False)
    elif r < 5:
        return cg.hex_neighbours_n(r, c, n=2, keep_origin=True, return_frontier=False)
    elif r < 15:
        return cg.hex_neighbours_n(r, c, n=1, keep_origin=True, return_frontier=False)
    else:
        return cg.hex_neighbours_n(r, c, n=0, keep_origin=True, return_frontier=False)

    
def brachium_region(r, c):
    return cg.hex_neighbours_n(r, c, n=1, keep_origin=True, return_frontier=False)

def draw_benihime(body_info_dict, N, detail_info, ver):
    grid_info_dict= dict()
    #face
    x,pdf =get_function('normal')
    sim = SIM.PDF_montecarlo((x, pdf), N=N, simulate_type='cdf', seed=1, spacing='linear')
    x_sim_pdf, y_sim_pdf = sim.pdf_sim()
    x_sim_cdf_left, cdf_draws_left, _ = sim.cdf_sim()  
    
    val = np.array([x, -pdf]).T
    sorted_val = np.array(sorted(val, key = lambda val: val[1]))
    
    select_face_left = (x_sim_pdf < 0.)
    
    grid_hex_rc_face = cg.fn2grid(x_sim_pdf, -y_sim_pdf,5 ,3 ,
                                  body_info_dict['head top'][0],body_info_dict['head top'][1], 
                                  sorted_val[0][0], sorted_val[0][1], detail_info)
    grid_hex_rc_face_left = [(r,c) for r,c in grid_hex_rc_face if c <body_info_dict['head top'][1]]
    grid_hex_rc_face_right = [(r,c) for r,c in grid_hex_rc_face if c >=body_info_dict['head top'][1]]
                                                                                
                                                                               
    grid_hex_rc_hair_left = cg.fn2grid(x_sim_cdf_left, cdf_draws_left ,4,0.8  ,0, 
                                       min(c for r,c in grid_hex_rc_face if r == max( min(r for r,c in grid_hex_rc_face), 0))+1, 
                                       max(x), 1, detail_info)
    
    x_sim_cdf_right, cdf_draws_right, _  = SIM.PDF_montecarlo((x[::-1], pdf), N=N, simulate_type='cdf', seed=1, spacing='linear').cdf_sim()  
    grid_hex_rc_hair_right = cg.fn2grid(x_sim_cdf_right, cdf_draws_right ,4,0.8  ,0, 
                                        max(c for r,c in grid_hex_rc_face if r == max( min(r for r,c in grid_hex_rc_face), 0))-1,
                                        min(x), 1, detail_info)

    valid_hair = []
       
    for (r,c) in list(set(grid_hex_rc_hair_left))+list(set(grid_hex_rc_hair_right)):
        valid_hair.extend(hair_region(r, c))
        '''
        if r <2:
            valid_hair.extend(cg.hex_neighbours_n(r, c, n=3, keep_origin = True, return_frontier=False))
        elif r <5:
            valid_hair.extend(cg.hex_neighbours_n(r, c, n=2, keep_origin = True, return_frontier=False))
        elif r <15:
            valid_hair.extend(cg.hex_neighbours_n(r, c, n=1, keep_origin = True, return_frontier=False))
        else:
            valid_hair.extend(cg.hex_neighbours_n(r, c, n=0, keep_origin = True, return_frontier=False))
        '''
    grid_hex_rc_hair_left_extra = cg.fn2grid(x_sim_cdf_left , cdf_draws_left,8,1 ,max( min(r for r,c in grid_hex_rc_face), 0), 
                                       max(c for r,c in grid_hex_rc_face if r ==max( min(r for r,c in grid_hex_rc_face), 0))+1,
                                        max(x), 1, detail_info
                          )+cg.fn2grid(x_sim_cdf_left , cdf_draws_left,8,1 ,max( min(r for r,c in grid_hex_rc_face), 0), 
                                       max(c for r,c in grid_hex_rc_face if r ==max( min(r for r,c in grid_hex_rc_face), 0))+2,
                                        max(x), 1, detail_info
                          )
    grid_hex_rc_hair_right_extra = cg.fn2grid(x_sim_cdf_right, cdf_draws_right  ,8,1, max( min(r for r,c in grid_hex_rc_face), 0), 
                                       min(c for r,c in grid_hex_rc_face if r == max( min(r for r,c in grid_hex_rc_face), 0))-1, 
                                       min(x), 1, detail_info
                          )+cg.fn2grid(x_sim_cdf_right, cdf_draws_right  ,8,1, max( min(r for r,c in grid_hex_rc_face), 0), 
                                       min(c for r,c in grid_hex_rc_face if r == max( min(r for r,c in grid_hex_rc_face), 0))-2, 
                                       min(x), 1, detail_info)
    valid_hair+=grid_hex_rc_hair_right_extra
    valid_hair+=grid_hex_rc_hair_left_extra
    
    
    x,pdf =get_function('maxwell')
    grid_hex_rc_red_left , _= draw_sym(x, pdf[::-1], 
                                        (max(r for r,c in grid_hex_rc_face), 
                                         min(c for r,c in grid_hex_rc_face if r == max(r for r,c in grid_hex_rc_face))-2
                                        ), 2,2, detail_info, N = N )
                                
    grid_hex_rc_red_right,_ = draw_sym(x, pdf, 
                                        (max(r for r,c in grid_hex_rc_face), 
                                         max(c for r,c in grid_hex_rc_face if r == max(r for r,c in grid_hex_rc_face) )+2
                                        ), 2,2, detail_info, N = N )
     
    
    left_shoulder_joint = (min(r for r,c in grid_hex_rc_red_left)+1,
                           min(c for r,c in grid_hex_rc_red_left if r == min(r for r,c in grid_hex_rc_red_left)+1)-2)
    right_shoulder_joint = (min(r for r,c in grid_hex_rc_red_right)+1,
                           max(c for r,c in grid_hex_rc_red_right if r == min(r for r,c in grid_hex_rc_red_right)+1)+2)
    print(left_shoulder_joint, right_shoulder_joint)#v(9, 7), (9,20)
    
    left_elbow_joint = (max(r for r,c in grid_hex_rc_red_left)-1,
                           min(c for r,c in grid_hex_rc_red_left if r == max(r for r,c in grid_hex_rc_red_left)-1))
    right_elbow_joint = (max(r for r,c in grid_hex_rc_red_right)-1,
                           max(c for r,c in grid_hex_rc_red_right if r == max(r for r,c in grid_hex_rc_red_right)-1))
    print(left_elbow_joint, right_elbow_joint)#(18,2), (18, 26)
    
    
    #right_joints =[]
    #left_joints = []
    sim_2d = SIM.PDF_montecarlo((np.array([0]), np.array([0])), N=N, seed=3)
    x_sim_2d, y_sim_2d = sim_2d.pdf_sim_2d(ellipse_pdf, bounds=(-1, 1, -1, 1))
    
    '''
    for loc in [left_shoulder_joint, left_elbow_joint]:
        grid_hex_rc = cg.fn2grid(x_sim_2d, y_sim_2d ,6,6,loc[0],  loc[1], 0,0, detail_info, square = True)
        left_joints+=grid_hex_rc
    for loc in [right_shoulder_joint, right_elbow_joint]:
        grid_hex_rc = cg.fn2grid(x_sim_2d, y_sim_2d ,6,6,loc[0],  loc[1], 0,0, detail_info, square = True)
        right_joints+=grid_hex_rc
    ''' 
    grid_hex_rc_left_shoulder_joint = cg.fn2grid(x_sim_2d, y_sim_2d ,6,6,
                                                 left_shoulder_joint[0],  left_shoulder_joint[1], 0,0, detail_info, square = True)
    grid_hex_rc_right_shoulder_joint = cg.fn2grid(x_sim_2d, y_sim_2d ,6,6,
                                                 right_shoulder_joint[0],  right_shoulder_joint[1], 0,0, detail_info, square = True)
    grid_hex_rc_left_elbow_joint = cg.fn2grid(x_sim_2d, y_sim_2d ,6,6,
                                                 left_elbow_joint[0],  left_elbow_joint[1], 0,0, detail_info, square = True)
    grid_hex_rc_right_elbow_joint = cg.fn2grid(x_sim_2d, y_sim_2d ,6,6,
                                                 right_elbow_joint[0],  right_elbow_joint[1], 0,0, detail_info, square = True)
    
        
    x,pdf =get_function('uniform')
    sim_left_brachium = SIM.PDF_montecarlo((x, pdf), N=N,  simulate_type='cdf', seed=1, spacing='linear')
    x_sim_cdf_left, cdf_draws_left, _ = sim_left_brachium.cdf_sim()  
    grid_hex_rc_left_brachium = cg.fn2grid(x_sim_cdf_left, cdf_draws_left ,6,3,
                             left_shoulder_joint[0],left_shoulder_joint[1], max(x ),1, detail_info)
    
    
    sim_right_brachium = SIM.PDF_montecarlo((x[::-1], pdf),N=N, simulate_type='cdf', seed=1, spacing='linear')
    x_sim_cdf_right, cdf_draws_right, _ = sim_right_brachium.cdf_sim()  
    grid_hex_rc_right_brachium = cg.fn2grid(x_sim_cdf_right, cdf_draws_right ,6,3,
                             right_shoulder_joint[0],right_shoulder_joint[1], min(x ),1, detail_info)
    
    
    
    grid_hex_rc_left_forearm_up = cg.fn2grid(x_sim_cdf_left, cdf_draws_left ,3,3,
                                      left_elbow_joint[0],left_elbow_joint[1], min(x ),0, detail_info)
    grid_hex_rc_right_forearm_up = cg.fn2grid(x_sim_cdf_right, cdf_draws_right ,3,3,
                                       right_elbow_joint[0],right_elbow_joint[1], max(x ),0, detail_info)
    
    
    grid_hex_rc_left_forearm_down = cg.fn2grid(x_sim_cdf_left, cdf_draws_left,10,4,
                                       left_elbow_joint[0],left_elbow_joint[1], max(x ),1, detail_info)
    grid_hex_rc_right_forearm_down = cg.fn2grid(x_sim_cdf_right, cdf_draws_right ,10,4,
                                           right_elbow_joint[0],right_elbow_joint[1], min(x ),1, detail_info)
    forearms = []
    if 'up' in ver: 
        grid_hex_rc_left_forearm = grid_hex_rc_left_forearm_up
        grid_hex_rc_right_forearm = grid_hex_rc_right_forearm_up
    else:
        grid_hex_rc_left_forearm = grid_hex_rc_left_forearm_down
        grid_hex_rc_right_forearm = grid_hex_rc_right_forearm_down
    for (r,c) in list(set(grid_hex_rc_left_forearm+grid_hex_rc_right_forearm)):
         forearms.extend(cg.hex_neighbours_n(r, c, n=0, keep_origin = True, return_frontier=False))

        
    brachiums = []
    for (r,c) in list(set(grid_hex_rc_left_brachium+grid_hex_rc_right_brachium)):
         brachiums.extend(brachium_region(r, c))# cg.hex_neighbours_n(r, c, n=1, keep_origin = True, return_frontier=False)
    
    
    if ver == 'raw':
        grid_info_dict['left hair'] = [sorted(grid_hex_rc_hair_left, key=lambda rc: rc[0]), [[0,0,0]]]
        grid_info_dict['right hair'] = [sorted(grid_hex_rc_hair_right, key=lambda rc: rc[0]), [[0,0,0]]]
        grid_info_dict['left hair extra'] = [sorted(grid_hex_rc_hair_left_extra, key=lambda rc: rc[0]), [[0,0,0]]]
        grid_info_dict['right hair extra'] = [sorted(grid_hex_rc_hair_right_extra, key=lambda rc: rc[0]), [[0,0,0]]]
        
        grid_info_dict['face'] = [grid_hex_rc_face, [[0.95, 0.95,0.95]]]
        grid_info_dict['left red'] = [grid_hex_rc_red_left,[cgc.hex_to_rgb("#4b000e")]]#v
        grid_info_dict['right red'] = [grid_hex_rc_red_right,[[1,0,0]]]##ff002e 
        
        grid_info_dict['left brachium'] = [sorted(grid_hex_rc_left_brachium, key=lambda rc: rc[0]), [[0.95, 0.95,0.95]]]
        grid_info_dict['right brachium'] = [sorted(grid_hex_rc_right_brachium, key=lambda rc: rc[0]), [[0.95, 0.95,0.95]]]
        
    
    elif 'up' in ver: 
        grid_info_dict['left forearm'] = [sorted(grid_hex_rc_left_forearm, key=lambda rc: rc[0], reverse = True), 
                                          [[0.95, 0.95,0.95]]]
        grid_info_dict['right forearm'] = [sorted(grid_hex_rc_right_forearm, key=lambda rc: rc[0], reverse = True), 
                                           [[0.95, 0.95,0.95]]]
    
    else: 
        grid_info_dict['left face'] = [grid_hex_rc_face_left, [[0.95, 0.95,0.95]]]
        grid_info_dict['right face'] = [grid_hex_rc_face_right, [[0.95, 0.95,0.95]]]


        grid_info_dict['left red'] = [grid_hex_rc_red_left,[cgc.hex_to_rgb("#d60600")]]#vcgc.hex_to_rgb("#4b000e")
        grid_info_dict['right red'] = [grid_hex_rc_red_right,[[1,0,0]]]##ff002e 
        
        grid_info_dict['brachiums'] = [brachiums, [[0.95, 0.95,0.95]]]

        grid_info_dict['forearms'] = [forearms, [[0.95, 0.95,0.95]]]
        grid_info_dict['left shoulder joint'] = [grid_hex_rc_left_shoulder_joint, [[0.95, 0.95,0.95]]]
        grid_info_dict['right shoulder joint'] = [grid_hex_rc_right_shoulder_joint, [[0.95, 0.95,0.95]]]
        grid_info_dict['left elbow joint'] = [grid_hex_rc_left_elbow_joint, [[0.95, 0.95,0.95]]]
        grid_info_dict['right elbow joint'] = [grid_hex_rc_right_elbow_joint, [[0.95, 0.95,0.95]]]
        
        
        grid_info_dict['hair'] = [valid_hair, [[0,0,0]]]
    
    return grid_info_dict


            
    
head_center_row = 12
head_center_col = 14
n_head = 3
N = 500
urahara_dict, urahara_info_dict = draw_urahara(head_center_row, head_center_col, n_head)
char_dict_ = {'bkgd': draw_bkgd(head_center_row,head_center_col, 30),
                'benihime': draw_benihime(urahara_info_dict, N, detail_info, 'mode1'),
                'urahara': urahara_dict,
             'clothing': draw_urahara_clothing(urahara_info_dict, N, detail_info, 'mode3'),
              
}
for dict_ in char_dict_.values(): 

    for key in dict_.keys():
        part  = list(set(dict_.get(key)[0]))
        colors = dict_.get(key)[1] 
        if key in ['brachiums', 'left face', 'right face', 'forearms', 'left joints', 'right joints', 'skin' ]:
            sigma_color = 0.
        if len(colors) == 1:
            
            select_part= cg.select_mask(part,hex_rc_arr)
            hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color*3) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01, mode = 'linear') 



pc.set_facecolor(hex_colors)
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'bleach_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')
