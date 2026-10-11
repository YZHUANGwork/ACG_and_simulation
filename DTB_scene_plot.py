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
import SOLN_waveinterference as SOLN

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
n_drops = 10

xs = rng.uniform(-5, 5, n_drops)
ys = rng.uniform(-5, 5, n_drops)
sources = tuple(np.array([x, y]) * u.m for x, y in zip(xs, ys))

wavelength =(0.8,) * n_drops * u.m 
amplitude = tuple(rng.uniform(2., 3., n_drops) *u.m) # (2.5,) * n_drops * u.m
freq = tuple(rng.uniform(1., 2., n_drops) *u.Hz) 
kick_time =(0 * u.s,)+ tuple(np.sort(rng.uniform(0, 13, n_drops-1)) * u.s)#np.arange(1, 18, 1.2)[:n_drops]* u.s

#
decay_tau = tuple(rng.uniform(0.2, 1, n_drops) * u.s)

is_ripple = (True,) * n_drops

kick_time_source = 13. * u.s
sources_constant = ([-3,-5] * u.m,)
wavelength_constant =(3.0 * u.m,)
amplitude_constant = (2.0 * u.m,)
freq_constant = (3.0 * u.Hz,)
kick_time_constant = (kick_time_source,)
is_ripple_constant =(False,)
decay_tau_constant =(None,)

WAVE = SOLN.WaveInterference2D(
    grid_size=200,
    x_range=[-8, 8] * u.m, y_range=[-8, 8] * u.m,
    sources= tuple(sources)+tuple(sources_constant),
    wavelength=tuple(wavelength)+tuple(wavelength_constant),
    amplitude=tuple(amplitude)+tuple(amplitude_constant),
    freq=tuple(freq)+tuple(freq_constant),
    kick_time=tuple(kick_time)+tuple(kick_time_constant),
    is_ripple=tuple(is_ripple)+tuple(is_ripple_constant),
    decay_tau = tuple(decay_tau)+tuple(decay_tau_constant)
)
    

COLORMAP = [
    (0.0,  '#000309'),   # dark pine forest, base
    (0.1, '#00246c'),   # treeline
    (0.3, '#0048d5'),
    (0.5, '#003295'),# bare grey rock
    (0.7, '#0036a0'), 
    (0.8, '#cea0c4'),#  rock dusted with snow
    (0.9, '#0048d5'),   # snowfield
    (0.95, '#8fb8ff'),   # snowfield
    (1.0,  '#ffffff'),   # summit snow
]


X = WAVE.X.ravel()
Y = WAVE.Y.ravel()

grid_hex_rc = cg.fn2grid(
    X, Y,
    DOMAIN_W_SCALE=0.95, DOMAIN_H_SCALE=0.95,
    PIVOT_ROW=min(r for r,c in hex_rc_arr), PIVOT_COL=(min(c for r,c in hex_rc_arr) + max(c for r,c in hex_rc_arr)) // 2,
    PIVOT_ROW_X=0, PIVOT_COL_Y=Y.max(),
    detail_info=detail_info, square=False,
)


              
current_time=6
VALUE = WAVE.field_at(current_time).ravel()
hex_positions, blended = cg.average_grid_to_hex_scene(
        grid_hex_rc, hex_rc_arr,VALUE, hex_colors, COLORMAP, alpha=1.0, weight_by_value=False
                                                        )
hex_colors[hex_positions] = blended

def draw_char(pivot, name , size):
    if name == '1':
        raw_body = cgd.draw_block((pivot[0], (pivot[1], pivot[1])), pivot[0]+size)
        core_raw = cg.hex_neighbours_n(pivot[0], pivot[1], n=1, keep_origin = True, return_frontier=False)
        core = [(min(r for r,c in core_raw), min(c for r,c in core_raw if r==min(r for r,c in core_raw))),
                (max(r for r,c in core_raw), min(c for r,c in core_raw if r==max(r for r,c in core_raw))),
                (pivot[0], max(c for r,c in core_raw if r==pivot[0]))
               ]

        main_body_part1 =  []
        for loc in raw_body:
            if loc[0]<pivot[0]+size//2:
                n=2
            else:
                n=1
            main_body_part1+=cg.hex_neighbours_n(loc[0], loc[1], n=n, keep_origin = True, return_frontier=False)
            
        main_body_part2 = cgd.draw_trapezoid(max(r for r,c in main_body_part1), 
                                             min(c for r,c in main_body_part1 if r==max(r for r,c in main_body_part1)), 
                                             max(c for r,c in main_body_part1 if r==max(r for r,c in main_body_part1)), 
                                             max(r for r,c in main_body_part1)+size//2, slope_left = '1.5', slope_right = '1.5',
                                             direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
        main_body_part3 = cgd.draw_trapezoid(max(r for r,c in main_body_part2), 
                                             min(c for r,c in main_body_part2 if r==max(r for r,c in main_body_part2)), 
                                             max(c for r,c in main_body_part2 if r==max(r for r,c in main_body_part2)), 
                                             max(r for r,c in main_body_part2)+1, slope_left = '8/3', slope_right = '8/3',
                                             direction = 'rl', bend_left = 'left', bend_right = 'right')[-2]
        main_body = main_body_part1+main_body_part2+main_body_part3
        keys = [ 'main_body', 'core_raw', 'core']#
        colors = [[cgc.hex_to_rgb("#a8fbfb"),cgc.hex_to_rgb("#ddf7ff") ],[cgc.hex_to_rgb("#e2f7fd")],[cgc.hex_to_rgb("#506267")],
         
          ]
    elif name == '2':
        head_center = pivot
        core_raw = cg.hex_neighbours_n(head_center[0], head_center[1], n=1, keep_origin = True, return_frontier=False)
        core = [(min(r for r,c in core_raw), min(c for r,c in core_raw if r==min(r for r,c in core_raw))),
                (max(r for r,c in core_raw), min(c for r,c in core_raw if r==max(r for r,c in core_raw))),
                (head_center[0], max(c for r,c in core_raw if r==head_center[0]))
               ]
        head_raw, raw_body, line_dict, head_dict, neck, torso_dict, upperlimb_joint_dict, botlimb_joint_dict, thigh_dict, calf_dict, feet_dict, upperarm_dict, forearm_dict, hand_dict, plot_temp = cgf.draw_human_body(head_center[0], head_center[1], version="5")
        
        sk_ =cgd.draw_trapezoid(line_dict['femoral_head_r'], 
                                min(c for r,c in thigh_dict['thigh'] if r==line_dict['femoral_head_r']), 
                                max(c for r,c in thigh_dict['thigh'] if r==line_dict['femoral_head_r']), 
                                line_dict['knee_r'], slope_left = '0.5', slope_right = '0.5',
                                direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
        body = raw_body+sk_
        
        hair_raw_part1 = cgd.draw_slope_0p5_diagonal(min(r for r,c in head_raw), 
                                               min(c for r,c in head_raw if r==min(r for r,c in head_raw)), min(r for r,c in head_raw)-1, 
                                               left_down=False, right_down=False, left_up=True, right_up=False
                   )+cgd.draw_slope_0p5_diagonal(min(r for r,c in head_raw), 
                                               max(c for r,c in head_raw if r==min(r for r,c in head_raw)), min(r for r,c in head_raw)-1, 
                                               left_down=False, right_down=False, left_up=False, right_up=True
                   )
        pony_tail_raw = cgd.draw_trapezoid(min(r for r,c in hair_raw_part1), 
                                min(c for r,c in hair_raw_part1 if r==min(r for r,c in hair_raw_part1)), 
                                max(c for r,c in hair_raw_part1 if r==min(r for r,c in hair_raw_part1)), 
                                head_center[0]-1, slope_left = '0.5', slope_right = '0.5',
                                direction = 'lr', bend_left = 'left', bend_right = 'right')[-2]
       
        pony_tail =  [(r,c) for r,c in pony_tail_raw if (r,c) not in body
                     ]+cg.hex_neighbours_n(head_center[0]-2-2, head_center[1], n=1, keep_origin = True, return_frontier=False
                     )+cgd.verticle_line(head_center[0], min(c for r,c in head_raw if r== head_center[0]), 
                                         head_center[0]+2, bend = 'right'
                     )+cgd.verticle_line(head_center[0], max(c for r,c in head_raw if r== head_center[0]), 
                                         head_center[0]+2, bend = 'left')
        
        keys = [ 'body', 'core_raw', 'core', "pony_tail",]#
        colors = [[cgc.hex_to_rgb("#a8fbfb"),cgc.hex_to_rgb("#ddf7ff") ],[cgc.hex_to_rgb("#e2f7fd")], [cgc.hex_to_rgb("#506267")],
                  [cgc.hex_to_rgb("#c3e9f4")], [[0,0,0]]
         
          ]
        
    elif name == 'Kirsi':
        #head_center = pivot
        shoulder_line = cgd.draw_slope_1p5_diagonal(pivot[0], pivot[1], pivot[0]-2, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                                                   )+cgd.draw_slope_1p5_diagonal(pivot[0], pivot[1], pivot[0]+2, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                                                   )

        left_arm_row = min(r for r,c in shoulder_line)
        right_arm_row =  max(r for r,c in shoulder_line)
        left_arm = cgd.draw_slope_0p5_diagonal(left_arm_row, min(c for r,c in shoulder_line if r==left_arm_row), 
                                               left_arm_row+size*2+1, left_down=True, right_down=False, left_up=False, right_up=False)
        right_arm = cgd.draw_slope_0p5_diagonal(right_arm_row, min(c for r,c in shoulder_line if r==right_arm_row), 
                                               right_arm_row+size*2+1, left_down=True, right_down=False, left_up=False, right_up=False)

        left_hand = cgd.draw_slope_0p5_diagonal(max(r for r,c in left_arm), 
                                                 min(c for r,c in left_arm if r==max(r for r,c in left_arm)), 
                                               max(r for r,c in left_arm)-1, 
                                                 left_down=False, right_down=False, left_up=True, right_up=False)



        right_hand = cgd.horizontal_lines([(max(r for r,c in right_arm), 
                                           (max(c for r,c in right_arm if r==max(r for r,c in right_arm)),
                                            max(c for r,c in right_arm if r==max(r for r,c in right_arm))+1))])

        head_line = cgd.draw_slope_0p5_diagonal(pivot[0], pivot[1], pivot[0]-size, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True)


        body_line = cgd.draw_slope_0p5_diagonal(pivot[0], pivot[1], pivot[0]+size*3, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                                                   )


        head_center = (pivot[0]-1, min(c for r,c in head_line if r==pivot[0]-1))
        head_raw = cg.hex_neighbours_n(head_center[0], head_center[1], n=size, keep_origin = True, return_frontier=False)

        face_line =cgd.draw_slope_1p5_diagonal(head_center[0], head_center[1], head_center[0]-2, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                                                   )+cgd.draw_slope_1p5_diagonal(head_center[0], head_center[1], head_center[0]+2, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                                                   )
        face_line_row_to_col = {r: c for r, c in face_line}

        sk_center = (max(r for r,c in body_line), min(c for r,c in body_line if r==max(r for r,c in body_line)))
        hand_line =  cgd.draw_slope_1p5_diagonal(sk_center[0], sk_center[1], sk_center[0]-3, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                                                   )+cgd.draw_slope_1p5_diagonal(sk_center[0], sk_center[1], sk_center[0]+3, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                                                   )
        chest_center = (max(r for r,c in head_raw)+1, min(c for r,c in body_line if r==max(r for r,c in head_raw)+1))
        chest_line = cgd.draw_slope_1p5_diagonal(chest_center[0], chest_center[1], chest_center[0]-1, 
                                                 left_down=False, right_down=False, left_up=True, right_up=False
                                                )+cgd.draw_slope_1p5_diagonal(chest_center[0], chest_center[1], chest_center[0]+1, 
                                                 left_down=False, right_down=True, left_up=False, right_up=False
                                                )+cgd.draw_slope_1p5_diagonal(chest_center[0], chest_center[1]+1, chest_center[0]-2, 
                                                 left_down=False, right_down=False, left_up=True, right_up=False
                                                )+cgd.draw_slope_1p5_diagonal(chest_center[0], chest_center[1]+1, chest_center[0]+1, 
                                                 left_down=False, right_down=True, left_up=False, right_up=False
                                                )
        top_raw =cg.hex_neighbours_n(chest_center[0], chest_center[1], n=2, keep_origin = False, return_frontier=True )[-1]
        bolero_raw = []
        for loc in chest_line:
            if loc not in body_line:
                bolero_raw += cg.hex_neighbours_n(loc[0], loc[1], n=1, keep_origin = True, return_frontier=False)

        left_leg_row = sk_center[0]-1
        right_leg_row = sk_center[0]+1

        left_leg_raw = cgd.draw_slope_0p5_diagonal(left_leg_row, min(c for r,c in hand_line if r==left_leg_row), 
                                               left_leg_row+size*5, left_down=True, right_down=False, left_up=False, right_up=False)
        right_leg_raw = cgd.draw_slope_0p5_diagonal(right_leg_row, min(c for r,c in hand_line if r==right_leg_row), 
                                               left_leg_row+size*5, left_down=True, right_down=False, left_up=False, right_up=False
                                                   )+cgd.draw_slope_0p5_diagonal(right_leg_row, min(c for r,c in hand_line if r==right_leg_row), 
                                               left_leg_row-size*5, left_down=False, right_down=False, left_up=False, right_up=True
                                                   )

        right_leg_raw_row_to_col = {r: c for r, c in right_leg_raw}

        sk_raw  = cg.hex_neighbours_n(sk_center[0], sk_center[1], 
                                      n=size*3, keep_origin = True, return_frontier=False)
        sk_dark_part1 = [(r, c) for r, c in sk_raw if r in right_leg_raw_row_to_col and 
                   c > right_leg_raw_row_to_col[r] and r >min(r for r,c in sk_raw)+size+size+1]

        sk_dark_part2 = cgd.draw_slope_0p5_diagonal(max(r for r,c in body_line), min(c for r,c in body_line if r==max(r for r,c in body_line)), 
                                     max(r for r,c in body_line)-1, left_down=False, right_down=False, left_up=True, right_up=False
                                    )+cgd.horizontal_lines([(max(r for r,c in body_line), 
                                           (min(c for r,c in body_line if r==max(r for r,c in body_line)),
                                            min(c for r,c in body_line if r==max(r for r,c in body_line))+1))]
                                    )+cgd.draw_slope_0p5_diagonal(max(r for r,c in body_line), 
                                     min(c for r,c in body_line if r==max(r for r,c in body_line)), 
                                     max(r for r,c in sk_raw), left_down=True, right_down=False, left_up=False, right_up=False
                                    )
        sk_dark_part3_raw = []
        for loc in sk_dark_part2:
            if loc[0]>=max(r for r,c in sk_raw)-size:
                sk_dark_part3_raw+=cg.hex_neighbours_n(loc[0], loc[1], n=1, keep_origin = True, return_frontier=False)
        sk_dark_part3 = [(r, c) for r, c in sk_dark_part3_raw  if (r, c) in sk_raw]
        sk_dark_part4 = [(r, c) for r, c in 
                         cgd.draw_slope_1p5_diagonal(sk_center[0], min(c for r,c in sk_raw if r==sk_center[0]), sk_center[0]-2, 
                                      left_down=False, right_down=False, left_up=False, right_up=True
                        )+cgd.draw_slope_1p5_diagonal(sk_center[0], min(c for r,c in sk_raw if r==sk_center[0])+1, sk_center[0]-3, 
                                      left_down=False, right_down=False, left_up=False, right_up=True
                        ) if (r, c) in sk_raw]
        sk_dark  = sk_dark_part1+sk_dark_part2+sk_dark_part3+sk_dark_part4
        raw_body = head_raw+shoulder_line+body_line+hand_line+chest_line+left_arm+right_arm

        sk_light_part1 = [(r, c) for r, c in cgd.draw_slope_8p3_diagonal(chest_center[0], chest_center[1], max(r for r,c in sk_raw), 
                                      left_down=True, right_down=False, left_up=False, right_up=False)+
                                      cgd.draw_slope_8p3_diagonal(chest_center[0], chest_center[1]-1, max(r for r,c in sk_raw), 
                                      left_down=True, right_down=False, left_up=False, right_up=False)+
                                      cgd.draw_slope_8p3_diagonal(chest_center[0], chest_center[1]-2, max(r for r,c in sk_raw), 
                                      left_down=True, right_down=False, left_up=False, right_up=False) if (r, c) in sk_raw]
        sk_light_part2 = [(r, c) for r, c in  sk_raw if r <=min(r for r,c in  sk_raw)+size]
        sk_light_part3_temp = [(r,c) for r,c in left_leg_raw if r >=max(r for r,c in  sk_raw)-size
                         ]
        sk_light_part3_raw = []
        for loc in sk_light_part3_temp:
            sk_light_part3_raw+=cg.hex_neighbours_n(loc[0], loc[1], n=1, keep_origin = True, return_frontier=False)
        sk_light_part3 = [(r,c) for r,c in sk_light_part3_raw if (r,c) in sk_raw and (r,c) not in sk_dark]
        sk_light_part4 = [(r,c) for r,c in right_leg_raw if r >=max(r for r,c in  sk_raw)-size]
        sk_light_part5 = cgd.draw_slope_0p5_diagonal(min(r for r,c in sk_light_part4)-1,
                                                     min(c for r,c in right_leg_raw if r==min(r for r,c in sk_light_part4)-1)-1, 
                                                     min(r for r,c in sk_light_part4)-1-size, 
                                      left_down=False, right_down=False, left_up=False, right_up=True)
        sk_light = sk_light_part1+sk_light_part2+sk_light_part3+sk_light_part4+sk_light_part5
        face = [(r, c) for r, c in head_raw if r in face_line_row_to_col and c <= face_line_row_to_col[r]]
        hair_raw = [(r, c) for r, c in head_raw if (r, c) not in face
                   ]+cg.hex_neighbours_n(head_center[0], head_center[1], n=size+1, 
                                         keep_origin = False, return_frontier=True
                   )[-1]+cg.hex_neighbours_n(min(r for r,c in head_line), min(c for r,c in head_line if r==min(r for r,c in head_line)), 
                                             n=size+1, keep_origin = False, return_frontier=True)[-1]


        bolero = [(r,c) for r,c in bolero_raw+left_arm+right_arm if (r,c) not in head_raw+left_hand+right_hand]
        hair = [(r,c) for r,c in hair_raw if (r,c) not in bolero]

        pony_tail_start = (min(r for r,c in hair), max(c for r,c in hair if r==min(r for r,c in hair)))
        pony_tail_raw = cgd.draw_slope_0p5_diagonal(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]-size, 
                                                    left_down=False, right_down=False, left_up=True, right_up=True
                       )+cgd.horizontal_lines([(pony_tail_start[0], (pony_tail_start[1],pony_tail_start[1]+size+1))
                      ])+cgd.draw_slope_0p5_diagonal(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]+size+size+size, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                       )
        pony_tail_part1 = cgd.verticle_line(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]-size-1, bend = 'left'
                                           )
        pony_tail_part2_raw = cgd.draw_slope_1p5_diagonal(min(r for r,c in pony_tail_part1),
                                                       min(c for r,c in pony_tail_part1 if r==min(r for r,c in pony_tail_part1)),
                                                       min(r for r,c in pony_tail_part1)+size+1, 
                                                    left_down=True, right_down=False, left_up=False, right_up=False
                       )
        pony_tail_part2 = []
        for loc in pony_tail_part2_raw:
            n=1 if loc[0]!=min(r for r,c in pony_tail_part2_raw) and loc[0]!=max(r for r,c in pony_tail_part2_raw) else 0

            pony_tail_part2 +=cg.hex_neighbours_n(loc[0], loc[1], n=n, keep_origin = True, return_frontier=False)

        pony_tail_raw = [(r,c) for r,c in pony_tail_raw if (r,c) not in hair]
        pony_tail_left = [(r,c) for r,c in pony_tail_part1+pony_tail_part2 if (r,c) not in hair+pony_tail_raw] 

        pony_tail_mid = cg.hex_neighbours_n(min(r for r,c in pony_tail_raw), 
                                            max(c for r,c in pony_tail_raw if r==min(r for r,c in pony_tail_raw)), n=1)

        pony_tail_back_raw = []
        for loc in pony_tail_raw:
            if loc[0]>pony_tail_start[0]:
                pony_tail_back_raw+=cg.hex_neighbours_n(loc[0], loc[1], n=2, keep_origin = True, return_frontier=False)
        pony_tail_back = [(r,c) for r,c in pony_tail_back_raw if (r,c) not in head_raw+hair+pony_tail_mid+pony_tail_left]

        pony_tail_right_raw_part1 = cgd.draw_slope_8p3_diagonal(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]-1, 
                                                    left_down=False, right_down=False, left_up=False, right_up=True)
        pony_tail_right_raw_part2 = cgd.verticle_line(min(r for r,c in pony_tail_right_raw_part1),
                                    max(c for r,c in pony_tail_right_raw_part1 if r==min(r for r,c in pony_tail_right_raw_part1))+1,
                                    min(r for r,c in pony_tail_right_raw_part1)+size+size+size+1, bend = 'right'
                                           )
        pony_tail_right_raw = pony_tail_right_raw_part1+pony_tail_right_raw_part2
        pony_tail_right = []
        for loc in pony_tail_right_raw:
            if loc[0]>pony_tail_start[0] and loc[0]<max(r for r,c in pony_tail_right_raw_part2)-2:
                pony_tail_right+=cg.hex_neighbours_n(loc[0], loc[1], n=1, keep_origin = True, return_frontier=False)
        pony_tail_right+=pony_tail_right_raw

        pony_tail_bottom_part1 = cgd.draw_slope_1p5_diagonal(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]+1, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False)
        pony_tail_bottom_part2 = cgd.verticle_line(max(r for r,c in pony_tail_bottom_part1),
                                    max(c for r,c in pony_tail_bottom_part1 if r==max(r for r,c in pony_tail_bottom_part1))+1,
                                    max(r for r,c in pony_tail_bottom_part1)+size+size+1, bend = 'left'
                                           )
        pony_tail_bottom = [(r, c) for r, c in  pony_tail_bottom_part1+pony_tail_bottom_part2 if (r,c) not in hair]

        hair_dark =[(r, c) for r, c in  cgd.draw_slope_1p5_diagonal(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]+1, 
                                                    left_down=False, right_down=True, left_up=False, right_up=False
                                               )+cgd.draw_slope_1p5_diagonal(pony_tail_start[0],pony_tail_start[1], pony_tail_start[0]-1, 
                                                    left_down=False, right_down=False, left_up=True, right_up=False
                                               )if (r,c) not in hair]
        print(hair_dark)
        top = [(r,c) for r,c in  top_raw if (r,c) not in bolero+head_raw]
        left_foot = [(r, c) for r, c in left_leg_raw if r> max(r for r,c in left_leg_raw)-size]
        right_foot = [(r, c) for r, c in right_leg_raw if r> max(r for r,c in right_leg_raw)-size]


        left_leg = [(r, c) for r, c in left_leg_raw if (r, c) not in sk_raw+left_foot and r >sk_center[0]]
        right_leg = [(r, c) for r, c in right_leg_raw if (r, c) not in sk_raw+right_foot and r >sk_center[0]]

        skin = face+[(r,c) for r,c in body_line if r==max(r for r,c in face)+2]+[(r,c) for r,c in left_hand+right_hand 
                                                                                 if (r,c) not in sk_light_part1]
        keys = [ 'raw_body', 'sk_raw', "sk_dark", "sk_light",
                "hair","pony_tail_raw", "pony_tail_back", 

                "pony_tail_left","pony_tail_mid","pony_tail_right","pony_tail_bottom", "hair_dark", 

                "skin","bolero","top", "left_leg", "right_leg", "left_foot", "right_foot", ]#
        colors = [[[0,0,0]],[cgc.hex_to_rgb("#5f11a5")], [cgc.hex_to_rgb("#471a73")], [cgc.hex_to_rgb("#6218c0")], ##5d10c0 #6218c0 #6724d5

                  [cgc.hex_to_rgb("#dcdaf2")],[cgc.hex_to_rgb("#a9a7be")],[cgc.hex_to_rgb("#a9a7be")], 

                  [cgc.hex_to_rgb("#c8c6de")],[cgc.hex_to_rgb("#e2e1f0")],[cgc.hex_to_rgb("#c8c6de")],[cgc.hex_to_rgb("#d5d3eb")],[cgc.hex_to_rgb("#bab8cb")],

                  [cgc.hex_to_rgb("#fff0db")],[[0,0,0]],[cgc.hex_to_rgb("#6724d5")],

                  [[0.4, 0.4, 0.4]], [[0.4, 0.4, 0.4]],[[0,0,0]],[[0,0,0]],
              ]
    local_vars = locals()
    
    return {k: [local_vars[k], c] for k, c in zip(keys, colors)}



char_dict_ = {
        'Kirsi':draw_char((max(r for r, c in hex_rc_arr) // 4, max(c for r, c in hex_rc_arr) // 4*3-5), 'Kirsi', 2),
            '1': draw_char((max(r for r, c in hex_rc_arr) // 4*3, max(c for r, c in hex_rc_arr) // 4+10), '1', 5),
     '2':draw_char((8, max(c for r, c in hex_rc_arr) // 4), '2', 3),
       
                }

for char, dict_ in char_dict_.items(): 
    if char =='Kirsi':
        alpha_part = 1
    else:
        alpha_part =0.8
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
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'DTB_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')