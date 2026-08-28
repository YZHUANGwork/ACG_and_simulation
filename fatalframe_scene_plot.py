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
import  DIAGRAM_periodigram_timeseries as DIAG
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


sigma_color = 0.02
    
#------read solution-----------------------
Period = 1*u.yr
f0 = (1/Period).to(1/u.s)
DIAG_full = DIAG.TimeSeriesPeriodogram(Ad=0.03,
                 Phase=0 * u.deg,
                 N_in_bin=10, Nbkg_in_bin=1,
                 P=Period, T=5 * u.yr, dt=5*u.day,
                 freqs=np.linspace(f0.value,1, 2000)/u.s  ,
                 noise_type='none',threshold = 0.1,
                 seed=1)

power = DIAG_full.run_periodogram().value
_, recurrence_matrix = DIAG_full.recurrence_plot()
t  = DIAG_full.time_bin_centers.to(u.day).value
i_idx, j_idx = np.where(recurrence_matrix)
X = t[i_idx]
Y = t[j_idx]

rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}


DOMAIN_W = X.max() - X.min()
DOMAIN_H = Y.max() - Y.min()
CANVAS_PHYSICAL_W = DOMAIN_W
CANVAS_PHYSICAL_H = DOMAIN_H
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
 
px0, py0 = cg.hex_center_pixel(22, 0, detail_info)
OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X.min()
OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y.min()

grid_hex_rc = cg.world_metres_to_hex_index(
    X + OFFSET_X, Y + OFFSET_Y, detail_info,
    canvas_physical_x_range=canvas_physical_x_range,
    canvas_physical_y_range=canvas_physical_y_range,
)
hex_to_grid_idx = {}
for flat_i, rc in enumerate(grid_hex_rc):
    hex_to_grid_idx.setdefault(rc, []).append(flat_i)
hex_pos_to_grid_idx = {
    rc_to_idx[rc]: idxs
    for rc, idxs in hex_to_grid_idx.items()
    if rc in rc_to_idx
}

bkgd_colors = [cgc.hex_to_rgb("#010049"), cgc.hex_to_rgb("#f1f1f1"), [0,0,1], [0,0,1] ]
blend_colors = [cgc.hex_to_rgb("#0601d1" ), cgc.hex_to_rgb("#9291c6" ), cgc.hex_to_rgb("#03008b" ) , cgc.hex_to_rgb("#cecdff" )  ]
color_copys = []

RECURRENCE_ALPHA = 1   

max_count = max((len(idxs) for idxs in hex_pos_to_grid_idx.values()), default=1)
for bkgd_color, blend_color in zip(bkgd_colors, blend_colors):
    hex_colors[[True]*len(hex_rc_arr)] =bkgd_color
    RECURRENCE_FILL_COLOR = blend_color
    hex_idx_to_color = {}
    for hex_idx in range(len(hex_rc_arr)):
        idxs = hex_pos_to_grid_idx.get(hex_idx, [])
        a = len(idxs) / max_count            # 0 at no recurrence, up to 1 at max density
        blend_weight = RECURRENCE_ALPHA * a  # 0 at a=0 (fully base), rising toward RECURRENCE_ALPHA
        hex_idx_to_color[hex_idx] = (
            (1 - blend_weight) * hex_colors[hex_idx]
            + blend_weight * np.array(RECURRENCE_FILL_COLOR)
        )

    hex_colors = np.array([
                cgc.select_normal_color([True], hex_idx_to_color[i], np.ones(3) * sigma_color)[0]
                for i in range(len(hex_rc_arr))
            ])
    color_copys.append(hex_colors.copy())


def draw_camera(center_row, center_col, n_lens):
    dn = 1
    n_lens_glass = n_lens+dn
    n_lens_frame = n_lens_glass+dn
    lens_reflect = cg.hex_neighbours_n(center_row, center_col, n=n_lens, keep_origin = True, return_frontier=False)
    lens_glass = cg.hex_neighbours_n(center_row, center_col, n=n_lens_glass, keep_origin = False, return_frontier=True)[-1]
    lens_frame = cg.hex_neighbours_n(center_row, center_col, n=n_lens_frame, keep_origin = False, return_frontier=True)[-1]
    
    frame_left_start = center_col-n_lens_frame-2
    frame_right_end = center_col+n_lens_frame+3
    frame_front = cgd.draw_block((center_row, (frame_left_start,  frame_right_end)),
                                 center_row-n_lens_frame-1
                               )+cgd.draw_block((center_row,(frame_left_start,  frame_right_end)),
                                 center_row+n_lens_frame+1
                               )
    
    lens_cap = cgd.draw_block((min(r for r,c in frame_front), (max(c for r,c in frame_front if r == min(r for r,c in frame_front))-2 ,  
                                           max(c for r,c in frame_front if r == min(r for r,c in frame_front))-1
                                                              )),
                                 max(r for r,c in frame_front)
                               )
    frame_top = cgd.draw_trapezoid(min(r for r,c in frame_front), 
                                   min(c for r,c in frame_front if r == min(r for r,c in frame_front))+1, 
                                   max(c for r,c in frame_front if r == min(r for r,c in frame_front))-2, 
                                   min(r for r,c in frame_front)-1, slope_left = '0.5', slope_right = '0.5', direction = 'rl',
                  bend_left = 'left', bend_right = 'right')[-2]
    lens_reflect+=[(min(r for r,c in frame_top)+1, (min(c for r, c in frame_top if r == min(r for r,c in frame_top))+
                                                 max(c for r, c in frame_top if r == min(r for r,c in frame_top)))//2)]
    return {
         'frame_top': [frame_top, [cgc.hex_to_rgb("#393530")]],
         'frame_front': [frame_front, [cgc.hex_to_rgb("#393530")]],
        'lens_frame': [lens_frame, [cgc.hex_to_rgb("#574432")]],
        'lens_glass': [lens_glass, [[0,0,0]]],
            'lens_reflect': [lens_reflect, [[1,1,1]]],
     'lens_cap': [lens_cap, [[0,0,0]]],
        
            }

    
def draw_snake_body(seg_dict):
    
    body_segs = []
    body_top_segs = []
    body_bottom_segs= []
    for key in seg_dict.keys():
        seg_size = seg_dict.get(key)[0]
        seg_centers = seg_dict.get(key)[1]
        for i in range(0, len(seg_centers)-1):

            current_body_seg = cg.hex_neighbours_n(seg_centers[i][0], seg_centers[i][1], n=seg_size, 
                                                   keep_origin = True, return_frontier=False
                            )
            next_body_seg = cg.hex_neighbours_n(seg_centers[i+1][0], seg_centers[i+1][1], 
                                                    n=seg_size, keep_origin = True, return_frontier=False
                            )
            current_top_body_seg = [(r,c) for r,c in current_body_seg if r <=seg_centers[i][0]]
            current_bottom_body_seg = [(r,c) for r,c in current_body_seg if r >seg_centers[i][0]]

            valid_top_body_seg = [(r,c) for r, c in current_top_body_seg if (r,c) not in next_body_seg]
            valid_bottom_body_seg = [(r,c) for r, c in current_bottom_body_seg if (r,c) not in next_body_seg]


            body_segs.extend(current_body_seg)
            body_top_segs.extend(valid_top_body_seg)
            body_bottom_segs.extend(valid_bottom_body_seg)

        last_body_seg = cg.hex_neighbours_n(seg_centers[-1][0], seg_centers[-1][1], 
                                                    n=seg_size, keep_origin = True, return_frontier=False)
        last_top_body_seg = [(r,c) for r,c in last_body_seg if r <=seg_centers[-1][0]]
        last_bottom_body_seg = [(r,c) for r,c in last_body_seg if r >seg_centers[-1][0]]
        body_segs.extend(last_body_seg)
        body_top_segs.extend(last_top_body_seg)
        body_bottom_segs.extend(last_bottom_body_seg)

        body_top_segs.extend(valid_top_body_seg)
        body_bottom_segs.extend(valid_bottom_body_seg)
        
    return { 
            'body_bottom': [body_bottom_segs, [[0,0,0]]], #cgc.hex_to_rgb("#ffd600")
        'body_top': [body_top_segs, [[1,1,1]]],#cgc.hex_to_rgb("#000b57") 
     
           
            }
def draw_snake_head(joint_center_r,joint_center_c, n_joint ):
    joint = cg.hex_neighbours_n(joint_center_r, joint_center_c, n=n_joint,  keep_origin = True, return_frontier=False)
    joint_top = [(r,c) for r,c in joint if r <=joint_center_r]
    joint_bottom = [(r,c) for r,c in joint if r >joint_center_r]

        
    size_joint = 2*n_joint
    head_part1 = cgd.draw_trapezoid(joint_center_r ,joint_center_c-n_joint ,joint_center_c+n_joint ,
                                    joint_center_r-n_joint-size_joint, 
                                    slope_left = '0.5', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                  )[-2]
    head_part2 = cgd.draw_trapezoid(min(r for r,c in head_part1)+1 ,
                                    min(c for r,c in head_part1 if r == min(r for r,c in head_part1)+1)+1 ,
                                    max(c for r,c in head_part1 if r == min(r for r,c in head_part1)+1)+1 ,
                                    min(r for r,c in head_part1)+1-n_joint-size_joint, 
                                    slope_left = '0.5', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                  )[-2]
    head_part3 = cgd.draw_trapezoid(min(r for r,c in head_part2)+1 ,
                                    min(c for r,c in head_part2 if r == min(r for r,c in head_part2)+1)+1 ,
                                    max(c for r,c in head_part2 if r == min(r for r,c in head_part2)+1)+1 ,
                                    min(r for r,c in head_part2)+1-n_joint-n_joint, 
                                    slope_left = '0.5', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                  )[-2]
    head_part4 = cgd.draw_trapezoid(min(r for r,c in head_part3) ,
                                    min(c for r,c in head_part3 if r == min(r for r,c in head_part3))+1 ,
                                    max(c for r,c in head_part3 if r == min(r for r,c in head_part3)) ,
                                    min(r for r,c in head_part3)-n_joint, 
                                    slope_left = '0.5', slope_right = '1.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                  )[-2]
    
    head = joint_top+head_part1+head_part2+head_part3+head_part4
    teeth = cgd.draw_slope_0p5_diagonal(min(r for r,c in head),
                              max(c for r,c in head if r == min(r for r,c in head)), min(r for r,c in head)+2, 
                              left_down= False, right_down=True , left_up=False, right_up= False
                             )+cgd.draw_slope_0p5_diagonal(min(r for r,c in head)+1,
                              max(c for r,c in head if r == min(r for r,c in head)+1), min(r for r,c in head)+1+2, 
                              left_down= False, right_down=True , left_up=False, right_up= False
                             )
    teeth_valid = [(r,c) for r,c in teeth if (r,c)not in head]
    
    
    jaw_part1 = cgd.draw_trapezoid(joint_center_r,joint_center_c+n_joint+1,joint_center_c+n_joint+1+size_joint, 
                             joint_center_r+n_joint, 
                             slope_left = '0.5', slope_right = '1.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                  )[-2]
    jaw_part2 = cgd.draw_trapezoid(max(r for r,c in jaw_part1) ,
                                    min(c for r,c in jaw_part1 if r == max(r for r,c in jaw_part1))+1 ,
                                    max(c for r,c in jaw_part1 if r == max(r for r,c in jaw_part1)) ,
                                    max(r for r,c in jaw_part1)+size_joint, 
                                    slope_left = '1.5', slope_right = '0.5', direction = 'rr',bend_left = 'left', bend_right = 'right'
                  )[-2]
    jaw_part3 = cgd.horizontal_lines([(max(r for r,c in jaw_part2), (
                                                                   max(c for r,c in jaw_part2 if r == max(r for r,c in jaw_part2)),
                                                                   max(c for r,c in jaw_part2 if r == max(r for r,c in jaw_part2))+1 ))])
    jaw = joint_bottom+jaw_part1+jaw_part2+jaw_part3
    
    full_head = jaw+head
    
    eye_r = joint_center_r-size_joint-size_joint
    eye = (eye_r, min(c for r,c in full_head if r ==eye_r)+1 )
    
    mouth = cgd.horizontal_lines([(joint_center_r-n_joint-n_joint-n_joint, 
                                   (max(c for r,c in full_head if r == joint_center_r-n_joint-n_joint-n_joint)+1,
                                    max(c for r,c in full_head if r == joint_center_r-n_joint-n_joint-n_joint)+1 )
                                  ),
                                  (joint_center_r-n_joint-n_joint, 
                                   (max(c for r,c in full_head if r == joint_center_r-n_joint-n_joint)+1,
                                    max(c for r,c in full_head if r == joint_center_r-n_joint-n_joint)+1)
                                  ),
                                  (joint_center_r-n_joint, 
                                   (max(c for r,c in full_head if r == joint_center_r-n_joint)+1,
                                    max(c for r,c in full_head if r == joint_center_r-n_joint)+size_joint+size_joint )
                                  ),
                                  (joint_center_r, 
                                   (max(c for r,c in full_head if r == joint_center_r)+1,
                                    max(c for r,c in full_head if r == joint_center_r)+size_joint+size_joint )),
                                  (joint_center_r+n_joint, 
                                   (max(c for r,c in full_head if r == joint_center_r+n_joint)+1,
                                    max(c for r,c in full_head if r == joint_center_r+n_joint)+size_joint+size_joint )),
                                 (joint_center_r+n_joint+n_joint, 
                                   (max(c for r,c in full_head if r == joint_center_r+n_joint+n_joint)+1,
                                    max(c for r,c in full_head if r == joint_center_r+n_joint+n_joint)+size_joint))])
    
    tongue_r = joint_center_r
    tongue_c = max(c for r,c in mouth if r ==  tongue_r)-1
    tongue_part1 = cgd.draw_slope_1p5_diagonal(tongue_r, tongue_c, eye_r, 
                                              left_down=False, right_down=False, left_up=False , right_up=True 
                              )+cgd.draw_slope_1p5_diagonal(tongue_r, tongue_c+1, eye_r, 
                                              left_down=False, right_down=False, left_up=False , right_up=True 
                              )
    tongue_part2 = cgd.horizontal_lines([(min(r for r,c in tongue_part1), 
                                   (max(c for r,c in tongue_part1 if r == min(r for r,c in tongue_part1)),
                                    max(c for r,c in tongue_part1 if r == min(r for r,c in tongue_part1))+size_joint+n_joint )
                                         )])
    tongue_part3 = cgd.draw_slope_0p5_diagonal(min(r for r,c in tongue_part2), 
                                               max(c for r,c in tongue_part2 if r == min(r for r,c in tongue_part2))-2,
                                               min(r for r,c in tongue_part2)+3, 
                                              left_down=False, right_down=True , left_up=False , right_up= False 
                                              )+cgd.draw_slope_0p5_diagonal(min(r for r,c in tongue_part2), 
                                               max(c for r,c in tongue_part2 if r == min(r for r,c in tongue_part2)),
                                               min(r for r,c in tongue_part2)+1, 
                                              left_down=False, right_down=True , left_up=False , right_up= False 
                                              )
                                        
                                        
    tongue = tongue_part1+tongue_part2+tongue_part3
    return { 
            
        'head': [head, [[0,0,1]]],
     'teeth': [teeth, [[0,0,0]]],
           'jaw': [jaw, [cgc.hex_to_rgb("#948a7d") ]],
        'mouth': [mouth, [ [0,0,0] ]],
         'eye': [[eye], [ [1,1,1] ]],
        'tongue': [tongue, [cgc.hex_to_rgb("#00f7ff")]],
            }

#hex_colors = color_copys[-1]
hex_colors[[True]*len(hex_rc_arr)] =[0.96, 0.96, 0.96]
camera_center_row = 9
camera_center_col = 27
n_lens = 0

start_center_part1 = (2, 7)

seg_centers_part1 = [start_center_part1, (start_center_part1[0]+1, start_center_part1[1]+3),
                                         (start_center_part1[0]+1, start_center_part1[1]+6),
                                         (start_center_part1[0]+2, start_center_part1[1]+9),
                                         (start_center_part1[0]+3, start_center_part1[1]+11)
                ]

start_center_part2 = (seg_centers_part1[-1][0]+2, seg_centers_part1[-1][1])
seg_centers_part2 = [ (start_center_part2[0]-1, start_center_part2[1]+1), start_center_part2,
                  (start_center_part2[0]+1, start_center_part2[1]),
                  (start_center_part2[0]+1, start_center_part2[1]-2),
                  (start_center_part2[0]+1, start_center_part2[1]-4),
                  (start_center_part2[0]+1, start_center_part2[1]-6),
                  (start_center_part2[0]+1, start_center_part2[1]-8),
                  (start_center_part2[0]+2, start_center_part2[1]-10),
                  (start_center_part2[0]+2, start_center_part2[1]-12),
                  (start_center_part2[0]+3, start_center_part2[1]-14),
                  (start_center_part2[0]+3, start_center_part2[1]-16),
                  (start_center_part2[0]+4, start_center_part2[1]-17),
                  (start_center_part2[0]+5, start_center_part2[1]-18)]

start_center_part3 = (seg_centers_part2[-1][0]+5, seg_centers_part2[-1][1])
seg_centers_part3 = [start_center_part3, (start_center_part3[0]-1, start_center_part3[1]+2),
                                         (start_center_part3[0]-1, start_center_part3[1]+4),
                     (start_center_part3[0]-2, start_center_part3[1]+5),
                     (start_center_part3[0]-2, start_center_part3[1]+7),
                     (start_center_part3[0]-2, start_center_part3[1]+9),
                     (start_center_part3[0]-2, start_center_part3[1]+11),
                     (start_center_part3[0]-3, start_center_part3[1]+12),
                     (start_center_part3[0]-3, start_center_part3[1]+14),
                     (start_center_part3[0]-3, start_center_part3[1]+16),
                     (start_center_part3[0]-2, start_center_part3[1]+16),
                     (start_center_part3[0]-1, start_center_part3[1]+15),
                     (start_center_part3[0]-1, start_center_part3[1]+13),
                     (start_center_part3[0]-1, start_center_part3[1]+11)
                                        ]

start_center_part4 = (seg_centers_part3[-1][0]+2, seg_centers_part3[-1][1])
seg_centers_part4 = [start_center_part4]

start_center_part5 = (0,2)
seg_centers_part5 = [start_center_part5, (start_center_part5[0]+2, start_center_part5[1]),
                                         (start_center_part5[0]+3, start_center_part5[1]),
                                         (start_center_part5[0]+5, start_center_part5[1]+1),
                    (start_center_part5[0]+6, start_center_part5[1]+3)]
start_center_part6 = (3,34)
seg_centers_part6 = [start_center_part6,
                     (start_center_part6[0]-2, start_center_part6[1]-1),
                     (start_center_part6[0], start_center_part6[1]-3),
                                         (start_center_part6[0]-1, start_center_part6[1]-6),
                     (start_center_part6[0]-2, start_center_part6[1]-9),
                    (start_center_part6[0]-3, start_center_part6[1]-11),
                     (start_center_part6[0]-4, start_center_part6[1]-14),]
start_center_part7 = (21,34)
seg_centers_part7 = [start_center_part7,
                     (start_center_part7[0]-1, start_center_part6[1]-3),
                     (start_center_part7[0], start_center_part6[1]-7),
                     (start_center_part7[0]+2, start_center_part6[1]-10),
                    ]

seg_dict = {'a': [2, seg_centers_part1],
           'b': [1, seg_centers_part2],
           'c': [2, seg_centers_part3] ,
           'd': [2, seg_centers_part4],
           'e': [2, seg_centers_part5],
           'f': [2, seg_centers_part6],
           'g': [2, seg_centers_part7]}
jaw_temp = cg.hex_neighbours_n(start_center_part4[0], start_center_part4[1], n=2,  keep_origin = True, return_frontier=False
                            )
jaw_center = (start_center_part4[0], max(c for r,c in jaw_temp if r == start_center_part4[0]+1)-1)
n_jaw = 1
char_dict_ = {'camera': draw_camera(camera_center_row, camera_center_col, n_lens),
             'tattoo snake body': draw_snake_body(seg_dict),
             'tattoo snake head': draw_snake_head(jaw_center[0],jaw_center[1], n_jaw )}
for dict_ in char_dict_.values(): 

    for key in dict_.keys():
        part  = dict_.get(key)[0]
        colors = dict_.get(key)[1] 
        
        if len(colors) == 1:

            select_part= cg.select_mask(part,hex_rc_arr)
            if key in ["body_top", 'frame_top', 'frame_front']:
                hex_colors[select_part] = color_copys[0][select_part]
            elif key in ['body_bottom' ,'lens_frame']:
                 hex_colors[select_part] = color_copys[1][select_part]
            elif key in ['head', 'tongue']: 
                hex_colors[select_part] = color_copys[2][select_part]
            elif key in ['jaw']: 
                hex_colors[select_part] = color_copys[3][select_part]
            else:
                hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 
        else:
            hex_colors = cgc.color_row_gradient(part, 
                                        colors[0],colors[1],
                                        hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01) 

pc.set_facecolor(hex_colors)
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)
ax.add_collection(pc)
pc.set_facecolor(hex_colors)
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'fatalframe_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')
