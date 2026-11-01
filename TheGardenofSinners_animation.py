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
import SOLN_harmonicosc as SOLN

import TheGardenofSinners_scene_plot as GS_SCENE
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100

fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=15, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)
IMG_W_SCENE, IMG_H_SCENE, HEX_R, dx_hex_center, dy_hex_center = detail_info
sigma_color = 0.03



n_marker = 14
center_row = (max(r for r, c in hex_rc_arr)+min(r for r, c in hex_rc_arr) )// 2
center_col = max(c for r, c in hex_rc_arr) // 2

hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#ff0000"), cgc.hex_to_rgb("#006889"), hex_rc_arr, hex_colors, 
                                    (center_row-4, center_col), 18,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='linear',
                        start_n=0, end_n=None)

hex_colors = cgc.color_hex_gradient(hex_rc_arr, cgc.hex_to_rgb("#006889"), cgc.hex_to_rgb("#73ffea"), hex_rc_arr, hex_colors, 
                                    (center_row-4, center_col), max(c for r, c in hex_rc_arr) ,
                        sigma_color=0.03, end_weight=0.2, period=None, mode='sigmoid',
                        start_n=n_marker+5, end_n=None)

background_hex_colors = hex_colors.copy()

bases = [{
     "Shiki": GS_SCENE.draw_obj(center_row-6, center_col+10, hex_rc_arr, "Shiki flip"),
     "Shirazumi": GS_SCENE.draw_obj(center_row+6, center_col-10, hex_rc_arr, "Shirazumi"),
     "building": GS_SCENE.draw_obj(center_row-4, center_col, hex_rc_arr, "building"),
    'left': GS_SCENE.draw_char(center_row, center_col-12, hex_rc_arr, 'left'),
    'right': GS_SCENE.draw_char(center_row, center_col+12, hex_rc_arr, 'right')

}
        ,
         { "Aozaki head": GS_SCENE.draw_obj(center_row-4, center_col, hex_rc_arr, "can"),},
         
         { "Aozaki head": GS_SCENE.draw_obj(center_row-4, center_col, hex_rc_arr, "can blood"),}
     
        ]

background_base_hex_colors = []
for char_dict_ in bases:
    hex_colors  = background_hex_colors.copy()
    for char, dict_ in char_dict_.items(): 
         for key in dict_.keys():

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
                            hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01, mode = mode, pivot=pivot ) 

                else:
                    if 'sigmoid' in sort_info:
                        mode = 'sigmoid'
                    else:
                        mode = 'linear'
                    print('mode', mode)

                    hex_colors = cgc.color_row_gradient(part, 
                            colors[0],colors[1],
                            hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = mode)
    background_base_hex_colors.append(hex_colors.copy() )


rng = np.random.default_rng(seed=0)

   
#-------------------READ SOLN-----------------------
t_final = 18
t_full = np.linspace(0, t_final, 300) * u.s

sol_left = SOLN.HarmonicOscillator(k=12.0 * u.kg / u.s ** 2, m=10.0 * u.kg, orientation="horizontal",
                                 q0=0.3 * u.m, v0=0.0 * u.m / u.s)
sol_right = SOLN.HarmonicOscillator(k=12.0 * u.kg / u.s ** 2, m=10.0 * u.kg, orientation="horizontal",
                                 q0=-0.3 * u.m, v0=0.0 * u.m / u.s)

sol_vert = SOLN.HarmonicOscillator(k=50.0 * u.kg / u.s ** 2, m=5.0 * u.kg, orientation="vertical",
                                q0=0.0 * u.m, v0=0.0 * u.m / u.s)
sol_coupled = SOLN.HarmonicOscillator(k=80.0 * u.kg / u.s ** 2, m=[2.0 * u.kg, 2.0 * u.kg],
                                q0=[0.2 * u.m, -0.2 * u.m], v0=[0.0 * u.m / u.s, 0.0 * u.m / u.s])

x1_c, x2_c = sol_coupled.coupled_trajectory(t=t_full)
#springlength = 1.0 * u.m   # x2_0 - x1_0 at t=0
#x2_c  += springlength.value


X_left = sol_left.trajectory_1d(t_full)

X_right = sol_right.trajectory_1d(t_full)


grid_hex_rc_left = cg.fn2grid(
    X_left, np.zeros_like(X_left),
    DOMAIN_W_SCALE=2, DOMAIN_H_SCALE=2,
    PIVOT_ROW=center_row, PIVOT_COL=center_col,
    PIVOT_ROW_X=X_left.max(),  PIVOT_COL_Y=0.,
    detail_info=detail_info, square=False,
)
grid_hex_rc_right = cg.fn2grid(
    X_right, np.zeros_like(X_right),
    DOMAIN_W_SCALE=2, DOMAIN_H_SCALE=2,
    PIVOT_ROW=center_row, PIVOT_COL=center_col,
    PIVOT_ROW_X=X_right.min(),  PIVOT_COL_Y=0.,
    detail_info=detail_info, square=False,
)

Y_vert = sol_vert.trajectory_1d(t_full) 

grid_hex_rc_revealed = cg.fn2grid(
    np.zeros_like(Y_vert), Y_vert,                
    DOMAIN_W_SCALE=1, DOMAIN_H_SCALE=1,
    PIVOT_ROW=center_row-10, PIVOT_COL=center_col,
    PIVOT_ROW_X=0.,  
    PIVOT_COL_Y=Y_vert.max(), 
    detail_info=detail_info, square=False,
)

grid_hex_rc_coupledleft = cg.fn2grid(
    x1_c, np.zeros_like(x1_c),
    DOMAIN_W_SCALE=4, DOMAIN_H_SCALE=2,
    PIVOT_ROW=center_row, PIVOT_COL=center_col-8,
    PIVOT_ROW_X=x1_c.max(),  PIVOT_COL_Y=0.,
    detail_info=detail_info, square=False,
)
grid_hex_rc_coupledright = cg.fn2grid(
    x2_c, np.zeros_like(x2_c),
    DOMAIN_W_SCALE=4, DOMAIN_H_SCALE=2,
    PIVOT_ROW=center_row, PIVOT_COL=center_col+8,
    PIVOT_ROW_X=x2_c.min(),  PIVOT_COL_Y=0.,
    detail_info=detail_info, square=False,
)
# ── UPDATE ──────────────────────────────────────────────────────────────
total_seconds = t_final

Phase0_end = 10
Phase1_end = len(t_full)   //4
total_frames = Phase0_end+ len(t_full)   
Phase2_end =   Phase1_end+Phase1_end
Phase3_end = Phase2_end+Phase1_end
print("total_frames", total_frames)

def update(snapshot):
    if snapshot<Phase0_end:
        current_hex_colors = background_base_hex_colors[0].copy()
    else:
        current_snapshot = snapshot-Phase0_end
            
            
        loc_left = grid_hex_rc_left[current_snapshot]
        loc_right = grid_hex_rc_right[current_snapshot]
        
        char_dict_ = {
             "building": GS_SCENE.draw_obj(center_row-4, center_col, hex_rc_arr, "building"),
            'left': GS_SCENE.draw_char(loc_left[0], loc_left[1], hex_rc_arr,  'left'),
            'right': GS_SCENE.draw_char(loc_right[0], loc_right[1], hex_rc_arr,  'right'),

        }
        if snapshot>=Phase0_end and snapshot< Phase1_end:
            current_hex_colors = background_base_hex_colors[1].copy()
         
            reveal = False
            
        
        elif snapshot>=Phase1_end and snapshot< Phase2_end:
            #can
            current_hex_colors = background_base_hex_colors[1].copy()
        
            reveal = True
            reveal_row = grid_hex_rc_revealed[current_snapshot][0]
        
         
        elif snapshot>=Phase2_end and snapshot< Phase3_end:
            #can blood
            current_hex_colors = background_base_hex_colors[2].copy()
        
            reveal = True
            reveal_row = grid_hex_rc_revealed[current_snapshot][0]
        
        else:
            current_hex_colors = background_hex_colors.copy()
            reveal = False
            
            loc_coupledleft = grid_hex_rc_coupledleft[current_snapshot]
            loc_coupledright = grid_hex_rc_coupledright[current_snapshot]
            char_dict_ = {
                "Shiki": GS_SCENE.draw_obj(center_row+8, loc_coupledleft[1], hex_rc_arr, "Shiki flip"),
                "Shirazumi": GS_SCENE.draw_obj(center_row-8, loc_coupledright[1], hex_rc_arr, "Shirazumi"),

            'left': GS_SCENE.draw_char(loc_left[0], loc_left[1], hex_rc_arr,  'left'),
            'right': GS_SCENE.draw_char(loc_right[0], loc_right[1], hex_rc_arr,  'right'),

        }
        for char, dict_ in char_dict_.items(): 

            for key in dict_.keys():

                part  = dict_.get(key)[0]
                if reveal and char == "building":
                    print('reveal row building')
                    part = [(r,c) for r,c in part if r<=reveal_row]
                colors = dict_.get(key)[1] 
                if len(colors) == 1:

                    select_part= cg.select_mask(part,hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color) 

                else:
                    sort_info = colors[-1]
                    if 'hex' in sort_info:

                        hex_colors = cgc.color_hex_gradient(part,colors[0],colors[1], hex_rc_arr, 
                                hex_colors, sort_info[0], sort_info[1],
                                sigma_color=sigma_color, end_weight=0.01, period=None, mode='linear',
                                start_n=1, end_n=None)
                    elif 'col' in sort_info:
                        
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
                        current_hex_colors = cgc.color_row_gradient(part, 
                                colors[0],colors[1],
                                hex_rc_arr, current_hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.01,
                                                                    mode = mode, pivot=pivot ) 

                    else:
                        if 'sigmoid' in sort_info:
                            mode = 'sigmoid'
                        else:
                            mode = 'linear'
                        print('mode', mode)

                        current_hex_colors = cgc.color_row_gradient(part, 
                                colors[0],colors[1],
                                hex_rc_arr, current_hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01, mode = mode) 

        
    pc.set_facecolor(current_hex_colors)
    return (pc,)



FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'TheGardenofSinners_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")