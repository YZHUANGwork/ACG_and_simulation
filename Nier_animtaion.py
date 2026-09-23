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

import TRAJECTORY_TwobodyScattering as SOLN
import Nier_scene_plot as NIER_SCENE

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

bkgd_base_hex_colors = hex_colors.copy()

COLLISION_PARAMS = [
    {"m1": 1*u.kg, "m2": 1e5*u.kg, "r_contact": (0,0, 0)*u.m,
     "v_i_1": (0,-1, 0)*u.m/u.s, "v_i_2": (0, 0, 0)*u.m/u.s, "e": 0, 
     "char1_name": "A2 type 4O sword", "char2_name": "king", "frame": "lab","plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//8*7, max(c for r,c in hex_rc_arr)//2)
    },
    
    {"m1": 100*u.kg, "m2": 5*u.kg, "r_contact": (0, 0, 5)*u.m,
     "v_i_1": (1, 0, 1)*u.m/u.s, "v_i_2": (-2, 0, 1)*u.m/u.s, "e": 1., 
     "char1_name": "A2 type 4O lance", "char2_name": "2B", "frame": "lab","plane": "xz", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//6, max(c for r,c in hex_rc_arr)//5)
    },
    
    {"m1": 1*u.kg, "m2": 2*u.kg, "r_contact": (1, 0.5, 0)*u.m,
     "v_i_1": (2, 1, 0)*u.m/u.s, "v_i_2": (-1, 0, 0)*u.m/u.s, "e": 0.9,
     "char1_name": "A2 type 4O lance", "char2_name": "2B", "frame": "lab", "plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2)
    },
    
    {"m1": 1*u.kg, "m2": 2*u.kg, "r_contact": (1, 0.5, 0)*u.m,
     "v_i_1": (0,-1, 0)*u.m/u.s, "v_i_2": (-1, 0, 0)*u.m/u.s, "e": 1, 
     "char1_name": "A2 type 4O lance", "char2_name": "9S", "frame": "lab","plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//5*3, max(c for r,c in hex_rc_arr)//5*3)
    },
    
    {"m1": 1*u.kg, "m2": 2*u.kg, "r_contact": (1, 0.5, 0)*u.m,
     "v_i_1": (0,-1, 0)*u.m/u.s, "v_i_2": (-1, 0, 0)*u.m/u.s, "e": 1, 
     "char1_name": "A2 type 4O lance", "char2_name": "2B", "frame": "com","plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//5*3, max(c for r,c in hex_rc_arr)//5*3)
    },
    
    {"m1": 1*u.kg, "m2": 2*u.kg, "r_contact": (1, 0.5, 0)*u.m,
     "v_i_1": (2, 1, 0)*u.m/u.s, "v_i_2": (-1, 0, 0)*u.m/u.s, "e": 0.9, 
     "char1_name": "A2 type 4O lance", "char2_name": "9S", "frame": "com","plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2)
    },
    
    {"m1": 1*u.kg, "m2": 1*u.kg, "r_contact": (1, 0.5, 0)*u.m,
     "v_i_1": (2, 0, 0)*u.m/u.s, "v_i_2": (-1, 0, 0)*u.m/u.s, "e": 1., 
     "char1_name": "A2 type 4O lance", "char2_name": "2B", "frame": "com","plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2)
    },
    
    
    {"m1": 1*u.kg, "m2": 1*u.kg, "r_contact": (0,0, 0)*u.m,
     "v_i_1": (1,0, 0)*u.m/u.s, "v_i_2": (0, 0, 0)*u.m/u.s, "e": 0, 
     "char1_name": "A2 Virtuous Contract", "char2_name": "2B", "frame": "lab","plane": "xy", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//2, max(c for r,c in hex_rc_arr)//2)
    },
    
   {"m1": 1*u.kg, "m2": 5*u.kg, "r_contact": (0, 0, 5)*u.m,
     "v_i_1": (2, 0, 1)*u.m/u.s, "v_i_2": (-2, 0, 1)*u.m/u.s, "e": 0.5, 
     "char1_name": "A2 Virtuous Contract", "char2_name": "9S", "frame": "lab","plane": "xz", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//7, max(c for r,c in hex_rc_arr)//2)
    },
    {"m1": 1*u.kg, "m2": 5*u.kg, "r_contact": (0, 0, 5)*u.m,
     "v_i_1": (2, 0, 1)*u.m/u.s, "v_i_2": (0, 0, 1)*u.m/u.s, "e": 0., 
     "char1_name": "A2 Virtuous Contract", "char2_name": "9S", "frame": "lab","plane": "xz", 
    "r_contact_grid": (max(r for r,c in hex_rc_arr)//7, max(c for r,c in hex_rc_arr)//4*3)
    },
]



all_solutions = []
for params in COLLISION_PARAMS:
    collision = SOLN.TwoBodyCollision(
        m1=params["m1"], m2=params["m2"], r_contact=params["r_contact"],
        v_i_1=params["v_i_1"], v_i_2=params["v_i_2"], e=params["e"],
    )
    t_before = np.linspace(-1, 0, 50) * u.s
    t_after = np.linspace(0, 1, 50)[1:] * u.s   # skip 0 here -- t_before already ends at 0
    t = np.concatenate([t_before.value, t_after.value]) * u.s
    
    
    if params["frame"] == "lab":
        frame_params = collision.lab_frame()
        r_contact_lab, v_i_1_lab, v_i_2_lab, v_f_1_lab, v_f_2_lab = frame_params
        r_contact = r_contact_lab.to_value(u.m)
        
    elif params["frame"] == "com":
        frame_params = collision.cm_frame()
        r_contact_com, v_i_1_com, v_i_2_com, v_f_1_com, v_f_2_com = frame_params
        r_contact = r_contact_com.to_value(u.m)
        
    if 'z' in params["plane"]:
        a_gravity = const.g0 * np.array([0, 0, -1])   # -z is "down" here, x-z plane
        r1, r2 = SOLN.solve_trajectory(frame_params, t, a=a_gravity)
    else:
        r1, r2 = SOLN.solve_trajectory(frame_params, t)
    x1, y1, z1 = r1.T.to_value(u.m)
    x2, y2, z2 = r2.T.to_value(u.m)
    #
    EPS_DOMAIN = 1e-6   # floor so a zero-motion axis doesn't zero out OFFSET_X/Y's anchor term
    max_x = max(max(x1.max(), x2.max()) - min(x1.min(), x2.min()), EPS_DOMAIN)
    max_y = max(max(y1.max(), y2.max()) - min(y1.min(), y2.min()), EPS_DOMAIN)
    max_z = max(max(z1.max(), z2.max()) - min(z1.min(), z2.min()), EPS_DOMAIN)
    axis_data = {
        "x": (x1, x2, max_x, 0),
        "y": (y1, y2, max_y, 1),
        "z": (z1, z2, max_z, 2),
    }
    plane = params["plane"]   # e.g. "xy", "xz" -- plane[0]=horizontal axis, plane[1]=vertical axis
    X_char1, X_char2, DOMAIN_W, r_contact_X_idx = axis_data[plane[0]]
    Y_char1, Y_char2, DOMAIN_H, r_contact_Y_idx = axis_data[plane[1]]
 

    canvas_physical_x_range = (0, DOMAIN_W)
    canvas_physical_y_range = (0, DOMAIN_H)
 
    
    r_contact_grid = params["r_contact_grid"]
    print('r_contact_grid', r_contact_grid)
    
    px0, py0 = cg.hex_center_pixel(r_contact_grid[0], r_contact_grid[1], detail_info)
    print(px0, py0)
    OFFSET_X = px0 * DOMAIN_W / IMG_W_SCENE - r_contact[r_contact_X_idx]
    OFFSET_Y = DOMAIN_H * (1.0 - py0 / IMG_H_SCENE) - r_contact[r_contact_Y_idx]
    
    
    char1_traj_raw = np.array(cg.world_metres_to_hex_index(
        X_char1 + OFFSET_X, Y_char1 + OFFSET_Y, detail_info,
        canvas_physical_x_range=canvas_physical_x_range, canvas_physical_y_range=canvas_physical_y_range))
    char2_traj_raw = np.array(cg.world_metres_to_hex_index(
        X_char2 + OFFSET_X, Y_char2 + OFFSET_Y, detail_info,
        canvas_physical_x_range=canvas_physical_x_range, canvas_physical_y_range=canvas_physical_y_range))
    
    valid_mask = [tuple(rc1) in hex_rc_arr and tuple(rc2) in hex_rc_arr for rc1, rc2 in zip(char1_traj_raw, char2_traj_raw)]
    char1_traj = char1_traj_raw[valid_mask]
    char2_traj = char2_traj_raw[valid_mask]
    t_valid = t[valid_mask]
    print("raw t frames", len(t), "t_valid" , len(t_valid))
          
    all_solutions.append({
        "char1_traj": char1_traj,
        "char2_traj": char2_traj,
        "t_len": len(t_valid),
        "char1_name": params["char1_name"],
        "char2_name": params["char2_name"],
    })

start_offsets = [0]
for sol in all_solutions[:-1]:
    start_offsets.append(start_offsets[-1] + sol["t_len"])
 
# ── UPDATE ──────────────────────────────────────────────────────────────
Phase0_end = 10           # intro frames that just hold the base scene
total_frames = Phase0_end + sum(sol["t_len"] for sol in all_solutions)   # se
print(sol["t_len"] for sol in all_solutions)
print("total_frames", total_frames)
def update(snapshot):
    if snapshot < Phase0_end:
        current_hex_colors = bkgd_base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot - Phase0_end
        current_hex_colors = bkgd_base_hex_colors.copy()
        #n_reveal = current_snapshot + 1
        
        current_dict = {}
        for i, sol in enumerate(all_solutions):
            local_idx = current_snapshot - start_offsets[i]
            if 0 <= local_idx < sol["t_len"]:          # this is the solution currently playing
                char1_loc = sol["char1_traj"][local_idx]
                char2_loc = sol["char2_traj"][local_idx]
                current_dict[f"sol{i}_char2"] = NIER_SCENE.draw_char_tiny(char2_loc[0], char2_loc[1], sol["char2_name"])
 
                current_dict[f"sol{i}_char1"] = NIER_SCENE.draw_char_tiny(char1_loc[0], char1_loc[1], sol["char1_name"])
                
        for dict_ in current_dict.values():
            for key in dict_.keys():
                part = dict_.get(key)[0]
                colors = dict_.get(key)[1]
                if len(colors) == 1:
                    select_part = cg.select_mask(part, hex_rc_arr)
                    current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)
                else:
                    current_hex_colors = cgc.color_row_gradient(part,
                                       colors[0], colors[1],
                                       hex_rc_arr, current_hex_colors, sort='row',
                                       sigma_color=sigma_color, end_weight=0.01, mode='linear')
        pc.set_facecolor(current_hex_colors)
    return (pc,)



total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'Nier_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")