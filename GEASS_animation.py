import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
import astropy.units as u
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import expected_value as EXP
import atomic_line_color_map as line_rgb
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,_ = cg.make_hex_scene(
    IMG_W=1000, IMG_H= 1200,HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)




sigma_color = 0.01
hex_colors = cgc.color_row_gradient(hex_rc_arr, cgc.hex_to_rgb("#4f0815"), cgc.hex_to_rgb("#3f161d"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.01)    

start_row = 25
start_col =  13

row_seg1 = 3
row_seg2 = 4
n = 2
n_helmet_glass = 13
n_helmet_metal = n_helmet_glass+7


helmet_glass = cg.hex_neighbours_n(start_row-n_helmet_glass-7, start_col, n=n_helmet_glass , keep_origin = True)
valid_helmet_glass = [(r,c)  for r,c in helmet_glass if c >=2 and c <=25 and r <start_row-7]
helmet_glass_bot_row = max(r for r, c in valid_helmet_glass)
_, _, _,_, glass_region, vertex = cgd.draw_trapezoid(helmet_glass_bot_row, 
                                                     min(c for r,c in valid_helmet_glass if r == helmet_glass_bot_row), 
                                                     max(c for r,c in valid_helmet_glass if r == helmet_glass_bot_row), 
                                                     helmet_glass_bot_row+row_seg1,
                                                      slope_left = '1.5', slope_right = '1.5', direction = 'rl')

FULL_helmet_glass_regions = valid_helmet_glass+glass_region
select_helmet_glass_regions = cg.select_mask(FULL_helmet_glass_regions  ,hex_rc_arr)

face_top_row = helmet_glass_bot_row-n_helmet_glass//2
face_region = [(r,c) for r,c in FULL_helmet_glass_regions if c >start_col and r >=face_top_row]
select_face_region = cg.select_mask(face_region  ,hex_rc_arr)

right_eyesocket_top = cgd.draw_slope_8p3_diagonal(face_top_row+3, 
                                          start_col+1, face_top_row, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                         )+cgd.draw_slope_8p3_diagonal(face_top_row+3, 
                                          start_col+2, face_top_row, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                         )+cgd.draw_slope_8p3_diagonal(face_top_row+3, 
                                          start_col+3, face_top_row, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                         )
right_eye_bounds_c2r = dict(sorted({c: max(r for r, cc in right_eyesocket_top if cc == c) 
                                 for c in set(c for _, c in right_eyesocket_top)}.items()))
valid_right_eyesocket_top = [(r,c) for r,c in right_eyesocket_top if (r,c) in face_region]

right_eyeball_r = face_top_row+2
right_eyeball_c = start_col+int((max(c for r,c in valid_right_eyesocket_top)-start_col)//1.5)
right_eyeball_n = 4
right_eye_ball = cg.hex_neighbours_n(right_eyeball_r, right_eyeball_c, n=right_eyeball_n, keep_origin = True, return_frontier=False)
valid_right_eye_ball = [(r,c) for r,c in right_eye_ball if r> right_eye_bounds_c2r.get(c, -1) and r <right_eyeball_r+right_eyeball_n-1
                        and (r, c) in face_region]

eyesocket_bot_r = max(r for r,c in right_eye_ball)-1
valid_right_eyesocket_bot = [(r,c) for r,c in right_eye_ball if r == eyesocket_bot_r]+[(min(c for r,c in right_eye_ball if r == 
                                                                           eyesocket_bot_r)-1, eyesocket_bot_r)]

select_eyesocket = cg.select_mask(valid_right_eyesocket_top+valid_right_eyesocket_bot,hex_rc_arr)


helmet_metal = cg.hex_neighbours_n(start_row-n_helmet_glass-7, start_col, n=n_helmet_metal , keep_origin = True)
helmet_metal_bot_row = max(r for r, c in helmet_metal)

_, _, _,_, metal_region, vertex = cgd.draw_trapezoid(helmet_metal_bot_row, 
                                                     min(c for r,c in helmet_metal if r == helmet_metal_bot_row), 
                                                     max(c for r,c in helmet_metal if r == helmet_metal_bot_row), 
                                                     helmet_metal_bot_row+row_seg1,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'rl')

_, _, _,_, metal_region2, vertex = cgd.draw_trapezoid(helmet_metal_bot_row+row_seg1, 
                                                     min(c for r,c in metal_region if r == helmet_metal_bot_row+row_seg1), 
                                                     max(c for r,c in metal_region if r == helmet_metal_bot_row+row_seg1), 
                                                     helmet_metal_bot_row+row_seg1+row_seg2,
                                                      slope_left = '1.5', slope_right = '1.5', direction = 'rl')

select_helmet_metal_regions = cg.select_mask(helmet_metal+metal_region+metal_region2,  hex_rc_arr)

neck = cgd.draw_block((helmet_metal_bot_row+row_seg1, (min(c for r,c in metal_region if r == helmet_metal_bot_row+row_seg1)+3,
                                                       max(c for r,c in metal_region if r == helmet_metal_bot_row+row_seg1)-3)), 37)
select_neck_regions = cg.select_mask(neck,  hex_rc_arr)

collar_row = 32
_, _, _,_, collar_region1, vertex = cgd.draw_trapezoid(collar_row, 
                                                     min(c for r,c in neck if r == collar_row)-1, 
                                                     max(c for r,c in neck if r == collar_row)+1, 
                                                     collar_row+3,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr')
collar_region2 = cgd.draw_block((collar_row+3, (min(c for r,c in collar_region1 if r == collar_row+3),
                                                max(c for r,c in collar_region1 if r == collar_row+3))
                      ), 37)
collar_region_back = collar_region1+collar_region2
select_collar_region_back = cg.select_mask(collar_region_back,  hex_rc_arr)

collar_front = cgd.draw_block((collar_row+4, (min(c for r,c in neck if r == collar_row+4)-1,
                                                max(c for r,c in neck if r == collar_row+4)+1)
                      ), 37)
select_collar_region_front = cg.select_mask(collar_front,  hex_rc_arr)

#geass marker 
marker = cg.hex_neighbours_n(start_row, start_col, n=n , keep_origin = False, return_frontier=True)[-1]+cg.hex_neighbours_n(start_row, start_col, n=n+1, keep_origin = False, return_frontier=True)[-1]

marker_top_row = min(r for r, c in marker)
marker_bot_row = max(r for r, c in marker)
valid_marker = [(r,c) for r,c in marker if r > marker_top_row+n-1]

end_row = marker_top_row-3
marker_top_row = min(r for r, c in valid_marker)
marker_L_vertex = (marker_top_row, min(c for r,c in valid_marker if r == marker_top_row)+1)
marker_R_vertex = (marker_top_row, max(c for r,c in valid_marker if r == marker_top_row)-1)
                      
L_lines = cgd.draw_slope_1p5_diagonal(marker_L_vertex[0], marker_L_vertex[1], end_row, 
                                      left_down=False, right_down=False, left_up=True, right_up=False
                                     )+cgd.draw_slope_1p5_diagonal(marker_L_vertex[0], marker_L_vertex[1]-1, end_row+1, 
                                      left_down=False, right_down=False, left_up=True, right_up=False
                                     )+cgd.draw_slope_1p5_diagonal(marker_L_vertex[0], marker_L_vertex[1]-2, end_row+2, 
                                      left_down=False, right_down=False, left_up=True, right_up=False
                                     )+cgd.draw_slope_0p5_diagonal(marker_L_vertex[0], marker_L_vertex[1], marker_bot_row, 
                                      left_down=True , right_down= False, left_up=False, right_up=False)

R_lines = cgd.draw_slope_1p5_diagonal(marker_R_vertex[0], marker_R_vertex[1], end_row, 
                                      left_down=False, right_down=False, left_up=False , right_up=True
                                     )+cgd.draw_slope_1p5_diagonal(marker_R_vertex[0], marker_R_vertex[1]+1, end_row+1, 
                                      left_down=False, right_down=False, left_up=False , right_up=True
                                     )+cgd.draw_slope_1p5_diagonal(marker_R_vertex[0], marker_R_vertex[1]+2, end_row+2, 
                                      left_down=False, right_down=False, left_up=False , right_up=True
                                     )+cgd.draw_slope_0p5_diagonal(marker_R_vertex[0], marker_R_vertex[1], marker_bot_row, 
                                      left_down=False, right_down=True , left_up=False, right_up=False)


bot_L_vertex = (marker_bot_row, min(c for r,c in L_lines if r == marker_bot_row))
bot_R_vertex = (marker_bot_row, max(c for r,c in R_lines if r == marker_bot_row))

L_bot_lines = cgd.draw_slope_1p5_diagonal(bot_L_vertex[0], bot_L_vertex[1], marker_bot_row+2, 
                                      left_down=False, right_down=True , left_up=False, right_up=False
                                     )+cgd.draw_slope_1p5_diagonal(bot_L_vertex[0], bot_L_vertex[1]+1 , marker_bot_row+2, 
                                      left_down=False, right_down=True , left_up=False, right_up=False
                                     )
R_bot_lines = cgd.draw_slope_1p5_diagonal(bot_R_vertex[0], bot_R_vertex[1], marker_bot_row+2, 
                                      left_down=True , right_down= False, left_up=False, right_up=False
                                     )+cgd.draw_slope_1p5_diagonal(bot_R_vertex[0], bot_R_vertex[1]-1, marker_bot_row+2, 
                                      left_down=True , right_down= False, left_up=False, right_up=False
                                     )


L_tip = cgd.draw_slope_0p5_diagonal(end_row, min(c for r,c in L_lines if r == end_row), end_row-7, 
                                      left_down=False, right_down=False, left_up=True, right_up=False
                                     )
R_tip = cgd.draw_slope_0p5_diagonal(end_row, max(c for r,c in R_lines if r == end_row), end_row-7, 
                                      left_down=False, right_down=False, left_up=False , right_up=True
                                     )
select_marker_regions = cg.select_mask(L_lines+R_lines+L_tip+R_tip+L_bot_lines+R_bot_lines+[(marker_bot_row+3,start_col)],  hex_rc_arr)


hex_colors[select_collar_region_back] = cgc.select_normal_color(select_collar_region_back, 
                                                               [0.7,0.7,0.7], np.ones(3)*sigma_color) 

hex_colors= cgc.color_row_gradient(neck, cgc.hex_to_rgb("#404041"), cgc.hex_to_rgb("#000000"),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color*3, end_weight= 0.01) 
hex_colors[select_collar_region_front] =  cgc.select_normal_color(select_collar_region_front, 
                                                               [0.9,0.9,0.9], np.ones(3)*sigma_color)  

hex_colors[select_helmet_metal_regions] = [0,0,0]
hex_colors[select_marker_regions] =cgc.select_normal_color(select_marker_regions, 
                                                               cgc.hex_to_rgb("#837232"), np.ones(3)*sigma_color)   
hex_colors =cgc.color_row_gradient([(r,c) for r,c in FULL_helmet_glass_regions if c <start_col], 
                                                                cgc.hex_to_rgb("#002761"), cgc.hex_to_rgb("#000000"),
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.1) 
hex_colors =cgc.color_row_gradient([(r,c) for r,c in FULL_helmet_glass_regions if c >=start_col], 
                                                                cgc.hex_to_rgb("#e5efff"), cgc.hex_to_rgb("#000000"), # #141b41
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.001) 



hex_colors=cgc.color_row_gradient(face_region, cgc.hex_to_rgb("#81786b"), cgc.hex_to_rgb("#c5bdb2"),
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.1) 

hex_colors =cgc.color_hex_gradient(valid_right_eye_ball,  cgc.hex_to_rgb("#000000") , cgc.hex_to_rgb("#5d2673" ),
                                    hex_rc_arr, hex_colors, (right_eyeball_r, right_eyeball_c),
                                    right_eyeball_n,  sigma_color=0.01, end_weight= 0.1)#  cgc.hex_to_rgb("#d954ea")
hex_colors[select_eyesocket] =[0,0,0]

base_hex_colors = hex_colors.copy()


select_eyeball = cg.select_mask(valid_right_eye_ball,hex_rc_arr)
select_eyeball = cg.select_mask(valid_right_eye_ball,hex_rc_arr)
atoms = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
atomic_linergb_dict = dict()
atomic_linenum_dict = dict()
full_rgbs = []

for atom in atoms:
    df_air = line_rgb.get_atom_obs_wl(atom)
    df_lines = line_rgb.calc_BR(df_air)
    df_lines_rgb=line_rgb.calc_rgb(df_lines)
    sorted_df_lines_rgb = df_lines_rgb.sort_values(by='intens', ascending=False)
    sorted_rgbs = np.array([list(sorted_df_lines_rgb['r']), list(sorted_df_lines_rgb['g']), list(sorted_df_lines_rgb['b'])]).T
    
    atomic_linergb_dict[atom] = sorted_rgbs
    atomic_linenum_dict[atom] = len(sorted_rgbs)
    full_rgbs.extend(sorted_rgbs)
    
Phase0_end = 30
eye_snapshots = atomic_linenum_dict.get('He')+atomic_linenum_dict.get('Ne')
marker_snapshots = sum(atomic_linenum_dict.values())

n_snapshots = atomic_linenum_dict.get('He')+Phase0_end+marker_snapshots


hex_colors[select_marker_regions]
def update(snapshot):
    if snapshot <Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_hex_colors = base_hex_colors.copy()
        
        if snapshot <Phase0_end+eye_snapshots:
            current_snapshot = snapshot-Phase0_end
            current_hex_colors[select_eyeball] = cgc.select_normal_color(select_eyeball, 
                                                                         full_rgbs[current_snapshot], 
                                                                         np.ones(3)*sigma_color*5)  
            

            
        if snapshot>= Phase0_end+ atomic_linenum_dict.get('He') and snapshot< Phase0_end+marker_snapshots+atomic_linenum_dict.get('He'):
            current_snapshot = snapshot-(Phase0_end+ atomic_linenum_dict.get('He'))
            current_hex_colors[select_marker_regions] = cgc.select_normal_color(select_marker_regions, 
                                                                     full_rgbs[current_snapshot], np.ones(3)*sigma_color*5)  

        pc.set_facecolor(current_hex_colors)
    return (pc,)   
        
total_seconds =18.0

# FPS [1/time] = total_frames / total_seconds
FPS = (n_snapshots / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
 
ani = animation.FuncAnimation(fig, update, frames=int(n_snapshots),
                               interval=time_gap, blit=False)
 
# ── SAVE ──────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'GEASS_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved → {OUTPUT_FILE}")
#plt.savefig('cg/'+cg_name, dpi=DPI, bbox_inches='tight')
#print('saved')