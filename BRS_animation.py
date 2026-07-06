
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap
import astropy.units as u
import os

import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc


import heat_equation_analytical as heat_eqn


rng = np.random.default_rng(42)


# ── FIGURE — built exactly ONCE, shared by every phase ──────────────────
IMG_W=1280
IMG_H=720
HEX_R=22
z_order_max = 5
DPI = 100

fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info = cg.make_hex_scene(
    IMG_W=IMG_W, IMG_H=IMG_H, HEX_R=HEX_R, DPI=DPI, hex_index=False, z_order_max=z_order_max)
sigma_color = 0.01

hex_colors = cgc.color_row_gradient(hex_rc_arr, cgc.hex_to_rgb("#c0c6ce"), cgc.hex_to_rgb("#d5d0c0"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.1)    

end_row = 18
row_seg1 = 7
row_seg2 = 7
row_seg3 = 6
start_col = (3, 14)

hair_region_mid1= cgd.draw_block((0,start_col) , 0+row_seg1)
left_vertex = (row_seg1, min(c for r, c in hair_region_mid1 if r == row_seg1))
right_vertex =  (row_seg1, max(c for r, c in hair_region_mid1 if r == row_seg1))

_, _, _, _, hair_region_left, vertex_hair =  cgd.draw_trapezoid(0,  0, left_vertex[1],row_seg1+row_seg2,
                                                          slope_left = 'inf', slope_right = '0.5', direction = 'll')

_, _, _, _, hair_region_right1, vertex_hair_right1 =  cgd.draw_trapezoid(0,  right_vertex[1], 35, row_seg1,
                                                          slope_left = '0.5', slope_right = 'inf', direction = 'rr')

_, _, _, _, hair_region_mid2, vertex_hair_mid2 =  cgd.draw_trapezoid(row_seg1, left_vertex[1], right_vertex[1],row_seg1+row_seg2,
                                                          slope_left = '0.5', slope_right = 'inf', direction = 'rr')
temp_left_vertex_mid2, temp_right_vertex_mid2 = vertex_hair_mid2[-2:]

_, _, _, _, hair_region_mid3, vertex_hair_mid3 =  cgd.draw_trapezoid(row_seg1+row_seg2, 
                                                          temp_left_vertex_mid2[1], temp_right_vertex_mid2[1],row_seg1+row_seg2+row_seg3, 
                                                                 slope_left = '1.5', slope_right = '0.5', direction = 'rr')
_, _, _, _, hair_region_right2, vertex_hair_right2 =  cgd.draw_trapezoid(row_seg1,  vertex_hair_right1[-2][1], 35, row_seg1+row_seg2,
                                                                         slope_left = '1.5', slope_right = 'inf', direction = 'rr')

_, _, _, _, hair_region_right3, vertex_hair_right3 =  cgd.draw_trapezoid(row_seg1+row_seg2,  vertex_hair_right2[-2][1], 35, 
                                                                         row_seg1+row_seg2+row_seg3,
                                                                         slope_left = '8/3', slope_right = 'inf', direction = 'rr')


hair = hair_region_mid1+hair_region_mid2+hair_region_mid3+hair_region_left+hair_region_right1+hair_region_right2+hair_region_right3
select_hair = cg.select_mask(hair, hex_rc_arr)

hex_colors = cgc.color_row_gradient(hair, cgc.hex_to_rgb("#000000") , cgc.hex_to_rgb("#141414" ), hex_rc_arr, hex_colors, sort = 'col', sigma_color = 0.05, end_weight= 0.2, period=10)


#hex_colors[select_hair] = [0,0,0]

right_eyesocket_r = row_seg1+row_seg2+row_seg3-4
right_eyesocket_c = vertex_hair_mid3[-1][1]-2
right_eyesocket_regions = cgd.draw_slope_8p3_diagonal(right_eyesocket_r, 
                                          right_eyesocket_c, right_eyesocket_r-5, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                         )+cgd.draw_slope_8p3_diagonal(right_eyesocket_r, 
                                          right_eyesocket_c+1, right_eyesocket_r-5, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                                                      )+cgd.draw_slope_8p3_diagonal(right_eyesocket_r, 
                                          right_eyesocket_c+2, right_eyesocket_r-5, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                                                      )
right_eye_bounds_c2r = dict(sorted({c: max(r for r, cc in right_eyesocket_regions if cc == c) 
                                 for c in set(c for _, c in right_eyesocket_regions)}.items()))


left_eyesocket_r = row_seg1+row_seg2+row_seg3-4
left_eyesocket_c = vertex_hair_mid3[-2][1]-8

left_eyesocket_regions = cgd.draw_slope_1p5_diagonal(left_eyesocket_r, 
                                          left_eyesocket_c, left_eyesocket_r-5, 
                                          left_down=False, right_down=False, left_up=True , right_up=False
                                         )+cgd.draw_slope_1p5_diagonal(left_eyesocket_r, 
                                          left_eyesocket_c-1, left_eyesocket_r-5, 
                                          left_down=False, right_down=False, left_up=True, right_up=False
                                                                      )+cgd.draw_slope_1p5_diagonal(left_eyesocket_r, 
                                          left_eyesocket_c-2, left_eyesocket_r-5, 
                                          left_down=False, right_down=False, left_up=True, right_up=False
                                                                      )

left_eye_bounds_c2r = dict(sorted({c: max(r for r, cc in left_eyesocket_regions if cc == c) 
                                 for c in set(c for _, c in left_eyesocket_regions)}.items()))
left_eye_bounds_2_c2r = {r: c for r, c in cgd.verticle_line(0, left_eyesocket_c-2, 22)}



right_eyeball_r = right_eyesocket_r
right_eyeball_c = right_eyesocket_c+int((max(c for r,c in right_eyesocket_regions)-right_eyesocket_c)//1.8)
right_eyeball_n = 5
right_eye_ball = cg.hex_neighbours_n(right_eyeball_r, right_eyeball_c, n=right_eyeball_n, keep_origin = True, return_frontier=False)

valid_right_eye_ball = [(r,c) for r,c in right_eye_ball if r>= right_eye_bounds_c2r.get(c)]

left_eyeball_r = left_eyesocket_r
left_eyeball_c = left_eyesocket_c-int((left_eyesocket_c-max(0, min(c for r,c in left_eyesocket_regions)))//2)
print(left_eyeball_r, left_eyeball_c)
left_eyeball_n = 5
left_eye_ball = cg.hex_neighbours_n(left_eyeball_r, left_eyeball_c, n=left_eyeball_n, keep_origin = True, return_frontier=False) 

valid_left_eye_ball= [(r,c) for r,c in left_eye_ball if r>= left_eye_bounds_c2r.get(c, -1) and c <= left_eye_bounds_2_c2r.get(r, -1)]

left_eyelashes = cgd.draw_slope_0p5_diagonal(left_eyesocket_r+2, 
                                          left_eyesocket_c-1, min(r for r, c in left_eyesocket_regions)+2, 
                                          left_down=False, right_down=False, left_up= True , right_up=False
                                         )+cgd.draw_slope_0p5_diagonal(left_eyesocket_r+2, 
                                          left_eyesocket_c-2, min(r for r, c in left_eyesocket_regions)+2, 
                                          left_down=False, right_down=False, left_up= True , right_up=False
                                         )+cgd.draw_slope_0p5_diagonal(left_eyesocket_r, 
                                          left_eyesocket_c-4, min(r for r, c in left_eyesocket_regions)+1, 
                                          left_down=False, right_down=False, left_up= True , right_up=False
                                         )
right_eyelashes = cgd.draw_slope_0p5_diagonal(right_eyesocket_r+4, 
                                          right_eyesocket_c+2, min(r for r, c in right_eyesocket_regions)+2, 
                                          left_down=False, right_down=False, left_up= False , right_up=True 
                                         )+cgd.draw_slope_0p5_diagonal(right_eyesocket_r+3, 
                                          right_eyesocket_c+3, min(r for r, c in right_eyesocket_regions)+2, 
                                          left_down=False, right_down=False, left_up= False , right_up=True 
                                         )+cgd.draw_slope_0p5_diagonal(right_eyesocket_r+1, 
                                          right_eyesocket_c+5, min(r for r, c in right_eyesocket_regions)+1, 
                                          left_down=False, right_down=False, left_up= False , right_up=True 
                                         )+cgd.draw_slope_0p5_diagonal(right_eyesocket_r, 
                                          right_eyesocket_c+7, min(r for r, c in right_eyesocket_regions), 
                                          left_down=False, right_down=False, left_up= False , right_up=True 
                                         )
select_eyelash_regions = cg.select_mask(left_eyelashes+right_eyelashes, hex_rc_arr)


select_eyesocket_regions = cg.select_mask(right_eyesocket_regions+left_eyesocket_regions, hex_rc_arr)

select_eyeball_regions = cg.select_mask(valid_right_eye_ball+valid_left_eye_ball, hex_rc_arr)

left_eyelid_r = left_eyesocket_r- 4
left_eyelid_regions = cgd.draw_slope_1p5_diagonal(left_eyelid_r, 
                                          left_eyesocket_c, left_eyelid_r-5, 
                                          left_down=False, right_down=False, left_up=True , right_up=False
                                         )+cgd.draw_slope_1p5_diagonal(left_eyelid_r, 
                                          left_eyesocket_c-1, left_eyelid_r-5, 
                                          left_down=False, right_down=False, left_up=True, right_up=False
                                                                      )
right_eyelid_r = right_eyesocket_r- 4
right_eyelid_regions = cgd.draw_slope_8p3_diagonal(right_eyelid_r, 
                                          right_eyesocket_c, right_eyelid_r-5, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                         )+cgd.draw_slope_8p3_diagonal(right_eyelid_r, 
                                          right_eyesocket_c+1, right_eyelid_r-5, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                                                      )+cgd.draw_slope_8p3_diagonal(right_eyelid_r, 
                                          right_eyesocket_c+2, right_eyelid_r-5, 
                                          left_down=False, right_down=False, left_up=False, right_up=True
                                                                      )
select_eyelid_regions = cg.select_mask(left_eyelid_regions+right_eyelid_regions, hex_rc_arr)

#eyeball color 
hex_colors = cgc.color_hex_gradient(valid_right_eye_ball,  cgc.hex_to_rgb("#0500ea") , cgc.hex_to_rgb("#b5d4d8" ),
                                    hex_rc_arr, hex_colors, (right_eyeball_r, right_eyeball_c),
                                    right_eyeball_n,  sigma_color=0.05, end_weight= 0.1)

hex_colors = cgc.color_hex_gradient(valid_left_eye_ball,  cgc.hex_to_rgb("#0500ea") , cgc.hex_to_rgb("#b5d4d8" ),
                                    hex_rc_arr, hex_colors, (left_eyeball_r, left_eyeball_c),
                                    left_eyeball_n,  sigma_color=0.05, end_weight= 0.1)

hex_colors[select_eyesocket_regions] = cgc.select_normal_color(select_eyesocket_regions, 
                                                               cgc.hex_to_rgb("#151b21"), np.ones(3)*sigma_color) 

hex_colors[select_eyelash_regions] = cgc.select_normal_color(select_eyelash_regions, 
                                                               cgc.hex_to_rgb("#021021"), np.ones(3)*sigma_color) 
hex_colors[select_eyelid_regions] = cgc.select_normal_color(select_eyelid_regions, cgc.hex_to_rgb("#3e4348"), np.ones(3)*sigma_color) 



base_hex_colors = hex_colors.copy()

hex_alphas = np.zeros(len(hex_rc_arr))
flame_cmap = LinearSegmentedColormap.from_list('fire_field',[
    (0.00, '#ffffff'),   # ignition
    (0.3, '#fdfeff'),
    (0.5, '#f5fdff'),
    (0.8, '#ebfffe'),   
    (0.9, '#d9fff8'),
    (0.95, '#a5ffee'),
    (0.98, '#38ffff'),
    (1.00, '#00e8ff'),  
])
norm = Normalize(vmin=0, vmax=1)


FLAME_ALPHA = 0.5
def burn_state_to_hex_colors(T_flat):
    for pos, idxs in hex_pos_to_grid_idx.items():
        val = T_flat[idxs].mean()
        a = np.clip(norm(val), 0.0, 1.0)
        flame_color = np.array(flame_cmap(a)[:3])
        hex_colors[pos] = (1 - FLAME_ALPHA) * base_hex_colors[pos] + FLAME_ALPHA * flame_color
        
        

T_MAX = 20.0
animation_snapshots = 100
times = np.linspace(0, T_MAX, animation_snapshots)



#analytical solution
frames = []
for i, t in enumerate(times):
    print(f"Solving snapshot {i+1}/{animation_snapshots}  (t={t:.2f}) ...")
    T = heat_eqn.T_full(t)
    
    #T_full(t)          # (NROW, NCOL) array, this ONE snapshot
    frames.append(T.ravel())  # flatten to match burn_state_to_hex_colors' expected shape

frames = np.stack(frames)     # shape: (n_snapshots, NROW*NCOL)
rc_to_idx   = {rc: i for i, rc in enumerate(hex_rc_arr)}

#flame in the right eye region
DOMAIN_W = 8
DOMAIN_H = 4
CANVAS_PHYSICAL_W, CANVAS_PHYSICAL_H = DOMAIN_W*0.8, DOMAIN_H+1 # bigger relative to DOMAIN_W/H = smaller flame; smaller = bigger flame
OFFSET_X = (CANVAS_PHYSICAL_W - DOMAIN_W) / 5      # keep within [0, CANVAS_PHYSICAL_W - DOMAIN_W]
OFFSET_Y = (CANVAS_PHYSICAL_H - DOMAIN_H) / 2.5 
print(OFFSET_X, OFFSET_Y)
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range  =(0, CANVAS_PHYSICAL_H)

grid_hex_rc = cg.world_metres_to_hex_index(heat_eqn.X.ravel() + OFFSET_X, heat_eqn.Y.ravel() + OFFSET_Y, 
                                           detail_info,
                                           canvas_physical_x_range=canvas_physical_x_range, 
                                           canvas_physical_y_range=canvas_physical_y_range)
hex_to_grid_idx = {}
for flat_i, rc in enumerate(grid_hex_rc):
    hex_to_grid_idx.setdefault(rc, []).append(flat_i)
hex_pos_to_grid_idx = {
    rc_to_idx[rc]: idxs
    for rc, idxs in hex_to_grid_idx.items()
    if rc in rc_to_idx
}

flame_hex_positions = list(hex_pos_to_grid_idx.keys())
Phase0_end = 10
n_snapshots = animation_snapshots+Phase0_end 
# ── UPDATE ──────────────────────────────────────────────────────────────
def update(snapshot):
    if snapshot <Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot-Phase0_end
        current_hex_colors = base_hex_colors.copy()   # fresh reset from base every frame
        T_flat = frames[current_snapshot]
        for pos in flame_hex_positions:
            idxs = hex_pos_to_grid_idx[pos]
            val = T_flat[idxs].mean()
            a = np.clip(norm(val), 0.0, 1.0)
            flame_color = np.array(flame_cmap(a)[:3])
            blend_weight = FLAME_ALPHA * a   # 0 at T=0 (fully transparent, pure base),
                                               # rising toward FLAME_ALPHA only as it heats up
            current_hex_colors[pos] = (1 - blend_weight) * base_hex_colors[pos] + blend_weight * flame_color
        pc.set_facecolor(current_hex_colors)
    

    return (pc,)

total_seconds=18.0
    
# FPS [1/time] = total_frames / total_seconds
FPS = (n_snapshots / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=int(n_snapshots),
                               interval=time_gap, blit=False)

# ── SAVE ──────────────────────────────────────────────────────────────────
FILE_NAME  = "BRS_animation.mp4"
OUTPUT_FOLDER = 'RESULT'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, FILE_NAME)



print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved → {OUTPUT_FILE}")


