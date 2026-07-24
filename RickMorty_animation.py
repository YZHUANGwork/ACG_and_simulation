import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection

from matplotlib.colors import LinearSegmentedColormap, Normalize
import astropy.units as u
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import expected_value as EXP
import fluid_eqn_solve as FLUID
from scipy.integrate import solve_ivp
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,detail_info = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)



sigma_color = 0.01

skyline = 10
select_skyline = [r == skyline for r,c in hex_rc_arr ]

ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
wall = [(r,c) for r,c in hex_rc_arr if r <skyline]
outside = cgd.draw_block((0,(2, 32)) , skyline)
hex_colors = cgc.color_row_gradient(wall, cgc.hex_to_rgb("#020c20"), cgc.hex_to_rgb("#747a89"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'col', sigma_color = sigma_color, end_weight= 0.1, period=5)    

hex_colors = cgc.color_row_gradient(ground, cgc.hex_to_rgb("#474746"), cgc.hex_to_rgb("#8d8d8b"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.05)    

head_row = 17
center_col = 17
n_head = 2

n_pickel = 2
pickel_start = 4
picke_end = 16

pickel = cg.hex_neighbours_n(pickel_start,center_col-3, n=n_pickel , keep_origin = True)+cgd.draw_block((pickel_start,(center_col-3-n_pickel, center_col-3+n_pickel)) , picke_end)+cg.hex_neighbours_n(picke_end,center_col-3, n=n_pickel , keep_origin = True)

pickel_front = cg.hex_neighbours_n(pickel_start,center_col-3, n=n_pickel-1 , keep_origin = True)+cgd.draw_block((pickel_start,(center_col-3-n_pickel+1, center_col-3+n_pickel-1)) , picke_end)+cg.hex_neighbours_n(picke_end,center_col-3, n=n_pickel-1 , keep_origin = True)

pickel_eye = [(pickel_start+n_pickel, min(c for r,c in pickel if r == pickel_start+n_pickel)+1),
              (pickel_start+n_pickel, max(c for r,c in pickel if r == pickel_start+n_pickel)-1)]
pickel_eyebrow = cgd.horizontal_lines([(pickel_start, (min(c for r,c in pickel if r == pickel_start)+1, 
                                                       max(c for r,c in pickel if r == pickel_start)-1))])

_, _, _,_, pickel_mouth, vertex = cgd.draw_trapezoid(pickel_start+n_pickel+2, 
                                                     min(c for r,c in pickel if r == pickel_start+n_pickel+2)+1, 
                                                     max(c for r,c in pickel if r == pickel_start+n_pickel+2)-1, 
                                                     pickel_start+n_pickel+3,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'rl')

pickel_tongue = [(max(r for r,c in pickel_mouth ),max(c for r,c in pickel_mouth if r == max(r for r,c in pickel_mouth )) ) ]

select_pickel = cg.select_mask(pickel  ,hex_rc_arr)
hex_colors[select_pickel] = cgc.select_normal_color(select_pickel,cgc.hex_to_rgb("#3b8d0e"), np.ones(3)*sigma_color*3) 

select_pickel_front = cg.select_mask(pickel_front  ,hex_rc_arr)
hex_colors[select_pickel_front] = cgc.select_normal_color(select_pickel_front,cgc.hex_to_rgb("#88d608"), np.ones(3)*sigma_color*3) 


select_pickel_eye = cg.select_mask(pickel_eye  ,hex_rc_arr)
hex_colors[select_pickel_eye] = cgc.select_normal_color(select_pickel_eye,[1,1,1], np.ones(3)*sigma_color) 

select_pickel_eyebrow = cg.select_mask(pickel_eyebrow  ,hex_rc_arr)
hex_colors[select_pickel_eyebrow] = cgc.select_normal_color(select_pickel_eyebrow, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color) 

select_pickel_mouth = cg.select_mask(pickel_mouth  ,hex_rc_arr)
hex_colors[select_pickel_mouth] = cgc.select_normal_color(select_pickel_mouth,cgc.hex_to_rgb("#42151c"), np.ones(3)*sigma_color) 

select_pickel_tongue = cg.select_mask(pickel_tongue  ,hex_rc_arr)
hex_colors[select_pickel_tongue] = cgc.select_normal_color(select_pickel_tongue,cgc.hex_to_rgb("#d44545"), np.ones(3)*sigma_color) 

n_leg = 2
leg_start = 5
leg_end = 20
leg_col = center_col+6


leg_head = cg.hex_neighbours_n(leg_start,leg_col, n=n_leg , keep_origin = True)
leg_hair = cgd.draw_triangle(leg_start-n_leg-n_leg, leg_col,  leg_start, 
                             slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                            )[-2]+cgd.draw_triangle(leg_start+n_leg, max(c for r,c in leg_head if r == leg_start+n_leg),  leg_start-n_leg, 
                             slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                                   )[-2]+cgd.draw_triangle(leg_start-n_leg, 
                                                                           max(c for r,c in leg_head if r == leg_start-n_leg),  
                                                                           leg_start+n_leg, 
                             slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                                   )[-2]
select_leg_hair = cg.select_mask(leg_hair ,hex_rc_arr)
hex_colors[select_leg_hair] = cgc.select_normal_color(select_leg_hair, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color*3) 


knee_r = leg_start+8
knee = cg.hex_neighbours_n(knee_r,leg_col+1, n=n_leg-1 , keep_origin = True)
thigh = cgd.draw_slope_0p5_diagonal(leg_start, min(c for r,c in leg_head if r == leg_start), knee_r,
                                   left_down=False, right_down=True, left_up=False, right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(leg_start, min(c for r,c in leg_head if r == leg_start)+1, leg_start+6,
                                   left_down=False, right_down=True, left_up=False, right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(knee_r, leg_col+1, leg_start+2,
                                   left_down=False, right_down=False, left_up=True , right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(leg_start, leg_col, leg_start+3,
                                   left_down=False, right_down=True , left_up= False, right_up=False
                                  )
calf = cgd.draw_slope_0p5_diagonal(knee_r, min(c for r,c in knee if r == knee_r), knee_r+5,
                                   left_down=False, right_down=True, left_up=False, right_up=False
                                  )+cgd.draw_slope_0p5_diagonal(knee_r, min(c for r,c in knee if r == knee_r)+1, knee_r+3,
                                   left_down=False, right_down=True, left_up=False, right_up=False)
ankle_r = max(r for r, c in calf)
foot = cgd.draw_slope_0p5_diagonal(ankle_r, min(c for r,c in calf if r == ankle_r), ankle_r+2,
                                   left_down=True , right_down=False, left_up=False, right_up=False
                                  )
leg = list(set(leg_head+knee +thigh+calf+foot))
select_leg = cg.select_mask(leg,hex_rc_arr)
hex_colors[select_leg] = cgc.select_normal_color(select_leg,cgc.hex_to_rgb("#c6c0b5"), np.ones(3)*sigma_color*3) 

leg_eye = [(leg_start+1, min(c for r,c in leg_head if r == leg_start+1)),
              (leg_start+1, max(c for r,c in leg_head if r == leg_start+1)-1)]
leg_eyebrow = cgd.horizontal_lines([(leg_start-1, (min(c for r,c in leg_head if r == leg_start-1), 
                                                       max(c for r,c in leg_head if r == leg_start-1)-1))])




select_leg_eye = cg.select_mask(leg_eye  ,hex_rc_arr)
hex_colors[select_leg_eye] = cgc.select_normal_color(select_leg_eye,[1,1,1], np.ones(3)*sigma_color) 

select_leg_eyebrow = cg.select_mask(leg_eyebrow  ,hex_rc_arr)
hex_colors[select_leg_eyebrow] = cgc.select_normal_color(select_leg_eyebrow, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color) 


foreground_points = []
Rick_head_loc = (head_row-3, center_col-8)
Rick_hair =  cgd.draw_triangle(Rick_head_loc[0]-4, Rick_head_loc[1],  head_row, 
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                               )[-2]+cgd.draw_triangle(Rick_head_loc[0]+6, Rick_head_loc[1],  head_row-4, 
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'rl'
                                                                      )[-2]+cgd.draw_triangle(Rick_head_loc[0], Rick_head_loc[1],  
                                                                                              head_row+2, 
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr'
                                                                      )[-2]

Rick_head = cg.hex_neighbours_n(Rick_head_loc[0], Rick_head_loc[1], n=1 , keep_origin = True)
Rick_neck_r = max(r for r,c in Rick_hair)
Rick_neck = cgd.horizontal_lines([(Rick_neck_r, (Rick_head_loc[1]-1, Rick_head_loc[1]+1))])

_, _, _,_, Rick_body, vertex = cgd.draw_trapezoid(Rick_neck_r+1, 
                                                     min(c for r,c in Rick_hair if r == Rick_neck_r-1), 
                                                     max(c for r,c in Rick_hair if r == Rick_neck_r-1), 
                                                     22,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr')


foreground_points.extend(Rick_hair)
foreground_points.extend(Rick_head)
foreground_points.extend(Rick_neck)
foreground_points.extend(Rick_body)

Morty_head = cg.hex_neighbours_n(head_row, center_col, n=n_head , keep_origin = True)


Morty_neck = [(r,c) for r,c in Morty_head if r == head_row+n_head]

_, _, _,_, Morty_body, vertex = cgd.draw_trapezoid(head_row+n_head, 
                                                     min(c for r,c in Morty_head if r == head_row+n_head), 
                                                     max(c for r,c in Morty_head if r == head_row+n_head), 
                                                     22,
                                                      slope_left = '0.5', slope_right = '0.5', direction = 'lr')


foreground_points.extend(Morty_head)
foreground_points.extend(Morty_body)
foreground_mask = cg.select_mask(list(set(foreground_points)), hex_rc_arr)


select_Morty_body = cg.select_mask(Morty_body  ,hex_rc_arr)
hex_colors[select_Morty_body] = cgc.select_normal_color(select_Morty_body,cgc.hex_to_rgb("#fffd3b"), np.ones(3)*sigma_color) 


select_Morty_head = cg.select_mask(Morty_head  ,hex_rc_arr)
#hex_colors[select_Morty_head] = cgc.select_normal_color(select_Morty_head, cgc.hex_to_rgb("#693f1d"), np.ones(3)*sigma_color) 
hex_colors = cgc.color_hex_gradient(Morty_head,  cgc.hex_to_rgb("#9f6230") , cgc.hex_to_rgb("#9f4830" ),
                                    hex_rc_arr, hex_colors, (head_row, center_col),
                                    n_head,  sigma_color=sigma_color, end_weight= 0.01)

select_Rick_neck = cg.select_mask(Rick_neck  ,hex_rc_arr)
hex_colors[select_Rick_neck] = cgc.select_normal_color(select_Rick_neck, cgc.hex_to_rgb("#c6c0b5"), np.ones(3)*sigma_color) 

select_Rick_hair = cg.select_mask(Rick_hair ,hex_rc_arr)
hex_colors[select_Rick_hair] = cgc.select_normal_color(select_Rick_hair, cgc.hex_to_rgb("#a2e8fa"), np.ones(3)*sigma_color*3) 

select_Rick_head = cg.select_mask(Rick_head  ,hex_rc_arr)
hex_colors[select_Rick_head] = cgc.select_normal_color(select_Rick_head, cgc.hex_to_rgb("#c6c0b5"), np.ones(3)*sigma_color) 


select_Rick_body = cg.select_mask(Rick_body  ,hex_rc_arr)
hex_colors[select_Rick_body] = cgc.select_normal_color(select_Rick_body,[0.95, 0.95, 0.95], np.ones(3)*sigma_color) 
base_hex_colors = hex_colors.copy()


SPIRAL_WINDS = 2.5   # how many winds across the radius -- tune this for tighter/looser coil

# solve the full physics -- default t_end/n_frames now, since the animation
# needs the whole swirl evolution, not just the initial coil
solution = FLUID.solve_fluid_flow()
ncol, nrow = solution.r.shape
x, y, X, Y, dx, dy = FLUID._build_grid(FLUID.DOMAIN_W_DEFAULT, FLUID.DOMAIN_H_DEFAULT, ncol, nrow)
r = solution.r
domain_r = solution.domain_r
cx, cy = FLUID.DOMAIN_W_DEFAULT / 2, FLUID.DOMAIN_H_DEFAULT / 2

# the mosquito-coil initial dye pattern
theta = np.arctan2(Y - cy, X - cx)
spiral_phase = theta - SPIRAL_WINDS * (r / domain_r) * 2 * np.pi
dye0 = np.where(r < domain_r, np.where(np.sin(spiral_phase) > 0, 1.0, 0.0), 0.15)

# dye is a passive tracer, not part of the physics solution -- it rides the
# already-solved velocity field, has zero effect back on it
KAPPA_DYE = 0.01
KX_dye, KY_dye, K2_dye, K2_inv_dye = FLUID._wavenumbers(
    FLUID.DOMAIN_W_DEFAULT, FLUID.DOMAIN_H_DEFAULT, ncol, nrow)

def _ddx_dye(fh): return np.real(np.fft.ifft2(1j * KX_dye * fh))
def _ddy_dye(fh): return np.real(np.fft.ifft2(1j * KY_dye * fh))
def _lap_dye(fh): return np.real(np.fft.ifft2(-K2_dye * fh))

def _dye_rhs(t, dye_flat):
    dye = dye_flat.reshape(ncol, nrow)
    dye_hat = np.fft.fft2(dye)
    vx, vy = solution.velocity(t)   # reads the ALREADY-SOLVED physics, doesn't affect it
    ddye = -(vx * _ddx_dye(dye_hat) + vy * _ddy_dye(dye_hat)) + KAPPA_DYE * _lap_dye(dye_hat)
    return ddye.flatten()

t_eval = solution._t_eval
dye_frames = []
_dye_state = dye0.flatten()
_t_prev = 0.0
for _t_now in t_eval:
    if _t_now > _t_prev:
        _dye_sol = solve_ivp(_dye_rhs, (_t_prev, _t_now), _dye_state, method="RK45",
                              rtol=1e-2, atol=1e-4, dense_output=False)
        _dye_state = _dye_sol.y[:, -1]
        _dye_2d = _dye_state.reshape(ncol, nrow)
        _dye_2d = np.where(r < domain_r, _dye_2d, dye0)   # hard clamp outside to rest value
        _dye_state = _dye_2d.flatten()
    dye_frames.append(_dye_state.reshape(ncol, nrow).copy())
    _t_prev = _t_now
 
# canvas sizing: must match IMG_W:IMG_H ratio or the circle stretches into an oval
DOMAIN_W, DOMAIN_H = FLUID.DOMAIN_W_DEFAULT, FLUID.DOMAIN_H_DEFAULT
IMG_W_SCENE, IMG_H_SCENE = detail_info[0], detail_info[1]
CANVAS_PHYSICAL_H = DOMAIN_H * 0.9
CANVAS_PHYSICAL_W = CANVAS_PHYSICAL_H * (IMG_W_SCENE / IMG_H_SCENE)
OFFSET_X = (CANVAS_PHYSICAL_W - DOMAIN_W) / 2
OFFSET_Y = (CANVAS_PHYSICAL_H - DOMAIN_H) / 2
canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)
 
# only map grid points actually inside the circular domain -- no square background
inside_circle_flat = (r.ravel() < domain_r)
X_circle = X.ravel()[inside_circle_flat]
Y_circle = Y.ravel()[inside_circle_flat]
circle_flat_indices = np.nonzero(inside_circle_flat)[0]
 
grid_hex_rc = cg.world_metres_to_hex_index(
    X_circle + OFFSET_X, Y_circle + OFFSET_Y, detail_info,
    canvas_physical_x_range=canvas_physical_x_range,
    canvas_physical_y_range=canvas_physical_y_range,
)
rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}
hex_to_grid_idx = {}
for circle_i, rc in enumerate(grid_hex_rc):
    flat_i = circle_flat_indices[circle_i]
    hex_to_grid_idx.setdefault(rc, []).append(flat_i)
hex_pos_to_grid_idx = {
    rc_to_idx[rc]: idxs
    for rc, idxs in hex_to_grid_idx.items()
    if rc in rc_to_idx
}
 
# plain flat background, only the coil gets colored -- nothing else in this plot
#base_hex_colors = np.full((len(patches), 3), 0.92)
portal_cmap = LinearSegmentedColormap.from_list('portal_field', [(0.0, '#e3ff00'), (1.0, '#00b300')])
norm = Normalize(vmin=0, vmax=1)

 
Phase0_end = 10          # frames before the portal appears (doorway still just red)
n_snapshots = len(t_eval) + Phase0_end
 
 
# ── UPDATE ──────────────────────────────────────────────────────────────
def update(snapshot):
    if snapshot < Phase0_end:
        current_hex_colors = base_hex_colors.copy()
        pc.set_facecolor(current_hex_colors)
    else:
        current_snapshot = snapshot - Phase0_end
        current_hex_colors = base_hex_colors.copy()   # fresh reset from base every frame
        dye_flat = dye_frames[current_snapshot].ravel()
        for pos, idxs in hex_pos_to_grid_idx.items():
            val = dye_flat[idxs].mean()
            a = np.clip(norm(val), 0.0, 1.0)
            current_hex_colors[pos] = np.array(portal_cmap(a)[:3])  
        current_hex_colors[foreground_mask] = base_hex_colors[foreground_mask]  # keep silhouette on top
        
        pc.set_facecolor(current_hex_colors)
    return (pc,)



total_seconds=18.0
    
# FPS [1/time] = total_frames / total_seconds
FPS = (n_snapshots / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=int(n_snapshots),
                               interval=time_gap, blit=False)

# ── SAVE ──────────────────────────────────────────────────────────────────
FILE_NAME  = "RickMorty_animation.mp4"
OUTPUT_FOLDER = 'RESULT'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, FILE_NAME)



print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved → {OUTPUT_FILE}")