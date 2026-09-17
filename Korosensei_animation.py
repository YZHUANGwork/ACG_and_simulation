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
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
import os
import cg_plot_fn as cg
import cg_draw_fn as cgd
import cg_color_fn as cgc
import cg_draw_figure as cgf

import SOLUTION_Schrodinger as SOLN
import Korosensei_scene_plot as K_SCENE
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


n_head = 3
head_center_row = 8
head_center_col = max(c for r,c in hex_rc_arr)//5


skyline = max(r for r,c in hex_rc_arr)//2
ground = [(r,c) for r,c in hex_rc_arr if r >=skyline]
wall = [(r,c) for r,c in hex_rc_arr if r <skyline]
#BLACKBOARD = cgd.draw_block((head_center_row-8,  (min(c for r,c in hex_rc_arr)+3,max(c for r,c in hex_rc_arr)-3 
#                                                 )
#                            ), head_center_row+20)

hex_colors = cgc.color_row_gradient(wall, cgc.hex_to_rgb("#f4f4f4"), cgc.hex_to_rgb("#d6d6d6"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.05, mode = 'linear')   
#hex_colors = cgc.color_row_gradient(BLACKBOARD, cgc.hex_to_rgb("#262b28"), cgc.hex_to_rgb("#444444"), # #cccac4
#                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.05, mode = 'linear')   



hex_colors =cgc.color_row_gradient(ground, cgc.hex_to_rgb("#d6d6d6"), cgc.hex_to_rgb("##343b37"), # #cccac4
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color, end_weight= 0.05, mode = 'linear')     


bkgd_base_hex_colors = hex_colors.copy()


BLOCK_COLOR = cgc.hex_to_rgb("#262b28")
CURVE_COLOR = cgc.hex_to_rgb("#ffd647")
dtheta = 6.0   # degrees swept per snapshot -- controls both speed and total frame count
BLOCK_RISE_SNAPSHOTS = 3


def _cell_angle_deg(cell, piv_x, piv_y):
    """Angle of a hex cell relative to a pivot, standard math convention
    (0=positive x, 90=straight up, counterclockwise positive). Pixel-y
    increases downward, so it's flipped here to get that convention."""
    r, c = cell
    cx, cy = cg.hex_center_pixel(r, c, detail_info)
    dx = cx - piv_x
    dy_math = -(cy - piv_y)
    return math.degrees(math.atan2(dy_math, dx))


def _clockwise_distance_from_zero(theta):
    """Degrees traveled going CLOCKWISE from positive-x (0 deg) to reach
    this angle -- the LONG way around for angles in the upper-left (Q2),
    short way for angles already below the x-axis (Q3/Q4)."""
    return -theta if theta <= 0 else (360.0 - theta)


def build_barrier_animation(x_nm, E, A, L_nm, V0, frame_offset):
    """Build everything needed for one barrier's full animation sequence
    (slide-in, block-rise, curve-reveal, sweep, travel) -- exactly the
    same structure/choreography as before, just parameterized by the
    barrier's own solution so it can be repeated with different barriers
    for a looping effect. `frame_offset` shifts this barrier's phase
    boundaries so multiple barriers' animations play back to back.
    """
    #------ solution PARAMS -----------------------
    sol_barrier = SOLN.FiniteBarrier(
        m=const.m_e * const.c ** 2,
        x=x_nm* u.nm,
        t=0 * u.fs,
        E=E * u.eV, A=A, L=L_nm * u.nm, V0=V0 * u.eV,
    )
    phi_barrier, _A, _B, _C, _D, _F = sol_barrier.solve(t=0 * u.fs)
    x_barrier = sol_barrier.x.to_value(u.nm)
    Lv_barrier = sol_barrier.L.to_value(u.nm)
    L = Lv_barrier

    dict_, tentacle_start_pts = K_SCENE.draw_Korosensei(head_center_row, head_center_col, n_head, view='side')
    grid_hex_rc_dict = dict()
    tentacle_start_pt = None
    col_x0 = col_xL = None

    for key, tentacle_start_pt in tentacle_start_pts.items():
        row, col = tentacle_start_pt
        if col < head_center_col:
            X = x_barrier[::-1]
            Y = phi_barrier.real
            domain_mid = x_barrier.min() + x_barrier.max()
            x0_ref, xL_ref = domain_mid - 0.0, domain_mid - L
        else:
            X = x_barrier
            Y = phi_barrier.real
            x0_ref, xL_ref = 0.0, L

        DOMAIN_W = X.max() - X.min()
        DOMAIN_H = Y.max() - Y.min()
        CANVAS_PHYSICAL_H = DOMAIN_H * 2.5
        CANVAS_PHYSICAL_W = DOMAIN_W * 1.2
        canvas_physical_x_range = (0, CANVAS_PHYSICAL_W)
        canvas_physical_y_range = (0, CANVAS_PHYSICAL_H)

        px0, py0 = cg.hex_center_pixel(row, col, detail_info)
        OFFSET_X = px0 * CANVAS_PHYSICAL_W / IMG_W_SCENE - X[0]
        OFFSET_Y = CANVAS_PHYSICAL_H * (1.0 - py0 / IMG_H_SCENE) - Y[0]

        if col < head_center_col:
            grid_hex_rc = cg.world_metres_to_hex_index(
                X[:20] + OFFSET_X, Y[:20] + OFFSET_Y, detail_info,
                canvas_physical_x_range=canvas_physical_x_range,
                canvas_physical_y_range=canvas_physical_y_range,
            )
        else:
            grid_hex_rc = cg.world_metres_to_hex_index(
                X + OFFSET_X, Y + OFFSET_Y, detail_info,
                canvas_physical_x_range=canvas_physical_x_range,
                canvas_physical_y_range=canvas_physical_y_range,
            )
        curve_cells = list(dict.fromkeys(
            (r, c) for r, c in grid_hex_rc if (r, c) in hex_rc_arr))
        grid_hex_rc_dict[key] = [curve_cells, dict_["head"][-1]]
        if 'LEFT' in key:
            dict_[key] = [curve_cells, dict_["head"][-1]]
        else:
            col_x0 = list(set(cg.world_metres_to_hex_index(
                np.array([x0_ref]) + OFFSET_X, np.array([0.0]),
                detail_info,
                canvas_physical_x_range=(0, CANVAS_PHYSICAL_W),
                canvas_physical_y_range=(0, 1.0))))[0][1]
            col_xL = list(set(cg.world_metres_to_hex_index(
                np.array([xL_ref]) + OFFSET_X, np.array([0.0]),
                detail_info,
                canvas_physical_x_range=(0, CANVAS_PHYSICAL_W),
                canvas_physical_y_range=(0, 1.0))))[0][1]

    barrier = cgd.draw_block((max(r for r, c in hex_rc_arr), (col_x0, col_xL)), 0)

    #---------------animation_base_hex_colors_phase1----------------------------
    dict_rawbody = dict_.copy()

    max_char_c = -1
    local_hex_colors = bkgd_base_hex_colors.copy()
    for key in dict_.keys():
        part = dict_.get(key)[0]
        max_char_c = max(max_char_c, max(c for r, c in part))
        colors = dict_.get(key)[1]
        if len(colors) == 1:
            select_part = cg.select_mask(part, hex_rc_arr)
            local_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)
        else:
            local_hex_colors = cgc.color_row_gradient(part,
                                        colors[0], colors[1],
                                        hex_rc_arr, local_hex_colors, sort='row', sigma_color=sigma_color,
                                        end_weight=0.01, mode='linear')
    animation_base_hex_colors = local_hex_colors.copy()

    final_lines = cgd.draw_slope_0p5_diagonal(tentacle_start_pt[0], tentacle_start_pt[1], 0,
                                        left_down=False, right_down=False, left_up=True, right_up=False)

    Phase0_end = 10
    Phase1_end = Phase0_end + max_char_c
    Phase2_end = Phase1_end + len(grid_hex_rc_dict['RIGHT1'][0])

    piv_x, piv_y = cg.hex_center_pixel(tentacle_start_pt[0], tentacle_start_pt[1], detail_info)

    rawbody_cells = list(dict.fromkeys(
        cell for key in dict_rawbody.keys() for cell in dict_rawbody.get(key)[0]))
    rawbody_angles = {cell: _cell_angle_deg(cell, piv_x, piv_y) for cell in rawbody_cells}

    pivot_cell = (tentacle_start_pt[0], tentacle_start_pt[1])
    protected_cells = set(final_lines)
    final_line_angles = [_cell_angle_deg(cell, piv_x, piv_y) for cell in final_lines if cell != pivot_cell]
    target_angle_upper = float(np.mean(final_line_angles)) if final_line_angles else 90.0

    upper_sweep_total = target_angle_upper
    lower_sweep_total = _clockwise_distance_from_zero(target_angle_upper)

    n_sweep_frames = int(np.ceil(max(upper_sweep_total, lower_sweep_total) / dtheta))
    Phase_sweep_end = Phase2_end + n_sweep_frames

    rawbody_cell_set = set(rawbody_cells)
    rc_to_idx = {rc: i for i, rc in enumerate(hex_rc_arr)}

    final_line_kept_cells = [cell for cell in final_lines if cell in rawbody_cell_set]

    def _cell_dist_from_pivot(cell):
        r, c = cell
        cx, cy = cg.hex_center_pixel(r, c, detail_info)
        return (cx - piv_x) ** 2 + (cy - piv_y) ** 2

    final_line_kept_cells.sort(key=_cell_dist_from_pivot)
    final_line_colors = [animation_base_hex_colors[rc_to_idx[cell]] for cell in final_line_kept_cells]

    len_final = len(final_line_kept_cells)
    if len_final == 0:
        print('WARNING: no final_lines cells belonged to dict_rawbody -- falling back to a single default-colored cell')
        final_line_kept_cells = [final_lines[0]]
        final_line_colors = [np.array([1.0, 1.0, 1.0])]
        len_final = 1

    combined_cells = list(reversed(final_line_kept_cells)) + curve_cells
    n_combined = len(combined_cells)
    travel_positions = n_combined + 1

    total_frames_local = Phase_sweep_end + travel_positions

    top_row = min(r for r, c in hex_rc_arr)
    bottom_row = max(r for r, c in hex_rc_arr)
    block_rows = list(range(bottom_row, top_row - 2, -BLOCK_RISE_SNAPSHOTS))

    dict_["barrier"] = [barrier, [BLOCK_COLOR]]
    dict_["curve_cells"] = [curve_cells, [CURVE_COLOR]]

    def _in_swept_wedge(cell, swept_upper, swept_lower):
        if cell in protected_cells:
            return False
        theta = rawbody_angles[cell]
        if 0 <= theta <= swept_upper:
            return True
        if _clockwise_distance_from_zero(theta) <= swept_lower:
            return True
        return False

    return dict(
        dict_=dict_, dict_rawbody=dict_rawbody, max_char_c=max_char_c,
        animation_base_hex_colors=animation_base_hex_colors,
        Phase0_end=Phase0_end + frame_offset, Phase1_end=Phase1_end + frame_offset,
        Phase2_end=Phase2_end + frame_offset, Phase_sweep_end=Phase_sweep_end + frame_offset,
        total_frames=total_frames_local + frame_offset, frame_offset=frame_offset,
        block_rows=block_rows, grid_hex_rc_dict=grid_hex_rc_dict,
        combined_cells=combined_cells, len_final=len_final, travel_positions=travel_positions,
        final_line_colors=final_line_colors, upper_sweep_total=upper_sweep_total,
        lower_sweep_total=lower_sweep_total, _in_swept_wedge=_in_swept_wedge,
    )


# ── build both barriers, back to back (second one's phases offset by the first's total length) ──
barrier1 = build_barrier_animation(x_nm = np.linspace(-2.2, 4,400) , E=1., A=1.0, L_nm=0.5, V0=1.3, frame_offset=0)
barrier2 = build_barrier_animation(x_nm = np.linspace(-1., 4, 400) , E=1., A=1.0, L_nm=2., V0=0.5, frame_offset=barrier1['total_frames'])
barrier3 = build_barrier_animation(x_nm = np.linspace(-1, 4, 400) ,E=1., A=1.0, L_nm=1., V0=0.8, frame_offset=barrier2['total_frames'])

barriers = [barrier1, barrier2, barrier3]
total_frames = barrier3['total_frames']

print('total_frames', total_frames)


def _run_phase(b, local_snapshot):
    """Runs the exact same 5-phase choreography as before, for whichever
    barrier bundle `b` and local (barrier-relative) snapshot applies."""
    dict_ = b['dict_']; dict_rawbody = b['dict_rawbody']; max_char_c = b['max_char_c']
    animation_base_hex_colors = b['animation_base_hex_colors']
    Phase0_end = b['Phase0_end'] - b['frame_offset']
    Phase1_end = b['Phase1_end'] - b['frame_offset']
    Phase2_end = b['Phase2_end'] - b['frame_offset']
    Phase_sweep_end = b['Phase_sweep_end'] - b['frame_offset']
    block_rows = b['block_rows']; grid_hex_rc_dict = b['grid_hex_rc_dict']
    combined_cells = b['combined_cells']; len_final = b['len_final']; travel_positions = b['travel_positions']
    final_line_colors = b['final_line_colors']
    upper_sweep_total = b['upper_sweep_total']; lower_sweep_total = b['lower_sweep_total']
    _in_swept_wedge = b['_in_swept_wedge']

    snapshot = local_snapshot
    if snapshot < Phase0_end:
        current_hex_colors = bkgd_base_hex_colors.copy()
    elif snapshot >= Phase0_end and snapshot < Phase1_end:
        current_snapshot = snapshot - Phase0_end
        current_hex_colors = bkgd_base_hex_colors.copy()

        for key in dict_rawbody.keys():
            part = dict_rawbody.get(key)[0]
            colors = dict_rawbody.get(key)[1]
            shifted_part = [(r, c - max_char_c + current_snapshot) for r, c in part]
            if len(colors) == 1:
                select_part = cg.select_mask(shifted_part, hex_rc_arr)
                current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)
            else:
                current_hex_colors = cgc.color_row_gradient(shifted_part, colors[0], colors[1],
                                            hex_rc_arr, current_hex_colors, sort='row', sigma_color=sigma_color,
                                            end_weight=0.01, mode='linear')

        if current_snapshot > len(block_rows) - 1:
            block_height = 0
        else:
            block_height = max(0, block_rows[current_snapshot])
        visible_barrier = [(r, c) for r, c in dict_["barrier"][0] if r >= block_height]
        select_barrier = cg.select_mask(visible_barrier, hex_rc_arr)
        current_hex_colors[select_barrier] = cgc.select_normal_color(select_barrier, dict_["barrier"][1][0], np.ones(3)*sigma_color)

    elif snapshot >= Phase1_end and snapshot < Phase2_end:
        current_hex_colors = animation_base_hex_colors.copy()

        select_barrier = cg.select_mask(dict_["barrier"][0], hex_rc_arr)
        current_hex_colors[select_barrier] = cgc.select_normal_color(select_barrier, dict_["barrier"][1][0], np.ones(3)*sigma_color)

        right_tentacle_snapshot = snapshot - Phase1_end
        n_reveal = right_tentacle_snapshot + 1

        visible_curve = grid_hex_rc_dict['RIGHT1'][0][:n_reveal]
        select_curve = cg.select_mask(visible_curve, hex_rc_arr)
        current_hex_colors[select_curve] = cgc.select_normal_color(select_curve, grid_hex_rc_dict['RIGHT1'][1][0],
                                                                           np.ones(3)*sigma_color)

    elif snapshot >= Phase2_end and snapshot < Phase_sweep_end:
        current_snapshot = snapshot - Phase2_end
        swept_upper = min((current_snapshot + 1) * dtheta, upper_sweep_total)
        swept_lower = min((current_snapshot + 1) * dtheta, lower_sweep_total)

        current_hex_colors = bkgd_base_hex_colors.copy()

        for key in dict_rawbody.keys():
            part = dict_rawbody.get(key)[0]
            colors = dict_rawbody.get(key)[1]
            remaining_part = [cell for cell in part
                               if not _in_swept_wedge(cell, swept_upper, swept_lower)]
            if not remaining_part:
                continue
            if len(colors) == 1:
                select_part = cg.select_mask(remaining_part, hex_rc_arr)
                current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)
            else:
                current_hex_colors = cgc.color_row_gradient(remaining_part, colors[0], colors[1],
                                            hex_rc_arr, current_hex_colors, sort='row', sigma_color=sigma_color,
                                            end_weight=0.01, mode='linear')

        for key in ("barrier", "curve_cells"):
            part = dict_.get(key)[0]
            colors = dict_.get(key)[1]
            select_part = cg.select_mask(part, hex_rc_arr)
            current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)

    else:
        current_snapshot = snapshot - Phase_sweep_end
        start_idx = min(current_snapshot, travel_positions - 1)
        traveling_cells = combined_cells[start_idx: start_idx + len_final]
        remaining_curve_cells = combined_cells[start_idx + len_final:]

        current_hex_colors = bkgd_base_hex_colors.copy()

        part = dict_.get("barrier")[0]
        colors = dict_.get("barrier")[1]
        select_part = cg.select_mask(part, hex_rc_arr)
        current_hex_colors[select_part] = cgc.select_normal_color(select_part, colors[0], np.ones(3)*sigma_color)

        if remaining_curve_cells:
            colors = dict_.get("curve_cells")[1]
            select_remaining = cg.select_mask(remaining_curve_cells, hex_rc_arr)
            current_hex_colors[select_remaining] = cgc.select_normal_color(select_remaining, colors[0], np.ones(3)*sigma_color)

        for i, cell in enumerate(traveling_cells):
            color_i = final_line_colors[i] if i < len(final_line_colors) else final_line_colors[-1]
            select_cell = cg.select_mask([cell], hex_rc_arr)
            current_hex_colors[select_cell] = cgc.select_normal_color(select_cell, color_i, np.ones(3)*sigma_color)

    return current_hex_colors


def update(snapshot):
    # dispatch to whichever barrier's animation this GLOBAL snapshot falls into
    for b in barriers:
        if snapshot < b['total_frames']:
            local_snapshot = snapshot - b['frame_offset']
            current_hex_colors = _run_phase(b, local_snapshot)
            pc.set_facecolor(current_hex_colors)
            return (pc,)
    return (pc,)



total_seconds = 18.0
FPS = (total_frames / (total_seconds * u.s))
time_gap = ((1 / FPS).to(u.ms)).value
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=time_gap, blit=False)
# ── SAVE ──────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'Korosensei_animation.mp4')

print(f"Encoding {OUTPUT_FILE} ...")
writer = animation.FFMpegWriter(
    fps=float(FPS.value), codec='libvpx-vp9',
    extra_args=['-b:v', '0', '-crf', '33', '-deadline', 'good', '-cpu-used', '2'],
)
ani.save(OUTPUT_FILE, writer=writer, dpi=DPI)
print(f"Saved -> {OUTPUT_FILE}")