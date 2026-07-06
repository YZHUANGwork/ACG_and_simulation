import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(42)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


def make_hex_scene(IMG_W=1280, IMG_H=720, HEX_R=22, DPI=100, hex_index = True, z_order_max = 5):
    HEX_apothem = np.sqrt(3)/2 * HEX_R
    HEX_SIDE_LEN = HEX_R
    
    dx_hex_center = HEX_apothem * 2
    dy_hex_center = HEX_SIDE_LEN * 0.5 + HEX_R
    n_cols = int(IMG_W / dx_hex_center) + 3
    n_rows = int(IMG_H / dy_hex_center) + 3
    
    detail_info = [IMG_W, IMG_H, HEX_R, dx_hex_center, dy_hex_center]
    
    hex_vertex_angles = np.radians(np.arange(30., 390, 60))
    hex_vertex_coord = HEX_R * np.column_stack([np.cos(hex_vertex_angles), np.sin(hex_vertex_angles)])

    patches = []
    hex_center_coords = []
    hex_rc_arr = []

    for row_i in range(-1, n_rows):
        for col_i in range(-1, n_cols):
            x_hex_center = col_i * dx_hex_center + (HEX_apothem if row_i % 2 == 1 else 0)
            y_hex_center = row_i * dy_hex_center
            verts = hex_vertex_coord + np.array([x_hex_center, y_hex_center])
            patches.append(MplPolygon(verts, closed=True))
            hex_center_coords.append((x_hex_center, y_hex_center))
            hex_rc_arr.append((row_i, col_i))

    hex_center_coords = np.array(hex_center_coords)
    hex_ntotal = len(patches)
    hex_colors = np.full((hex_ntotal, 3), 0.96)

    fig, ax = plt.subplots(figsize=(IMG_W/DPI, IMG_H/DPI), dpi=DPI)
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#f5f5f5')
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    pc = PatchCollection(patches, facecolor=hex_colors,
                         edgecolor='#cccccc', linewidth=0.4, zorder=0)
    ax.add_collection(pc)
    
    if hex_index:
        for i, (row_i, col_i) in enumerate(hex_rc_arr):
            x_hex_center, y_hex_center = hex_center_coords[i]
            if -HEX_R <= x_hex_center <= IMG_W + HEX_R and -HEX_R <= y_hex_center <= IMG_H + HEX_R:
                ax.text(x_hex_center, y_hex_center-3, str(row_i), ha='center', va='center', fontsize=3.5, color='#993333', zorder=z_order_max)
                ax.text(x_hex_center, y_hex_center+4, str(col_i), ha='center', va='center', fontsize=3.5, color='#333399', zorder=z_order_max)

        ax.add_collection(pc)
    
    
    return fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc, detail_info


def hex_neighbours(row, col):
    """6 neighbours for offset hex grid (flat-top pointy-top)."""
    if row % 2 == 0:
        return [
            (row-1, col-1), (row-1, col),
            (row,   col-1), (row,   col+1),
            (row+1, col-1), (row+1, col),
        ]
    else:
        return [
            (row-1, col),   (row-1, col+1),
            (row,   col-1), (row,   col+1),
            (row+1, col),   (row+1, col+1),
        ]
    

def hex_neighbours_n(row, col, n=1, keep_origin = True, return_frontier=False):
    region = {(row, col)}
    frontier = {(row, col)}
    for _ in range(n):
        new_frontier = set()
        for r, c in frontier:
            for nb in hex_neighbours(r, c):
                if nb not in region:
                    new_frontier.add(nb)
                    region.add(nb)
        frontier = new_frontier
    if keep_origin== False:
        region.discard((row, col))
    if return_frontier:
        return list(region), list(frontier)
    return list(region)

'''
def world_metres_to_pixel(x,y, IMG_W, IMG_H):
    """
    Map a trajectory position (x, y) to pixel coordinates (px_x, px_y).
    (x,y) can be any two of the cartesian coordinate
    vertical is flipped because world-vertical points up but image-vertical points down.
    """
    px_x = (x - x.min()) / (x.max() - x.min()) * IMG_W
    px_y = (1.0 - (y - y.min()) / (y.max() - y.min())) * IMG_H
    return px_x, px_y
'''
def world_metres_to_pixel(x, y, IMG_W, IMG_H, canvas_physical_x_range=None, canvas_physical_y_range=None):
    """
    Map a trajectory position (x, y) to pixel coordinates (px_x, px_y).
    (x,y) can be any two of the cartesian coordinate
    vertical is flipped because world-vertical points up but image-vertical points down.
 
    canvas_physical_x_range, canvas_physical_y_range: optional (min, max)
    tuples giving the physical size that the FULL CANVAS (the whole
    IMG_W x IMG_H image) represents. If omitted (default), falls back to
    the original behavior of normalizing against x.min()/x.max() of the
    data itself -- which stretches whatever's passed in to fill the
    entire canvas. Pass an explicit canvas physical size (bigger than the
    data's own extent) to place a small region as a sub-area instead.
    """
    x_min, x_max = canvas_physical_x_range if canvas_physical_x_range is not None else (x.min(), x.max())
    y_min, y_max = canvas_physical_y_range if canvas_physical_y_range is not None else (y.min(), y.max())
    px_x = (x - x_min) / (x_max - x_min) * IMG_W
    px_y = (1.0 - (y - y_min) / (y_max - y_min)) * IMG_H
    return px_x, px_y
'''
def world_metres_to_hex_index(x,y, detail_info):
    """
    Map arrays of physical positions (x_m, z_m) to flat hex indices.

    Steps:
      1. Convert world metres -> pixel coords via world_metres_to_pixel()
      2. Snap pixel-y to the nearest hex row  (divide by dy_hex_center)
      3. Snap pixel-x to the nearest hex col  (subtract odd-row stagger, divide by dx_hex_center)
      4. Look up (row, col) in rc_to_idx to get the flat hex index
    """
    IMG_W, IMG_H, HEX_R, dx_hex_center, dy_hex_center = detail_info
    px_x, px_y = world_metres_to_pixel(x,y, IMG_W, IMG_H)
    hex_row = np.round(px_y / dy_hex_center).astype(int)
    
    HEX_apothem = np.sqrt(3)/2 * HEX_R
    odd_row_stagger = np.where(hex_row % 2 == 1, HEX_apothem, 0.0)
    hex_col = np.round((px_x - odd_row_stagger) / dx_hex_center).astype(int)

    # nearest hex row
    hex_row = np.round(px_y / dy_hex_center).astype(int)

    # odd rows are staggered right by half a column width
    odd_row_stagger = np.where(hex_row % 2 == 1, HEX_apothem, 0.0)

    # nearest hex col
    hex_col = np.round((px_x - odd_row_stagger) / dx_hex_center).astype(int)

    
    return list(zip(hex_row, hex_col))#[(row1,col1), (row2,col2), (row3,col3), ...]
'''
def world_metres_to_hex_index(x, y, detail_info, canvas_physical_x_range=None, canvas_physical_y_range=None):
    """
    Map arrays of physical positions (x_m, z_m) to flat hex indices.
 
    Steps:
      1. Convert world metres -> pixel coords via world_metres_to_pixel()
      2. Snap pixel-y to the nearest hex row  (divide by dy_hex_center)
      3. Snap pixel-x to the nearest hex col  (subtract odd-row stagger, divide by dx_hex_center)
      4. Look up (row, col) in rc_to_idx to get the flat hex index
 
    canvas_physical_x_range, canvas_physical_y_range: optional (min, max)
    physical size the full canvas represents, passed straight through to
    world_metres_to_pixel -- see its docstring. Omit for the original
    fill-the-whole-canvas behavior.
    """
    IMG_W, IMG_H, HEX_R, dx_hex_center, dy_hex_center = detail_info
    px_x, px_y = world_metres_to_pixel(x, y, IMG_W, IMG_H,
                                        canvas_physical_x_range=canvas_physical_x_range,
                                        canvas_physical_y_range=canvas_physical_y_range)
    hex_row = np.round(px_y / dy_hex_center).astype(int)
    
    HEX_apothem = np.sqrt(3)/2 * HEX_R
    odd_row_stagger = np.where(hex_row % 2 == 1, HEX_apothem, 0.0)
    hex_col = np.round((px_x - odd_row_stagger) / dx_hex_center).astype(int)
 
    # nearest hex row
    hex_row = np.round(px_y / dy_hex_center).astype(int)
 
    # odd rows are staggered right by half a column width
    odd_row_stagger = np.where(hex_row % 2 == 1, HEX_apothem, 0.0)
 
    # nearest hex col
    hex_col = np.round((px_x - odd_row_stagger) / dx_hex_center).astype(int)
 
    
    return list(zip(hex_row, hex_col))#[(row1,col1), (row2,col2), (row3,col3), ...]



def steps_to_Q(n_steps, end_weight=0.05):
    # after n recursive steps with fixed Q, color fraction remaining = sigmoid(Q)^n
    # solve sigmoid(Q)^n = end_weight for Q
    w = end_weight ** (1.0 / n_steps)
    Q = np.log(w / (1 - w))
    return np.array([[Q]])


def get_number_of_terms(last_term, first_term, diff=1):
    return (last_term - first_term )/diff+1


def select_mask(sub_coord, total_coord):
    sub_rcs = set(map(tuple, sub_coord))
    mask = np.array([tuple(rc) in sub_rcs for rc in total_coord])
    return mask


def random_walks(start, n_walks, hex_rc_arr, down_weight=1, left_weight=1, right_weight=1, up_weight=1, seed=None):

    valid_cells = {tuple(rc) for rc in hex_rc_arr}

    def _single_walk(start_row_i, start_col_i):
        path = []
        row_i, col_i = start_row_i, start_col_i
        visited = set()

        while True:
            path.append((row_i, col_i))
            visited.add((row_i, col_i))

            nbrs = hex_neighbours(row_i, col_i)
            weighted = []
            for nr, nc in nbrs:
                if (nr, nc) in visited or (nr, nc) not in valid_cells:
                    continue
                if nr > row_i:
                    w = down_weight
                elif nr < row_i:
                    w = up_weight
                elif nc < col_i:
                    w = left_weight
                else:
                    w = right_weight
                weighted.extend([(nr, nc)] * w)

            if not weighted:
                break

            row_i, col_i = weighted[rng.integers(len(weighted))]

        return path

    rng = np.random.default_rng(seed)
    start_row_i, start_col_i = start
    return [_single_walk(start_row_i, start_col_i) for _ in range(n_walks)]