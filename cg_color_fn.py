import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
import cg_plot_fn as cg
import expected_value as EXP
rng = np.random.default_rng(42)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    
def select_normal_color(select, rgb_mean, rgb_std):
    n_color  = sum(select)
    colors = np.column_stack([
    rng.normal(rgb_mean[0], rgb_std[0], n_color),
    rng.normal(rgb_mean[1], rgb_std[1], n_color),
    rng.normal(rgb_mean[2], rgb_std[2], n_color),
    ]).clip(0, 1)
    return colors


def select_uniform_color(select, rgb_start, rgb_end):
    n_color  = sum(select)
    colors = np.column_stack([
    rng.uniform(rgb_start[0], rgb_end[0], n_color),
    rng.uniform(rgb_start[1], rgb_end[1], n_color),
    rng.uniform(rgb_start[2], rgb_end[2], n_color),
    ]).clip(0, 1)
    return colors


'''
def color_row_gradient(arr, color_i, color_f, hex_rc_arr, hex_colors, sort = 'row', sigma_color = 0.03, end_weight= 0.2):
    Y = np.array([[1.0], [0.0]])
    
    idc = 0 if sort == 'row' else 1

    sorted_idc = sorted(list(set([x[idc] for x in arr])))
    n_steps = len(sorted_idc)
    X = cg.steps_to_Q(n_steps, end_weight=end_weight)
    
    for i in sorted_idc:
        select = [r==i and (r,c) in arr for r, c in hex_rc_arr]
        V_input = np.array([color_i,color_f])
        Color_output, _ = EXP.expected_value(X, Y, V_input)
        color_i = Color_output[0]
    
        hex_colors[select] = select_normal_color(select, color_i, np.ones(3)*sigma_color)   
    
    return hex_colors

'''
 
def color_row_gradient(arr, color_i, color_f, hex_rc_arr, hex_colors, sort = 'row', sigma_color = 0.03, end_weight= 0.2, period=None):
    Y = np.array([[1.0], [0.0]])
    
    idc = 0 if sort == 'row' else 1
 
    sorted_idc = sorted(list(set([x[idc] for x in arr])))
    n_steps = len(sorted_idc)
    X = cg.steps_to_Q(n_steps, end_weight=end_weight)
    
    target = color_f
    for step, i in enumerate(sorted_idc):
        if period and step > 0 and step % period == 0:
            color_i, target = target, color_i
 
        select = [r==i and (r,c) in arr for r, c in hex_rc_arr] if sort == 'row' else  [c==i and (r,c) in arr for r, c in hex_rc_arr]
        V_input = np.array([color_i,target])
        Color_output, _ = EXP.expected_value(X, Y, V_input)
        color_i = Color_output[0]
    
        hex_colors[select] = select_normal_color(select, color_i, np.ones(3)*sigma_color)   
    
    return hex_colors

def color_hex_gradient(arr, color_i, color_f, hex_rc_arr, hex_colors, center, n_layers, sigma_color=0.03, end_weight=0.2, period=None):
    Y = np.array([[1.0], [0.0]])
    row, col = center
 
    frontiers = [[center]]
    for n in range(1, n_layers + 1):
        _, frontier = cg.hex_neighbours_n(row, col, n=n, return_frontier=True)
        ring = [h for h in frontier if h in arr]
        if not ring:
            break
        frontiers.append(ring)
 
    n_steps = len(frontiers)
    X = cg.steps_to_Q(n_steps, end_weight=0.2)
 
    target = color_f
    for i, frontier in enumerate(frontiers):
        if period and i > 0 and i % period == 0:
            color_i, target = target, color_i
 
        select = [(r,c) in frontier for r, c in hex_rc_arr]
        V_input = np.array([color_i, target])
        Color_output, _ = EXP.expected_value(X, Y, V_input)
        color_i = Color_output[0]
 
        hex_colors[select] = select_normal_color(select, color_i, np.ones(3)*sigma_color)
 
    return hex_colors
'''
def color_hex_gradient(arr, color_i, color_f, hex_rc_arr, hex_colors, center, n_layers, sigma_color=0.03, end_weight=0.2):
    Y = np.array([[1.0], [0.0]])
    row, col = center
 
    frontiers = [[center]]
    for n in range(1, n_layers + 1):
        _, frontier = cg.hex_neighbours_n(row, col, n=n, return_frontier=True)
        ring = [h for h in frontier if h in arr]
        if not ring:
            break
        frontiers.append(ring)
 
    n_steps = len(frontiers)
    X = cg.steps_to_Q(n_steps, end_weight=end_weight)
 
    for frontier in frontiers:
        select = [(r,c) in frontier for r, c in hex_rc_arr]
        V_input = np.array([color_i,color_f])
        Color_output, _ = EXP.expected_value(X, Y, V_input)
        color_i = Color_output[0]
 
        hex_colors[select] = select_normal_color(select, color_i, np.ones(3)*sigma_color)
 
    return hex_colors
'''