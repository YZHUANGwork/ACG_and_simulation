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
import blackbody_color_map as BB
rng = np.random.default_rng(42)

DOC = 'png'
if DOC == 'png':
    HEX_INDEX = False
elif DOC == 'pdf':
    HEX_INDEX = True
    
z_order_max = 5
DPI = 100
fig, ax, patches, hex_colors, hex_center_coords, hex_rc_arr, pc,_ = cg.make_hex_scene(
    IMG_W=1280, IMG_H=720, HEX_R=22, DPI=DPI, hex_index = HEX_INDEX, z_order_max = z_order_max)



sigma_color = 0.01
row_boundary = 10

hex_colors = cgc.color_row_gradient(hex_rc_arr, 
                                    cgc.hex_to_rgb("#fafff4"), cgc.hex_to_rgb("#b1b8a6"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    
 

skyline = row_boundary-1

locs = [(0,2), (-2,16)]
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline+5,
                                                        slope_left = '1.5', slope_right = '1.5', direction = 'lr')
    
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#e9f3f0"), cgc.hex_to_rgb("#c5d1cd"), 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

    
    
loc = (7, 3)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline+5,
                                                    slope_left = '8/3', slope_right = '8/3', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

loc = (3, 18)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                    slope_left = '1.5', slope_right = '1.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

loc = (3, 30)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                    slope_left = '1.5', slope_right = '1.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    


loc = (1, 10)
_, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                    slope_left = '0.5', slope_right = '0.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#f5fee8"), cgc.hex_to_rgb("#162b1f"), 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    


locs = [(6, 7), (7,9), (6,12), (7,14), (7,16), (7,18), (7,20), (6, 22), (5, 25), (7,28), (7,30), (7,32)]
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  skyline,
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr')
    
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#1f614f"), cgc.hex_to_rgb("#1c241e"), 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    

_, _, start_LINE,_, entire_region, vertex =  cgd.draw_trapezoid(row_boundary, 5, 34, 22, 
                                                                slope_left = '1.5', slope_right = '1.5', direction = 'lr')

hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#151d1b"), cgc.hex_to_rgb("#536864") , 
                                    hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)    
select_skyline = cg.select_mask([(r-1, c) for r,c in start_LINE], hex_rc_arr)
hex_colors[select_skyline] = cgc.select_normal_color(select_skyline, cgc.hex_to_rgb("#595d3e"), np.ones(3)*sigma_color) 


locs = [(10,0), (19,14)]
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  22,
                                                        slope_left = '0.5', slope_right = '0.5', direction = 'lr')
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#234431"), [0,0,0], 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color)  
    
locs = [(18,8), (20,20)] 
for loc in locs:
    _, _, _, entire_region, vertex =  cgd.draw_triangle(loc[0], loc[1],  22,
                                                        slope_left = '1.5', slope_right = '1.5', direction = 'lr')
    
    hex_colors = cgc.color_row_gradient(entire_region, cgc.hex_to_rgb("#234431"), [0,0,0], 
                                        hex_rc_arr, hex_colors, sort = 'row', sigma_color = sigma_color) 
    
pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)


OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'valley_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')