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


pc = PatchCollection(patches, facecolor=hex_colors,
                        edgecolor='#bbba90', linewidth=0.4, zorder=z_order_max-1)


OUTPUT_FOLDER = 'RESULT'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, 'BRS_scene.'+DOC)

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
print('saved')