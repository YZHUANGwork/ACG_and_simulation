import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
import cg_plot_fn as cg


rng = np.random.default_rng(42)


def draw_triangle(start_row, start_col,  end_row, slope_left = '0.5', slope_right = '0.5', direction = 'lr'):   
    left_LINE, right_LINE, start_LINE, end_LINE, entire_region, details = draw_trapezoid(start_row, start_col, start_col, end_row, 
                                                                                         slope_left = slope_left,
                                                                                        slope_right = slope_right,
                                                                                        direction = direction)
    [(start_row, start_col), (start_row, start_col), end_left_vertex,  end_right_vertex] = details
    
    return left_LINE, right_LINE, end_LINE, entire_region, [(start_row, start_col), end_left_vertex,  end_right_vertex]

def draw_trapezoid(start_row, start_left_col, start_right_col, end_row, slope_left = '0.5', slope_right = '0.5', direction = 'lr'):
    
    if direction == 'lr':
        left_line_dir = [True, False, False, False]
        right_line_dir = [False, True, False, False]
        if end_row<start_row:
            left_line_dir = [ False, False,True, False]
            right_line_dir = [False,  False, False,True]
    if direction == 'rl':
        left_line_dir = [False, True , False, False]
        right_line_dir = [True , False, False , False]
        if end_row<start_row:
            left_line_dir = [ False, False,False, True ]
            right_line_dir = [False,  False,True, False]
            
    elif direction == 'll':
        left_line_dir = [True, False, False, False]
        right_line_dir = [True, False, False, False]
        if end_row<start_row:
            left_line_dir = [ False, False,True, False]
            right_line_dir = [False,  False, True,False]
            
    elif direction == 'rr':
        left_line_dir = [False, True, False, False]
        right_line_dir = [False, True, False, False]
        if end_row<start_row:
            left_line_dir = [ False, False,False, True]
            right_line_dir = [False,  False, False,True]        
    if slope_left == '0.5':
        
        left_LINE  = draw_slope_0p5_diagonal(start_row, start_left_col, end_row, 
                                          left_down=left_line_dir[0], 
                                          right_down=left_line_dir[1], 
                                          left_up=left_line_dir[2], 
                                          right_up=left_line_dir[3])
    if slope_left == '1.5':
        left_LINE  = draw_slope_1p5_diagonal(start_row, start_left_col, end_row, 
                                          left_down=left_line_dir[0], 
                                          right_down=left_line_dir[1], 
                                          left_up=left_line_dir[2], 
                                          right_up=left_line_dir[3])
    if slope_left == '8/3':
        left_LINE  = draw_slope_8p3_diagonal(start_row, start_left_col, end_row, 
                                          left_down=left_line_dir[0], 
                                          right_down=left_line_dir[1], 
                                          left_up=left_line_dir[2], 
                                          right_up=left_line_dir[3])
    if slope_left == 'inf':
        #line_info = (start_row, (start_left_col, start_left_col)) 
        left_LINE  = verticle_line(start_row, start_left_col, end_row)
        print(left_LINE)
    if slope_right == '0.5':
        right_LINE = draw_slope_0p5_diagonal(start_row, start_right_col, end_row, 
                                          left_down=right_line_dir[0], 
                                          right_down=right_line_dir[1] , 
                                          left_up=right_line_dir[2], 
                                          right_up=right_line_dir[3])
    
    if slope_right == '1.5':
        right_LINE = draw_slope_1p5_diagonal(start_row, start_right_col, end_row, 
                                          left_down=right_line_dir[0], 
                                          right_down=right_line_dir[1] , 
                                          left_up=right_line_dir[2], 
                                          right_up=right_line_dir[3])
    
    if slope_right == '8/3':
        right_LINE = draw_slope_8p3_diagonal(start_row, start_right_col, end_row, 
                                          left_down=right_line_dir[0], 
                                          right_down=right_line_dir[1] , 
                                          left_up=right_line_dir[2], 
                                          right_up=right_line_dir[3])
        
    if slope_right == 'inf':
        #line_info = (start_row, (start_right_col, start_right_col)) 
        right_LINE  = verticle_line(start_row, start_right_col, end_row)
        print(right_LINE)
    
    
    end_left_vertex  = (end_row, min(c for r, c in left_LINE if r ==end_row))                      
    end_right_vertex = (end_row, max(c for r, c in right_LINE if r ==end_row))    
    print('end_left_vertex=',end_left_vertex, 'end_right_vertex = ',end_right_vertex)
    
    
    sorted_left_LINE  = sorted(left_LINE,  key=lambda x: x[0])
    sorted_right_LINE = sorted(right_LINE, key=lambda x: x[0])
    
    entire_region = horizontal_lines([( r1, (c1, c2)) 
                                            for (r1, c1), (r2, c2) in zip(sorted_left_LINE, sorted_right_LINE)])
    
    start_LINE = horizontal_lines([( start_row, (start_left_col, start_right_col)) ])
    end_LINE = horizontal_lines([( end_row, (end_left_vertex[1], end_right_vertex[1])) ])
    
    return left_LINE, right_LINE, start_LINE, end_LINE, entire_region, [(start_row, start_left_col), 
                                                                        (start_row, start_right_col),
                                                                        end_left_vertex, 
                                                                        end_right_vertex]

def horizontal_lines(lines_info):
    result = []
    for row, (col_1, col_2) in lines_info:
        col_start = min(col_1, col_2)
        col_end = max(col_1, col_2)
        result.extend((row, col) for col in range(col_start, col_end + 1))
    return result

def verticle_line(start_row, start_col, end_row):
    result = [(start_row, start_col)]
    step = 1 if end_row > start_row else -1
    for row in range(start_row, end_row, step):
        if start_row % 2 == 1:
            next_col = start_col
        else:
            next_col = start_col-1 if row % 2 == 0 else start_col
        result.append((row + step, next_col))
    return result

            
def draw_block(line_info, end_row):
    """
    Generate a filled hexagonal block between two rows.

    Starting from a horizontal line of hexes, extends it row by row
    toward end_row, adjusting column bounds at each step to account
    for the hex grid's alternating row offset. Returns all hex
    coordinates covered by the block.

    Parameters
    ----------
    line_info : tuple (start_row, (col1, col2))
        The starting row and the inclusive column range of the first line.
    end_row : int
        The row to extend toward (exclusive). Can be above or below start_row.

    Returns
    -------
    Result of horizontal_lines() — all hex (row, col) coordinates in the block.
    """
    
    start_row, (orig_col1, orig_col2) = line_info
    result = [tuple(line_info)]
    step = 1 if end_row > start_row else -1
    for row in range(start_row, end_row, step):
        if start_row % 2 == 1:
            next_cols = (orig_col1+1, orig_col2) if row % 2 == 1 else (orig_col1, orig_col2)
        else:
            next_cols = (orig_col1, orig_col2-1) if row % 2 == 0 else (orig_col1, orig_col2)
        if next_cols[0] <= next_cols[1]:
            result.append((row + step, next_cols))
    return horizontal_lines(result)


def draw_slope_8p3_diagonal(start_row, start_col, end_row, left_down=True, right_down=True, left_up=True, right_up=True):
    
    step = 1 if end_row > start_row else -1
    shift = -1 if (left_down or left_up) else 1
    result = [(start_row, start_col)]
    
    for next_row in range(start_row + step, end_row + step, step):
        #print(draw_slope_1p5_diagonal(start_row, start_col, next_row, left_down, right_down, left_up, right_up))
        
        start_col = draw_slope_1p5_diagonal(start_row, start_col, next_row, left_down, right_down, left_up, right_up)[-1][1] + shift
        #print(start_col, start_row)
        start_row = next_row
        result.append((start_row, start_col))
        #print((start_row, start_col))
    
    return result

def draw_slope_1p5_diagonal(start_row, start_col, end_row, left_down=True, right_down=True, left_up=True, right_up=True):
    slope = 1.5
    is_odd = start_row % 2 == 1
    result = [(start_row, start_col)]
    
    def left_col(diff):
        d = abs(diff)
        return math.floor(start_col - d*slope + 0.5) if is_odd else math.ceil(start_col - d*slope - 0.5)
    
    def right_col(diff):
        d = abs(diff)
        return math.floor(start_col + d*slope + 0.5) if is_odd else math.ceil(start_col + d*slope - 0.5)
    
    if end_row > start_row:
        step = 1
        if left_down:
            result += [(row, left_col(row - start_row)) for row in range(start_row + step, end_row + step, step)]
        if right_down:
            result += [(row, right_col(row - start_row)) for row in range(start_row + step, end_row + step, step)]
    
    if end_row < start_row:
        step = -1
        if left_up:
            result += [(row, left_col(row - start_row)) for row in range(start_row + step, end_row + step, step)]
        if right_up:
            result += [(row, right_col(row - start_row)) for row in range(start_row + step, end_row + step, step)]
    
    return result



def draw_slope_0p5_diagonal(start_row, start_col, end_row, left_down=True, right_down=True, left_up=True, right_up=True):
    is_odd = start_row % 2 == 1
    result = [(start_row, start_col)]
    for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
        if row == start_row:
            continue
        rows_away = abs(row - start_row)
        if row > start_row:
            if left_down:  result.append((row, start_col - (math.floor(rows_away/2) if is_odd else math.ceil(rows_away/2))))
            if right_down: result.append((row, start_col + (math.ceil(rows_away/2) if is_odd else math.floor(rows_away/2))))
        if row < start_row:
            if left_up:    result.append((row, start_col - (math.floor(rows_away/2) if is_odd else math.ceil(rows_away/2))))
            if right_up:   result.append((row, start_col + (math.ceil(rows_away/2) if is_odd else math.floor(rows_away/2))))
    return result
    
