import numpy as np
import plotly.express as px
import colorsys
import matplotlib.colors as mcolors

# Get the dictionary of color names and their hex codes
color_hex_dict = mcolors.CSS4_COLORS

def darken_lighten_color(color, a=0.):
    try:
        c = np.array(px.colors.hex_to_rgb(color)) / 255.0
        c = colorsys.rgb_to_hls(*c)
        a_m = 0.5 - a
        a_p = 0.5 + a
        l_c = colorsys.hls_to_rgb(c[0], c[1],a_m*c[2])
        d_c = colorsys.hls_to_rgb(c[0], c[1],a_p*c[2])
        l_c = f'rgb({int(l_c[0]*255)}, {int(l_c[1]*255)}, {int(l_c[2]*255)})'
        d_c = f'rgb({int(d_c[0]*255)}, {int(d_c[1]*255)}, {int(d_c[2]*255)})'
        return l_c, d_c
    except ValueError:
        raise ValueError(f"Cannot lighten color {color}")
    


def plot_3d_scatter(morphoframe, axis_labels, conditions, colors, amount, size, title, show=True):
    """
    Plots a 3D interactive scatter plot and saves it as an HTML file.

    Parameters:
    - morphoframe: np.ndarray, shape (n_points, n_dimensions)
        Array containing the data points.
    - conditions: list
        List of columns to concatenate for labels.
    - html_filename: str
        Filename to save the HTML plot.
    """
    
    # create a column with joined conditions.
    morphoframe['condition'] = morphoframe[conditions].apply(lambda x: '_'.join(x), axis=1)

    condition_list = morphoframe['condition'].unique()
    condition_list = condition_list.tolist()
    
    # get colormap
    color_map = {}
    nb_colors = len(colors)
    nb_conditions = len(condition_list)

    if nb_colors == 0:
        color_map = None

    elif nb_colors == nb_conditions:
        for i in range(len(colors)):
            color_name = colors[i]
            color_map[condition_list[i]] = color_name

    elif nb_colors == int(nb_conditions/2):
        for i in range(len(colors)):
            color_name = colors[i]
            color_hex = color_hex_dict[color_name]
            lighten_c, darken_c = darken_lighten_color(color_hex, a=amount)
            color_map[condition_list[2*i]] = lighten_c
            color_map[condition_list[2*i+1]] = darken_c   

    
    fig = px.scatter_3d(morphoframe, 
                        x = axis_labels[0], 
                        y = axis_labels[1], 
                        z = axis_labels[2], 
                        color = 'condition',
                        color_discrete_map = color_map,
                        title = title,
                       )
    
    fig.update_traces(marker=dict(size=size),
                        
                    )
    
    # Update layout to adjust legend size
    fig.update_layout(
        legend=dict(
            font=dict(size=20),  # Adjust the font size as needed
            itemsizing='constant', # Adjust the legend marker size
        )
    )

    if show:
        fig.show()

    return fig