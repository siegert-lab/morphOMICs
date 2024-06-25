import numpy as np
import plotly.express as px
import colorsys
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# Get the dictionary of color names and their hex codes
color_hex_dict = mcolors.CSS4_COLORS

def _darken_lighten_color(color, a=0.):
    # Takes a color and returns a lighter and a darker shade of this color.
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
    
def _set_colormap(colors, condition_list, amount):
    # Define the colormap for the functions that plot.
    if isinstance(colors, dict):
        color_map = colors
    else:
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
                lighten_c, darken_c = _darken_lighten_color(color_hex, a=amount)
                color_map[condition_list[2*i]] = lighten_c
                color_map[condition_list[2*i+1]] = darken_c   
    
    return color_map


def plot_3d_scatter(morphoframe, axis_labels, conditions, colors, amount, size, title, circle_color=None, show=True):
    """
    Plots a 3D interactive scatter plot and optionally displays it.

    Parameters:
    - morphoframe: pd.DataFrame
        DataFrame containing the data points.
    - axis_labels: list of str, length 2
        Labels for the x-axis and y-axis.
    - conditions: list of str
        Columns in morphoframe to concatenate for condition labels.
    - colors: dict or list of str
        Dictionary mapping conditions to colors or a list of colors of data points.
    - circle_color: dict or None
        Dictionary mapping conditions to colors of the circle around data points.   
    - amount: float between 0 and 1
        Influence the shade (darker/lighter) of the color.
    - size: int
        Size of data points in the scatter plot.
    - title: str
        Title of the plot.
    - show: bool, optional, default=True
        Whether to display the plot (True) or not (False).

    Returns:
    - fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    # Create a column with joined conditions
    morphoframe['condition'] = morphoframe[conditions].apply(lambda x: '-'.join(x), axis=1)

    condition_list = morphoframe['condition'].unique()
    condition_list = condition_list.tolist()
    
    color_map = _set_colormap(colors, condition_list, amount)
    
    if circle_color is None:
        fig = px.scatter_3d(morphoframe, 
                            x = axis_labels[0], 
                            y = axis_labels[1], 
                            z = axis_labels[2], 
                            color = 'condition',
                            color_discrete_map = color_map,
                            title = title,
                        )
    

    else:
        fig = go.Figure()
        for condition in condition_list:
            # Filter data for each condition
            condition_data = morphoframe[morphoframe['condition'] == condition]
            
            # Add scatter trace for each condition
            fig.add_trace(go.Scatter3d(
                x=condition_data[axis_labels[0]],
                y=condition_data[axis_labels[1]],
                z=condition_data[axis_labels[2]],
                mode='markers',
                name=condition,
                marker=dict(
                    color = color_map[condition],
                    line = dict(
                        color=circle_color[condition],
                        width=0.2
                    )
                )
            ))
    
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



def plot_2d_scatter(morphoframe, axis_labels, conditions, colors, circle_color, amount, size, title, show=True):
    """
    Plots a 2D interactive scatter plot and optionally displays it.

    Parameters:
    - morphoframe: pd.DataFrame
        DataFrame containing the data points.
    - axis_labels: list of str, length 2
        Labels for the x-axis and y-axis.
    - conditions: list of str
        Columns in morphoframe to concatenate for condition labels.
    - colors: dict or list of str
        Dictionary mapping conditions to colors or a list of colors.
    - size: int
        Size of markers in the scatter plot.
    - title: str
        Title of the plot.
    - show: bool, optional, default=True
        Whether to display the plot (True) or not (False).

    Returns:
    - fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    # Create a column with joined conditions
    morphoframe['condition'] = morphoframe[conditions].apply(lambda x: '-'.join(x), axis=1)

    condition_list = morphoframe['condition'].unique()
    condition_list = condition_list.tolist()
    
    color_map = _set_colormap(colors, condition_list, amount)
    
    if circle_color is None:
        # Create the Plotly scatter plot
        fig = px.scatter(morphoframe, 
                        x=axis_labels[0], 
                        y=axis_labels[1], 
                        color='condition',
                        color_discrete_map=color_map,
                        title=title
                        )

    else:
        fig = go.Figure()
        for condition in condition_list:
            # Filter data for each condition
            condition_data = morphoframe[morphoframe['condition'] == condition]
            
            # Add scatter trace for each condition
            fig.add_trace(go.Scatter(
                x=condition_data[axis_labels[0]],
                y=condition_data[axis_labels[1]],
                mode='markers',
                name=condition,
                marker=dict(
                    color = color_map[condition],
                    line = dict(
                        color=circle_color[condition],
                        width=0.2
                    )
                )
            ))
    
    fig.update_traces(marker=dict(size=size))
    
    # Update layout to adjust legend size
    fig.update_layout(
        title = title,
        legend=dict(
            font=dict(size=20),  # Adjust the font size as needed
            itemsizing='constant', # Adjust the legend marker size
        ),
        # Enforce square aspect ratio
    autosize=False,
    width=800,  # Adjust as needed
    height=500,  # Adjust as needed
    )

    if show:
        fig.show()

    return fig