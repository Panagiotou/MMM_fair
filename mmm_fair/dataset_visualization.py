import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

def generate_nested_pie_chart(df, columns, title=None, color_scheme=None):
    """
    Generate a nested pie chart (sunburst) where the same values in the same ring
    have the same color while ensuring distinct colors between rings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(columns, list) or len(columns) < 1:
        raise TypeError("columns must be a list with at least 1 column name")
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Determine a numeric column to use as values
    value_col = None
    for col in df.columns:
        if col not in columns and pd.api.types.is_numeric_dtype(df[col]):
            value_col = col
            break

    # List of color maps to use for each column
    color_palettes = [
        px.colors.qualitative.Set1,
        px.colors.qualitative.Set2,
        # px.colors.qualitative.Set3,
        # px.colors.qualitative.Plotly,
        # px.colors.qualitative.D3
    ]  # Add more palettes if needed

    # Create a color mapping for each level
    color_map = {}
    for j, col in enumerate(columns):
        unique_values = df[col].unique()
        # Select a color palette for the current column, cycling through available palettes
        color_palette = color_palettes[j % len(color_palettes)]
        color_map[col] = {
            val: color_palette[i % len(color_palette)]
            for i, val in enumerate(unique_values)
        }

    # Create the sunburst chart
    fig = px.sunburst(
        df,
        path=columns,
        values=value_col,
        hover_name=columns[-1],
        hover_data=None if value_col is None else [value_col],
    )

    global_color_map = {}

    for level in columns:
        if level in color_map:  # Make sure the level has a color map
            for val, color in color_map[level].items():
                if str(val) not in global_color_map:
                    global_color_map[str(val)] = color

    for i, trace in enumerate(fig.data):
        # Initialize the 'colors' list for the trace if it's None
        if trace.marker.colors is None:
            trace.marker.colors = []

        # Initialize an empty list to store colors for each label
        colors = []

        # For each segment, apply the color based on its label
        for j, label in enumerate(trace.labels):
            color = global_color_map.get(
                str(label), "#000000"
            )  # Default to black if not found
            colors.append(color)

        # Assign the list of colors to the trace's marker colors
        trace.marker.colors = colors

    # Create dummy traces for the custom legend
    legend_entries = []
    for col, col_map in color_map.items():
        # Add feature label entry (i.e., 'Feature 1', 'Feature 2')
        legend_entries.append(
            go.Scatter(
                x=[None],
                y=[None],  # Empty points for the legend item
                mode="markers",
                marker=dict(
                    color="rgba(0,0,0,0)", size=10
                ),  # Invisible marker for the feature label
                name=f"{col}:",
                legendgroup=col,  # Group all values under the feature group
                showlegend=True,
                line=dict(
                    width=0
                ),  # Ensure the dummy trace doesn't create lines on the plot
            )
        )

        # Add value entries under the feature label
        for val, color in col_map.items():
            legend_entries.append(
                go.Scatter(
                    x=[None],
                    y=[None],  # Empty points for the legend item
                    mode="markers",
                    marker=dict(color=color, size=15),
                    name=f"{val}",
                    legendgroup=col,  # Group the values under the feature name
                    showlegend=True,
                    line=dict(
                        width=0
                    ),  # Ensure the dummy trace doesn't create lines on the plot
                )
            )

    # Add the legend entries to the layout
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        uniformtext=dict(minsize=10, mode="hide"),
        font=dict(color="#2B2B2B"),
        title={
            "text": title if title else f"Nested Visualization of {', '.join(columns)}",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        legend=dict(
            orientation="h",  # Horizontal layout for the legend
            yanchor="bottom",  # Align legend to the bottom
            y=-0.15,  # Position the legend below the chart
            x=+1.05,  # Position the legend below the chart
            tracegroupgap=10,  # Spacing between legend items
        ),
        # Set background color of plot and chart area to transparent
        plot_bgcolor="#F5E8D8", #rgba(112,128,144,1)",  # Transparent plot background
        paper_bgcolor= "#F5E8D8",  #"rgba(0,0,0,0)",  # Transparent paper background
        # Disable grid and axes lines
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),  # No grid or ticks on x-axis
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),  # No grid or ticks on y-axis
    )

    customdata = [
        " → ".join([f"{columns[i]}: {val}" for i, val in enumerate(i.split("/"))])
        for i in trace.ids
    ]

    # Pass the full path information as custom data
    fig.update_traces(
        branchvalues="total",
        customdata=customdata,  # Full path data as customdata
        hovertemplate="<b>Intersection</b>: %{customdata}<br><b>Percentage</b>: %{percentRoot:.1%}<extra></extra>",
        textinfo="label+percent root",
        insidetextorientation="radial",
        texttemplate="%{label}<br>%{percentRoot:.1%}",
    )

    # Add the legend entries for custom color patches
    for legend_entry in legend_entries:
        fig.add_trace(legend_entry)

    # chart_height = 400
    # fig.update_layout(height=chart_height)

    # fig_json = pio.to_json(fig)
    # print(fig_json)
    # return fig_json
    return fig.to_html(include_plotlyjs="cdn", full_html=False)
