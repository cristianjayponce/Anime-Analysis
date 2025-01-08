# --- IMPORT REQUIRED LIBRARIES AND SETTINGS ---
import warnings

import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.display import HTML, display  # Added HTML import
from sklearn.preprocessing import MinMaxScaler  # Added for normalization

# --- PLOTLY TEMPLATE AND COLOR SETTINGS ---
pio.templates.default = "simple_white"
antq_blue = "#336699"
antq_purple = "#ac39ac"
antq_red = "#e63900"
antq_orange = "#ff8c1a"
faded_grey = "#e6e6e6"
light_grey = "#d9d9d9"
palette = px.colors.qualitative.Set2

# --- HTML STYLE SETTINGS ---
HTML("""
<style>
th { background-color: #ecc6d9; }
td { background-color: lavender; }
</style>
""")

# --- WARNINGS FILTER ---
warnings.filterwarnings('ignore')

# --- LOAD THE DATASET ---
general = pd.read_csv("Anime.csv")

# --- DATA CLEANING ---
general["Type"] = general["Type"].fillna("Unknown").str.strip()
general["Release_year"] = pd.to_numeric(general["Release_year"], errors='coerce')
general = general.dropna(subset=["Release_year", "Rating"])

# --- NORMALIZATION ---
scaler = MinMaxScaler()
general[["Release_year", "Rating"]] = scaler.fit_transform(general[["Release_year", "Rating"]])

# Display the normalized dataset
print("Normalized Dataset:")
display(general.head())

# --- SCATTER PLOT: TYPES OF ANIME OVER THE YEARS ---
fig = px.scatter(
    general,
    y="Rating",
    x="Release_year",
    color="Type",
    color_discrete_sequence=palette,
    hover_data=["Name"]
)
fig.update_layout(
    hoverlabel=dict(bgcolor="#e6ccff", font_size=14),
    font=dict(color="Purple")
)
fig.show()

# --- PIE CHART: DISTRIBUTION OF ANIME TYPES ---
type_list = general["Type"].unique()
type_values = general["Type"].value_counts().reindex(type_list, fill_value=0).values

fig = px.pie(
    values=type_values,
    names=type_list
)
fig.update_traces(
    marker=dict(colors=palette),
    hole=.3,
    textposition='inside',
    textinfo='percent+label'
)
fig.update_layout(
    annotations=[dict(text='Type', x=0.5, y=0.5, font_size=24, showarrow=False)],
    hoverlabel=dict(bgcolor="#e6ccff", font_size=14),
    legend=dict(yanchor="top", y=0.90, xanchor="left", x=0.80),
    font=dict(color="purple")
)
fig.show()
# --- BAR CHART: TOP 20 RATED ANIME --- 
top_20_rated = general.sort_values(by='Rating', ascending=False).head(20)

fig = px.bar(
    top_20_rated,
    y="Name",
    x="Rating",
    color="Type",
    color_discrete_sequence=palette,
    orientation='h',
    height=600,
    labels={"Name": "Anime"}
)

# Add detailed text and hovertemplate
fig.update_traces(
    text=top_20_rated["Rating"].apply(lambda r: f"{r:.3f}"),  # Display exact rating with 3 decimal places
    textposition='outside',  # Position the text outside the bars
    hovertemplate=(
        '<b>Anime:</b> %{y}<br>'
        '<b>Rating (Normalized):</b> %{x:.3f}<br>'
        '<b>Type:</b> %{marker.color}<extra></extra>'  # Remove extra hover info box
    ),
    selector=dict(type="bar")
)

# Adjust layout for x-axis ticks and range
fig.update_layout(
    title="Top Rated Anime & Their Type",
    xaxis_title="Normalized Rating (0.0 - 1.0)",  # Explicitly indicate the full normalized range
    yaxis_title="Anime",
    xaxis=dict(
        range=[0.0, 1.0],  # Ensure x-axis spans the full normalized range
        tickvals=[i / 10 for i in range(11)],  # Generate ticks at intervals of 0.1
        ticktext=[f"{i / 10:.1f}" for i in range(11)],  # Display ticks as 0.0, 0.1, ..., 1.0
    ),
    yaxis={
        'categoryorder': 'total ascending'
    },
    hoverlabel=dict(
        bgcolor="#e6ccff",
        font_size=14
    ),
    legend=dict(
        title="Type of Anime",
        yanchor="top",
        y=0.50,
        xanchor="left",
        x=0.80,
        bgcolor="#e6ccff",
        bordercolor="purple",
        borderwidth=1
    ),
    font=dict(
        color="Purple",
        size=12
    )
)

fig.show()

# --- LINE CHART: BEST STUDIOS RATINGS OVER THE YEARS ---
# Filter for the best studios
best_studios_df = general[general.Studio.isin(["ufotable", "MAPPA", "Production I.G", "Kyoto Animation"])]

# Group by Studio and Release_year to calculate mean ratings
best_studios_ratings = (
    best_studios_df.groupby(["Studio", "Release_year"])["Rating"]
    .mean()
    .reset_index()
)

# Plot the line chart
fig = px.line(
    best_studios_ratings,
    x="Release_year",
    y="Rating",
    color="Studio",
    markers=True,
    line_shape='spline',
    color_discrete_sequence=palette,
    labels={"Release_year": "Year", "Rating": "Average Rating"},
    title="Best Studios and Their Ratings Over the Years"
)

fig.update_layout(
    hoverlabel=dict(font_size=14),
    font=dict(color="purple", size=14),
    legend=dict(title="Studio", yanchor="top", y=1, xanchor="left", x=1.02)
)
fig.show()



# --- ADDITIONAL ANALYSIS: OTHER STUDIOS ---
additional_studios = general[general.Studio.isin(["Toei Animation", "Sunrise"])]
for studio in ["Toei Animation", "Sunrise"]:
    studio_data = additional_studios[additional_studios.Studio == studio]
    print(f"{studio} Analysis:")
    print(studio_data.describe())
