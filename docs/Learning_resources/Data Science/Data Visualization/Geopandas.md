# Geopandas Data Visualization Guide

Geopandas extends pandas to enable spatial operations and geographic visualizations. Here's a comprehensive guide to creating maps and spatial visualizations with Geopandas.

## Table of Contents
1. [Basic Map Plotting](#basic-map-plotting)
2. [Choropleth Maps](#choropleth-maps)
3. [Point Maps](#point-maps)
4. [Heatmaps](#heatmaps)
5. [Line Maps](#line-maps)
6. [Interactive Maps](#interactive-maps)
7. [Customizing Maps](#customizing-maps)

---

## Basic Map Plotting

### Simple Boundary Plot
**Purpose**: Visualize geographic boundaries  
**Best for**: Showing regions/countries/states

```python
import geopandas as gpd

# Load example dataset
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot countries
ax = world.plot(figsize=(10, 6))
ax.set_title("World Countries")
```

### Multiple Layers
**Purpose**: Combine different geographic layers  
**Best for**: Contextual mapping

```python
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

ax = world.plot(color='lightgrey', edgecolor='black', figsize=(12, 8))
cities.plot(ax=ax, color='red', markersize=5)
ax.set_title("World Map with Major Cities")
```

---

## Choropleth Maps

### Basic Choropleth
**Purpose**: Color regions by attribute values  
**Best for**: Thematic mapping of statistics

```python
ax = world.plot(column='pop_est', 
                legend=True,
                figsize=(12, 8),
                cmap='OrRd',
                scheme='quantiles')
ax.set_title("World Population Estimate")
```

### Custom Classification Schemes
**Purpose**: Control how data is binned  
**Best for**: Highlighting specific value ranges

```python
ax = world.plot(column='gdp_md_est',
                legend=True,
                figsize=(12, 8),
                cmap='Blues',
                scheme='natural_breaks',
                k=5)
ax.set_title("World GDP Estimate (Natural Breaks)")
```

---

## Point Maps

### Scatter Plot on Map
**Purpose**: Show point data with geographic context  
**Best for**: Event locations, site analysis

```python
ax = world.plot(color='lightgrey', figsize=(12, 8))
cities.plot(ax=ax, 
           column='name', 
           markersize=cities['pop_max']/1000000,
           cmap='viridis',
           legend=True)
ax.set_title("World Cities by Population")
```

### Bubble Map
**Purpose**: Show magnitude with point size  
**Best for**: Comparing quantitative values at points

```python
ax = world.plot(color='lightgrey', figsize=(12, 8))
cities.plot(ax=ax, 
           markersize=cities['pop_max']/500000,
           color='red',
           alpha=0.5)
ax.set_title("City Population Bubble Map")
```

---

## Heatmaps

### Density Heatmap
**Purpose**: Show point density  
**Best for**: Hotspot analysis

```python
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(12, 8))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

world.plot(ax=ax, color='lightgrey')
cities.plot(ax=ax, 
           markersize=1,
           color='darkred',
           alpha=0.3)
ax.set_title("City Density Heatmap")
```

---

## Line Maps

### Route Visualization
**Purpose**: Show paths/routes/networks  
**Best for**: Transportation analysis

```python
# Example with NYC taxi data
routes = gpd.read_file('taxi_routes.geojson')
nyc = gpd.read_file('nyc_boundary.geojson')

ax = nyc.plot(color='lightgrey', figsize=(12, 8))
routes.plot(ax=ax, linewidth=0.5, alpha=0.3, color='blue')
ax.set_title("Taxi Routes in NYC")
```

---

## Interactive Maps

### Folium Integration
**Purpose**: Create interactive web maps  
**Best for**: Exploratory analysis

```python
import folium
from folium.plugins import HeatMap

m = folium.Map(location=[20, 0], zoom_start=2)

# Add GeoJSON layer
folium.GeoJson(world).add_to(m)

# Add points
for idx, row in cities.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color='red',
        fill=True
    ).add_to(m)

m.save('interactive_map.html')
```

---

## Customizing Maps

### Advanced Styling
**Purpose**: Create publication-quality maps  
**Best for**: Final presentation

```python
fig, ax = plt.subplots(figsize=(15, 10))

# Base map
world.plot(ax=ax, 
           color='#f0f0f0', 
           edgecolor='#999999', 
           linewidth=0.5)

# Highlight specific countries
world[world['continent'] == 'Africa'].plot(
    ax=ax, 
    color='#2ca25f', 
    edgecolor='black'
)

# Add cities
cities.plot(ax=ax, 
            color='red', 
            markersize=5, 
            alpha=0.7)

# Customize
ax.set_title("Africa with Major Cities", fontsize=16)
ax.set_axis_off()
plt.tight_layout()
```

### Adding Map Elements
**Purpose**: Enhance map readability  
**Best for**: Professional maps

```python
import contextily as ctx

fig, ax = plt.subplots(figsize=(12, 8))

# Plot your geodataframe
nyc.plot(ax=ax, alpha=0.5, edgecolor='k')

# Add basemap
ctx.add_basemap(ax, crs=nyc.crs, source=ctx.providers.Stamen.TonerLite)

# Add scalebar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
scalebar = AnchoredSizeBar(ax.transData,
                          10000,  # 10km
                          '10 km', 
                          'lower right',
                          pad=0.5,
                          color='black',
                          frameon=False,
                          size_vertical=1)
ax.add_artist(scalebar)

ax.set_title("New York City with Basemap")
ax.set_axis_off()
```

---

## Key Geopandas Features

1. **CRS Support**: Handle different coordinate reference systems
2. **Spatial Joins**: Combine data based on geographic relationships
3. **Geometric Operations**: Buffers, intersections, unions
4. **Attribute Queries**: Filter data using spatial predicates
5. **Integration**: Works with matplotlib, folium, plotly, and more

To install required packages:
```bash
pip install geopandas matplotlib folium contextily
```

Geopandas provides the foundation for geographic data science in Python, combining pandas' data manipulation with geographic operations and visualization capabilities.