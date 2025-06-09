# Data Visualization using Plotly

Plotly is an interactive, open-source visualization library that supports over 40 chart types. It provides high-level APIs for Python (Plotly Express) and low-level graph objects (Plotly Graph Objects) for more customization.

## Table of Contents
1. [Basic Charts](#basic-charts)
2. [Statistical Charts](#statistical-charts)
3. [Financial Charts](#financial-charts)
4. [Maps](#maps)
5. [3D Charts](#3d-charts)
6. [Subplots](#subplots)
7. [Animation](#animation)
8. [Customization](#customization)

---

## Basic Charts

### Scatter Plot
**Syntax (Plotly Express):**
```python
px.scatter(data_frame, x, y, color, size, hover_name, trendline)
```

**Example:**
```python
import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", 
                 color="species", size="petal_length",
                 hover_name="species", trendline="ols")
fig.show()
```

**Syntax (Graph Objects):**
```python
go.Scatter(x, y, mode, name, marker)
```

**Example:**
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["sepal_width"],
    y=df["sepal_length"],
    mode="markers",
    marker=dict(
        size=df["petal_length"]*2,
        color=df["species_id"],
        showscale=True
    )
))
fig.show()
```

### Line Plot
**Syntax:**
```python
px.line(data_frame, x, y, color, line_group)
```

**Example:**
```python
df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x="year", y="lifeExp", 
              title="Life expectancy in Canada")
fig.show()
```

### Bar Chart
**Syntax:**
```python
px.bar(data_frame, x, y, color, barmode)
```

**Example:**
```python
df = px.data.tips()
fig = px.bar(df, x="day", y="total_bill", 
             color="sex", barmode="group")
fig.show()
```

### Pie Chart
**Syntax:**
```python
px.pie(data_frame, names, values, color_discrete_sequence)
```

**Example:**
```python
df = px.data.tips()
fig = px.pie(df, names="day", values="total_bill",
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
```

---

## Statistical Charts

### Histogram
**Syntax:**
```python
px.histogram(data_frame, x, y, color, marginal, nbins)
```

**Example:**
```python
df = px.data.tips()
fig = px.histogram(df, x="total_bill", color="sex", 
                   marginal="rug", nbins=30)
fig.show()
```

### Box Plot
**Syntax:**
```python
px.box(data_frame, x, y, color, points)
```

**Example:**
```python
fig = px.box(df, x="day", y="total_bill", 
             color="smoker", points="all")
fig.show()
```

### Violin Plot
**Syntax:**
```python
px.violin(data_frame, x, y, color, box)
```

**Example:**
```python
fig = px.violin(df, x="day", y="total_bill", 
                color="sex", box=True)
fig.show()
```

### Density Heatmap
**Syntax:**
```python
px.density_heatmap(data_frame, x, y, z, nbinsx, nbinsy)
```

**Example:**
```python
fig = px.density_heatmap(df, x="total_bill", y="tip", 
                         nbinsx=20, nbinsy=20)
fig.show()
```

---

## Financial Charts

### Candlestick Chart
**Syntax:**
```python
go.Candlestick(x, open, high, low, close)
```

**Example:**
```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(go.Candlestick(
    x=df['Date'],
    open=df['AAPL.Open'],
    high=df['AAPL.High'],
    low=df['AAPL.Low'],
    close=df['AAPL.Close']
))
fig.show()
```

### OHLC Chart
**Syntax:**
```python
go.Ohlc(x, open, high, low, close)
```

**Example:**
```python
fig = go.Figure(go.Ohlc(
    x=df['Date'],
    open=df['AAPL.Open'],
    high=df['AAPL.High'],
    low=df['AAPL.Low'],
    close=df['AAPL.Close']
))
fig.show()
```

---

## Maps

### Scatter Map
**Syntax:**
```python
px.scatter_geo(data_frame, lat, lon, color, size)
```

**Example:**
```python
df = px.data.gapminder().query("year == 2007")
fig = px.scatter_geo(df, locations="iso_alpha",
                     color="continent",
                     size="pop",
                     projection="natural earth")
fig.show()
```

### Choropleth Map
**Syntax:**
```python
px.choropleth(data_frame, locations, color, locationmode)
```

**Example:**
```python
fig = px.choropleth(df, locations="iso_alpha",
                    color="lifeExp",
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()
```

---

## 3D Charts

### 3D Scatter Plot
**Syntax:**
```python
px.scatter_3d(data_frame, x, y, z, color)
```

**Example:**
```python
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', 
                    y='sepal_width', z='petal_width',
                    color='species')
fig.show()
```

### Surface Plot
**Syntax:**
```python
go.Surface(z, colorscale)
```

**Example:**
```python
import numpy as np
z = np.random.rand(10,10)
fig = go.Figure(go.Surface(z=z, colorscale="Viridis"))
fig.show()
```

---

## Subplots

### Make Subplots
**Syntax:**
```python
make_subplots(rows, cols, subplot_titles)
```

**Example:**
```python
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6]), row=1, col=1)
fig.add_trace(go.Bar(x=[1,2,3], y=[3,2,1]), row=1, col=2)

fig.update_layout(height=400, width=800, title_text="Subplots")
fig.show()
```

---

## Animation

### Animated Charts
**Syntax:**
```python
px.scatter(data_frame, animation_frame, animation_group)
```

**Example:**
```python
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 hover_name="country",
                 animation_frame="year",
                 animation_group="country",
                 range_x=[100,100000], range_y=[25,90])
fig.show()
```

---

## Customization

### Update Layout
**Syntax:**
```python
fig.update_layout(title, xaxis_title, yaxis_title)
```

**Example:**
```python
fig.update_layout(
    title="Customized Chart",
    xaxis_title="X Axis Title",
    yaxis_title="Y Axis Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
```

### Update Traces
**Syntax:**
```python
fig.update_traces(marker_size, line_width)
```

**Example:**
```python
fig.update_traces(
    marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)
```

---

## Key Plotly Features

1. **Interactivity**: Hover tooltips, zoom, pan, and click events
2. **Themes**: Built-in templates like `plotly`, `plotly_white`, `plotly_dark`
3. **Export**: Save as PNG, JPEG, SVG, or PDF
4. **Dash Integration**: Build interactive web applications
5. **Web Support**: Embed charts in websites and notebooks

To use Plotly in Jupyter notebooks, you may need to install the required extensions:
```python
pip install jupyter-dash
```

Plotly offers both a high-level API (Plotly Express) for quick visualization and a low-level API (Graph Objects) for detailed customization. The library works seamlessly with Pandas DataFrames and integrates well with Dash for building analytical web applications.