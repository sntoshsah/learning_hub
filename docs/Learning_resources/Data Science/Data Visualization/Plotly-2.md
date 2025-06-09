# Data Visualization with Plotly: Essential Chart Types Explained

Plotly is a powerful Python library for creating interactive, publication-quality visualizations. Here's a concise guide to its main chart types with examples:

## Basic Charts

### 1. Scatter Plots
**Purpose**: Show relationships between two continuous variables  
**Best for**: Correlation analysis, clusters, outliers  
```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()
```

### 2. Line Charts
**Purpose**: Display trends over time/ordered categories  
**Best for**: Time series, progress tracking  
```python
df = px.data.stocks()
fig = px.line(df, x="date", y="GOOG", title="Google Stock Price")
fig.show()
```

### 3. Bar Charts
**Purpose**: Compare quantities across categories  
**Best for**: Performance comparison, survey results  
```python
df = px.data.tips()
fig = px.bar(df, x="day", y="total_bill", color="sex")
fig.show()
```

### 4. Pie Charts
**Purpose**: Show proportional composition  
**Best for**: Market share, budget allocation  
```python
df = px.data.tips()
fig = px.pie(df, names="day", values="total_bill")
fig.show()
```

## Statistical Charts

### 5. Histograms
**Purpose**: Display data distribution  
**Best for**: Understanding spread, skewness  
```python
fig = px.histogram(df, x="total_bill", nbins=20)
fig.show()
```

### 6. Box Plots
**Purpose**: Show quartiles and outliers  
**Best for**: Statistical comparison, outlier detection  
```python
fig = px.box(df, x="day", y="total_bill", color="smoker")
fig.show()
```

### 7. Violin Plots
**Purpose**: Combine box plot with density estimation  
**Best for**: Distribution shape comparison  
```python
fig = px.violin(df, x="day", y="total_bill", box=True)
fig.show()
```

## Advanced Visualizations

### 8. Heatmaps
**Purpose**: Visualize matrix data with colors  
**Best for**: Correlation matrices, confusion matrices  
```python
fig = px.imshow([[1, 20, 30], [20, 1, 60], [30, 60, 1]])
fig.show()
```

### 9. 3D Scatter Plots
**Purpose**: Explore 3-variable relationships  
**Best for**: Multidimensional data analysis  
```python
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width')
fig.show()
```

### 10. Maps (Choropleth)
**Purpose**: Geographic data visualization  
**Best for**: Regional comparisons, election results  
```python
df = px.data.gapminder().query("year == 2007")
fig = px.choropleth(df, locations="iso_alpha", color="gdpPercap")
fig.show()
```

## Financial Charts

### 11. Candlestick Charts
**Purpose**: Show financial market data  
**Best for**: Stock price movements  
```python
df = px.data.stocks(indexed=True)
fig = px.line(df, facet_col="company", facet_col_wrap=2)
fig.show()
```

## Interactive Features

Plotly charts automatically include:
- Hover tooltips
- Zoom/pan controls
- Click legend to toggle traces
- Download as image option

To customize:
```python
fig.update_layout(
    title="Custom Title",
    xaxis_title="X Label",
    yaxis_title="Y Label"
)
```

For Jupyter notebooks, ensure you have the required renderer:
```python
import plotly.io as pio
pio.renderers.default = "notebook"
```

Plotly excels at creating interactive visualizations that can be embedded in web applications (using Dash) or exported as standalone HTML files. The library offers over 40 chart types with extensive customization options while maintaining simplicity through its Plotly Express high-level API.