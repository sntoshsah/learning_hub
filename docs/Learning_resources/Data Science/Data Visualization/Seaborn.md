# Data Visualization using Seaborn

Seaborn is a Python data visualization library built on top of Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. Below is a comprehensive guide covering various types of Seaborn plots with explanations, syntax, and example code.

## Table of Contents
1. [Relational Plots](#relational-plots)
2. [Distribution Plots](#distribution-plots)
3. [Categorical Plots](#categorical-plots)
4. [Regression Plots](#regression-plots)
5. [Matrix Plots](#matrix-plots)
6. [Multi-plot Grids](#multi-plot-grids)
7. [Styling and Themes](#styling-and-themes)

---

## Relational Plots
Used to visualize relationships between variables.

### Scatter Plot (`relplot` or `scatterplot`)
**Syntax:**
```python
sns.scatterplot(x, y, data, hue, size, style)
# or
sns.relplot(x, y, data, kind='scatter', hue, size, style)
```

**Example:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Using scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='time', size='size')
plt.title('Scatter Plot of Tips vs Total Bill')
plt.show()

# Using relplot
sns.relplot(x='total_bill', y='tip', data=tips, kind='scatter', 
            hue='day', style='time', size='size')
plt.show()
```

### Line Plot (`lineplot`)
**Syntax:**
```python
sns.lineplot(x, y, data, hue, style, ci)
```

**Example:**
```python
fmri = sns.load_dataset('fmri')

plt.figure(figsize=(10, 6))
sns.lineplot(x='timepoint', y='signal', data=fmri, 
             hue='region', style='event', ci=68)
plt.title('FMRI Signal Over Time')
plt.show()
```

---

## Distribution Plots
Used to visualize distributions of data.

### Histogram (`histplot`)
**Syntax:**
```python
sns.histplot(data, x, y, bins, kde, hue)
```

**Example:**
```python
penguins = sns.load_dataset('penguins')

plt.figure(figsize=(8, 6))
sns.histplot(data=penguins, x='flipper_length_mm', 
             bins=20, kde=True, hue='species')
plt.title('Distribution of Flipper Length')
plt.show()
```

### Kernel Density Estimate (`kdeplot`)
**Syntax:**
```python
sns.kdeplot(data, x, y, hue, shade)
```

**Example:**
```python
plt.figure(figsize=(8, 6))
sns.kdeplot(data=penguins, x='bill_length_mm', 
            hue='species', shade=True)
plt.title('KDE of Bill Length')
plt.show()
```

### Joint Plot (`jointplot`)
**Syntax:**
```python
sns.jointplot(x, y, data, kind='scatter'|'kde'|'hist'|'reg'|'hex')
```

**Example:**
```python
sns.jointplot(x='total_bill', y='tip', data=tips, 
              kind='reg', hue='time')
plt.show()
```

### Pair Plot (`pairplot`)
**Syntax:**
```python
sns.pairplot(data, hue, vars, kind, diag_kind)
```

**Example:**
```python
sns.pairplot(penguins, hue='species', 
             vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'])
plt.show()
```

---

## Categorical Plots
Used to visualize categorical data.

### Bar Plot (`barplot`)
**Syntax:**
```python
sns.barplot(x, y, data, hue, ci, estimator)
```

**Example:**
```python
plt.figure(figsize=(8, 6))
sns.barplot(x='day', y='total_bill', data=tips, 
            hue='sex', ci=68, estimator=sum)
plt.title('Total Bill by Day and Gender')
plt.show()
```

### Count Plot (`countplot`)
**Syntax:**
```python
sns.countplot(x, data, hue)
```

**Example:**
```python
plt.figure(figsize=(8, 6))
sns.countplot(x='day', data=tips, hue='sex')
plt.title('Count of Customers by Day')
plt.show()
```

### Box Plot (`boxplot`)
**Syntax:**
```python
sns.boxplot(x, y, data, hue)
```

**Example:**
```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips, hue='time')
plt.title('Box Plot of Total Bill by Day and Time')
plt.show()
```

### Violin Plot (`violinplot`)
**Syntax:**
```python
sns.violinplot(x, y, data, hue, split)
```

**Example:**
```python
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', data=tips, 
               hue='sex', split=True)
plt.title('Violin Plot of Total Bill by Day and Gender')
plt.show()
```

### Swarm Plot (`swarmplot`)
**Syntax:**
```python
sns.swarmplot(x, y, data, hue)
```

**Example:**
```python
plt.figure(figsize=(10, 6))
sns.swarmplot(x='day', y='total_bill', data=tips, hue='sex')
plt.title('Swarm Plot of Total Bill by Day')
plt.show()
```

---

## Regression Plots
Used to visualize relationships with regression fits.

### Regression Plot (`regplot`)
**Syntax:**
```python
sns.regplot(x, y, data, ci, order)
```

**Example:**
```python
plt.figure(figsize=(8, 6))
sns.regplot(x='total_bill', y='tip', data=tips, 
            ci=95, order=2)
plt.title('Quadratic Regression of Tips vs Total Bill')
plt.show()
```

### LM Plot (`lmplot`)
**Syntax:**
```python
sns.lmplot(x, y, data, hue, col, row)
```

**Example:**
```python
sns.lmplot(x='total_bill', y='tip', data=tips, 
           hue='smoker', col='time', row='sex')
plt.show()
```

---

## Matrix Plots
Used for matrix-like data visualization.

### Heatmap (`heatmap`)
**Syntax:**
```python
sns.heatmap(data, annot, fmt, cmap)
```

**Example:**
```python
flights = sns.load_dataset('flights')
flights_matrix = flights.pivot('month', 'year', 'passengers')

plt.figure(figsize=(10, 8))
sns.heatmap(flights_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Passengers Heatmap by Year and Month')
plt.show()
```

### Cluster Map (`clustermap`)
**Syntax:**
```python
sns.clustermap(data, standard_scale, z_score)
```

**Example:**
```python
plt.figure(figsize=(10, 8))
sns.clustermap(flights_matrix, standard_scale=1, cmap='coolwarm')
plt.show()
```

---

## Multi-plot Grids
For creating multiple plots in a grid.

### Facet Grid (`FacetGrid`)
**Syntax:**
```python
g = sns.FacetGrid(data, col, row, hue)
g.map(plot_type, x, y)
```

**Example:**
```python
g = sns.FacetGrid(tips, col='time', row='smoker', hue='sex')
g.map(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()
plt.show()
```

### Pair Grid (`PairGrid`)
**Syntax:**
```python
g = sns.PairGrid(data, vars, hue)
g.map_upper(func)
g.map_diag(func)
g.map_lower(func)
```

**Example:**
```python
g = sns.PairGrid(penguins, vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'], hue='species')
g.map_upper(sns.scatterplot)
g.map_diag(sns.histplot)
g.map_lower(sns.kdeplot)
g.add_legend()
plt.show()
```

---

## Styling and Themes
Seaborn provides several built-in themes.

**Syntax:**
```python
sns.set_style('darkgrid'|'whitegrid'|'dark'|'white'|'ticks')
sns.set_context('paper'|'notebook'|'talk'|'poster')
sns.set_palette('palette_name')
```

**Example:**
```python
sns.set_style('whitegrid')
sns.set_palette('husl')
sns.set_context('talk')

plt.figure(figsize=(8, 6))
sns.barplot(x='day', y='total_bill', data=tips)
plt.title('Styled Bar Plot')
plt.show()
```

---

## Common Seaborn Functions

| Function | Description |
|----------|-------------|
| `sns.load_dataset()` | Load example datasets |
| `sns.set()` | Set aesthetic parameters |
| `sns.despine()` | Remove spines from plot |
| `sns.color_palette()` | Return a list of colors |
| `sns.palplot()` | Plot a color palette |
| `sns.axes_style()` | Return the parameters of the style |

Seaborn works best with Pandas DataFrames, where variables are stored in columns. Most functions accept `x`, `y`, and `hue` parameters to specify variables from the DataFrame. 

<!-- Remember to include `plt.show()` when not using Jupyter notebooks, or use the `%matplotlib inline` magic command in Jupyter environments. -->