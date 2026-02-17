"""
Streamlit app for exploratory data analysis using the Iris dataset.
This app allows users to visualize and analyze the Iris dataset interactively.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Configure the Streamlit page
st.set_page_config(page_title="Iris EDA", layout="wide")

# Title and description
st.title("🌸 Iris Dataset Exploratory Data Analysis")
st.markdown(
    "This app provides interactive visualization and analysis of the Iris dataset."
)

# Load the Iris dataset
@st.cache_data
def load_data():
    """Load the Iris dataset from scikit-learn."""
    iris = load_iris()
    iris_df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    iris_df['Species'] = iris.target_names[iris.target]
    return iris_df


# Load data
iris_data = load_data()

# Display dataset overview
st.header("📊 Dataset Overview")

# Show first rows
col1, col2 = st.columns(2)
with col1:
    st.subheader("First 5 Rows")
    st.dataframe(iris_data.head(), use_container_width=True)

with col2:
    st.subheader("Dataset Information")
    st.write(f"**Total rows:** {len(iris_data)}")
    st.write(f"**Total columns:** {len(iris_data.columns)}")
    st.write(f"**Species:** {iris_data['Species'].unique().tolist()}")

# Display summary statistics
st.header("📈 Summary Statistics")
st.dataframe(iris_data.describe(), use_container_width=True)

# Interactive visualizations section
st.header("🎨 Interactive Visualizations")

# Get numeric columns (exclude Species)
numeric_columns = iris_data.select_dtypes(include=[np.number]).columns.tolist()

# Sidebar for user selections
st.sidebar.header("Visualization Options")

# Column selection for histogram
selected_column_hist = st.sidebar.selectbox(
    "Select column for histogram:",
    numeric_columns,
    help="Choose a numeric column to display in the histogram"
)

# Column selections for scatter plot
col_x = st.sidebar.selectbox(
    "Select X-axis for scatter plot:",
    numeric_columns,
    help="Choose a numeric column for the X-axis"
)

col_y = st.sidebar.selectbox(
    "Select Y-axis for scatter plot:",
    numeric_columns,
    index=1 if len(numeric_columns) > 1 else 0,
    help="Choose a numeric column for the Y-axis"
)

# Create visualizations
viz_col1, viz_col2 = st.columns(2)

# Histogram
with viz_col1:
    st.subheader(f"📊 Histogram: {selected_column_hist}")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(iris_data[selected_column_hist], bins=20, color="skyblue", edgecolor="black")
    ax_hist.set_xlabel(selected_column_hist)
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(axis="y", alpha=0.3)
    st.pyplot(fig_hist)

# Scatter plot
with viz_col2:
    st.subheader(f"📍 Scatter Plot: {col_x} vs {col_y}")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    
    # Color points by species
    species_colors = {"setosa": "red", "versicolor": "green", "virginica": "blue"}
    for species in iris_data["Species"].unique():
        species_data = iris_data[iris_data["Species"] == species]
        ax_scatter.scatter(
            species_data[col_x],
            species_data[col_y],
            label=species,
            color=species_colors.get(species, "gray"),
            alpha=0.6,
            s=80
        )
    
    ax_scatter.set_xlabel(col_x)
    ax_scatter.set_ylabel(col_y)
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.3)
    st.pyplot(fig_scatter)

# Display records grouped by species
st.header("🔍 Data by Species")
species_selected = st.selectbox(
    "Select species to view records:",
    iris_data["Species"].unique()
)
species_data = iris_data[iris_data["Species"] == species_selected]
st.dataframe(species_data, use_container_width=True)
st.write(f"**Count:** {len(species_data)} records")
