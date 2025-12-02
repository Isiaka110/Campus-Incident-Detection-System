# utils/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def create_heatmap(df, x_col, y_col, title):
    """
    Generates a heatmap of incident frequency (FR7, R4).
    The heatmap is saved to a base64 string for Streamlit display.
    """
    plt.figure(figsize=(10, 6))
    
    # Check if the columns are available before creating the pivot table
    if x_col in df.columns and y_col in df.columns:
        # Calculate frequency
        heatmap_data = df.groupby([y_col, x_col]).size().unstack(fill_value=0)
        
        # Plot the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(title, fontsize=14)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()
    
    else:
        # Handle case where columns might be missing after filtering
        print(f"Warning: Columns {x_col} or {y_col} not found in DataFrame for heatmap.")
        plt.close()
        return None


def create_bar_chart(df, col, title, top_n=10):
    """
    Generates a bar chart of the top N frequencies (FR7).
    """
    plt.figure(figsize=(10, 6))
    
    if col in df.columns:
        top_data = df[col].value_counts().nlargest(top_n)
        sns.barplot(x=top_data.index, y=top_data.values, palette='viridis')
        plt.title(title, fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Incident Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()
    
    else:
        print(f"Warning: Column {col} not found in DataFrame for bar chart.")
        plt.close()
        return None