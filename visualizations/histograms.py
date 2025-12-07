from matplotlib import pyplot as plt
import pandas as pd


def generate_histogram(df, column, title='Histogram', xlabel='Value', ylabel='Frequency', bins=30):
    """Generate and display a histogram for a specified column in the DataFrame."""
    plt.figure(figsize=(8, 6))
    plt.hist(df[column].dropna(), bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)

    # Save high-res
    plt.savefig(f'visualizations/images/histograms/{title}_histogram.png', dpi=300, bbox_inches='tight')