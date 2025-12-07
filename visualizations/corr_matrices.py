import seaborn as sns
import matplotlib.pyplot as plt

def generate_correlation_heatmap(df, title='Correlation Heatmap'):
    """Generate and display a correlation heatmap for the given DataFrame."""
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Pairwise Correlation Heatmap')
    plt.tight_layout()
    plt.show()