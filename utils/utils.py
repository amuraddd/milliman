def plot_class_distribution(target, save_fig=False):
    """
    Plot target distribution given a binary target variable
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # generate a historgam
    ax = sns.histplot(target, discrete=True, shrink=0.6)
    
    #set ticks for x axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'], ha='center')
    
    # Add counts on the bars
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom')

    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.title("Distribution (0 and 1)")
    if save_fig:
        plt.savefig('figures/class_distribution.pdf', format="pdf", bbox_inches="tight")
    plt.show()
        
def plot_pca_components_and_variance(mod, save_fig=False):
    """
    Plot explained variance from a PCA decomposition.
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    plt.grid()
    sns.lineplot(
        np.cumsum(mod.explained_variance_ratio_ * 100), 
        marker='o',
        color='green',
        markeredgecolor='black'
    )
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    if save_fig:
        num_components = len(mod.explained_variance_ratio_)
        plt.savefig(f'figures/pca_variance_{num_components}_components.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def generate_2d_pca_distribution_plot(components, target):
    """
    Generate a 2D plot given PCA with 2 components
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    pca_df = pd.DataFrame(components)
    pca_df['target'] = target.reset_index().drop(columns=['UniqueID'])
    sns.scatterplot(pca_df, x=0, y=1, hue='target')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('figures/2d_pca_distribution_plot.pdf', format="pdf", bbox_inches="tight")
    
def generate_3d_pca_distribution_plot(components, target):
    """
    Generate a 3D plot given PCA with 3 components
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    pca_df = pd.DataFrame(components)
    pca_df['target'] = target.reset_index().drop(columns=['UniqueID'])
    colors = np.where(pca_df['target'] == 0, '#FF9B2F', '#27667B')

    # Create 3D plot
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot=
    ax.scatter(pca_df[0], pca_df[1], pca_df[2], c=colors, s=10, depthshade=True)

    # Labels
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target 0', markerfacecolor='#FF9B2F', markersize=4),
        Line2D([0], [0], marker='o', color='w', label='Target 1', markerfacecolor='#27667B', markersize=4)
    ]
    ax.legend(handles=legend_elements, title="Target")
    plt.savefig('figures/3d_pca_distribution_plot.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def notebook_line_magic():
    """
    Avoid having to restart kernel when working with python scripts
    """
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    print("Line Magic Set")
    