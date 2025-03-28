import matplotlib.pyplot as plt
import numpy as np

def simple_plot(font_size):
    x = [0, 300, 600, 1000]
    y = [1.336, 1.3475, 1.358, 1.3725]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel('Sucrose Concentration (mOsm)', fontsize=font_size)
    ax.set_ylabel('Refractive Index', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    plt.title('Refractive Index VS Sucrose Concentration', fontsize=font_size)
    ax.scatter(x, y, color='red')

    plt.tight_layout()  # Adjust layout to make room for labels
    plt.show()

simple_plot(font_size=16)
