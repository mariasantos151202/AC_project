import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rnn_final.sources.config as config

HISTORIES:dict

def plot_histories(names, model_title=None, ylim_loss=None, save=False):
    # Loss and accuracy curves for several experiments. Dashed = train, solid = validation.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    
    for i, name in enumerate(names):
        h = HISTORIES[name]
        c = f"C{i}"
        
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], "--", color="blue", alpha=0.7)
        axes[0].plot(ep, h["val_loss"], "-", color="blue", label=name)
        k = int(np.argmin(h["val_loss"]))                       # epoch with the lowest validation loss
        axes[0].plot(k + 1, h["val_loss"][k], "o", color=c, markersize=7, markeredgecolor="black")
        
        axes[1].plot(ep, h["accuracy"], "--", color="orange", alpha=0.7)
        axes[1].plot(ep, h["val_accuracy"], "-", color="orange", label=name)
    
    axes[0].set_title(f"Loss plot {model_title}")
    axes[1].set_title(f"Accuracy plot {model_title}")
    
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)

    #Definição dos tamanhos dos eixos
    if ylim_loss is not None:
        axes[0].set_ylim(*ylim_loss)
    
  
    axes[0].legend(
        [
            Line2D([], [], ls="--", color="gray"), 
            Line2D([], [], ls="-", color="gray"),
            Line2D([], [], ls="", marker="o", color="gray", markeredgecolor="black")
        ],
        ["training", "validation", "min. validation loss"], 
        fontsize=8
    )

    if (save):
        plt.savefig(RESULTS_PATH/"history_plot.png", dpi=150)

    plt.tight_layout()
    plt.show()
