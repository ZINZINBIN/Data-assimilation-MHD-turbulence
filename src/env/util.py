import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional

def plot_contourf(field:np.ndarray, savedir:str, filename:str, title:str, dpi:int = 160):
    # check directory
    filepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="white", dpi=dpi)
    ax.imshow(field, cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)

def generate_contourf_gif(record:List, savedir:str, filename:str, title:str, xmin:float, xmax:float, plot_freq:int = 32):
    # check directory
    filepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)

    T = len(record)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="white")
    ax.cla()
    ax.imshow(record[0], cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()

    def _update(idx):
    
        ax.cla()
        ax.imshow(record[idx], cmap="jet")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.tight_layout()

    ani = animation.FuncAnimation(fig, _update, frames=T, interval = 1000// plot_freq, blit=False)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq, bitrate = False))
    plt.close(fig)

def generate_comparison_gif(record:List, record_filtered:List, savedir:str, filename:str, title:str, plot_freq:int = 32):
    # check directory
    filepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)

    T = min(len(record), len(record_filtered))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")
    axes = axes.ravel()

    axes[0].cla()
    axes[0].imshow(record[0], cmap="jet")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].cla()
    axes[1].imshow(record_filtered[0], cmap="jet")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    
    fig.tight_layout()

    def _update(idx):
        axes[0].cla()
        axes[0].imshow(record[idx], cmap="jet")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        axes[1].cla()
        axes[1].imshow(record_filtered[idx], cmap="jet")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        
        fig.tight_layout()

    ani = animation.FuncAnimation(fig, _update, frames=T, blit=False)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq, bitrate = False))
    plt.close(fig)

def generate_comparison_snapshot(u:np.ndarray, u_filtered:np.ndarray, savedir:str, filename:str, title:str):
    # check directory
    filepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")
    axes = axes.ravel()

    axes[0].cla()
    axes[0].imshow(u, cmap="jet")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("{} without noise".format(title))

    axes[1].cla()
    axes[1].imshow(u_filtered, cmap="jet")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("{} with noise".format(title))

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)
    return fig, axes

def generate_snapshots(
    snapshot_list:List,
    title:str,
    savedir:Optional[str],
    filename:Optional[str],
    Tmax:float = 0.5,
):
    # check directory
    filepath = os.path.join(savedir, filename)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Snapshot info
    Nt = len(snapshot_list)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120)
    axes = axes.ravel()

    axes[0].cla()
    axes[0].imshow(snapshot_list[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("{} at $t=0$".format(title))

    axes[1].cla()
    axes[1].imshow(snapshot_list[Nt//2])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("{} at $t={:.2f}$".format(title, Tmax / 2))

    axes[2].cla()
    axes[2].imshow(snapshot_list[-1])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("{} at $t={:.2f}$".format(title, Tmax))

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)
    return fig, axes
