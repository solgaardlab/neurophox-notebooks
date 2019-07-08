from typing import Callable, Dict, Optional

import numpy as np
import matplotlib.animation as manimation
from scipy.signal import savgol_filter
from neurophox.helpers import to_absolute_theta

DARK_RED = (0.7, 0, 0)
LIGHT_RED = (0.85, 0.5, 0.5)
DARK_ORANGE = (0.7, 0.35, 0)
DARK_BLUE = (0, 0.2, 0.6)
LIGHT_BLUE = (0.5, 0.6, 0.8)
DARK_GREEN = (0, 0.4, 0)
GRAY = (0.5, 0.5, 0.5)
DARK_PURPLE = (0.4, 0, 0.6)
LIGHT_PURPLE = (0.7, 0.5, 0.8)

class MOResultsVisualizer:
    def __init__(self, model_name: str, model_results: Dict, label_fontsize: int=7, title_fontsize: int=8):
        self.model_name = model_name
        self.model_results = model_results
        self.dim = self.model_results[0]['theta_checkerboard'].shape[0]
        self.label_fontsize = label_fontsize
        self.title_fontsize = title_fontsize

    def plot_theta_checkerboard(self, ax, plt, i: int, cbar_shrink: float=1):
        """Plot theta checkerboard snapshot after ith epoch.

        Args:
            ax: `matplotlib.axes.Axes`
            plt: `matplotlib.pyplot`
            i: Epoch index
            cbar_shrink: Shrink factor for colorbar

        Returns:

        """
        ax.set_xlabel(r"Layer ($\ell$)", fontsize=self.label_fontsize)
        ax.set_ylabel(r"Input ($n$)", fontsize=self.label_fontsize)
        ax.set_title(r"$\theta_{n\ell} / 2$", fontsize=self.title_fontsize)
        plot_handle = ax.imshow(to_absolute_theta(self.model_results[i]["theta_checkerboard"][:-2]), cmap="hot")
        cbar = plt.colorbar(
            mappable=plot_handle,
            ticks=[0, np.pi / 2, np.pi],
            ax=ax,
            shrink=cbar_shrink)
        cbar.set_ticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])
        cbar.ax.tick_params(labelsize=self.label_fontsize)
        ax.set_xticks([]), ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plot_handle.set_clim((0, np.pi))
        return plot_handle

    def plot_phi_checkerboard(self, ax, plt, i: int, cbar_shrink: float=1):
        """Plot phi checkerboard snapshot after ith epoch.

        Args:
            ax: `matplotlib.axes.Axes`
            plt: `matplotlib.pyplot`
            i: Epoch index
            cbar_shrink: Shrink factor for colorbar

        Returns:

        """
        ax.set_xlabel(r"Layer ($\ell$)", fontsize=self.label_fontsize)
        ax.set_ylabel(r"Input ($n$)", fontsize=self.label_fontsize)
        ax.set_title(r"$\phi_{n\ell}$", fontsize=self.title_fontsize)
        plot_handle = ax.imshow(np.mod(self.model_results[i]["phi_checkerboard"][:-2], 2 * np.pi), cmap="hot")
        cbar = plt.colorbar(
            mappable=plot_handle,
            ticks=[0, np.pi, 2 * np.pi],
            ax=ax,
            shrink=cbar_shrink)
        cbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        cbar.ax.tick_params(labelsize=self.label_fontsize)
        ax.set_xticks([]), ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plot_handle.set_clim((0, 2 * np.pi))
        return plot_handle

    def plot_estimate(self, ax, plt, i: int, cbar_shrink: float=1):
        """Plot transformer matrix estimate snapshot after ith epoch.

        Args:
            ax: `matplotlib.axes.Axes`
            plt: `matplotlib.pyplot`
            i: Epoch index
            cbar_shrink: Shrink factor for colorbar

        Returns:

        """
        if "prm" in self.model_name:
            ax.set_title(
                r"\textbf{Estimate magnitude}: $|\hat{U}_{\mathrm{PR}}|$",
                fontsize=self.title_fontsize
            )
        else:
            ax.set_title(
                r"\textbf{Estimate magnitude}: $|\hat{U}_{\mathrm{R}}|$",
                fontsize=self.title_fontsize
            )
        plot_handle = ax.imshow(self.model_results[i]["estimate_mag"], cmap="hot")
        cbar = plt.colorbar(mappable=plot_handle, ax=ax, shrink=cbar_shrink)
        cbar.ax.tick_params(labelsize=self.label_fontsize)
        ax.axis("off")
        return plot_handle

    def plot_error(self, ax, plt, i: int, cbar_shrink: float=1, clim=(0, 0.2)):
        """Plot transformer matrix error snapshot after ith epoch.

        Args:
            ax: `matplotlib.axes.Axes`
            plt: `matplotlib.pyplot`
            i: Epoch index
            cbar_shrink: Shrink factor for colorbar

        Returns:

        """
        if "prm" in self.model_name:
            ax.set_title(
                r"\textbf{Error magnitude}: $|\hat{U}_{\mathrm{PR}} - U|$",
                fontsize=self.title_fontsize
            )
        else:
            ax.set_title(
                r"\textbf{Error magnitude}: $|\hat{U}_{\mathrm{R}} - U|$",
                fontsize=self.title_fontsize
            )
        plot_handle = ax.imshow(self.model_results[i]["error_mag"], cmap="hot")
        cbar = plt.colorbar(mappable=plot_handle, ax=ax, shrink=cbar_shrink)
        cbar.ax.tick_params(labelsize=self.label_fontsize)
        if np.max(self.model_results[i]["error_mag"]) > 0.2:
            plot_handle.autoscale()
        else:
            plot_handle.set_clim(clim)
        ax.axis("off")
        return plot_handle

    def _get_smoothed_thetas(self, i, num_bins: int, smooth_window: int=None):
        theta_freqs, thetas = np.histogram(to_absolute_theta(self.model_results[i]['theta_list']), bins=num_bins, normed=True)
        if smooth_window is not None:
            theta_freqs = savgol_filter(theta_freqs, smooth_window, 3)
        vals = theta_freqs / len(self.model_results[i]["theta_list"]) * num_bins
        return thetas[1:], vals

    def _get_smoothed_phis(self, i, num_bins: int, smooth_window: Optional[int]=None):
        phi_freqs, phis = np.histogram(np.mod(self.model_results[i]["phi_list"], 2 * np.pi), bins=num_bins, normed=True)
        if smooth_window is not None:
            phi_freqs = savgol_filter(phi_freqs, smooth_window, 3)
        vals = phi_freqs / len(self.model_results[i]["phi_list"]) * num_bins
        return [0] + list(phis[1:]) + [2 * np.pi], [0] + list(vals) + [0]

    def _get_thetas(self, i, num_bins: int):
        theta_freqs, thetas = np.histogram(to_absolute_theta(self.model_results[i]['theta_list']), bins=num_bins, normed=True)
        return thetas[1:], theta_freqs

    def _get_phis(self, i, num_bins: int):
        phi_freqs, phis = np.histogram(self.model_results[i]['phi_list'], bins=num_bins, normed=True)
        return [0] + list(phis[1:]) + [2 * np.pi], [0] + list(phi_freqs) + [0]

    def plot_thetas(self, ax, i: int, color='black', alpha=1, use_smoothing=False, smooth_window: int=3, num_bins: int=200):
        if use_smoothing:
            thetas, vals = self._get_smoothed_thetas(i, num_bins, smooth_window)
        else:
            thetas, vals = self._get_thetas(i, num_bins)
        plot_handle = ax.plot(thetas, vals, color=color, alpha=alpha)
        ax.set_xticks([0, np.pi / 2, np.pi])
        ax.set_xticklabels(["$0$", r"$\pi/4$", r"$\pi/2$"], fontsize=self.label_fontsize)
        ax.set_xlabel(r'$\theta / 2$', fontsize=self.label_fontsize)
        ax.set_title(r'\textbf{Histogram}: $\theta_{n\ell} / 2$', fontsize=self.title_fontsize)
        ax.get_yaxis().set_ticks([])
        return plot_handle

    def plot_phis(self, ax, i: int, color='black', smooth_window: Optional[int]=None, num_bins: int=50):
        phis, vals = self._get_smoothed_phis(i, num_bins, smooth_window)
        plot_handle = ax.plot(phis, vals, color=color)
        ax.set_xticks([0., np.pi, 2 * np.pi])
        ax.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"], fontsize=self.label_fontsize)
        ax.set_xlabel(r'$\phi$', fontsize=self.label_fontsize)
        ax.set_title(r'\textbf{Histogram}: $\phi_{n\ell}$', fontsize=self.title_fontsize)
        ax.get_yaxis().set_ticks([])
        return plot_handle

    def plot_losses(self, ax, log=False):
        if log:
            ax.plot(np.log10(self.model_results['losses'] / self.dim), color='black')
        else:
            ax.plot(self.model_results['losses'] / self.dim, color='black')

    def plot_snapshot(self, plt, i, dpi=500, smooth_window=None, num_bins=50, cbar_shrink=0.825):
        fig, axes = plt.subplots(2, 3, dpi=dpi)
        self.plot_theta_checkerboard(axes[0, 0], plt, i, cbar_shrink)
        self.plot_phi_checkerboard(axes[0, 1], plt, i, cbar_shrink)
        self.plot_thetas(axes[1, 0], i, use_smoothing=False, smooth_window=smooth_window, num_bins=num_bins)
        self.plot_phis(axes[1, 1], i)
        self.plot_estimate(axes[0, 2], plt, i, cbar_shrink)
        self.plot_error(axes[1, 2], plt, i, cbar_shrink)

    def plot_meaningful_snapshot(self, plt, i, dpi=500, smooth_window=None, num_bins=50, cbar_shrink=0.825):
        fig, axes = plt.subplots(2, 2, dpi=dpi)
        self.plot_theta_checkerboard(axes[0, 0], plt, i, cbar_shrink)
        self.plot_thetas(axes[0, 1], i, use_smoothing=False, smooth_window=smooth_window, num_bins=num_bins)
        self.plot_estimate(axes[1, 0], plt, i, cbar_shrink)
        self.plot_error(axes[1, 1], plt, i, cbar_shrink)

    def plot_movie(self, plt, title, save_path='data/',
                   pbar_handle: Callable=None,
                   start_iteration=0, end_iteration=100,
                   dpi=500, smooth_window=None, num_bins=50, cbar_shrink=0.825,
                   movie_fileext="mp4", display_epoch=True, epoch_fontsize=11):
        filetype_dict = {
            "gif": "imagemagick",
            "mp4": "ffmpeg"
        }
        if not (movie_fileext in filetype_dict):
            raise Exception('Must specify either mp4 or gif (mp4 highly recommended).')
        ffmpeg_writer = manimation.writers[filetype_dict[movie_fileext]]
        writer = ffmpeg_writer(fps=10, metadata=dict(title=f"movie-{self.model_name}", artist="Matplotlib"))
        fig, axes = plt.subplots(2, 3)

        plot_handle_00 = self.plot_theta_checkerboard(axes[0, 0], plt, start_iteration, cbar_shrink)
        plot_handle_01 = self.plot_phi_checkerboard(axes[0, 1], plt, start_iteration, cbar_shrink)
        plot_handle_10 = self.plot_thetas(axes[1, 0], start_iteration,
                                          use_smoothing=False, smooth_window=smooth_window, num_bins=num_bins)
        plot_handle_11 = self.plot_phis(axes[1, 1], start_iteration)
        plot_handle_02 = self.plot_estimate(axes[0, 2], plt, start_iteration, cbar_shrink)
        plot_handle_12 = self.plot_error(axes[1, 2], plt, start_iteration, cbar_shrink)

        plt.subplots_adjust(top=0.95, right=1.1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(title, fontsize=12)

        iterator = pbar_handle(range(start_iteration + 1, end_iteration)) if pbar_handle \
            else range(start_iteration + 1, end_iteration)

        epoch_text = None
        if display_epoch:
            epoch_text = plt.figtext(0.99, 0.1, rf"\textbf{{Epoch}}: {1:03}",
                                   horizontalalignment='right',
                                   fontsize=epoch_fontsize)

        with writer.saving(fig, f"{save_path}/movie-{self.model_name}.{movie_fileext}", dpi=dpi):
            writer.grab_frame()
            for i in iterator:
                plot_handle_00.set_data(to_absolute_theta(self.model_results[i]["theta_checkerboard"][:-2]))
                plot_handle_01.set_data(np.mod(self.model_results[i]["phi_checkerboard"][:-2], 2 * np.pi))
                plot_handle_10[0].set_data(*(self._get_smoothed_thetas(i, num_bins, smooth_window)))
                axes[1, 0].relim()
                axes[1, 0].autoscale_view(True, True, True)
                plot_handle_11[0].set_data(*(self._get_smoothed_phis(i, num_bins, smooth_window)))
                axes[1, 1].relim()
                axes[1, 1].autoscale_view(True, True, True)
                plot_handle_02.set_data(self.model_results[i]["estimate_mag"])
                plot_handle_12.set_data(self.model_results[i]["error_mag"])
                if np.max(self.model_results[i]["error_mag"]) > 0.2:
                    plot_handle_12.autoscale()
                else:
                    plot_handle_12.set_clim((0, 0.2))
                if display_epoch:
                    epoch_text.remove()
                    epoch_text = plt.figtext(0.99, 0.1, rf"\textbf{{Epoch}}: {i + 1:03}",
                                           horizontalalignment='right',
                                           fontsize=epoch_fontsize)
                writer.grab_frame()
