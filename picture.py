import numpy as np
import matplotlib.pyplot as plt


def plot_spectral_curve(spectral_data, title="Spectral Curve", save_path=None):
    """
    Plot spectral curve for 8 spectral features.
    """

    wavelength = np.array([415, 445, 480, 515, 555, 590, 630, 680])
    spectral_data = np.array(spectral_data)

    # Check whether the input contains exactly 8 values
    if len(spectral_data) != 8:
        raise ValueError("spectral_data must contain exactly 8 values.")

    # Simple smoothing using linear interpolation
    x_smooth = np.linspace(wavelength.min(), wavelength.max(), 400)
    y_smooth = np.interp(x_smooth, wavelength, spectral_data)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Background colour
    ax.set_facecolor("#eaeaea")

    # Vertical dashed lines
    for x in wavelength:
        ax.axvline(
            x,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=0
        )

    # Smoothed curve
    ax.plot(
        x_smooth,
        y_smooth,
        color="seagreen",
        linewidth=2,
        zorder=2
    )

    # Original spectral points
    ax.plot(
        wavelength,
        spectral_data,
        linestyle="none",
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor="seagreen",
        zorder=3
    )

    # Axis settings
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Spectral Value", fontsize=12)
    ax.set_xticks(wavelength)
    ax.set_xlim(400, 700)
    ax.tick_params(labelsize=10)

    # Bottom spectral colour band
    ax_bar = fig.add_axes([0.125, 0.08, 0.775, 0.06])
    gradient = np.linspace(0, 1, 600).reshape(1, -1)

    ax_bar.imshow(
        gradient,
        aspect="auto",
        cmap="nipy_spectral",
        extent=[wavelength.min(), wavelength.max(), 0, 1]
    )

    ax_bar.set_yticks([])
    ax_bar.set_xticks(wavelength)
    ax_bar.tick_params(labelsize=10)

    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(bottom=0.28)

    # Save only when save_path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"The image has been saved as {save_path}")

    return fig


# Example usage
if __name__ == "__main__":
    spectral_data = np.array([0.04, 0.03, 0.05, 0.11, 0.14, 0.16, 0.17, 0.09])

    fig = plot_spectral_curve(
        spectral_data=spectral_data,
        title="Spectral Curve",
        save_path="result.png"
    )

    plt.show()