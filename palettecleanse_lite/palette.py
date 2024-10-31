"""
Palette class
"""

from functools import cached_property
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter
from PIL import Image
from sklearn.cluster import KMeans

try:
    from palettecleanse.utils import PaletteTypes, convert_rgb_palette_to_hex

except ImportError:
    # runs from pytest
    from .utils import PaletteTypes, convert_rgb_palette_to_hex

np.random.seed(42)  # to keep generated palette consistent


class Palette:
    def __init__(self, image_fpath: str, n_colors: int = 5) -> None:
        """
        Initialize the Palette object with an image, palette type, and number of colors

        Args:
            image_fpath (str/Path): filepath to image
            n_colors (int): number of colors to cluster in clustering algorithm. Default to 5
        """
        self.image_fpath = Path(image_fpath)
        self.n_colors = n_colors
        self.image = self.__load_image()

    def __load_image(self) -> Image:
        """
        Load the image, convert to RGB, and compress image.
        Compression primarily helps with clustering later on - large image files will still have a delay
        """
        img = Image.open(self.image_fpath).convert("RGB")
        img = img.resize((100, 100))
        return img

    @cached_property
    def rgb_values(self) -> np.ndarray:
        """
        Extract the `n_colors` dominant colors using KMeans clustering.
        Stores normalized rgb values

        Returns:
            (np.ndarray): array of normalized extracted colors
        """
        image_array = np.array(self.image)
        # reshape image array to 2D (n_pixels, 3) where 3 == RGB channels
        pixels = image_array.reshape((-1, 3))

        # kmeans clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_
        # most libraries use normalized colors
        return colors / 255.0

    @cached_property
    def sequential(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a sequential palette
        """
        # sort RGB array by sorting on least significant to most significant
        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/38194077#38194077
        colors = self.rgb_values[self.rgb_values[:, 2].argsort()]
        colors = colors[colors[:, 1].argsort(kind="mergesort")]
        colors = colors[colors[:, 0].argsort(kind="mergesort")]

        return mcolors.LinearSegmentedColormap.from_list("sequential", colors, N=256)

    @cached_property
    def diverging(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a diverging palette
        """
        # find midpoint of colors and split palette into less or greater than midpoint
        if len(self.rgb_values) >= 2:
            midpoint = len(self.rgb_values) // 2
            diverging_colors = np.vstack(
                (self.rgb_values[0], self.rgb_values[midpoint:], self.rgb_values[-1])
            )
            return mcolors.LinearSegmentedColormap.from_list(
                "diverging", diverging_colors, N=256
            )
        else:
            raise ValueError(
                "Image must contain at least two colors for diverging palette"
            )

    @cached_property
    def cyclic(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a cyclic palette
        """
        # repeat the first color at the end
        cyclic_colors = np.vstack((self.rgb_values, self.rgb_values[0]))
        return mcolors.LinearSegmentedColormap.from_list("cyclic", cyclic_colors, N=256)

    @cached_property
    def qualitative(self) -> np.ndarray:
        """
        Generates a raw array of colors (self.qualitative) corresponding to qualitative palette.

        Some plotting libraries utilize a raw array of colors as opposed
        to a mcolors object for qualitative palettes

        Returns:
            (np.ndarray): a shuffled array of colors spaced within palette
        """
        colors = self.sequential(np.linspace(0, 1, self.n_colors))
        np.random.shuffle(colors)  # this modifies in line
        return colors

    @cached_property
    def qualitative_palette(self) -> mcolors.ListedColormap:
        """
        Generates a qualitative palette (self.qualitative_palette)

        Note - the `qualitative` method is likely preferred
        for certain plotting libraries

        Returns:
            (mcolors.ListedColormap): a shuffled array of colors spaced within palette
        """
        return mcolors.ListedColormap(self.rgb_values, name="qualitative")

    @cached_property
    def hex_values(self) -> list:
        """hex codes separate attribute
        for ease of access

        Returns:
            list: list containing converted hex palette
        """
        return convert_rgb_palette_to_hex(list(self.rgb_values))

    def display_all_palettes(self) -> None:
        """
        Displays all possible palette options
        """
        palette_names = [x.name.lower() for x in PaletteTypes]

        # ideally this would extract from the PaletteTypes enum but
        # unclear as to how to reference from within Palette itself
        palette_types = [
            self.sequential,
            self.diverging,
            self.cyclic,
            self.qualitative_palette,
        ]
        n_palettes = len(palette_types)

        # remember what initiated with in order to reset after
        # iterating through
        init_colors = self.rgb_values

        _, axes = plt.subplots(n_palettes, 1, figsize=(6, 3))

        # generate the gradient for palette display
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # iterate over each palette type and display it
        for ax, palette, name in zip(axes, palette_types, palette_names):
            ax.imshow(gradient, aspect="auto", cmap=palette)
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()

        # reset colors to original. This is to avoid the scenario
        # in which displaying all palettes changes the colors var
        self.rgb_values = init_colors

    def display_example_plots(self) -> None:
        """
        Applies palette to selection of preprogrammed plots for ease of data
        visualization.

        This function follows bad code practice but given size of
        library, opted to keep all as single function instead of dispersing
        """
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 6))

        axes[0, 0].imshow(plt.imread(self.image_fpath))
        axes[0, 0].set_title("Image")

        # scatter plot
        s = axes[0, 1].scatter(
            np.random.rand(100),
            np.random.rand(100),
            c=range(10, 1010, 10),
            cmap=self.sequential,
        )
        axes[0, 1].set_title("Scatter Plot")
        fig.colorbar(s, ax=axes[0, 1])

        # bar plot
        axes[0, 2].barh(
            ["cat", "dog", "fish", "owl", "whale"],
            [15, 30, 45, 60, 20],
            color=self.qualitative,
        )
        axes[0, 2].set_title("Pareto Plot")

        # stackplot
        x = list(range(10))
        values = [sorted(np.random.rand(10)) for _ in range(5)]
        y = dict(zip(x, values))
        axes[0, 3].stackplot(x, y.values(), alpha=0.8, colors=self.qualitative)
        axes[0, 3].set_title("Stack Plot")

        # Kaplan-Meier plot
        n = 100
        populations = 5
        T = [np.random.exponential(20 * i, n) for i in range(populations)]
        E = [np.random.binomial(1, 0.15, n) for _ in range(populations)]
        kmf = KaplanMeierFitter()

        for i in range(populations):
            kmf.fit(T[i], E[i])
            kmf.plot_survival_function(
                ax=axes[1, 0], color=self.qualitative[i], alpha=0.8
            )
        axes[1, 0].legend().remove()
        axes[1, 0].set_title("Survival Plot")

        # violin plot
        violin_data = [
            np.random.normal(5, 1.5, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(10, 2, 100),
            np.random.normal(3, 5, 100),
        ]
        p = axes[1, 1].violinplot(violin_data, showmedians=False, showmeans=False)

        for i, pc in enumerate(p["bodies"]):
            pc.set_facecolor(self.qualitative[i])
            pc.set_edgecolor(self.qualitative[0])
            pc.set_alpha(0.8)
            # set extrema bars to be last indexed color in palette
            for partname in ("cbars", "cmins", "cmaxes"):
                p[partname].set_color(self.qualitative[-1])
        axes[1, 1].set_title("Violin Plot")

        # kde
        kde_data = [
            np.random.normal(size=100, loc=10, scale=2),
            np.random.normal(size=50, loc=70, scale=4),
            np.random.normal(size=200, loc=20, scale=6),
            np.random.normal(size=70, loc=0, scale=3),
        ]
        for i in range(len(kde_data)):
            axes[1, 2].hist(
                kde_data[i], density=True, color=self.qualitative[i], bins=10
            )
        axes[1, 2].set_title("Histogram Plot")

        # heat map
        axes[1, 3] = sns.heatmap(
            sns.load_dataset("glue").pivot(
                index="Model", columns="Task", values="Score"
            ),
            linewidth=0.5,
            annot=True,
            cmap=self.sequential,
        )
        axes[1, 3].set_title("Heat Map")

        # turn off all labels
        for ax in axes.flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
