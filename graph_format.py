import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


# === Global Plot Style Settings ===
sns.set_theme(context="notebook", style="whitegrid", font="garamond", palette="deep", font_scale=1.3)
custom_palette = [
    "#3B4252",  # Dark Slate
    "#A3BE8C",  # Olive Green
    "#EBCB8B",  # Light Ochre
    "#81A1C1",  # Muted Blue
    "#BF616A",  # Dusty Red
    "#5E81AC",  # Desaturated Navy
    "#88C0D0",  # Icy Blue
    "#D08770",  # Burnt Orange
    "#B48EAD",  # Soft Purple
    "#434C5E",  # Slate Gray
]

sns.set_palette(custom_palette)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 120,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "medium",
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "lines.linewidth": 2.5,
    "lines.markersize": 6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.facecolor": "#f5f5f5",
    "legend.edgecolor": "#dddddd",
})

COLOR_CYCLE = custom_palette
DEFAULT_SAVE_DIR = "Andet/Pictures/"

def save_figure(fig, filename):
    fig.savefig(DEFAULT_SAVE_DIR + filename, bbox_inches="tight", dpi=300)



# === Function to Plot Multiple Subplots in Grid ===
def plot_simulation_grid(data_dict, time, title=None, ncols=2, save_title=None):
    n = len(data_dict)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, (label, content) in enumerate(data_dict.items()):
        data = content["data"]
        unit = content.get("unit", "")
        axes[i].plot(time, data, label=label, color=custom_palette[i % len(custom_palette)])
        axes[i].set_title(label, fontweight="semibold")
        axes[i].set_xlabel("Time Periods")
        axes[i].set_ylabel(f"Value ({unit})" if unit else "Value")
        axes[i].grid(True, linestyle="--", alpha=0.7)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)


    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if save_title:
        save_figure(fig, save_title)
    plt.show()

def plot_simulation_grid_percentiles(data_dict, time, title=None, ncols=2, save_title=None):

    n = len(data_dict)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()


    for i, (label, content) in enumerate(data_dict.items()):
        data = content["data"]
        unit = content.get("unit", "")

        # Convert boolean arrays to float
        if data.dtype == bool:
            data = data.astype(float)

        # Ensure proper shape
        mean = np.nanmean(data, axis=0).squeeze()
        p5 = np.nanpercentile(data, 2.5, axis=0).squeeze()
        p95 = np.nanpercentile(data, 97.5, axis=0).squeeze()
        p16 = np.nanpercentile(data, 16, axis=0).squeeze()
        p84 = np.nanpercentile(data, 84, axis=0).squeeze()




        if mean.shape != time.shape:
            raise ValueError(f"Shape mismatch in '{label}': mean shape {mean.shape} vs time shape {time.shape}")

        ax = axes[i]
        color = custom_palette[0]

        ax.plot(time, mean, label="Mean", color=color)
        ax.fill_between(time, p5, p95, color=color, alpha=0.2, label="5–95% range")
        ax.fill_between(time, p16, p84, color=color, alpha=0.4, label="5–95% range")

        ax.set_title(label, fontweight="semibold")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel(f"Value ({unit})" if unit else "Value")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_title:
        save_figure(fig, save_title)

    plt.show()

def plot_simulation_grid_variance(data_dict, time, title=None, ncols=2, save_title=None):

    n = len(data_dict)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    colors = custom_palette

    for i, (label, content) in enumerate(data_dict.items()):
        data = content["data"]
        unit = content.get("unit", "")

        # Convert boolean arrays to float
        if data.dtype == bool:
            data = data.astype(float)

        # Calculate stats
        mean = np.nanmean(data, axis=0).squeeze()
        std = np.nanstd(data, axis=0).squeeze()

        # ±1 std
        lower1 = mean - std
        upper1 = mean + std
        # ±2 std
        lower2 = mean - 2 * std
        upper2 = mean + 2 * std

        if mean.shape != time.shape:
            raise ValueError(f"Shape mismatch in '{label}': mean shape {mean.shape} vs time shape {time.shape}")

        ax = axes[i]
        color = colors[0]

        # Plot mean and shaded areas
        ax.plot(time, mean, label="Mean", color=color)
        ax.fill_between(time, lower2, upper2, color=color, alpha=0.2, label="±2 Std Dev")
        ax.fill_between(time, lower1, upper1, color=color, alpha=0.4, label="±1 Std Dev")

        ax.set_title(label, fontweight="semibold")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel(f"Value ({unit})" if unit else "Value")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_simulation_one_graph(data_dict, time, figsize =(10, 6), title="Simulation Results Over Time", 
                              xlabel="Time Periods", ylabel="Value", save_title=None):
    fig, ax = plt.subplots(figsize=figsize)
    colors = custom_palette

    for (label, content), color in zip(data_dict.items(), colors):
        data = content["data"]
        unit = content.get("unit", "")
        plt.plot(time, data, label=f"{label} ({unit})" if unit else label, color=color)

    plt.title(title, fontsize=17, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()

def plot_variable_panels_over_individuals(sim, par, variables_dict, title=None, ncols=2, figsize_per_plot=(6, 3), max_i=None, save_title=None):
    """
    Plot one panel per variable, showing time series for many individuals.

    Parameters:
        sim (object): model.sim-like object with arrays
        par (object): contains par.T
        variables_dict (dict): {label: lambda sim, i -> array}, how to extract each variable per individual
        title (str): optional supertitle
        ncols (int): number of subplot columns
        figsize_per_plot (tuple): per subplot size
        max_i (int): maximum number of individuals to include (optional)
    """
    n_vars = len(variables_dict)
    nrows = math.ceil(n_vars / ncols)
    time = np.arange(par.T)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    axes = axes.flatten()

    I = sim.k.shape[0] if max_i is None else min(max_i, sim.k.shape[0])

    for ax, (label, accessor) in zip(axes, variables_dict.items()):
        for i in range(I):
            try:
                y = accessor(sim, i)
                ax.plot(time, y, alpha=0.4)
            except Exception:
                continue  # in case of shape mismatch or NaN
        ax.set_title(label, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(n_vars, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93 if title else 0.96)
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_all_solution_trajectories(model, par, t_end=60, save_title=None):
    """
    Plot solution trajectories for all state combinations in a single figure with 5 subplots.

    Parameters:
        model: Object with `.sol` containing solution variables
        par: Object with `.T`, `.N_s`, `.N_k`, `.N_a`, `.e_grid`
        t_end: End time period (default 60)
    """
    time = np.arange(t_end)

    # Define which variables to plot
    variables_dict = {
        "Welfare (V)": lambda sol, a, s, k, e: sol.V[:t_end, a, s, k, e, 2],
        "Hours (hours)": lambda sol, a, s, k, e: sol.h[:t_end, a, s, k, e, 2],
        "Extensive Margin (ex)": lambda sol, a, s, k, e: sol.ex[:t_end, a, s, k, e, 2],
        "Consumption (c)": lambda sol, a, s, k, e: sol.c[:t_end, a, s, k, e, 2],
        "Assets (a)": lambda sol, a, s, k, e: sol.a[:t_end, a, s, k, e, 2],
    }

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    for label_idx, (label, accessor) in enumerate(variables_dict.items()):
        ax = axes[label_idx]
        for s in range(par.N_s):
            for k in range(par.N_k):
                for a in range(par.N_a):
                    for e in range(len(par.e_grid)):
                        try:
                            y = accessor(model.sol, a, s, k, e)
                            ax.plot(time, y, alpha=0.4)
                        except Exception:
                            continue
        ax.set_title(label, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide the unused subplot if there's only 5
    if len(variables_dict) < len(axes):
        for i in range(len(variables_dict), len(axes)):
            axes[i].axis("off")

    fig.suptitle("Solution Trajectories Across All States", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if save_title:
        save_figure(fig, save_title)

    plt.show()

def plot_event_histogram(values1, xlabel, title,
                         values2=None,
                         label1="Event 1", label2="Event 2",
                         figsize=(10, 6), bins=None, save_title=None):
    """
    Plot one or two overlaid histograms of individual event timing (e.g., retirement or last work).

    Parameters:
        values1 (array-like): First event timing (e.g., last working period)
        xlabel (str): Label for x-axis
        title (str): Plot title
        values2 (array-like or None): Optional second event timing to compare
        label1 (str): Label for first dataset
        label2 (str): Label for second dataset
        figsize (tuple): Figure size
        bins (array-like or None): Custom bins (optional)
    """
    values1 = np.array(values1)
    valid1 = values1[~np.isnan(values1)]

    if values2 is not None:
        values2 = np.array(values2)
        valid2 = values2[~np.isnan(values2)]
    else:
        valid2 = None

    # Auto binning if not supplied
    if bins is None:
        all_vals = valid1 if valid2 is None else np.concatenate((valid1, valid2))
        bins = np.arange(int(np.min(all_vals)), int(np.max(all_vals)) + 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    weights1 = np.ones_like(valid1) / len(valid1)
    ax.hist(valid1, bins=bins, edgecolor='black', alpha=0.7, weights=weights1, label=label1)

    if valid2 is not None:
        weights2 = np.ones_like(valid2) / len(valid2)
        ax.hist(valid2, bins=bins, edgecolor='black', alpha=0.5, weights=weights2, label=label2)

    # Formatting
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Share of Individuals", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

    if valid2 is not None:
        ax.legend(fontsize=12)

    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()

def plot_bar_series(values, xlabel="Time", ylabel="Count", title="Bar Plot", normalize=False, figsize=(10, 6), t_start=0, t_end=40, save_title=None):
    """
    Plot a bar chart over time (e.g., retirements per period).

    Parameters:
        values (array-like): Values to plot (e.g., sum over individuals per time)
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        title (str): Plot title
        normalize (bool): If True, show values as shares (sum to 1)
        figsize (tuple): Figure size
    """
    values = np.array(values)
    time = np.arange(len(values))

    values = values[t_start:t_end]
    time = time[t_start:t_end]

    if normalize:
        values = values / np.sum(values)
        ylabel = "Share" if ylabel == "Count" else ylabel

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(time, values, color=custom_palette[0], edgecolor="black", alpha=0.8)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_event_bar_series(values1, xlabel, title,
                          values2=None,
                          label1="Event 1", label2="Event 2",
                          figsize=(10, 6), bins=None, save_title=None):
    """
    Plot one or two event distributions as overlapping bar series (normalized shares).

    Parameters:
        values1 (array-like): First event timing (e.g., retirement year)
        xlabel (str): X-axis label
        title (str): Plot title
        values2 (array-like): Optional second dataset
        label1 (str): Legend label for first dataset
        label2 (str): Legend label for second dataset
        figsize (tuple): Figure size
        bins (array-like): Optional bin edges
    """
    values1 = np.array(values1)
    valid1 = values1[~np.isnan(values1)].astype(int)

    if values2 is not None:
        values2 = np.array(values2)
        valid2 = values2[~np.isnan(values2)].astype(int)
    else:
        valid2 = None

    # Define bin range
    if bins is None:
        all_vals = valid1 if valid2 is None else np.concatenate((valid1, valid2))
        bins = np.arange(int(np.min(all_vals)), int(np.max(all_vals)) + 1)

    # Frequencies (normalized)
    count1 = np.array([np.sum(valid1 == b) for b in bins])
    freq1 = count1 / count1.sum()

    if valid2 is not None:
        count2 = np.array([np.sum(valid2 == b) for b in bins])
        freq2 = count2 / count2.sum()

    fig, ax = plt.subplots(figsize=figsize)

    # Overlapping bars with full width (0.8)
    ax.bar(bins, freq1, width=0.8, label=label1, edgecolor='black',
           color=custom_palette[0], alpha=0.7)

    if valid2 is not None:
        ax.bar(bins, freq2, width=0.8, label=label2, edgecolor='black',
               color=custom_palette[1], alpha=0.5)

    # Formatting
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Share of Individuals", fontsize=14)
    ax.set_xticks(bins)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

    if valid2 is not None:
        ax.legend(fontsize=12)

    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_bar_series_comparison(values1, values2, label1="Old", label2="New",
                                xlabel="Time", ylabel="Share of Individuals", title="Bar Comparison",
                                normalize=True, figsize=(12, 6), t_start=0, t_end=None, save_title=None):
    """
    Plot two overlaid bar series (e.g. retirements per period) with optional normalization.

    Parameters:
        values1, values2 (array-like): Time series data (counts)
        normalize (bool): If True, convert to shares
        t_start (int): Start index for plotting
        t_end (int or None): End index for plotting
    """
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    if t_end is None:
        t_end = len(values1)

    values1 = values1[t_start:t_end]
    values2 = values2[t_start:t_end]
    time = np.arange(t_start, t_end)

    if normalize:
        values1 = values1 / np.sum(values1)
        values2 = values2 / np.sum(values2)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(time, values1, width=0.8, color=custom_palette[0], alpha=0.7, label=label1, edgecolor='black')
    ax.bar(time, values2, width=0.8, color=custom_palette[1], alpha=0.5, label=label2, edgecolor='black')

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()





def plot_labor_margins_by_age(intensive_age, extensive_age, total_age, avg_intensive, avg_extensive, avg_total, age_start, title_prefix="", save_title=None):
    ages = np.arange(age_start, age_start + len(intensive_age))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    
    # --- Intensive Margin ---
    ax = axes[0]
    ax.plot(ages, intensive_age * 100, color=custom_palette[0], linewidth=2.5, label="Intensive Margin by Age")
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(avg_intensive * 100, color='red', linestyle='-', linewidth=1.5, label="Average Effect on Intensive Margin")
    ax.set_title(f"{title_prefix}Intensive Margin", fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Change (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Extensive Margin ---
    ax = axes[1]
    ax.plot(ages, extensive_age * 100, color=custom_palette[0], linewidth=2.5, label="Extensive Margin by Age")
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(avg_extensive * 100, color='red', linestyle='-', linewidth=1.5, label="Average Effect on Extensive Margin")
    ax.set_title(f"{title_prefix}Extensive Margin", fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Change (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Total Margin ---
    ax = axes[2]
    ax.plot(ages, total_age * 100, color=custom_palette[0], linewidth=2.5, label="Total Effect by Age")
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(avg_total * 100, color='red', linestyle='-', linewidth=1.5, label="Average Total Effect")
    ax.set_title(f"{title_prefix}Total Effect", fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Change (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()



def plot_comparison_single_panel(sim_og, sim_new, variables, time, figsize=(12, 6), title="Key Variable Comparison", save_title=None):
    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, var in enumerate(variables):
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(time, sim_og[var], label=f"{var} (Original)", linestyle='-', color=color)
        plt.plot(time, sim_new[var], label=f"{var} (New)", linestyle='--', color=color)

    plt.xlabel("Time Periods")
    plt.ylabel("Value")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_title:
        save_figure(fig, save_title)

    plt.show()

def plot_comparison_grid(data_og, data_new, time, title=None, ncols=2, save_title=None):

    keys = list(data_og.keys())
    n = len(keys)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        axes[i].plot(time, data_og[key], label="Original", linestyle='-', color=custom_palette[0])
        axes[i].plot(time, data_new[key], label="New", linestyle='--', color=custom_palette[1])
        axes[i].set_title(key)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if title else 0.93)
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_difference_grid(data_diff, time, title="Difference: Original - New", ncols=2, save_title=None):

    keys = list(data_diff.keys())
    n = len(keys)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        axes[i].plot(time, data_diff[key], color=custom_palette[0], linewidth=2)
        axes[i].axhline(0, linestyle="--", color="black", linewidth=1)
        axes[i].set_title(key)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Difference")
        axes[i].grid(True, linestyle='--', alpha=0.6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_ln_wage(ln_wage, man_hourly_salary, title="Log Wage Simulation vs Real Data", 
                 xlabel="Time (Years)", ylabel="ln(Wage)", save_title=None):

    # Convert real hourly salary to log scale
    ln_real_wage = np.log(man_hourly_salary)
    time = np.arange(len(ln_wage))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot simulated ln(wage)
    ax.plot(time, ln_wage, marker="o", linestyle="-", label="Simulated ln(Wage)", color=custom_palette[0])

    # Plot real ln(wage)
    ax.plot(time, ln_real_wage, marker="s", linestyle="--", label="Real ln(Wage)", color=custom_palette[1])

    # Axes formatting
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", frameon=True, fontsize=12)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save if title provided
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_model_vs_data_grid(a_dict, title=None, save_title=None):
    keys = list(a_dict.keys())
    nrows = len(keys)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 4 * nrows), sharex=False)

    if nrows == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        ax = axes[i]
        sim_data, empirical_data = a_dict[key]
        T_sim = len(sim_data)
        T_emp = len(empirical_data)

        if key in ['hours', 'extensive']:
            age_start, age_end = 30, 72
        else:
            age_start, age_end = 30, 100

        time = np.arange(age_start, age_start + max(T_sim, T_emp))

        ax.plot(time[:T_emp], empirical_data, label="Empirical", color=custom_palette[0], linewidth=2)
        ax.plot(time[:T_sim], sim_data, label="Simulated", color=custom_palette[1], linestyle='--', linewidth=2)

        ax.set_title(key.capitalize(), fontsize=12, fontweight="semibold")
        
        ax.set_xlim(age_start, age_end)

        if key == 'hours':
            ax.set_ylim(0.2, 1)
            ax.set_ylabel("Full time equivavlent hours", fontsize=11)
        elif key == 'extensive':
            ax.set_ylim(0, 1)
            ax.set_ylabel("Percent", fontsize=11)
        elif key == 'liquid':
            ax.set_ylabel("DKK", fontsize=11)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10)

    axes[-1].set_xlabel("Age", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if save_title:
        save_figure(fig, save_title)

    plt.show()


def plot_model_vs_data_grid_oos(a_dict, title=None, save_title=None):
    keys = list(a_dict.keys())
    nrows = len(keys)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)

    if nrows == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        ax = axes[i]
        sim_data, empirical_data = a_dict[key]
        T_sim = len(sim_data)
        T_emp = len(empirical_data)

        if key in ['illiquid']:
            age_start, age_end = 30, 100
        else:
            age_start, age_end = 30, 60

        time = np.arange(age_start, age_start + max(T_sim, T_emp))

        ax.plot(time[:T_emp], empirical_data, label="Empirical", color=custom_palette[0], linewidth=2)
        ax.plot(time[:T_sim], sim_data, label="Simulated", color=custom_palette[1], linestyle='--', linewidth=2)

        ax.set_title(key.capitalize(), fontsize=12, fontweight="semibold")
        
        ax.set_xlim(age_start, age_end)

        if key == 'illiquid':
            ax.set_ylim(0, 3)
            ax.set_ylabel("Million DKK", fontsize=11)
        elif key == 'wages':
            ax.set_ylim(400_000, 600_000)
            ax.set_ylabel("DKK", fontsize=11)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10)

    axes[-1].set_xlabel("Age", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if save_title:
        save_figure(fig, save_title)

    plt.show()
