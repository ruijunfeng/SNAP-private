import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from utils.json_utils import load_json
from data_module import HelocDataModule


# Font definitions
FONT1 = {"family": "serif", "color": "Black", "weight": "bold", "size": 12} # global figure
FONT2 = {"family": "serif", "color": "Black", "weight": "bold", "size": 10} # plot-level title
FONT3 = {"family": "serif", "color": "Black", "weight": "bold", "size": 8}  # axis labels, ticks, legend

# Premium style map: Unique deep colors and distinct markers for every single method
STYLE_MAP = {
    "SNAP": {"color": "#990000", "marker": "D"},               # Deep Crimson, Diamond
    "CALM": {"color": "#d35400", "marker": "s"},               # Deep Burnt Orange, Square
    "Informed GPT": {"color": "#2980b9", "marker": "^"},       # Strong Ocean Blue, Triangle Up
    "LightGBM": {"color": "#16a085", "marker": "v"},           # Sea Green, Triangle Down
    "XGBoost": {"color": "#27ae60", "marker": "p"},            # Medium Green, Pentagon
    "GBDT": {"color": "#117a65", "marker": "o"},   # Deep Teal, Circle
    "RF": {"color": "#8e44ad", "marker": "*"},       # Deep Purple, Star
    "SVM": {"color": "#2c3e50", "marker": "h"},                # Dark Navy Blue, Hexagon
    "LR": {"color": "#34495e", "marker": "X"}, # Charcoal, X
    "MLP": {"color": "#f39c12", "marker": "<"},                # Golden Rod, Triangle Left
    "KNN": {"color": "#7f8c8d", "marker": ">"},                # Slate Grey, Triangle Right
    "NB": {"color": "#95a5a6", "marker": "d"},         # Medium Grey, Thin Diamond
    "DT": {"color": "#bdc3c7", "marker": "P"},       # Light Grey, Plus
}

def set_ax_style(
    ax,
    *, # force keyword arguments
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = False,
    legend_kwargs: dict | None = None,
    grid_axis: str | None = "y",
):
    """Apply unified axis, tick, label, legend, and grid styles."""
    # 1) Tick and border settings
    ax.tick_params(which="major", axis="x", length=2, width=0.6)
    ax.tick_params(which="major", axis="y", length=2, width=0.6)
    ax.tick_params(which="minor", axis="x", length=1, width=0.6)
    ax.tick_params(which="minor", axis="y", length=1, width=0.6)
    for side in ["bottom", "left", "right", "top"]:
        ax.spines[side].set_linewidth(1)
    
    # 2) Tick label font
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname(FONT3["family"])
        label.set_color(FONT3["color"])
        label.set_fontweight(FONT3["weight"])
        label.set_fontsize(FONT3["size"])
    
    # 3) Axis labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontdict=FONT3)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontdict=FONT3)
    if title is not None:
        ax.set_title(title, fontdict=FONT2)
    
    # 4) Legend
    if show_legend:
        lg_kwargs = {"prop": {"family": FONT3["family"], "size": FONT3["size"], "weight": FONT3["weight"]}}
        if legend_kwargs:
            lg_kwargs.update(legend_kwargs)
        leg = ax.legend(**lg_kwargs)
        if leg is not None:
            leg.get_frame().set_linewidth(0.8)
            leg.get_frame().set_alpha(0.85)
    
    # 5) Grid
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.5)


def save_fig(fig, output_dir: str, sub_dir: str, file_name: str):
    """Apply tight layout, save figure, and close it."""
    os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{sub_dir}/{file_name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{sub_dir}/{file_name}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def prediction_density_visualization(
    probas_0: torch.Tensor,
    probas_1: torch.Tensor,
    title: str,
    output_dir: str,
    file_name: str,
):
    """KDE comparison of predicted probabilities for Non-default vs Default (Individual plots)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    sns.kdeplot(probas_0.to(torch.float32), label="Non-default (y=0)", fill=True, alpha=0.3, color="#1a5276", linewidth=1.5, ax=ax)
    sns.kdeplot(probas_1.to(torch.float32), label="Default (y=1)", fill=True, alpha=0.3, color="#7b241c", linewidth=1.5, ax=ax)
    
    set_ax_style(
        ax,
        title=title,
        xlabel="Predicted Probability of Default",
        ylabel="Density",
        show_legend=True,
        legend_kwargs={"loc": "upper right"},
        grid_axis="y",
    )
    
    ax.set_xlim(-0.1, 1.1) 
    save_fig(fig, output_dir, "kde_density", file_name)


def roc_curve_visualization(model_probas: dict, y_true: np.ndarray, output_dir: str):
    """ROC Curve comparison encompassing all methods with consistent markers, and save key points to CSV."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    # Data dictionary to store interpolated TPR values for the CSV
    target_fprs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    roc_data_for_csv = {}
    
    for name, probas in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)
        
        # Interpolate TPR at the specific FPR key points for the CSV
        interp_tpr = np.interp(target_fprs, fpr, tpr)
        roc_data_for_csv[name] = interp_tpr
        
        style = STYLE_MAP.get(name, {"color": "#7f8c8d", "marker": "o"})
        
        ax.plot(
            fpr, tpr, 
            color=style["color"], 
            marker=style["marker"], 
            markevery=0.1,       
            markersize=3,        
            lw=1.2,              
            zorder=10 if name == "SNAP" else 5, 
            label=f'{name} (AUC={roc_auc:.4f})',
            alpha=0.9
        )
    
    ax.plot([0, 1], [0, 1], color='black', lw=1.2, linestyle='--', alpha=0.6)
    
    set_ax_style(
        ax,
        title="ROC Curve Comparison",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        show_legend=True,
        legend_kwargs={"loc": "lower right", "ncol": 2, "prop": {"family": FONT3["family"], "size": 6.5, "weight": "normal"}},
        grid_axis="both"
    )
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    
    save_fig(fig, output_dir, "roc_curve", "roc_curve")
    
    # --- Save Key Points to CSV ---
    df_roc = pd.DataFrame.from_dict(roc_data_for_csv, orient='index', columns=[f"FPR={fpr}" for fpr in target_fprs])
    df_roc.index.name = "Methods"
    csv_path = os.path.join(output_dir, "roc_curve", "roc_curve.csv")
    df_roc.to_csv(csv_path)
    print(f"ROC curve data saved to {csv_path}")


def decile_binning_visualization(model_probas: dict, y_true: np.ndarray, output_dir: str):
    """Decile capture rate encompassing all methods with perfectly corresponding styles, and save rates to CSV."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    x = np.arange(1, 11)
    
    # Data dictionary to store decile rates for the CSV
    decile_data_for_csv = {}
    
    for name, probas in model_probas.items():
        df = pd.DataFrame({'proba': probas, 'true_label': y_true})
        df_sorted = df.sort_values(by='proba', ascending=False).reset_index(drop=True)
        df_sorted['decile'] = pd.qcut(df_sorted.index, q=10, labels=False)
        
        decile_rates = df_sorted.groupby('decile')['true_label'].mean().values
        decile_data_for_csv[name] = decile_rates
        
        style = STYLE_MAP.get(name, {"color": "#7f8c8d", "marker": "o"})
        
        ax.plot(
            x, decile_rates, 
            color=style["color"], 
            marker=style["marker"], 
            markersize=3,        
            lw=1.2,              
            zorder=10 if name == "SNAP" else 5, 
            label=name, 
            alpha=0.9
        )
    
    set_ax_style(
        ax,
        title="Actual Default Rate Comparison by Risk Decile",
        xlabel="Risk Decile (1 = Riskiest, 10 = Safest)",
        ylabel="Actual Default Rate",
        show_legend=True,
        legend_kwargs={"loc": "upper right", "ncol": 2, "prop": {"family": FONT3["family"], "size": 7, "weight": "normal"}},
        grid_axis="y"
    )
    
    ax.set_ylim([-0.05, 1.1])
    ax.set_xticks(x)
    
    save_fig(fig, output_dir, "decile_binning", "decile_binning")
    
    # --- Save Decile Rates to CSV ---
    df_decile = pd.DataFrame.from_dict(decile_data_for_csv, orient='index', columns=[f"Decile_{i}" for i in x])
    df_decile.index.name = "Methods"
    csv_path = os.path.join(output_dir, "decile_binning", "decile_binning.csv")
    df_decile.to_csv(csv_path)
    print(f"Decile binning data saved to {csv_path}")


if __name__ == "__main__":
    
    data_module = HelocDataModule()
    y_true = torch.tensor([item["labels"] for item in data_module.get_profile_dataset(data_module.test_indices)])
    y_true_np = y_true.numpy()
    
    output_base_dir = "results/summary"
    all_model_probas = {}
    
    def process_and_plot(name: str, y_proba_list: list):
        probas_np = torch.tensor(y_proba_list).numpy()
        all_model_probas[name] = probas_np
        
        probas_0 = torch.tensor(probas_np[y_true_np == 0])
        probas_1 = torch.tensor(probas_np[y_true_np == 1])
        
        file_name = name.lower().replace(" ", "_")
        
        prediction_density_visualization(
            probas_0=probas_0,
            probas_1=probas_1,
            title=name,
            output_dir=output_base_dir,
            file_name=file_name
        )
        print(f"Saved KDE plot for {name}")
    
    # Machine Learning Models
    model_names = [
        "LR", "KNN", "MLP", "SVM", "NB", "DT", 
        "RF", "GBDT", "XGBoost", "LightGBM",
    ]
    for name in model_names:
        model_results = load_json(f"results/machine_learning/{name}/predictions.json")
        y_proba = [model_results[str(i)]["y_proba"] for i in data_module.test_indices]
        process_and_plot(name, y_proba)
    
    # Informed GPT
    informed_gpt_results = load_json("results/informed_gpt/predictions.json")
    y_proba_gpt = [informed_gpt_results[str(i)]["y_proba"] for i in data_module.test_indices]
    process_and_plot("Informed GPT", y_proba_gpt)
    
    # CALM
    lora_results = load_json("results/calm/version_0/predictions.json")
    y_proba_calm = [lora_results[str(i)]["y_proba"] for i in data_module.test_indices]
    process_and_plot("CALM", y_proba_calm)
    
    # SNAP
    snap_results = load_json("results/snap/full_model/version_0/predictions.json")
    y_proba_snap = [snap_results[str(i)]["y_proba"] for i in data_module.test_indices]
    process_and_plot("SNAP", y_proba_snap)
    
    print("All individual density plots generated successfully.")
    
    # Generate Summary Plots and their corresponding CSV files
    roc_curve_visualization(all_model_probas, y_true_np, output_base_dir)
    print("Summary ROC curve and CSV generated successfully.")
    
    decile_binning_visualization(all_model_probas, y_true_np, output_base_dir)
    print("Summary Decile binning plot and CSV generated successfully.")
