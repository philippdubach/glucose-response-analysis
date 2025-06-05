"""Visualization utilities for glucose response analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import shap
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GlucosePlotter:
    """Plotting utilities for glucose response analysis."""

    def __init__(self):
        self.config = config
        self.figure_size = self.config.get("visualization.figure_size", [10, 6])
        self.dpi = self.config.get("visualization.dpi", 150)
        self.save_plots = self.config.get("visualization.save_plots", False)
        self.plot_format = self.config.get("visualization.plot_format", "png")

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_glucose_curve_fit(
        self, result: Dict, meal_number: int, save_path: str = None
    ):
        """Plot glucose curve fitting results."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Plot data and fit
        ax.scatter(
            result["t"],
            result["y_true"],
            label="Observations",
            alpha=0.7,
            s=30,
            color="blue",
        )
        ax.plot(
            result["t"],
            result["y_fit"],
            "r-",
            label="Fitted Curve",
            linewidth=2,
            color="red",
        )

        ax.set_xlabel("Time (minutes)", fontsize=12)
        ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
        ax.set_title(
            f"Meal {result['meal']} - {result['userID']}: "
            f"Glucose Response Curve Fitting",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)

        # Add parameter information
        info_text = (
            f"A = {result['A']:.2f} mg/dL\n"
            f"δ = {result['delta']:.2f} min\n"
            f"σ = {result['sigma']:.2f} min\n"
            f"b = {result['baseline']:.2f} mg/dL\n"
            f"R² = {result['r_squared']:.3f}"
        )

        ax.text(
            0.02,
            0.98,
            info_text,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.legend(fontsize=10)
        plt.tight_layout()

        if save_path or self.save_plots:
            if save_path is None:
                save_path = config.get_results_path("figures")
                save_path = (
                    Path(save_path) / f"glucose_fit_{meal_number}.{self.plot_format}"
                )
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, ax

    def plot_parameter_distributions(
        self, df_fitted_params: pd.DataFrame, save_path: str = None
    ):
        """Plot distributions of fitted parameters A, δ, σ."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)

        parameters = ["A", "delta", "sigma", "baseline"]
        titles = [
            "Amplitude (A)",
            "Time to Peak (δ)",
            "Curve Width (σ)",
            "Baseline (b)",
        ]
        units = ["mg/dL", "min", "min", "mg/dL"]

        for i, (param, title, unit) in enumerate(zip(parameters, titles, units)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            if param in df_fitted_params.columns:
                data = df_fitted_params[param].dropna()

                # Histogram
                ax.hist(
                    data,
                    bins=30,
                    alpha=0.7,
                    density=True,
                    color="skyblue",
                    edgecolor="black",
                )

                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                q25, q75 = data.quantile([0.25, 0.75])

                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.2f}",
                )
                ax.axvline(
                    q25,
                    color="orange",
                    linestyle=":",
                    linewidth=1,
                    label=f"Q25: {q25:.2f}",
                )
                ax.axvline(
                    q75,
                    color="orange",
                    linestyle=":",
                    linewidth=1,
                    label=f"Q75: {q75:.2f}",
                )

                ax.set_xlabel(f"{title} ({unit})")
                ax.set_ylabel("Density")
                ax.set_title(f"Distribution of {title}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.save_plots:
            if save_path is None:
                save_path = (
                    config.get_results_path("figures")
                    / f"parameter_distributions.{self.plot_format}"
                )
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, axes

    def plot_model_performance(self, model_results: Dict, save_path: str = None):
        """Plot model performance metrics for all targets."""
        targets = list(model_results.keys())
        n_targets = len(targets)

        fig, axes = plt.subplots(
            2, n_targets, figsize=(5 * n_targets, 10), dpi=self.dpi
        )

        if n_targets == 1:
            axes = axes.reshape(-1, 1)

        for i, target_name in enumerate(targets):
            model = model_results[target_name]["model"]
            test_metrics = model_results[target_name]["test_metrics"]

            # Prediction vs True plot
            ax1 = axes[0, i]
            y_true = model.y_test_
            y_pred = model.predict(model.X_test_)

            ax1.scatter(y_true, y_pred, alpha=0.6)
            min_val, max_val = (
                min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max()),
            )
            ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

            ax1.set_xlabel(f"True {target_name}")
            ax1.set_ylabel(f"Predicted {target_name}")
            ax1.set_title(
                f"{target_name.title()} - Predictions vs Truth\n"
                f"R² = {test_metrics['r_squared']:.3f}, "
                f"RMSE = {test_metrics['rmse']:.3f}"
            )
            ax1.grid(True, alpha=0.3)

            # Residuals plot
            ax2 = axes[1, i]
            residuals = y_pred - y_true
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color="r", linestyle="--")
            ax2.set_xlabel(f"Predicted {target_name}")
            ax2.set_ylabel("Residuals")
            ax2.set_title(f"{target_name.title()} - Residuals")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.save_plots:
            if save_path is None:
                save_path = (
                    config.get_results_path("figures")
                    / f"model_performance.{self.plot_format}"
                )
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, axes

    def plot_bland_altman(
        self, bland_altman_data: Dict, target_name: str, save_path: str = None
    ):
        """Create Bland-Altman plot for model evaluation."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        means = bland_altman_data["means"]
        differences = bland_altman_data["differences"]
        bias = bland_altman_data["bias"]
        la95_upper = bland_altman_data["la95_upper"]
        la95_lower = bland_altman_data["la95_lower"]

        # Scatter plot
        ax.scatter(means, differences, alpha=0.6, s=30)

        # Bias line
        ax.axhline(
            bias, color="red", linestyle="-", linewidth=2, label=f"Bias: {bias:.2f}"
        )

        # Limits of agreement
        ax.axhline(
            la95_upper,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Upper LA: {la95_upper:.2f}",
        )
        ax.axhline(
            la95_lower,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Lower LA: {la95_lower:.2f}",
        )

        ax.set_xlabel(f"Mean of True and Predicted {target_name}")
        ax.set_ylabel(f"Difference (Predicted - True) {target_name}")
        ax.set_title(f"Bland-Altman Plot - {target_name.title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation info
        corr_text = (
            f"Bias correlation: r = {bland_altman_data['bias_correlation']:.3f}\n"
        )
        corr_text += f"p-value: {bland_altman_data['bias_p_value']:.4f}"
        ax.text(
            0.02,
            0.98,
            corr_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path or self.save_plots:
            if save_path is None:
                save_path = (
                    config.get_results_path("figures")
                    / f"bland_altman_{target_name}.{self.plot_format}"
                )
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, ax

    def plot_shap_values(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        target_name: str,
        save_path: str = None,
    ):
        """Plot SHAP values for feature importance."""

        # Create two separate plots instead of subplots
        # Plot 1: Bar plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {target_name.title()}")
        plt.tight_layout()

        if save_path or self.save_plots:
            # Convert Path to string if needed
            save_path_str = str(save_path) if save_path else None
            bar_path = (
                save_path_str.replace(".png", "_bar.png") if save_path_str else None
            )
            if bar_path is None:
                bar_path = (
                    config.get_results_path("figures")
                    / f"shap_values_{target_name}_bar.{self.plot_format}"
                )
            plt.savefig(bar_path, dpi=self.dpi, bbox_inches="tight")

        # Plot 2: Beeswarm plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary Plot - {target_name.title()}")
        plt.tight_layout()

        if save_path or self.save_plots:
            # Convert Path to string if needed
            save_path_str = str(save_path) if save_path else None
            summary_path = (
                save_path_str.replace(".png", "_summary.png") if save_path_str else None
            )
            if summary_path is None:
                summary_path = (
                    config.get_results_path("figures")
                    / f"shap_values_{target_name}_summary.{self.plot_format}"
                )
            plt.savefig(summary_path, dpi=self.dpi, bbox_inches="tight")

        return None  # Return None since we're not using subplots

    def plot_macronutrient_distributions(
        self, df_ml: pd.DataFrame, save_path: str = None
    ):
        """Plot macronutrient distributions as mentioned in the research paper."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)

        macronutrients = ["CHO", "PRO", "FAT"]
        titles = ["Carbohydrates (CHO)", "Protein (PRO)", "Fat (FAT)"]
        colors = ["lightcoral", "lightblue", "lightgreen"]

        for i, (macro, title, color) in enumerate(zip(macronutrients, titles, colors)):
            if macro in df_ml.columns:
                data = df_ml[macro].dropna()

                # Density plot
                axes[i].hist(
                    data,
                    bins=30,
                    alpha=0.7,
                    density=True,
                    color=color,
                    edgecolor="black",
                )

                # Add statistics
                mean_val = data.mean()
                median_val = data.median()

                axes[i].axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.1f}g",
                )
                axes[i].axvline(
                    median_val,
                    color="blue",
                    linestyle=":",
                    linewidth=2,
                    label=f"Median: {median_val:.1f}g",
                )

                axes[i].set_xlabel(f"{title} (g)")
                axes[i].set_ylabel("Density")
                axes[i].set_title(f"Distribution of {title}")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.save_plots:
            if save_path is None:
                save_path = (
                    config.get_results_path("figures")
                    / f"macronutrient_distributions.{self.plot_format}"
                )
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, axes

    def create_comprehensive_report(
        self,
        df_fitted_params: pd.DataFrame,
        model_results: Dict,
        df_ml: pd.DataFrame,
        xgboost_regressor=None,
    ):
        """Create comprehensive visualization report."""
        logger.info("Creating comprehensive visualization report...")

        figures_path = config.get_results_path("figures")
        figures_path.mkdir(parents=True, exist_ok=True)

        # 1. Parameter distributions
        self.plot_parameter_distributions(
            df_fitted_params,
            figures_path / f"parameter_distributions.{self.plot_format}",
        )

        # 2. Macronutrient distributions
        self.plot_macronutrient_distributions(
            df_ml, figures_path / f"macronutrient_distributions.{self.plot_format}"
        )

        # 3. Model performance
        self.plot_model_performance(
            model_results, figures_path / f"model_performance.{self.plot_format}"
        )

        # 4. Bland-Altman plots for each target
        if xgboost_regressor:
            for target_name in model_results.keys():
                bland_altman_data = xgboost_regressor.create_bland_altman_analysis(
                    target_name
                )
                self.plot_bland_altman(
                    bland_altman_data,
                    target_name,
                    figures_path / f"bland_altman_{target_name}.{self.plot_format}",
                )

                # SHAP plots
                if len(df_ml) > 100:  # Only if we have enough samples
                    X_sample = df_ml.sample(
                        min(100, len(df_ml))
                    )  # Sample for SHAP analysis
                    X_features, _ = xgboost_regressor.prepare_features(X_sample)
                    shap_values = xgboost_regressor.calculate_shap_values(
                        X_features, target_name
                    )
                    self.plot_shap_values(
                        shap_values,
                        X_features,
                        target_name,
                        figures_path / f"shap_values_{target_name}.{self.plot_format}",
                    )

        logger.info(f"Comprehensive report saved to {figures_path}")
