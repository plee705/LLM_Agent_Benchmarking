from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(__file__).resolve().parent / "temp_testing.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "analysis_plots"

PRIMARY_TEXT_COLOR = "#4D4D4F"
SECONDARY_COLOR = "#BFBFBF"
PRIMARY_PLOT_COLOR_1 = "#324458"
PRIMARY_PLOT_COLOR_2 = "#F47B20"
ACCESSORY_LINE_COLOR = "#868687"


def load_temperature_test_data(csv_path: Path) -> pd.DataFrame:
	"""Read the temperature test CSV using pandas."""
	return pd.read_csv(csv_path)


def normalize_temperature_test_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Coerce data types and add derived columns used by downstream analyses."""
	normalized = df.copy()

	numeric_columns = ["temp", "elapsed_sec", "total_tokens", "output_tokens"]
	for col in numeric_columns:
		normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

	normalized["timed_out"] = (
		normalized["timed_out"].astype(str).str.strip().str.upper().eq("TRUE")
	)

	normalized["tokens_per_second"] = np.where(
		normalized["elapsed_sec"] > 0,
		normalized["total_tokens"] / normalized["elapsed_sec"],
		np.nan,
	)

	normalized["time_per_token_sec"] = np.where(
		normalized["total_tokens"] > 0,
		normalized["elapsed_sec"] / normalized["total_tokens"],
		np.nan,
	)

	return normalized


def calculate_total_model_run_time(df: pd.DataFrame) -> float:
	return float(df["elapsed_sec"].sum())


def calculate_total_overall_tokens(df: pd.DataFrame) -> int:
	return int(df["total_tokens"].sum())


def calculate_total_output_tokens(df: pd.DataFrame) -> int:
	return int(df["output_tokens"].sum())


def calculate_test_wide_time_per_token(df: pd.DataFrame) -> float:
	total_time = calculate_total_model_run_time(df)
	total_tokens = calculate_total_overall_tokens(df)
	return float(total_time / total_tokens) if total_tokens > 0 else np.nan


def calculate_test_wide_tokens_per_second(df: pd.DataFrame) -> float:
	total_time = calculate_total_model_run_time(df)
	total_tokens = calculate_total_overall_tokens(df)
	return float(total_tokens / total_time) if total_time > 0 else np.nan


def calculate_test_wide_timeout_rate(df: pd.DataFrame) -> float:
	return float(df["timed_out"].mean())


def calculate_test_wide_metrics(df: pd.DataFrame) -> pd.Series:
	metrics = {
		"total_model_run_time_sec": calculate_total_model_run_time(df),
		"total_overall_tokens": calculate_total_overall_tokens(df),
		"total_output_tokens": calculate_total_output_tokens(df),
		"time_per_token_sec": calculate_test_wide_time_per_token(df),
		"tokens_per_second": calculate_test_wide_tokens_per_second(df),
		"timeout_rate": calculate_test_wide_timeout_rate(df),
	}
	return pd.Series(metrics)


def calculate_temperature_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		df.groupby("temp", as_index=False)
		.agg(
			average_category_run_time_sec=("elapsed_sec", "mean"),
			average_total_tokens=("total_tokens", "mean"),
			average_output_tokens=("output_tokens", "mean"),
			total_token_use=("total_tokens", "sum"),
			average_tokens_per_sec=("tokens_per_second", "mean"),
			timeout_rate=("timed_out", "mean"),
			category_tests=("run_id", "count"),
			category_timeouts=("timed_out", "sum"),
		)
		.sort_values("temp")
	)
	return grouped


def print_test_wide_metrics(metrics: pd.Series) -> None:
	print("\n=== Test-Wide Metrics ===")
	print(f"Total model run time (sec): {metrics['total_model_run_time_sec']:.2f}")
	print(f"Total overall tokens: {int(metrics['total_overall_tokens'])}")
	print(f"Total output tokens: {int(metrics['total_output_tokens'])}")
	print(f"Time per token (sec/token): {metrics['time_per_token_sec']:.6f}")
	print(f"Tokens per second: {metrics['tokens_per_second']:.3f}")
	print(f"Timeout rate: {metrics['timeout_rate']:.2%}")


def print_temperature_category_metrics(category_metrics: pd.DataFrame) -> None:
	print("\n=== Temperature Category Metrics ===")
	display_df = category_metrics.copy()
	display_df["timeout_rate"] = display_df["timeout_rate"].map(lambda x: f"{x:.2%}")
	print(display_df.to_string(index=False))


def _mean_by_temperature(
	df: pd.DataFrame, column: str, exclude_timeouts: bool = False
) -> pd.Series:
	"""Compute mean of a column by temperature with optional timeout filtering."""
	data = df.loc[~df["timed_out"]] if exclude_timeouts else df
	temps = sorted(df["temp"].dropna().unique())
	series = data.groupby("temp")[column].mean()
	return series.reindex(temps)


def _style_axis(ax: plt.Axes) -> None:
	"""Apply shared axis styling for text and boundaries."""
	for spine in ax.spines.values():
		spine.set_color(PRIMARY_TEXT_COLOR)
	ax.tick_params(colors=PRIMARY_TEXT_COLOR)
	ax.xaxis.label.set_color(PRIMARY_TEXT_COLOR)
	ax.yaxis.label.set_color(PRIMARY_TEXT_COLOR)
	ax.title.set_color(PRIMARY_TEXT_COLOR)
	ax.grid(axis="y", linestyle=":", color=SECONDARY_COLOR, alpha=0.9)


def _place_legend_below_axis(ax: plt.Axes, ncol: int = 2) -> None:
	"""Place legend below the x-axis."""
	legend = ax.legend(
		loc="upper center",
		bbox_to_anchor=(0.5, -0.2),
		ncol=ncol,
		frameon=False,
	)
	if legend is not None:
		for text in legend.get_texts():
			text.set_color(PRIMARY_TEXT_COLOR)


def plot_parameter_dual_bar(
	df: pd.DataFrame,
	column: str,
	title: str,
	y_label: str,
	output_path: Path,
	include_overall_lines: bool = False,
	value_scale: float = 1.0,
) -> None:
	"""Plot include-timeouts vs exclude-timeouts bar chart by temperature."""
	if value_scale <= 0:
		raise ValueError("value_scale must be greater than 0")

	include_series = _mean_by_temperature(df, column, exclude_timeouts=False)
	exclude_series = _mean_by_temperature(df, column, exclude_timeouts=True)

	temps = include_series.index.astype(str)
	x = np.arange(len(temps))
	bar_width = 0.38

	fig, ax = plt.subplots(figsize=(9, 5))
	ax.bar(
		x - bar_width / 2,
		include_series.values / value_scale,
		bar_width,
		label="Include Timeouts",
		color=PRIMARY_PLOT_COLOR_1,
	)
	ax.bar(
		x + bar_width / 2,
		exclude_series.values / value_scale,
		bar_width,
		label="Exclude Timeouts",
		color=PRIMARY_PLOT_COLOR_2,
	)

	if include_overall_lines:
		overall_with = df[column].mean() / value_scale
		overall_without = df.loc[~df["timed_out"], column].mean() / value_scale
		ax.axhline(
			overall_with,
			color=ACCESSORY_LINE_COLOR,
			linestyle="--",
			linewidth=1.6,
			label="Overall Avg (with timeouts)",
		)
		ax.axhline(
			overall_without,
			color=ACCESSORY_LINE_COLOR,
			linestyle=":",
			linewidth=1.8,
			label="Overall Avg (without timeouts)",
		)

	ax.set_xticks(x)
	ax.set_xticklabels(temps)
	ax.set_xlabel("Temperature")
	ax.set_ylabel(y_label)
	ax.set_title(title)
	_style_axis(ax)
	_place_legend_below_axis(ax, ncol=2)

	fig.subplots_adjust(bottom=0.28, right=0.86)
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_timeout_rate_single_bar(df: pd.DataFrame, output_path: Path) -> None:
	"""Plot timeout rate by temperature as a single-bar chart."""
	timeout_rate = df.groupby("temp")["timed_out"].mean().sort_index()

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.bar(
		timeout_rate.index.astype(str),
		timeout_rate.values * 100,
		color=PRIMARY_PLOT_COLOR_1,
	)
	ax.set_xlabel("Temperature")
	ax.set_ylabel("Timeout Rate (%)")
	ax.set_title("Timeout Rate by Temperature")
	_style_axis(ax)

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_fastest_vs_slowest_solve_time(df: pd.DataFrame, output_path: Path) -> None:
	"""Plot fastest solve times with a ghost bar for slowest solve times."""
	non_timeout_df = df.loc[~df["timed_out"]].copy()
	if non_timeout_df.empty:
		return

	agg = (
		non_timeout_df.groupby("temp", as_index=False)
		.agg(
			fastest_solve_time_sec=("elapsed_sec", "min"),
			slowest_solve_time_sec=("elapsed_sec", "max"),
		)
		.sort_values("temp")
	)

	timeout_limit = (
		float(df["timeout_sec"].dropna().iloc[0])
		if "timeout_sec" in df.columns and not df["timeout_sec"].dropna().empty
		else 150.0
	)

	x = np.arange(len(agg))
	fig, ax = plt.subplots(figsize=(9, 5))

	ax.bar(
		x,
		agg["slowest_solve_time_sec"],
		width=0.62,
		color=SECONDARY_COLOR,
		alpha=0.45,
		label="Slowest Solve Time (ghost)",
	)
	ax.bar(
		x,
		agg["fastest_solve_time_sec"],
		width=0.38,
		color=PRIMARY_PLOT_COLOR_1,
		label="Fastest Solve Time",
	)
	ax.axhline(
		timeout_limit,
		color=ACCESSORY_LINE_COLOR,
		linestyle="--",
		linewidth=1.8,
		label=f"Timeout Threshold ({timeout_limit:.0f} sec)",
	)

	ax.set_xticks(x)
	ax.set_xticklabels(agg["temp"].astype(str))
	ax.set_xlabel("Temperature")
	ax.set_ylabel("Solve Time (sec)")
	ax.set_title("Fastest Solve Time vs Slowest (Ghost) by Temperature")
	_style_axis(ax)
	_place_legend_below_axis(ax, ncol=2)

	fig.subplots_adjust(bottom=0.28, right=0.86)
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_temperature_metrics(df: pd.DataFrame, output_dir: Path) -> list[Path]:
	"""Create all requested temperature analysis plots and return output paths."""
	output_dir.mkdir(parents=True, exist_ok=True)

	output_paths = [
		output_dir / "avg_runtime_vs_temp_dual_include_exclude_timeouts.png",
		output_dir / "avg_total_tokens_vs_temp_dual_include_exclude_timeouts.png",
		output_dir / "avg_output_tokens_vs_temp_dual_include_exclude_timeouts.png",
		output_dir / "avg_tokens_per_sec_vs_temp_dual_include_exclude_timeouts.png",
		output_dir / "timeout_rate_vs_temp_single_bar.png",
		output_dir / "fastest_vs_slowest_solve_time_vs_temp.png",
	]

	plot_parameter_dual_bar(
		df=df,
		column="elapsed_sec",
		title="Average Run Time vs Temperature (Include/Exclude Timeouts)",
		y_label="Average Run Time (sec)",
		output_path=output_paths[0],
		include_overall_lines=True,
	)
	plot_parameter_dual_bar(
		df=df,
		column="total_tokens",
		title="Average Total Tokens vs Temperature (Include/Exclude Timeouts)",
		y_label="Average Total Tokens (thousands)",
		output_path=output_paths[1],
		value_scale=1000.0,
	)
	plot_parameter_dual_bar(
		df=df,
		column="output_tokens",
		title="Average Output Tokens vs Temperature (Include/Exclude Timeouts)",
		y_label="Average Output Tokens (thousands)",
		output_path=output_paths[2],
		value_scale=1000.0,
	)
	plot_parameter_dual_bar(
		df=df,
		column="tokens_per_second",
		title="Average Tokens/Sec vs Temperature (Include/Exclude Timeouts)",
		y_label="Average Tokens/Sec",
		output_path=output_paths[3],
	)

	plot_timeout_rate_single_bar(df, output_paths[4])
	plot_fastest_vs_slowest_solve_time(df, output_paths[5])

	return output_paths


def build_metric_metadata_table(
	test_wide_metrics: pd.Series, category_metrics: pd.DataFrame
) -> pd.DataFrame:
	"""Build a long-form table of metric values and calculation definitions."""
	records = []

	test_metric_definitions = [
		(
			"total_model_run_time_sec",
			"sec",
			"sum(elapsed_sec)",
			"Total model run time across all tests.",
		),
		(
			"total_overall_tokens",
			"tokens",
			"sum(total_tokens)",
			"Total overall token use across all tests.",
		),
		(
			"total_output_tokens",
			"tokens",
			"sum(output_tokens)",
			"Total output token use across all tests.",
		),
		(
			"time_per_token_sec",
			"sec/token",
			"sum(elapsed_sec) / sum(total_tokens)",
			"Test-wide average time per token.",
		),
		(
			"tokens_per_second",
			"tokens/sec",
			"sum(total_tokens) / sum(elapsed_sec)",
			"Test-wide token throughput.",
		),
		(
			"timeout_rate",
			"ratio",
			"sum(timed_out) / count(tests)",
			"Timeout rate across all tests.",
		),
	]

	for metric_name, unit, formula, description in test_metric_definitions:
		records.append(
			{
				"scope": "test_wide",
				"temperature": "ALL",
				"metric": metric_name,
				"value": float(test_wide_metrics[metric_name]),
				"unit": unit,
				"formula": formula,
				"description": description,
			}
		)

	category_metric_definitions = {
		"average_category_run_time_sec": (
			"sec",
			"mean(elapsed_sec) within temperature",
			"Average runtime per test in this temperature category.",
		),
		"average_total_tokens": (
			"tokens",
			"mean(total_tokens) within temperature",
			"Average total tokens per test in this temperature category.",
		),
		"average_output_tokens": (
			"tokens",
			"mean(output_tokens) within temperature",
			"Average output tokens per test in this temperature category.",
		),
		"total_token_use": (
			"tokens",
			"sum(total_tokens) within temperature",
			"Total token use in this temperature category.",
		),
		"average_tokens_per_sec": (
			"tokens/sec",
			"mean(total_tokens / elapsed_sec) within temperature",
			"Average throughput in this temperature category.",
		),
		"timeout_rate": (
			"ratio",
			"sum(timed_out) / count(category tests)",
			"Timeout rate in this temperature category.",
		),
		"category_tests": (
			"count",
			"count(run_id) within temperature",
			"Number of tests in this temperature category.",
		),
		"category_timeouts": (
			"count",
			"sum(timed_out) within temperature",
			"Number of timed out tests in this temperature category.",
		),
	}

	for _, row in category_metrics.iterrows():
		temp_value = row["temp"]
		for metric_name, (unit, formula, description) in category_metric_definitions.items():
			records.append(
				{
					"scope": "temperature_category",
					"temperature": temp_value,
					"metric": metric_name,
					"value": float(row[metric_name]),
					"unit": unit,
					"formula": formula,
					"description": description,
				}
			)

	return pd.DataFrame(records)


def save_metrics_and_metadata(
	test_wide_metrics: pd.Series, category_metrics: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
	"""Write metrics and calculation metadata to CSV and TXT outputs."""
	output_dir.mkdir(parents=True, exist_ok=True)

	metadata_table = build_metric_metadata_table(test_wide_metrics, category_metrics)

	csv_output_path = output_dir / "temp_analysis_metrics_metadata.csv"
	txt_output_path = output_dir / "temp_analysis_metrics_metadata.txt"

	metadata_table.to_csv(csv_output_path, index=False)

	with txt_output_path.open("w", encoding="utf-8") as f:
		f.write("Temperature Test Analysis Report\n")
		f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
		f.write(f"Source CSV: {CSV_PATH}\n\n")

		f.write("Test-Wide Metrics\n")
		for metric_name, value in test_wide_metrics.items():
			f.write(f"- {metric_name}: {value}\n")

		f.write("\nTemperature Category Metrics\n")
		f.write(category_metrics.to_string(index=False))
		f.write("\n\nCalculation Metadata\n")

		metadata_cols = [
			"scope",
			"temperature",
			"metric",
			"unit",
			"formula",
			"description",
		]
		metadata_unique = metadata_table[metadata_cols].drop_duplicates()
		f.write(metadata_unique.to_string(index=False))
		f.write("\n")

	return csv_output_path, txt_output_path


def main() -> None:
	df = load_temperature_test_data(CSV_PATH)
	df = normalize_temperature_test_data(df)

	test_wide_metrics = calculate_test_wide_metrics(df)
	category_metrics = calculate_temperature_category_metrics(df)

	print_test_wide_metrics(test_wide_metrics)
	print_temperature_category_metrics(category_metrics)
	plot_paths = plot_temperature_metrics(df, PLOTS_DIR)
	csv_output_path, txt_output_path = save_metrics_and_metadata(
		test_wide_metrics, category_metrics, OUTPUT_DIR
	)
	print(f"\nSaved CSV metadata output: {csv_output_path}")
	print(f"Saved TXT metadata output: {txt_output_path}")
	print("Saved plot outputs:")
	for path in plot_paths:
		print(f"- {path}")


if __name__ == "__main__":
	main()

