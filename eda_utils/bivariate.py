
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col, count, mean
from pyspark.sql.types import StringType, NumericType
from pyspark.ml.feature import QuantileDiscretizer

def run_numerical_bivariate_analysis(df_spark, target_col, num_cols=None, num_bins=5, top_features=None):
    from pyspark.sql.functions import col, count, mean
    from pyspark.ml.feature import QuantileDiscretizer
    import pandas as pd

    results = {}
    num_cols = [col for col in num_cols if col != target_col]

    for col_name in num_cols:
        binned_col = f"{col_name}_bin"
        try:
            discretizer = QuantileDiscretizer(
                numBuckets=num_bins,
                inputCol=col_name,
                outputCol=binned_col,
                handleInvalid="skip",
                relativeError=0.01
            )
            model = discretizer.fit(df_spark)
            splits = model.getSplits()

            df_binned = model.transform(df_spark)

            agg_df = (
                df_binned.groupBy(binned_col)
                .agg(count("*").alias("count"), mean(target_col).alias("target_rate"))
                .orderBy(binned_col)
            )

            pdf = agg_df.toPandas()
            # Add readable bin range
            bin_labels = [f"[{splits[i]:.2f}, {splits[i+1]:.2f})" for i in range(len(splits)-1)]
            pdf["bin_label"] = bin_labels[:len(pdf)]
            results[col_name] = pdf[["bin_label", "count", "target_rate"]]

        except Exception as e:
            print(f"Skipped {col_name}: {e}")

    if top_features is not None:
        sorted_cols = sorted(results.items(), key=lambda x: x[1]["target_rate"].std(), reverse=True)
        results = dict(sorted_cols[:top_features])

    return results
  
def run_categorical_bivariate_analysis(df_spark, target_col, cat_cols, top_k=10, top_features=None):
    results = {}
    cat_cols = [c for c in cat_cols if c != target_col]

    for col_name in cat_cols:
        agg_df = (
            df_spark.groupBy(col_name)
            .agg(count("*").alias("count"), mean(target_col).alias("target_rate"))
            .orderBy("count", ascending=False)
            .limit(top_k)
        )
        pdf = agg_df.toPandas()
        results[col_name] = pdf

    if top_features is not None:
        sorted_cols = sorted(results.items(), key=lambda x: x[1]["target_rate"].std(), reverse=True)
        results = dict(sorted_cols[:top_features])

    return results

def plot_bivariate_categorical(results_cat, plots_per_row=2):
    num_features = len(results_cat)
    n_rows = (num_features + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(6 * plots_per_row, 4 * n_rows))
    axes = axes.flatten()

    for idx, (feature, df) in enumerate(results_cat.items()):
        ax = axes[idx]
        bars = ax.bar(df[feature].astype(str), df["target_rate"])
        ax.set_title(f"{feature} vs. Target Rate")
        ax.set_xlabel(feature)
        ax.set_ylabel("Target Rate")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[feature].astype(str), rotation=45)
        ax.grid(False)

        for bar, rate in zip(bars, df["target_rate"]):
            label_y = bar.get_height() * 1.02
            ax.text(bar.get_x() + bar.get_width()/2, label_y, f"{rate * 100:.2f}%", 
                    ha='center', va='bottom')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_bivariate_numerical(results_num, plots_per_row=2):
    num_features = len(results_num)
    n_rows = (num_features + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(6 * plots_per_row, 4 * n_rows))
    axes = axes.flatten()

    for idx, (feature, df) in enumerate(results_num.items()):
        ax = axes[idx]
        bars = ax.bar(df["bin_label"], df["target_rate"])
        ax.set_title(f"{feature} (binned) vs. Target Rate")
        ax.set_xlabel("Bin Range")
        ax.set_ylabel("Target Rate")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["bin_label"], rotation=45)
        ax.grid(False)

        for bar, rate in zip(bars, df["target_rate"]):
            label_y = bar.get_height() * 1.02
            ax.text(bar.get_x() + bar.get_width()/2, label_y, f"{rate * 100:.2f}%", 
                    ha='center', va='bottom')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def run_bivariate_analysis(df_spark, target_col, cat_cols=None, num_cols=None,
                           num_bins=5, top_k=10, top_features=None, plot=True):
    if cat_cols is None:
        cat_cols = [f.name for f in df_spark.schema.fields if isinstance(f.dataType, StringType)]
    if num_cols is None:
        num_cols = [f.name for f in df_spark.schema.fields if isinstance(f.dataType, NumericType)]

    cat_results = run_categorical_bivariate_analysis(df_spark, target_col, cat_cols, top_k, top_features)
    num_results = run_numerical_bivariate_analysis(df_spark, target_col, num_cols, num_bins, top_features)

    if plot:
        plot_bivariate_categorical(cat_results)
        plot_bivariate_numerical(num_results)

    return cat_results, num_results
