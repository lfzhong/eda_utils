from pyspark.sql.functions import col, count, isnan, stddev as spark_stddev, skewness as spark_skewness
from pyspark.sql.types import NumericType, StringType
import pandas as pd
import matplotlib.pyplot as plt

def describe_numeric_features(df_spark, features=None):
    """
    Summarizes numeric features in a Spark DataFrame.

    Parameters:
    - df_spark: Spark DataFrame
    - features: list of column names to include (default: all numeric columns)

    Returns:
    - Pandas DataFrame with summary statistics for numeric features
    """
    total_rows = df_spark.count()
    summary = []

    if features is None:
        features = [f.name for f in df_spark.schema.fields if isinstance(f.dataType, NumericType)]
    else:
        features = [col for col in features if isinstance(df_spark.schema[col].dataType, NumericType)]
        if not features:
            print("No numeric features found in features.")
            return pd.DataFrame()

    for col_name in features:
        field = df_spark.schema[col_name]

        missing_count = df_spark.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        missing_pct = round(missing_count / total_rows * 100, 2)
        n_unique = df_spark.select(col_name).distinct().count()

        stats = df_spark.select(
            spark_stddev(col(col_name)).alias("std"),
            skewness(col(col_name)).alias("skew")
        ).collect()[0]
        std = stats["std"]
        skew = stats["skew"]

        summary.append({
            "Feature": col_name,
            "Type": field.dataType.simpleString(),
            "% Missing": missing_pct,
            "Std Dev": round(std, 4) if std is not None else None,
            "Skewness": round(skew, 4) if skew is not None else None,
            "Unique Values": n_unique
        })

    return pd.DataFrame(summary).sort_values(by="% Missing", ascending=False)


def describe_categorical_features(df_spark, features=None, top_k=5):
    """
    Summarizes categorical (string) features in a Spark DataFrame.

    Parameters:
    - df_spark: Spark DataFrame
    - features: list of column names to include (default: all string columns)
    - top_k: number of top categories to show

    Returns:
    - Pandas DataFrame with frequency summaries for top categories and % counts
    """
    from pyspark.sql.functions import col, count, isnan
    from pyspark.sql.types import StringType
    import pandas as pd

    total_rows = df_spark.count()
    summary = []

    if features is None:
        features = [f.name for f in df_spark.schema.fields if isinstance(f.dataType, StringType)]
    else:
        features = [col for col in features if isinstance(df_spark.schema[col].dataType, StringType)]
        if not features:
            print("⚠️ No categorical features found in features.")
            return pd.DataFrame()

    for col_name in features:
        field = df_spark.schema[col_name]

        missing_count = df_spark.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        missing_pct = round(missing_count / total_rows * 100, 2)
        n_unique = df_spark.select(col_name).distinct().count()

        top_categories = (
            df_spark.groupBy(col_name)
            .count()
            .orderBy("count", ascending=False)
            .limit(top_k)
            .toPandas()
        )
        top_categories["pct"] = (top_categories["count"] / total_rows * 100).round(2)

        row = {
            "Feature": col_name,
            "Type": field.dataType.simpleString(),
            "% Missing": missing_pct,
            "Unique Values": n_unique
        }
        for i, row_cat in top_categories.iterrows():
            row[f"Top{i+1}"] = f"{row_cat[col_name]} ({row_cat['pct']}%)"

        summary.append(row)

    return pd.DataFrame(summary).sort_values(by="% Missing", ascending=False)

def plot_feature_distributions(df_spark,top_k, features=None, sample_frac=0.05, plots_per_row=2):
    """
    Plots distributions of selected features (numeric as histograms, categorical as bar plots).
    Arranges plots in a grid layout with multiple plots per row.
    Shows % on top of bars for categorical plots.

    Parameters:
    - df_spark: Spark DataFrame
    - features: list of columns to plot (default: all numeric and string)
    - sample_frac: fraction to sample for numeric plots
    - top_k: number of top categories for bar plots
    - plots_per_row: number of plots to display in each row
    """
    from pyspark.sql.types import NumericType, StringType
    import matplotlib.pyplot as plt

    if features is None:
        features = [f.name for f in df_spark.schema.fields if isinstance(f.dataType, (NumericType, StringType))]

    valid_cols = [col for col in features if isinstance(df_spark.schema[col].dataType, (NumericType, StringType))]
    n_plots = len(valid_cols)
    n_rows = (n_plots + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(6 * plots_per_row, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    total_rows = df_spark.count()

    for idx, col_name in enumerate(valid_cols):
        field = df_spark.schema[col_name]
        ax = axes[idx]

        if isinstance(field.dataType, NumericType):
            df_sample = df_spark.select(col_name).dropna().sample(fraction=sample_frac, seed=42).toPandas()
            ax.hist(df_sample[col_name], bins=30, edgecolor='k')
            ax.set_title(f"Distribution of {col_name}")
            ax.set_xlabel(col_name)
            ax.set_ylabel("Frequency")
            ax.grid(False)

        elif isinstance(field.dataType, StringType):
            top_df = (
                df_spark.groupBy(col_name)
                .count()
                .orderBy("count", ascending=False)
                .limit(top_k)
                .toPandas()
            )
            top_df["pct"] = (top_df["count"] / total_rows * 100).round(2)
            bars = ax.bar(top_df[col_name].astype(str), top_df["pct"])
            ax.set_title(f"Top {top_k} Categories: {col_name}")
            ax.set_xlabel(col_name)
            ax.set_ylabel("Percentage")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(False)

            # Add % labels on top of each bar
            for bar, pct in zip(bars, top_df["pct"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{pct:.1f}%", ha='center', va='bottom')

    for i in range(len(valid_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def run_univariate_analysis(df_spark, features=None, top_k=5, plot=True):
    """
    Runs univariate analysis for both numeric and categorical features.

    Parameters:
    - df_spark: Spark DataFrame
    - features: list of column names to include (default: all numeric and categorical)
    - low_std_thresh: optional stddev threshold for flagging low variance (unused)
    - top_k: number of top categories to show
    - plot: whether to show distribution plots

    Returns:
    - Tuple of (numeric_summary_df, categorical_summary_df)
    """
    numeric_summary = describe_numeric_features(df_spark, features)
    categorical_summary = describe_categorical_features(df_spark, features, top_k)

    if plot:
        plot_feature_distributions(df_spark, top_k, features)

    return numeric_summary, categorical_summary
