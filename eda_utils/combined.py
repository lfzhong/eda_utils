def plot_univariate_bivariate(df_spark, target_col, features=None, top_k=10, num_bins=5):
    """
    Plots univariate and bivariate charts side by side for each feature (excluding the target_col).
    Converts binary integer features (with only two unique values) to strings for better categorical plots.
    """
    import matplotlib.pyplot as plt
    from pyspark.sql.types import StringType, NumericType, IntegerType
    from pyspark.sql.functions import count, mean, col
    from pyspark.ml.feature import QuantileDiscretizer
    import pandas as pd

    total_rows = df_spark.count()

    # Step 1: Identify binary integer columns (with exactly 2 unique values)
    binary_cols = [
        f.name for f in df_spark.schema.fields
        if isinstance(f.dataType, IntegerType) and df_spark.select(f.name).distinct().count() == 2
    ]

    # Step 2: Create string-casted columns for binary variables
    for col_name in binary_cols:
        df_spark = df_spark.withColumn(f"{col_name}", col(col_name).cast("string"))

    # Step 3: Automatically select features if not provided
    if features is None:
        features = []
        for f in df_spark.schema.fields:
            if f.name == target_col:
                continue
            elif isinstance(f.dataType, (NumericType, StringType)):
                features.append(f.name)
    else:
        features = [f for f in features if f != target_col]

    n_features = len(features)
    if n_features == 0:
        print("No features to plot after excluding target_col.")
        return

    fig, axes = plt.subplots(n_features, 2, figsize=(14, 4 * n_features))
    if n_features == 1:
        axes = [axes]  # wrap single row

    for i, feature in enumerate(features):
        dtype = df_spark.schema[feature].dataType
        ax1, ax2 = axes[i]

        # --- Univariate ---
        if isinstance(dtype, StringType):
            top_df = (
                df_spark.groupBy(feature)
                .count()
                .orderBy("count", ascending=False)
                .limit(top_k)
                .toPandas()
            )
            top_df["pct"] = (top_df["count"] / total_rows * 100).round(2)
            bars = ax1.bar(top_df[feature].astype(str), top_df["pct"])
            ax1.set_title(f"{feature}")
            ax1.set_ylabel("Percentage")
            ax1.set_xticks(range(len(top_df)))
            ax1.set_xticklabels(top_df[feature].astype(str), rotation=45, ha="right")
            ax1.margins(x=0.01)
            for bar, pct in zip(bars, top_df["pct"]):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{pct:.1f}%", ha='center', va='bottom')

        elif isinstance(dtype, NumericType):
            df_sample = df_spark.select(feature).dropna().toPandas()
            ax1.hist(df_sample[feature], bins=30, edgecolor='k')
            ax1.set_title(f"{feature}")
            ax1.set_ylabel("Frequency")

        # --- Bivariate ---
        if isinstance(dtype, StringType):
            bivar_df = (
                df_spark.groupBy(feature)
                .agg(count("*").alias("count"), mean(target_col).alias("target_rate"))
                .orderBy("count", ascending=False)
                .limit(top_k)
                .toPandas()
            )
            bars = ax2.bar(bivar_df[feature].astype(str), bivar_df["target_rate"])
            ax2.set_title(f"{feature} vs. {target_col}")
            ax2.set_ylabel("Target Rate")
            ax2.set_xticks(range(len(bivar_df)))
            ax2.set_xticklabels(bivar_df[feature].astype(str), rotation=45, ha="right")
            ax2.set_ylim(0, bivar_df["target_rate"].max() * 1.3)
            ax2.margins(x=0.01)
            for bar, rate in zip(bars, bivar_df["target_rate"]):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         rate + bivar_df["target_rate"].max() * 0.05,
                         f"{rate * 100:.2f}%", ha='center', va='bottom')

        elif isinstance(dtype, NumericType):
            discretizer = QuantileDiscretizer(numBuckets=num_bins, inputCol=feature, outputCol="bin_idx", handleInvalid="skip")
            model = discretizer.fit(df_spark)
            df_binned = model.transform(df_spark).dropna(subset=["bin_idx", target_col])

            bin_summary = (
                df_binned.groupBy("bin_idx")
                .agg(mean(target_col).alias("target_rate"), count("*").alias("count"))
                .orderBy("bin_idx")
                .toPandas()
            )
            splits = model.getSplits()
            bin_label_map = {
                i: f"{'-∞' if i==0 else round(splits[i], 2)} – {'∞' if i==len(splits)-2 else round(splits[i+1], 2)}"
                for i in range(len(splits) - 1)
            }
            bin_summary["bin_label"] = bin_summary["bin_idx"].map(bin_label_map)

            bars = ax2.bar(bin_summary["bin_label"], bin_summary["target_rate"])
            ax2.set_title(f"{feature} (binned) vs. {target_col}")
            ax2.set_ylabel("Target Rate")
            ax2.set_xticks(range(len(bin_summary)))
            ax2.set_xticklabels(bin_summary["bin_label"], rotation=45, ha="right")
            ax2.set_ylim(0, bin_summary["target_rate"].max() * 1.3)
            ax2.margins(x=0.01)
            for bar, rate in zip(bars, bin_summary["target_rate"]):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         rate + bin_summary["target_rate"].max() * 0.05,
                         f"{rate * 100:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
