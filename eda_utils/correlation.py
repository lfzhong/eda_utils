from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import NumericType
import pandas as pd

def compute_spark_correlation_matrix(df_spark, features=None, dropna=True):
    """
    Computes Pearson correlation matrix among numeric features in a Spark DataFrame.

    Parameters:
    - df_spark: Spark DataFrame
    - features: List of numeric feature names to include (optional)
    - dropna: Whether to drop rows with any nulls in selected features

    Returns:
    - Pandas DataFrame of correlation matrix
    """
    # Auto-select numeric features
    if features is None:
        features = [f.name for f in df_spark.schema.fields if isinstance(f.dataType, NumericType)]

    # Drop rows with nulls (Spark VectorAssembler cannot handle nulls)
    if dropna:
        df_filtered = df_spark.select(*features).dropna()
    else:
        df_filtered = df_spark.select(*features)

    # Assemble into vector column
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    df_vector = assembler.transform(df_filtered).select("features")

    # Compute Pearson correlation
    corr_matrix = Correlation.corr(df_vector, "features", "pearson").head()[0].toArray()
    corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)

    return corr_df
