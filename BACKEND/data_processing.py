import pandas as pd
import re
import streamlit as st
import difflib
#import pandas as pd

#import pandas as pd

#import pandas as pd
#import re

@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)  # Removed infer_datetime_format
    # Use regex-based conversion for date-like columns if needed:
    date_pattern = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?$")
    for col in data.select_dtypes(include=["object"]).columns:
        sample = data[col].dropna().head(5)
        if not sample.empty and sample.apply(lambda x: bool(date_pattern.match(x.strip())) if isinstance(x, str) else False).all():
            try:
                data[col] = data[col].apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
                data[col] = pd.to_datetime(data[col], errors="coerce")  # No infer_datetime_format
            except Exception as e:
                print(f"Failed to convert {col} to datetime: {e}")
    return data





def get_summary_stats(data):
    """Return summary statistics of the DataFrame."""
    return data.describe()

def get_numerical_columns(data):
    """Return a list of numerical columns in the DataFrame."""
    return data.select_dtypes(include=["number"]).columns.tolist()

def get_categorical_columns(data):
    """Return a list of categorical columns in the DataFrame."""
    return data.select_dtypes(include=["object", "category"]).columns.tolist()

def get_datetime_columns(data):
    """Return a list of columns that are datetime64[ns]."""
    return data.select_dtypes(include=["datetime64[ns]"]).columns.tolist()


def get_selectable_columns(data):
    """Return a list of columns that are either numerical, categorical, or datetime."""
    numeric = get_numerical_columns(data)
    categorical = get_categorical_columns(data)
    datetime_cols = get_datetime_columns(data)
    return numeric + categorical + datetime_cols

def get_column_statistics(data, column):
    """Generate a string with key statistics for the selected numerical column."""
    summary = data[column].describe()
    cardinality = data[column].nunique()
    nan_count = data[column].isna().sum()
    stats_string = (
        f"Mean: {summary['mean']:.2f}, SD: {summary['std']:.2f}, "
        f"Min: {summary['min']}, Max: {summary['max']}, "
        f"Cardinality: {cardinality}, null values: {nan_count}"
    )
    return stats_string

def get_categorical_frequencies(data, column):
    """Return frequency counts for the selected categorical column."""
    return data[column].value_counts()

def get_all_categorical_summaries(data):
    """Return a dictionary of summary statistics for each categorical column."""
    summaries = {}
    cat_cols = get_categorical_columns(data)
    for col in cat_cols:
        summaries[col] = get_categorical_summary(data, col)
    return summaries


def get_categorical_summary(data, column):
    """Return summary statistics for a categorical column."""
    frequencies = data[column].value_counts()
    mode_value = frequencies.idxmax() if not frequencies.empty else None
    mode_freq = frequencies.max() if not frequencies.empty else None
    cardinality = data[column].nunique()
    total_count = data[column].count()
    missing_count = data[column].isna().sum()  # Count missing values

    summary = {
        "mode": mode_value,
        "mode_frequency": mode_freq,
        "cardinality": cardinality,
        "total_count": total_count,
        "missing_values": missing_count,      # Include missing-value info
        "frequency_distribution": frequencies.to_dict()
    }
    return summary

def is_duplicate(candidate, ai_list, threshold=0.8):
    """
    Returns True if the candidate visualization name is similar enough
    to any name in the ai_list based on a similarity threshold.
    """
    candidate_lower = candidate.lower()
    for ai_vis in ai_list:
        # Compute similarity ratio between candidate and ai suggestion
        ratio = difflib.SequenceMatcher(None, candidate_lower, ai_vis.lower()).ratio()
        if ratio >= threshold:
            return True
    return False

def row_to_text(row):
    return ", ".join([f"{col}: {row[col]}" for col in row.index])
        
