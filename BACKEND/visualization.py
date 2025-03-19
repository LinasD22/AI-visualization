import plotly.express as px
import pandas as pd


def create_histogram(data, column):
    """Create a histogram for a given numerical column."""
    fig = px.histogram(data, x=column, title=f"Histogram of {column}")
    return fig

def create_bar_chart(data, column):
    """Create a bar chart for a given categorical column."""
    counts = data[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.bar(counts, x=column, y='count', title=f"Bar Chart of {column}")
    return fig

def create_line_chart(data, column):
    """
    Create a line chart for a datetime column by grouping entries by date.
    """
    # Ensure the column is datetime
    data[column] = pd.to_datetime(data[column], errors='coerce')
    # Create a new column with just the date (without time)
    data['date_only'] = data[column].dt.date
    # Group by the new date-only column and count entries
    aggregated = data.groupby('date_only').size().reset_index(name='count')
    # Optionally, sort by date if needed
    aggregated = aggregated.sort_values('date_only')
    fig = px.line(aggregated, x='date_only', y='count', title=f"Line Chart of {column} over time")
    return fig

def create_line_chart_with_anomalies(data, datetime_col, value_col, threshold=2):
    # Ensure the datetime column is in datetime format
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
    data = data.sort_values(datetime_col)
    
    # Compute moving average and standard deviation
    data['moving_avg'] = data[value_col].rolling(window=10, min_periods=1).mean()
    data['moving_std'] = data[value_col].rolling(window=10, min_periods=1).std()
    
    # Flag anomalies where the difference exceeds the threshold * standard deviation
    data['anomaly'] = abs(data[value_col] - data['moving_avg']) > threshold * data['moving_std']
    
    # Create the line chart
    fig = px.line(data, x=datetime_col, y=value_col, title=f"Line Chart: {datetime_col} vs {value_col}")
    
    # Overlay anomalies
    anomaly_data = data[data['anomaly']]
    if not anomaly_data.empty:
        fig.add_scatter(x=anomaly_data[datetime_col], y=anomaly_data[value_col],
                        mode='markers', marker=dict(color='red', size=10),
                        name='Anomalies')
    return fig
