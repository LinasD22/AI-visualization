import google.generativeai as genai
import json
import pandas as pd

def get_gemini_suggestions(model, column, stats_string):
    """
    Build a prompt using the column name and its statistics,
    then call the Gemini model to get suggestions.
    """
    prompt = (
        f"This dataset has a variable named '{column}'. "
        f"The basic statistics are: {stats_string}. "
        "Summarize based on this information: should we keep this variable? "
        "Please provide your suggestions and reasoning. DO NOT INCLUDE ANY CODE"
    )
    response = model.generate_content(prompt)
    return response.text


def select_important_features(model, data):
    """
    Ask the AI to return a strict JSON object with two keys:
      "features": an object mapping the top 5 most important feature names (numerical or categorical)
                  to a one-sentence explanation for why that feature is important,
      "overall_explanation": a one-sentence explanation describing why these features were selected overall.
    
    We build a summary for each numerical and categorical feature as context.
    """
    # Get numerical and categorical columns
    numerical_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    # For categorical, consider both object and categorical dtypes
    categorical_cols = [col for col in data.columns if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col])]

    summary_str = ""
    # Build summary for numerical features
    for col in numerical_cols:
        stats = data[col].describe()
        summary_str += f"{col} (numerical): mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']}, max={stats['max']}\n"
    # Build summary for categorical features
    for col in categorical_cols:
        frequencies = data[col].value_counts()
        mode = frequencies.idxmax() if not frequencies.empty else "N/A"
        cardinality = data[col].nunique()
        summary_str += f"{col} (categorical): mode={mode}, cardinality={cardinality}\n"
    
    prompt = (
        "Given the following summary statistics for numerical and categorical features of a dataset:\n\n"
        f"{summary_str}\n\n"
        "Assume that the most important features are those that have anomalies, high amount of outliers, missing values or something else that attracts attention. "
        "or that are known to have a strong impact on the target variable. 15 important features is the maximum."
        "Please return a strict JSON object with two keys:\n"
        "  \"features\": an object where each key is one of the top 5 most important feature names for further analysis and each value is a one-sentence explanation for why that feature is important,\n"
        "  \"overall_explanation\": a one-sentence explanation describing why these features were selected overall. "
        "Do not include any additional text.\n\n"
        "Example response: {\"features\": {\"LotArea\": \"LotArea has high variance and is strongly correlated with house value.\", "
        "\"OverallQual\": \"OverallQual is a strong indicator of the quality of the house and has a direct impact on the sale price.\"}, "
        "\"overall_explanation\": \"These features were chosen because they exhibit high variability or low diversity, indicating their impact on the target variable.\"}"
    )
    
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    print("Raw AI response:", raw_text)  # Debug print

    # Remove markdown formatting if present (e.g., triple backticks)
    if raw_text.startswith("```json"):
        raw_text = raw_text.replace("```json", "").strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()
    
    try:
        result = json.loads(raw_text)
        if isinstance(result, dict) and "features" in result and "overall_explanation" in result:
            if isinstance(result["features"], dict) and all(isinstance(v, str) for v in result["features"].values()):
                return result
    except json.JSONDecodeError as e:
        print(f"Error parsing AI response: {e}")
    
    return {"features": {}, "overall_explanation": "No important features identified."}



def get_concise_insights(model, feature, summary_text):
    """
    Generate concise insights for a given feature using the AI model.
    """
    prompt = (
        f"Based on the following summary for the feature '{feature}':\n{summary_text}\n\n"
        "Please provide concise, actionable insights about what this feature indicates in the dataset, and mention any potential data issues, like typos and other data anomalies. Once again, try to be concise."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def choose_visualizations(model, data):
    """
    Ask the AI model to recommend visualization methods for the given dataset.
    Returns a list of recommended visualization names (strings).
    """

    # Identify different column types for context in the prompt
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    categorical_cols = [col for col in data.columns if pd.api.types.is_string_dtype(data[col])]
    datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]

    # Build some summary info to pass into the prompt
    numeric_summary = []
    for col in numeric_cols:
        stats = data[col].describe()
        numeric_summary.append(
            f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']}, max={stats['max']}"
        )

    # Create a single string with summary for numeric columns
    numeric_summary_str = "\n".join(numeric_summary)

    # Basic context about columns
    prompt = (
        "We have a dataset with the following column types:\n\n"
        f"Numeric columns: {numeric_cols}\n"
        f"Categorical columns: {categorical_cols}\n"
        f"Datetime columns: {datetime_cols}\n\n"
        "Here are summary statistics for the numeric columns:\n"
        f"{numeric_summary_str}\n\n"
        "Please provide a strict JSON array of recommended visualization methods (by name only) "
        "that would be most useful for exploring this dataset. Think of histograms, bar charts, box plots, scatter plots, line charts, density plots, correlation heatmaps, etc. "
        "Do not include any code, explanations, or text outside of a valid JSON array.\n"
        "Example:\n\n"
        "[\"Histogram\", \"Box Plot\", \"Scatter Plot\"]"
    )

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Debug (optional): print out the raw text
    print("Raw AI response for visualize suggestions:", raw_text)

    # Remove any triple backticks if present (e.g., ```json ... ```)
    # Remove any markdown formatting if present (e.g., triple backticks and the language tag)
    if raw_text.startswith("```json"):
        # Remove the starting ```json and ending ``` if they exist
        raw_text = raw_text.replace("```json", "").strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    # Attempt to parse the response as JSON
    try:
        suggestions = json.loads(raw_text)
        # Must be a list of strings to be valid
        if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
            return suggestions
    except Exception as e:
        print(f"Error parsing visualization suggestions: {e}")

    # Fallback if parsing or format validation fails
    return ["Histogram"]


def suggest_visualization_for_feature(model, feature, summary_text):
    """
    Given a feature name and its summary text, ask the AI which visualization type is most appropriate.
    Respond with exactly one of these options: Histogram, Bar Chart, Line Chart, Scatter Plot, or Density Plot.
    """
    prompt = (
        f"Based on the following summary for the feature '{feature}':\n{summary_text}\n\n"
        "Which type of visualization would best represent this data? "
        "Respond with exactly one of these options: Histogram, Bar Chart, Line Chart, Scatter Plot, Pie chart or Density Plot."
    )
    response = model.generate_content(prompt)
    vis_type = response.text.strip()
    return vis_type

def get_ai_response_chat(model, context, user_prompt):
    """
    Generate concise, actionable insights using the AI model by combining retrieved context and the user's question.
    """
    prompt = (
        "You are an expert data analyst. Below is some context extracted from a dataset and a user's question.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{user_prompt}\n\n"
        "Based on the above, please provide concise, actionable insights based on the Question or provided data, and justify your answer if needed. Do not provide code."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

