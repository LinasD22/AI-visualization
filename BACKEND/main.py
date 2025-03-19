import streamlit as st
import os
import data_processing as dp
import visualization as viz
import ai_utils as ai
import google.generativeai as genai
import pandas as pd
import numpy as np
import streamlit as st
import os
import data_processing as dp
import visualization as viz
import ai_utils as ai
import google.generativeai as genai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import re
import plotly.express as px
import json
from dotenv import load_dotenv
import os

st.set_page_config(
    page_title="AI visualazations",
    page_icon="🤖",  # or any emoji/unicode you like
    layout="centered"
)

def force_numeric_axis(fig):
    """If a figure is provided, force the x-axis to be type='linear'."""
    if fig is not None:
        fig.update_xaxes(type="linear")
    return fig

Data_previe_tab1, visualazation_tab2, Ai_chat_tab3 = st.tabs(
    ["📊 Data Preview", "🤖 AI Analysis", "💬 AI Chat"]
)

load_dotenv()
API_KEY = os.getenv('API_KEY')
os.environ["GOOGLE_API_KEY"] = API_KEY

# Configure the Generative AI module
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Choose the Gemini model
model = genai.GenerativeModel("models/gemini-2.0-flash-exp")

# Streamlit app title

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "data" not in st.session_state:
    st.session_state["data"] = None
if "data_preview" not in st.session_state:
    st.session_state["data_preview"] = None
if "summary_stats" not in st.session_state:
    st.session_state["summary_stats"] = None
if "cat_summary_df" not in st.session_state:
    st.session_state["cat_summary_df"] = None
if "ai_feature_response" not in st.session_state:
    st.session_state["ai_feature_response"] = {"features": {}, "overall_explanation": ""}
if "ai_suggestions" not in st.session_state:
    st.session_state["ai_suggestions"] = {}
if "ai_viz" not in st.session_state:
    st.session_state["ai_viz"] = []
if "selected_ai_feature" not in st.session_state:
    st.session_state["selected_ai_feature"] = "(None)"

with Data_previe_tab1:
    st.header("AI-Powered Automatic Data Visualization and Highlighting")
    # File uploader widget
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            if st.session_state["data"] is None:
                with st.spinner("Processing file..."):
                    st.session_state["data"] = dp.load_data(uploaded_file)
            data = st.session_state["data"]

            #st.write(data.dtypes)

            # st.write(data["DateTime"].head())
            # st.write(data["DateTime"].dtype)

            # --- Sampling Option ---
            #st.markdown("---")
            st.subheader("Data Sampling Option")

            # Load original data once and store it separately if not already stored
            if st.session_state["data"] is not None and "data_original" not in st.session_state:
                st.session_state["data_original"] = st.session_state["data"].copy()

            # Ask the user if they want to work with a sample
            sample_option = st.checkbox("The file is large. Would you like to work with a sample?")
            if sample_option:
                sample_frac = st.slider("Select sample fraction", min_value=0.05, max_value=1.0, value=0.15, step=0.05)
                if st.button("Apply Sample"):
                    with st.spinner("Sampling data..."):
                        # Always sample from the original data
                        original_data = st.session_state.get("data_original", st.session_state["data"])
                        sampled_data = original_data.sample(frac=sample_frac, random_state=42)
                        st.session_state["data"] = sampled_data
                        # Update cached preview and summary
                        st.session_state["data_preview"] = sampled_data.head()
                        st.session_state["summary_stats"] = dp.get_summary_stats(sampled_data)
                    st.success(f"Data sampled to {int(sample_frac*100)}% of the original size.")

            
            # Cache data preview
            if st.session_state["data_preview"] is None:
                st.session_state["data_preview"] = data.head()
            st.subheader("Data Preview")
            st.dataframe(st.session_state["data_preview"])

            # Cache numerical summary
            if st.session_state["summary_stats"] is None:
                st.session_state["summary_stats"] = dp.get_summary_stats(data)
            st.subheader("Summary Statistics (Numerical)")
            st.write(st.session_state["summary_stats"])

            #st.write(data.dtypes)
            num_cols = dp.get_numerical_columns(data)
            additional_stats = pd.DataFrame({
                "Skewness": data[num_cols].skew(),
                "Kurtosis": data[num_cols].kurt(),
                "Missing (%)": data[num_cols].isna().mean() * 100
            })
            st.session_state["additional_stats"] = additional_stats
            st.subheader("Additional Numerical Statistics")
            st.write(additional_stats)


            # Cache categorical summary (if available)
            categorical_columns = dp.get_categorical_columns(data)
            if categorical_columns:
                if st.session_state["cat_summary_df"] is None:
                    cat_summaries = dp.get_all_categorical_summaries(data)
                    row_labels = ["mode", "mode_frequency", "cardinality", "total_count", "missing_values"]
                    cat_summary_df = pd.DataFrame(index=row_labels, columns=cat_summaries.keys())
                    for col, summary_dict in cat_summaries.items():
                        cat_summary_df.loc["mode", col] = summary_dict["mode"]
                        cat_summary_df.loc["mode_frequency", col] = summary_dict["mode_frequency"]
                        cat_summary_df.loc["cardinality", col] = summary_dict["cardinality"]
                        cat_summary_df.loc["total_count", col] = summary_dict["total_count"]
                        cat_summary_df.loc["missing_values", col] = summary_dict["missing_values"]
                    cat_summary_df = cat_summary_df.astype(str)
                    st.session_state["cat_summary_df"] = cat_summary_df
                st.subheader("Summary Statistics (Categorical)")
                st.dataframe(st.session_state["cat_summary_df"])
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

            
        with visualazation_tab2:
            st.header("AI-Powered Automatic Data Visualization and Highlighting")
            
            # Unified dropdown for all selectable columns
            st.subheader("Explore Columns Manually")
            selectable_columns = dp.get_selectable_columns(data)
            if selectable_columns:
                selected_column = st.selectbox("Select a column", selectable_columns)
                # Check the type of the selected column and branch logic accordingly:
                if selected_column in dp.get_numerical_columns(data):
                    # Visualize numerical data
                    fig = viz.create_histogram(data, selected_column)
                    st.plotly_chart(fig)
                    
                    stats_string = dp.get_column_statistics(data, selected_column)
                    #if st.button("Get Gemini suggestions for this numerical column"):
                        #suggestion = ai.get_gemini_suggestions(model, selected_column, stats_string)
                        #st.subheader("Gemini API Suggestions")
                        #st.write(suggestion)
                        
                elif selected_column in dp.get_categorical_columns(data):
                    # Visualize categorical data
                    fig = viz.create_bar_chart(data, selected_column)
                    st.plotly_chart(fig)
                    
                    frequencies = dp.get_categorical_frequencies(data, selected_column)
                    st.subheader("Frequency Distribution")
                    st.write(frequencies)
                    
                    if st.button("Get Gemini suggestions for this categorical column"):
                        prompt = (
                            f"This dataset has a categorical variable named '{selected_column}'. "
                            f"The frequency distribution is: {frequencies.to_dict()}. "
                            "Based on this information, should we keep this variable? "
                            "Please provide your suggestions and reasoning."
                        )
                        response = model.generate_content(prompt)
                        st.subheader("Gemini API Suggestions")
                        st.write(response.text)
                        
                elif selected_column in dp.get_datetime_columns(data):
                    # Visualize datetime data
                    fig = viz.create_line_chart(data, selected_column)
                    st.plotly_chart(fig)
                    
                    # (Optional) Provide a Gemini prompt regarding time series insights
                    if st.button("Get Gemini suggestions for this datetime column"):
                        prompt = (
                            f"This dataset has a datetime variable named '{selected_column}'. "
                            "Analyze the trend over time and summarize any notable patterns or anomalies."
                        )
                        response = model.generate_content(prompt)
                        st.subheader("Gemini API Suggestions")
                        st.write(response.text)
            else:
                st.info("No numeric, categorical, or datetime columns found in this dataset.")
                


                
            # --- AI-Driven Feature Selection and Insights ---
            st.markdown("---")
            st.subheader("AI-Driven Feature Selection and Insights")

            # Button to call the AI for important features
            if st.button("Select Important Features"):
                with st.spinner("Selecting important features..."):
                    ai_response = ai.select_important_features(model, data)
                st.session_state["ai_feature_response"] = ai_response
                st.session_state["selected_ai_feature"] = "(None)"
                st.session_state["ai_suggestions"] = {}

            # Display overall AI explanation
            overall_explanation = st.session_state["ai_feature_response"].get("overall_explanation", "")
            st.markdown("**Overall AI Explanation for Selected Features:**")
            st.write(overall_explanation if overall_explanation else "No important features identified.")

            # List each AI-suggested feature with its explanation
            ai_features_dict = st.session_state["ai_feature_response"].get("features", {})
            if ai_features_dict:
                st.markdown("**AI-Suggested Features and Their Explanations:**")
                for feature, explanation in ai_features_dict.items():
                    st.markdown(f"- **{feature}**: {explanation}")
                
                # Provide a selectbox for the user to choose a feature for additional insights
                selected_ai_feature = st.selectbox(
                    "Select an AI-suggested feature for additional insights",
                    options=["(None)"] + list(ai_features_dict.keys()),
                    key="selected_ai_feature"
                )

                if selected_ai_feature and selected_ai_feature != "(None)":
                    #st.markdown("**AI Explanation for this Feature:**")
                    #st.write(ai_features_dict.get(selected_ai_feature, "No explanation provided."))

                    # If insights are not already cached, fetch them with a spinner
                    if selected_ai_feature not in st.session_state["ai_suggestions"]:
                        with st.spinner("Fetching concise insights..."):
                            if selected_ai_feature in dp.get_numerical_columns(data):
                                summary_text = dp.get_column_statistics(data, selected_ai_feature)
                            elif selected_ai_feature in dp.get_categorical_columns(data):
                                summary_text = str(dp.get_categorical_summary(data, selected_ai_feature))
                            elif selected_ai_feature in dp.get_datetime_columns(data):
                                summary_text = "This is a datetime feature. It may show trends over time."
                            else:
                                summary_text = "No summary available."
                            concise_insights = ai.get_concise_insights(model, selected_ai_feature, summary_text)
                            st.session_state["ai_suggestions"][selected_ai_feature] = concise_insights

                    st.subheader(f"Concise Insights for {selected_ai_feature}")
                    st.write(st.session_state["ai_suggestions"][selected_ai_feature])
            else:
                st.info("No AI-suggested features available.")

            # --- Adding more variables to anilaze ---
            
            
            # if ai_features_dict != "(None)":
            #     other_columns = [col for col in selectable_columns if col not in ai_features_dict]
            #     multiselect_more = st.multiselect("Add more variables to analize:", other_columns)
            #     all_features = multiselect_more + list(ai_features_dict.keys())
            # else:
            #     st.write("Please choose ai insights first")

            # if st.button("Create visualizations"):
            #     with st.spinner("AI is selecting visualizations..."):
            #         # Store the AI's suggestion in session state
            #         st.session_state["ai_viz"] = ai.choose_visualizations(model, data)    

            
            # if "ai_viz" in st.session_state and st.session_state["ai_viz"]:
            #     st.subheader("AI Selected Visualizations")
            #     st.write("The AI suggests the following visualizations:")
            #     for value in st.session_state["ai_viz"]:
            #         st.write(f"- {value}")  
                

            # # add to choose more than ai 
            # visualazations = [
            #     "Histograms",
            #     "Scatter Plots",
            #     "Line charts",
            #     "Bar charts",
            #     "Density plots",
            #     "Outlier detection",
            #     "Cleaning the data with ai reasoning",
            #     "Anomoly detection",
            #     "Relationship insights",
            #     "Dimensionality Reduction Visualizations"
            # ]

            # # TODO if visualazations not in ai_viz then create a multilist with the new variable where there is no duplicates
            # # -------------------------------------------------------------------------
            # not_in_ai_viz = []
            # ai_viz = st.session_state.get("ai_viz", [])
            # if ai_viz and isinstance(ai_viz, list):
            #     for v in visualazations:
            #         if not dp.is_duplicate(v, ai_viz, threshold=0.8):
            #             not_in_ai_viz.append(v)
            # else:
            #     not_in_ai_viz = visualazations

            # selected_options = st.multiselect("Choose what visualizations to add", not_in_ai_viz)

            # if st.button("Procced with visualizations"):
            #     st.write("AI-suggested visualizations (if any):", st.session_state["ai_viz"] if st.session_state["ai_viz"] else "No AI suggestions yet.")
            #     st.write("Additionally chosen visualizations:", selected_options)
            #     st.success("Visualizations have been selected! (You can now generate them as needed.)")
                

            # --- Automatic Graphing of Selected Variables ---
            # --- Automatic Graphing with AI-Suggested Visualization ---
            # --- Automatic Graphing of Selected Variables ---
            # --- Automatic Graphing of AI-Selected Features ---
            # --- Automatic Graphing of Selected Variables ---
            # --- Automatic Graphing of Selected Variables ---
            # --- Automatic Graphing of Selected Variables ---
            st.markdown("---")
            st.subheader("Automatic Graphing of Selected Variables")

            ai_features_dict = st.session_state["ai_feature_response"].get("features", {})

            if ai_features_dict:
                ai_features = list(ai_features_dict.keys())
                selectable_columns = dp.get_selectable_columns(data)
                other_columns = [col for col in selectable_columns if col not in ai_features]
                extra_features = st.multiselect("Add extra variables to graph:", options=other_columns)
                final_features = list(set(ai_features + extra_features))
                
                if final_features:
                    st.markdown("**Final Features for Visualization:**")
                    st.write(final_features)
                    
                    features_to_graph = st.multiselect("Select features to visualize", options=final_features)
                    
                    if features_to_graph:
                        for i, feature in enumerate(features_to_graph):
                            if feature in dp.get_numerical_columns(data):
                                summary_text = dp.get_column_statistics(data, feature)
                            elif feature in dp.get_categorical_columns(data):
                                summary_text = str(dp.get_categorical_summary(data, feature))
                            elif feature in dp.get_datetime_columns(data):
                                summary_text = "This is a datetime feature. It may show trends over time."
                            else:
                                summary_text = "No summary available."
                            
                            with st.spinner(f"Selecting visualization for {feature}..."):
                                suggested_viz = ai.suggest_visualization_for_feature(model, feature, summary_text)
                            
                            st.markdown(f"**{feature}:** AI suggests **{suggested_viz}**")
                            
                            fig = None
                            if suggested_viz.lower() == "histogram" and feature in dp.get_numerical_columns(data):
                                fig = viz.create_histogram(data, feature)
                            elif suggested_viz.lower() == "bar chart":
                                if feature in dp.get_numerical_columns(data):
                                    temp_data = data.copy()
                                    temp_data[feature] = temp_data[feature].astype(str)
                                    fig = viz.create_bar_chart(temp_data, feature)
                                else:
                                    fig = viz.create_bar_chart(data, feature)
                            elif suggested_viz.lower() == "line chart" and feature in dp.get_datetime_columns(data):
                                fig = viz.create_line_chart(data, feature)
                            elif suggested_viz.lower() == "scatter plot":
                                st.write("Scatter Plot requires two variables. Skipping scatter plot for this feature.")
                            elif suggested_viz.lower() == "density plot" and feature in dp.get_numerical_columns(data):
                                st.write("Density plot not implemented yet. Using histogram as a fallback.")
                                fig = viz.create_histogram(data, feature)
                            else:
                                if feature in dp.get_numerical_columns(data):
                                    fig = viz.create_histogram(data, feature)
                                elif feature in dp.get_categorical_columns(data):
                                    fig = viz.create_bar_chart(data, feature)
                                elif feature in dp.get_datetime_columns(data):
                                    fig = viz.create_line_chart(data, feature)
                            
                            if fig:
                                st.plotly_chart(fig, key=f"plot_{feature}_{i}")
                    else:
                        st.info("No features selected to visualize.")
                else:
                    st.info("No features selected for automatic graphing.")
            else:
                st.info("Please run the AI-driven feature selection first.")



            # --- Correlation Pairs Selection ---
            st.markdown("---")
            st.subheader("Correlation Pairs Selection and AI Explanations")
            if 'final_features' in locals() and final_features:
                numeric_features = [f for f in final_features if f in dp.get_numerical_columns(data)]
                if len(numeric_features) >= 2:
                    corr_matrix = data[numeric_features].corr()
                    threshold = 0.4
                    pair_list = []
                    pair_map = {}
                    for i in range(len(numeric_features)):
                        for j in range(i+1, len(numeric_features)):
                            f1 = numeric_features[i]
                            f2 = numeric_features[j]
                            corr_value = corr_matrix.loc[f1, f2]
                            if abs(corr_value) >= threshold:
                                pair_str = f"{f1} & {f2} (Correlation: {corr_value:.2f})"
                                pair_list.append(pair_str)
                                pair_map[pair_str] = (f1, f2, corr_value)
                    
                    options = ["(None)"] + pair_list
                    selected_pair_str = st.selectbox("Select a correlated pair to view", options=options)
                    
                    if selected_pair_str != "(None)":
                        f1, f2, corr_value = pair_map[selected_pair_str]
                        st.markdown(f"**Selected Pair:** {f1} and {f2} (Correlation: {corr_value:.2f})")
                        summary_text_f1 = dp.get_column_statistics(data, f1)
                        summary_text_f2 = dp.get_column_statistics(data, f2)
                        combined_summary = f"{f1}: {summary_text_f1}\n{f2}: {summary_text_f2}"
                        prompt = (
                            f"Given that the features '{f1}' and '{f2}' have a correlation of {corr_value:.2f}, "
                            f"and considering the following summary statistics:\n{combined_summary}\n"
                            "Please provide a concise two-sentence explanation on why these features might be correlated."
                        )
                        with st.spinner("Fetching AI explanation for the correlation..."):
                            ai_corr_explanation = model.generate_content(prompt).text.strip()
                        st.markdown("**AI Correlation Explanation:**")
                        st.write(ai_corr_explanation)
                        fig_scatter = px.scatter(data, x=f1, y=f2, title=f"Scatter Plot: {f1} vs {f2}")
                        st.plotly_chart(fig_scatter, key=f"scatter_{f1}_{f2}")
                    else:
                        st.info("No correlated pair selected.")
                else:
                    st.info("Not enough numeric features selected for correlation analysis.")
            else:
                st.info("No final features selected for correlation analysis.")

        with Ai_chat_tab3:
            st.header("Chat with AI Agent")
            for entry in st.session_state.chat_history:
                if entry["role"] == "user":
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #28a745;
                            padding: 30px;
                            margin: 30px 0;
                            border-radius: 5px;
                            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; 
                            font-size: 18px;
                            line-height: 1.5;
                            min-height: 80px;
                        '>
                            <strong>User:</strong> {entry['message']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"**AI:** {entry['message']}")

            response_placeholder = st.empty()

            # Build a richer context for the AI agent using extra numerical statistics
            if st.session_state["summary_stats"] is not None and st.session_state.get("additional_stats") is not None:
                context = (
                    f"Dataset shape: {data.shape}. Columns: {', '.join(data.columns)}. "
                    f"Numerical Summary (first 3 rows):\n{st.session_state['summary_stats'].head(20).to_csv(index=False)}\n"
                    f"Additional Numerical Statistics:\n{st.session_state['additional_stats'].to_csv(index=False)}"
                )
            else:
                context = (
                    f"Dataset shape: {data.shape}. Columns: {', '.join(data.columns)}. "
                    "No additional numerical statistics available."
                )


            #context = "\n".join(data.head(5).to_csv(index=False).splitlines())

            #context = "\n".join(data.to_csv(index=False).splitlines())

            with st.form("chat_form", clear_on_submit=False):
                user_input = st.text_input("Type your message here:")
                submitted = st.form_submit_button("Send")

            if submitted and user_input:
                st.session_state.chat_history.append({"role": "user", "message": user_input})
                # Limit chat history to the last 10 messages
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                ai_chat_response = ai.get_ai_response_chat(model, context, user_input)
                st.session_state.chat_history.append({"role": "ai", "message": ai_chat_response})
                response_placeholder.write("**AI:**\n\n" + ai_chat_response)

                            

                # --- Integrated Graph Query Section in Chat ---
                visualization_keywords = ["draw", "show", "plot", "graph", "visualize", "display"]
                if any(keyword in user_input.lower() for keyword in visualization_keywords):
                    with st.spinner("Interpreting graph command..."):
                        graph_prompt = (
                            "You are an expert data visualization assistant. You cannot generate any code. Given that the dataset has the following columns: " 
                            + ", ".join(dp.get_selectable_columns(data)) + ". "
                            "Context: " + context + ". "
                            "A user has asked: \"" + user_input + "\". "
                            "Please convert this request into a strict JSON object with exactly two keys: "
                            "\"visualization_type\" and \"columns\". "
                            "Allowed visualization types are: histogram, bar chart, line chart, scatter plot, correlation heatmap, "
                            "box plot, violin plot, pie chart. "
                            "For example, if the user said \"Show a histogram for SalePrice\", respond with: "
                            "{\"visualization_type\": \"histogram\", \"columns\": [\"SalePrice\"]}. "
                            "Respond with exactly valid JSON and nothing else."
                        )
                        graph_ai_response = model.generate_content(graph_prompt)
                        raw_graph_response = graph_ai_response.text.strip()
                        try:
                            match = re.search(r'\{.*\}', raw_graph_response, re.DOTALL)
                            json_text = match.group(0) if match else raw_graph_response
                            parsed = json.loads(json_text)
                        except Exception as e:
                            st.error("Error parsing the AI graph command: " + str(e))
                            parsed = None

                    # Wrap a single command dict in a list if needed.
                    if parsed is None:
                        commands = []
                    elif isinstance(parsed, list):
                        commands = parsed
                    elif isinstance(parsed, dict):
                        commands = [parsed]
                    else:
                        commands = []
                    
                    # Prepare lists of categorical and numeric columns.
                    categorical_columns = dp.get_categorical_columns(data)
                    selectable_columns = dp.get_selectable_columns(data)
                    # Assume that numeric columns are those in selectable_columns not in categorical_columns.
                    numeric_columns = [col for col in selectable_columns if col not in categorical_columns]

                    # Define a helper to force a numeric x-axis.
                    def force_numeric_axis(fig):
                        if fig is not None:
                            fig.update_xaxes(type="linear")
                        return fig

                    # Define a mapping of visualization types to functions that check data types.
                    viz_functions = {
                        "histogram": lambda cols: (
                            viz.create_histogram(data, cols[0])
                            if len(cols) == 1 and cols[0] in numeric_columns
                            else viz.create_bar_chart(data, cols[0])
                            if len(cols) == 1 and cols[0] in categorical_columns
                            else None
                        ),
                        "bar chart": lambda cols: viz.create_bar_chart(data, cols[0]) if len(cols) == 1 else None,
                        "line chart": lambda cols: force_numeric_axis(
                            # If one column, use index as x-axis.
                            px.line(data, x=data.index, y=cols[0], title=f"Line Chart: {cols[0]}")
                            if len(cols) == 1 and cols[0] in numeric_columns
                            # If two columns and both numeric, use first as x and second as y.
                            else px.line(data, x=cols[0], y=cols[1], title=f"Line Chart: {cols[0]} vs {cols[1]}")
                            if len(cols) == 2 and cols[0] in numeric_columns and cols[1] in numeric_columns
                            else None
                        ),
                        "scatter plot": lambda cols: (
                            px.scatter(data, x=cols[0], y=cols[1], title=f"Scatter Plot: {cols[0]} vs {cols[1]}")
                            if len(cols) == 2 and cols[0] in numeric_columns and cols[1] in numeric_columns
                            else None
                        ),
                        "correlation heatmap": lambda cols: (
                            px.imshow(data[cols].corr(), text_auto=True, aspect="auto", title="Correlation Heatmap")
                            if len([col for col in cols if col in numeric_columns]) >= 2
                            else None
                        ),
                        "box plot": lambda cols: (
                            viz.create_boxplot(data, cols[0])
                            if len(cols) == 1 and cols[0] in numeric_columns
                            else None
                        ),
                        "violin plot": lambda cols: (
                            viz.create_violinplot(data, cols[0])
                            if len(cols) == 1 and cols[0] in numeric_columns
                            else None
                        ),
                        "pie chart": lambda cols: (
                            viz.create_pie_chart(data, cols[0])
                            if len(cols) == 1 and cols[0] in categorical_columns
                            else None
                        ),
                    }
                    
                    # Process each visualization command.
                    for command in commands:
                        viz_type = command.get("visualization_type", "").lower()
                        cols = command.get("columns", [])
                        if viz_type and cols:
                            fig = None
                            if viz_type in viz_functions:
                                fig = viz_functions[viz_type](cols)
                            if fig is not None:
                                st.plotly_chart(fig, key=f"chat_query_{viz_type}_{'_'.join(cols)}")
                            else:
                                st.info(f"Could not generate a plot for '{viz_type}' with columns {cols}.")
                        else:
                            st.info("The agent did not generate a valid visualization command.")