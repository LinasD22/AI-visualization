# AI Visualizations with RAG

**AI Visualizations with RAG** is an AI-powered web application built with Streamlit that leverages Retrieval Augmented Generation (RAG) techniques to automatically analyze CSV datasets, generate insightful visualizations, and provide AI-driven data insights. The application integrates Google Gemini for generative content and interactive AI chat functionality.

## Overview

- **Automatic Data Processing:** Upload CSV files to view data previews, numerical summaries, and categorical insights.
- **Dynamic Visualizations:** Generate histograms, bar charts, line charts, scatter plots, and more based on your data types.
- **AI-Driven Analysis:** Use RAG techniques to select important features, explain correlations, and get concise AI-generated insights.
- **Interactive Chat:** Engage with an AI agent to ask questions and receive detailed explanations about your dataset.

## Prerequisites

- **Python 3.x**
- **Streamlit** – For building the web application.
- Other required Python packages:
  - `pandas`
  - `numpy`
  - `plotly`
  - `google-generativeai`
  - `python-dotenv`
  - `sentence-transformers`
  - `faiss`
  - *(and any other dependencies required by the custom modules: `data_processing`, `visualization`, and `ai_utils`)*

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/AI-visualization.git
   cd AI-visualization/BACKEND
