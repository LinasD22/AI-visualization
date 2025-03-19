# AI-Powered Data Visualization and Analysis

This project provides an AI-enhanced data visualization and analysis tool using **Retrieval-Augmented Generation (RAG)**. It enables users to upload CSV files, explore data through AI-generated insights, and interact with an AI assistant powered by Google's Gemini API for deeper analysis.

## Features
- 📊 **Data Preview**: Upload and explore CSV datasets with automatic sampling options.
- 📈 **AI-Driven Visualizations**: Automatic and manual visualizations of numerical, categorical, and datetime features.
- 🤖 **AI-Enhanced Insights**: Feature selection and summary statistics generated using AI.
- 💬 **AI Chatbot**: A built-in chatbot for discussing data and generating visualizations based on user queries.

## Installation
To run this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up the API key for Google Gemini
This project uses **Google Gemini API** for AI-powered insights. To obtain an API key:

1. Go to the [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account.
3. Navigate to the API key section.
4. Generate a new API key and copy it.
5. Create a `.env` file in the project root and add:
   ```
   API_KEY=your-google-gemini-api-key
   ```

### 5. Run the application
Start the Streamlit app by running:
```bash
streamlit run app.py
```

## Usage
### Uploading Data
- Click on **Upload your CSV file**.
- Choose a dataset to analyze.
- Select if you want to sample a fraction of the data for efficiency.

### AI-Powered Insights
- Explore **Summary Statistics** of numerical and categorical features.
- Let AI suggest **important features** and explain their relevance.
- Get **AI-generated visualizations** based on your dataset.

### Chat with AI
- Type queries in the **AI Chat** tab.
- Ask for data summaries, pattern analysis, or feature importance.
- Request specific visualizations (e.g., "Show a histogram for column X").

## License
This project is licensed under the MIT License. Feel free to use and modify it.

## Contributing
Pull requests and feature suggestions are welcome! Open an issue to discuss improvements.

---
_Enjoy exploring your data with AI-powered insights! 🚀_

