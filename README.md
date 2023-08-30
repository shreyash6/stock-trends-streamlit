# Stock Trend Analysis App

This is a simple Stock Trend Analysis application built with Streamlit and LangChain. 

## About

This app allows users to enter a company name and get an overview of the company, its historical stock price trends, and potential reasons for major financial events based on news and reports.

The application uses the LangChain library to generate text summaries

## Running the app

### Install requirements

```
pip install -r requirements.txt
```

### Run the app

```
streamlit run main.py
```

The app will be served at http://localhost:8501/


## Workflow

The application workflow is:

1. User enters a company name
2. First AI model generates a company overview 
3. Second AI model provides historical stock price trends
4. Third AI model summarizes potential reasons for major financial events

The summary and insights are displayed back to the user.

## Customization

The application can be customized by modifying the prompt templates in `main.py` to change the input questions and output summaries.

## Credits

This application uses the following open source libraries:

- [Streamlit](https://streamlit.io/) for creating the web app
- [LangChain](https://github.com/hwchase17/langchain) for chaining the AI models