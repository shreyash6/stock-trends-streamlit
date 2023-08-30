## Integrate our code OpenAI API
import os
import dotenv
#from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

dotenv.load_dotenv() 

#os.environ["OPENAI_API_KEY"]=openai_key
os.getenv("OPENAI_API_KEY")

# streamlit framework

st.title('Stock Trends')
input_text=st.text_input("Search the company name")

# Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['Company'],
    template="Using sources like Yahoo Finance, Google Finance, Investing.com, and the official website of {Company}. provide a brief overview of the company"
)

# Memory

person_memory = ConversationBufferMemory(input_key='Company', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='Overview', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='Trends', memory_key='description_history')

## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='Overview',memory=person_memory)

# Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=['Overview'],
    template="Using financial data sources like Alpha Vantage, Quandl, Yahoo Finance API, EOD Historical Data, and Financial Modeling Prep API, provide historical stock price trends or key financial metrics for that company over the {Overview}."
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='Trends',memory=dob_memory)
# Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=['Trends'],
    template="Based on insights from platforms like Seeking Alpha, Bloomberg, Reuters, MarketWatch, and Google News specific to that company during the time period ({Trends}), provide potential reasons for observed financial trends, such as product launches, mergers, global events, etc."
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)
parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['Company'],output_variables=['Overview','Trends','description'],verbose=True)



if input_text:
    st.write(parent_chain({'Company':input_text}))

    with st.expander('Overview'): 
        st.info(person_memory.buffer)

    with st.expander('Trends'): 
        st.info(descr_memory.buffer)
