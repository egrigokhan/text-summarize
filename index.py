from langchain.llms import OpenAI
from langchain import PromptTemplate


import os

from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)


def run(message, history):

    llm = OpenAI(temperature=0)

    one_input_prompt = PromptTemplate(
        input_variables=["instructions", "input"], template="""Please summarize the following text in detail, putting emphasis on the most important sections.
      {instructions}
      
      Text:
      
      {input}""")
    prompt = one_input_prompt.format(
        input=message, instructions=os.environ["instructions"])

    return llm(prompt)


def setup(config):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["instructions"] = config["instructions"]
