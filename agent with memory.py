from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import SerpAPIWrapper
from langchain.schema.output_parser import OutputParserException
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import logging
import sys
import os

llm = OpenAI(temperature=0)
PERSIST = True
if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = DirectoryLoader("./data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    logging.info("Created index from data/ directory")
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])  # Create the vectorstore database from the data


# Finally, create an interface for it to act as a tool.
dataretrieval = RetrievalQA.from_llm(llm=llm,
                                     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))

# The description of our custom tool needs to be adjusted, since
# the agent uses it to determine from where to grab data for a response.
# TODO: improve the descriptions. They are incredibly important.
# Tools are like plugins for the agent. They are the agent's "skills".
tools = [Tool(name="dataretrieval",
              func=dataretrieval.run,
              description="useful for answering questions about the Click PLCs."),
         Tool(name="math",
              func=load_tools(["llm-math"], llm=llm)[0].run,
              description="useful for when you need to solve math questions")]

prefix = """Your name is GPT. Have a conversation with a human, answering the following prompts as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Prompt: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)
# agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

while True:
    try:
        query = input("User (type 'exit' to quit): ")
        if query == "exit":
            sys.exit()
        result = agent_chain.run(input=query)
        print(f"Insider: {result}")
    except OutputParserException as e:
        logging.error(e)
        print("Insider: Sorry, I don't understand. Could you rephrase that?")