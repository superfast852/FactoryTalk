# Tracemark Technologies Inc. 2023
import os
import sys
import logging
import pickle  # Not belonging to langchain :/

# Everything data related
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Add-Ons
from langchain.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory

# Actual Model Inference
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import load_tools, Tool, ZeroShotAgent, AgentExecutor, initialize_agent, AgentType
# from langchain.chat_models import ChatOpenAI  # Here in case I want to change
from langchain import OpenAI
import langchain.schema.output_parser  # just the error stuff


def save2env(key, value):
    os.environ[key] = value
    if os.uname().sysname == "Linux":
        os.system(f"echo export {key}={os.environ[key]} >> ~/.profile")
    elif os.uname().sysname == "Windows":
        os.system(f"setx {key}={value}")


def create_persist(folder: str):
    loader = DirectoryLoader(folder)
    with open("filelist.pkl", "wb")as m:
        pickle.dump(None, m)
    files = os.listdir(folder)
    with open("filelist.pkl", "wb") as f:
        pickle.dump(files, f)
    return VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])


if os.environ.get("OPENAI_API_KEY") is None:
    save2env("OPENAI_API_KEY",
             input("Enter your OpenAI API key [https://platform.openai.com/account/api-keys]: ").strip())

if os.environ.get("SERPAPI_API_KEY") is None:
    save2env("SERPAPI_API_KEY", input("Enter your SerpAPI API key [https://serpapi.com/manage-api-key]: ").strip())

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s]: %(message)s")
logging.info("Initialized!")

# Instantiate access to the OpenAI API
llm = OpenAI(temperature=0)

# Prepare the data in the data/ directory for the agent to access
# If no new data has been added, load the data from the pickle file
if os.path.exists("persist"):
    print("Reusing index...\n")
    if not os.path.exists("filelist.pkl"):
        with open("filelist.pkl", "wb") as n:
            pickle.dump(None, n)
    with open("filelist.pkl", "rb") as listing:
        filelist = pickle.load(listing)
    if filelist != os.listdir("./data/"):
        os.system("rm -rf ./persist")
        os.remove("filelist.pkl")
        print("Index is out of date. Recreating index...\n")
        index = create_persist("./data/")
    else:
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    index = create_persist("./data/")
    # logging.info("Created index from data/ directory")


# Finally, create an interface for it to act as a tool.
dataretrieval = RetrievalQA.from_llm(llm=llm,
                                     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))

# The description of our custom tool needs to be adjusted, since
# the agent uses it to determine from where to grab data for a response.
# TODO: improve the descriptions. They are incredibly important.
# Tools are like plugins for the agent. They are the agent's "skills".
tools = [Tool(name="dataretrieval",
              func=dataretrieval.run,
              description="useful for answering questions only about the Click PLCs."),
         Tool(name="Search",
              func=SerpAPIWrapper().run,
              description="useful for when you need to answer questions about current events, "
                          "the current state of the world, or other plcs."),
         Tool(name="math",
              func=load_tools(["llm-math"], llm=llm)[0].run,
              description="useful for when you need to solve math questions")]


# The conversational agent follows a conversation and can respond based on previous context.
def ConversationalAgent(context: str = "Have a conversation with a human, "
                                       "answering the following questions as best you can"):
    prefix = f"{context}.You have access to the following tools:"
    suffix = """Begin!"
    
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(  # A prompt is a template for the agent to use to generate a response.
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")  # The agent will remember the conversation history

    llm_chain = LLMChain(llm=llm, prompt=prompt)  # The chain is just a wrapper for the llm model
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)  # Here we unite the chain with the tools.
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)


def ReactConversationalAgent():
    memory = ConversationBufferMemory(memory_key="chat_history")  # The agent will remember the conversation history
    agent = initialize_agent(tools=tools, llm=llm, verbose=True,
                             agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)
    return agent


# The SIAgent responds to a single query and does not have context about previous answers.
def SingleInstanceAgent():
    # This doesn't implement memory, so it's as simple as joining the LLM with the tools.
    return initialize_agent(tools=tools, llm=llm, verbose=True)


agent = ConversationalAgent()  # Choose the agent to use.

while True:
    try:
        # Simple interface for demonstration purposes.
        query = input("User (type 'exit' to quit): ")
        if query == "exit":
            sys.exit()
        result = agent.run(input=query)
        print(f"Insider: {result}")
    except langchain.schema.output_parser.OutputParserException:
        print("Insider: Sorry, I don't understand. Could you rephrase that?")
