from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.llms.openai.base import OpenAI
import chainlit as cl
from chainlit.input_widget import Select, TextInput
import openai
from index_wikipages import create_index
from utils import get_apikey

index = None
agent = None

@cl.on_chat_start
async def on_chat_start():
    global index
    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id= "MODEL",
                label= "OpenAI - Model",
                values=["gpt-4"],
                initial_index=0,
            ),
            
            TextInput(id="WikiPageRequest", label="Request Wikipage"),
        ]
    ).send()
    # Initialize session state (use ONE store consistently)
    cl.user_session.set("agent", None)
    cl.user_session.set("index", None)

def wikisearch_engine(index):
    query_engine = index.as_query_engine(
        response_mode="compact", verbose=True, similarity_top_k=10
    )
    return query_engine


def create_react_agent(MODEL, index):
    """Creates a ReAct agent with specific instructions and its required tools."""
    # Directly create the query engine and tool list
    query_engine = wikisearch_engine(index)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="Wikipedia",
                description="Useful for performing searches on the wikipedia knowledgebase",
            ),
        )
    ]

    llm = OpenAI(model=MODEL)

    agent = ReActAgent.from_tools(
        tools=query_engine_tools,
        llm=llm,
        verbose=True,
    )
    return agent


@cl.on_settings_update
async def setup_agent(settings):
    """Sets up the agent based on user settings."""
    request_query = str(settings.get("WikiPageRequest", ""))
    model_name = str(settings.get("MODEL", "gpt-3.5-turbo"))
    
    if not request_query:
        await cl.Message(author="Agent", content="Please provide a wikipage request.").send()
        return

    index = create_index(request_query)
    
    # Pass the index to the agent creation function
    agent = create_react_agent(model_name, index)
    
    cl.user_session.set("agent", agent)
    
    await cl.Message(
        author="Agent", content=f'Wikipage(s) "{request_query}" successfully indexed'
    ).send()
    

@cl.on_message
async def main(message: cl.Message):
    """Processes incoming messages from the user."""
    agent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(
            author="Agent", 
            content="Agent is not available. Please set up the agent in the settings."
        ).send()
        return

    response = await cl.make_async(agent.chat)(message)
    
    if isinstance(response, AgentChatResponse):
        final_response = response.response
    else:
        final_response = str(response)

    await cl.Message(author="Agent", content=final_response).send()