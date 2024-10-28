import os
import streamlit as st
from typing import List, Optional
from PIL import Image
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai  # type: ignore
import json

# Set the environment variable for Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/macbookpro/Desktop/Agents/Conservational Agent/local-llm-435705-7c523e3da00a.json"

class ImageAnalysisTool(BaseTool):
    name: str = "image_analysis"  # Must be valid as per API naming conventions
    description: str = "Analyzes images and returns detailed descriptions"
    
    def _run(self, image_path: str) -> str:
        model = genai.GenerativeModel('gemini-1.5-flash')
        image = Image.open(image_path)
        response = model.generate_content(["Describe this image in detail", image])
        return response.text


class TextGenerationTool(BaseTool):
    name: str = "text_generation"  # Tool name following naming conventions
    description: str = "Generates text based on a given prompt"

    def _run(self, prompt: str) -> str:
        """
        Generates text based on the input prompt using the language model.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt])
        return response.text


def create_gemini_agent(google_api_key: str):
    os.environ['GOOGLE_API_KEY'] = google_api_key
    genai.configure(api_key=google_api_key)
    
    # Initialize Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create tools with explicit arguments
    tools = [
        ImageAnalysisTool(name="image_analysis", description="Analyzes images and returns detailed descriptions"),
        TextGenerationTool(name="text_generation", description="Generates text based on input prompts")
    ]
    
    # Load prompt template from LangChain hub
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


class GeminiAssistant:
    def __init__(self, google_api_key: str):
        self.agent = create_gemini_agent(google_api_key)
        
    async def process_message(self, message: str, image_path: Optional[str] = None) -> str:
        """
        Process a message with optional image input
        """
        try:
            context = {"input": message}
            if image_path:
                image_description = self.agent.tools[0]._run(image_path)
                context["image_description"] = image_description
                
            response = await self.agent.ainvoke(context)
            return response["output"]
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    async def process_text(self, text_data: str) -> dict:
        """
        Process text data using the agent
        """
        try:
            response = await self.agent.ainvoke({
                "input": text_data
            })
            return response["output"]
        except Exception as e:
            return {"error": f"Error processing text: {str(e)}"}

# Streamlit app
st.title("Ahtasham AI Assistant")
google_api_key = st.text_input("Enter your API Key", type="password")

if google_api_key:
    assistant = GeminiAssistant(google_api_key=google_api_key)

    # Image Analysis Section
    st.subheader("Image Analysis")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing image..."):
            description = assistant.agent.tools[0]._run(uploaded_file)
        st.write("Image Description:", description)

# Text Generation Section
st.subheader("Text Generation")
text_input = st.text_area("Enter text prompt for generation")
if st.button("Generate Text") and text_input:
    with st.spinner("Generating text..."):
        text_generation_tool = assistant.agent.tools[1]  # Access the text generation tool
        response_text = text_generation_tool._run(text_input)
    st.write("Generated Text Output:", response_text)
