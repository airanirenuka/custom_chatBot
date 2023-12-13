from llama_index import SimpleDirectoryReader, GPTListIndex,load_index_from_storage, StorageContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    print(directory_path)
    max_input_size =  4096
    num_outputs =  512
    max_chunk_overlap = 0.2   # Change this to a float between 0 and 1
    chunk_size_limit = 600
    print("Calling PromptHelper")
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    print("Called PromptHelper")
    print("Calling LLMPredictor")
    # Note: Adjust the temperature and max_tokens based on the model requirements
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    print("Called LLMPredictor")
    documents = SimpleDirectoryReader(r"D:\Repo_chatBot\Custom_ChatGPT\docs").load_data()
    #print(documents)
    service_context = ServiceContext.from_defaults(
                                      llm_predictor=llm_predictor, prompt_helper=prompt_helper  )
    
    index = GPTVectorStoreIndex.from_documents(documents)  #,service_context=service_context)
    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist('storage')

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='storage')
   # index.save_to_disk('index.json')
       # load index
    index = load_index_from_storage(storage_context,  index_id="vector_index")
    return index    

def chatbot(input_text):
    # Reduce the length of the input_text to meet the model's maximum context length
    # input_text = input_text[:8192]
    input_text = input_text[:4096]
    # response = index.query(input_text, response_mode="compact")
    # return response.response
    query_engine=index.as_query_engine()
    response=query_engine.query(input_text)
    return response.response


index = construct_index("D:\Repo_chatBot\Custom_ChatGPT\docs")  # Use raw string (r) for paths
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")
iface.launch(share=True)
