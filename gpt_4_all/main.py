import os
from langchain import PromptTemplate, LLMChain

from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from gpt4all import GPT4All as OrigGPT4All

BASE_PATH = os.path.dirname(__file__)
# Set Up Question to pass to LLM
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Specify Model
# To run locally, download a compatible ggml-formatted model.

model_name = "orca-mini-3b.ggmlv3.q4_0.bin"

gpt4all_models_path = os.path.join(
    BASE_PATH, "models", model_name
)  # replace with your desired local file path


# model_path = OrigGPT4All.download_model(
#     model_filename=model_name, model_path=gpt4all_models_path, verbose=True, url=None
# )

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
# llm = GPT4All(model_name, allow_download=True, verbose=True)
# output = llm.generate("The capital of France is ", max_tokens=3)
# print(output)
# # If you want to use a custom model add the backend parameter
# # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(model=gpt4all_models_path, allow_download=True, verbose=True)

# print(GPT4All.list_models())
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)
