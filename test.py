import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import huggingface_hub

huggingface_hub.login()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True)

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    #model_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    model_kwargs={"quantization_config": quantization_config},
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=0.7,
    ),
    device=0,
    batch_size=1,
)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

chat_model = ChatHuggingFace(llm=llm)

res = chat_model.invoke(messages)
print(res.content)
