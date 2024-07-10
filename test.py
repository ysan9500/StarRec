import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
    device=0,
    batch_size=2,
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