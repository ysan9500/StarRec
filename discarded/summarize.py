import dotenv, gc
from langchain_core.load import loads
import json
from langchain_core import prompts
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def summarize(news):
    # return list of summarization for each news
    summaries = []

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True)

    llm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        # model_id = "facebook/bart-large-cnn",
        model_kwargs={"quantization_config": quantization_config},
        task="text-generation",
        # task="summarization",
        pipeline_kwargs=dict(
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=0.7,
        ),
        device=0,
        batch_size=1,
    )

    prompt_template = """You are a professional newswriter. 
    You will be given a content of a webpage that contains news article. 
    Write a concise and easy summary of the main news in 3 sentences.
    Following is a news to summarize:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = prompts.PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm

    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="news")


    print(f"Type of news: {type(news)}")
    for doc in news[:5]:
        # debugging
        print(f"doc Type: {type(doc)}")
        # print(doc)

        summary = stuff_chain.invoke([doc])["output_text"]
        summaries.append(summary)
        # print(summary)

        gc.collect()

    return summaries


if __name__=='__main__':
    dotenv.load_dotenv()
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    response = summarize(preferred_news)
    print(type(response))
    print(response)