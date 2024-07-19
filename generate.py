import dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.load import dumpd, dumps, load, loads

import gc

import json
from langchain_core.documents import Document



quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True)

def summarize(news):
    # Define LLM chain
    llm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        # model_id = "facebook/bart-large-cnn",
        model_kwargs={"quantization_config": quantization_config},
        task="text-generation",
        # task="summarization",
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.5,
        ),
        device=0,
        batch_size=1,
    )

    summaries = []

    # Define prompt
    prompt_template = """You are a professional newswriter. 
    You will be given a content of a webpage that contains news article in English. DO NOT repeat my prompt. 
    Write a concise and easy summary of the main news in 3 sentences. Answer in English. 
    Following is a news to summarize:
    "{context}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)


    print(f"Type of news: {type(news)}")
    for doc in news[:5]:
        # debugging
        print(f"doc Type: {type(doc)}")
        # print(doc)

        summary = chain.invoke({"context": [doc]})
        summaries.append(Document(page_content=summary))
        # print(summary)

        gc.collect()

    string_representation = dumps(summaries, pretty=True)
    with open("database/summaries.json", "w") as fp:
        json.dump(string_representation, fp)

    return summaries

def cleanup(news):
    llm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        # model_id = "facebook/bart-large-cnn",
        model_kwargs={"quantization_config": quantization_config},
        task="text-generation",
        # task="summarization",
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.5,
        ),
        device=0,
        batch_size=1,
    )

    cleaned = []

    prompt_template = """You will be provided scraped news from a website in English. It contains one main news along with other unnecessary texts. 
    Remove all the unnecessary parts, while preserving the main article. DO NOT EDIT OR SUMMARIZE THE MAIN ARTICLE!!
    DO NOT repeat my prompt. 
    Following is a news to clean up:
    "{context}"
    CLEANED UP NEWS:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)

    for doc in news[:5]:
        # debugging
        print(f"doc Type: {type(doc)}")
        # print(doc)

        result = chain.invoke({"context": [doc]})
        cleaned.append(Document(page_content=result))
        # print(summary)
        gc.collect()

    string_representation = dumps(cleaned, pretty=True)
    with open("database/cleaned_temp.json", "w") as fp:
        json.dump(string_representation, fp)

    return cleaned


if __name__=='__main__':
    dotenv.load_dotenv()
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    # with open("database/cleaned_temp.json", "r") as fp:
    #     clean_news = loads(json.load(fp))
    clean_news = cleanup(preferred_news)
    print(type(clean_news[0]))
    summaries = summarize(clean_news)
    print(type(summaries))
    print(summaries)