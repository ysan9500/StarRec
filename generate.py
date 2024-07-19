import dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import huggingface_hub

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
import gc


from langchain_core.load import dumpd, dumps, load, loads
import json


from langchain.schema import Document


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True)

from langchain.schema import Document  # 필요한 임포트 추가

def summarize(news):
    # Define LLM chain
    llm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        #model_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        model_kwargs={"quantization_config": quantization_config},
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=0.7,
        ),
        device=0,
        batch_size=1,
    )

    summaries = []

    # Define prompt
    prompt_template = """You are a professional newswriter. 
    You will be given a content of a webpage that contains news article. 
    Write a concise and easy summary of the main news in 3 sentences.
    Following is a news to summarize:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")


    print(f"Type of news: {type(news)}")
    summary = stuff_chain.invoke(news)["output_text"]
    # for doc in news[:5]:
    #     # debugging
    #     print(f"doc Type: {type(doc)}")
    #     # print(doc)
    #
    #     # "input_documents" 키를 사용하여 StuffDocumentsChain에 올바른 입력 제공
    #     summary = stuff_chain.invoke(doc)["output_text"]
    #     summaries.append(summary)
    #     # print(summary)
    # #
    #
    #     gc.collect()
    return summary

    # map_template = """The following is a set of documents
    # {docs}
    # Based on this list of docs, please identify the main themes 
    # Helpful Answer:"""
    # map_prompt = PromptTemplate.from_template(map_template)
    # map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # reduce_template = """The following is set of summaries:
    # {docs}
    # Take these and distill it into a final, consolidated summary of the main themes. 
    # Helpful Answer:"""
    # reduce_prompt = PromptTemplate.from_template(reduce_template)
    # reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=reduce_chain, document_variable_name="docs"
    # )

    # # Combines and iteratively reduces the mapped documents
    # reduce_documents_chain = ReduceDocumentsChain(
    #     # This is final chain that is called.
    #     combine_documents_chain=combine_documents_chain,
    #     # If documents exceed context for `StuffDocumentsChain`
    #     collapse_documents_chain=combine_documents_chain,
    #     # The maximum number of tokens to group documents into.
    #     token_max=4000,
    # )
        
    # # Combining documents by mapping a chain over them, then combining results
    # map_reduce_chain = MapReduceDocumentsChain(
    #     # Map chain
    #     llm_chain=map_chain,
    #     # Reduce chain
    #     reduce_documents_chain=reduce_documents_chain,
    #     # The variable name in the llm_chain to put the documents in
    #     document_variable_name="docs",
    #     # Return the results of the map steps in the output
    #     return_intermediate_steps=False,
    # )

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=1000, chunk_overlap=0
    # )
    # split_docs = text_splitter.split_documents(docs)

    # result = map_reduce_chain.invoke(split_docs)

    # print(result["output_text"])


if __name__=='__main__':
    dotenv.load_dotenv()
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    response = summarize(preferred_news)
    print(type(response))
    print(response)