import dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.load import dumpd, dumps, load, loads
from langchain_core.output_parsers import StrOutputParser

import gc

import json
from langchain_core.documents import Document

from langchain_core.messages import HumanMessage, SystemMessage

output_parser = StrOutputParser()

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



    print(f"Type of news: {type(news)}")
    for doc in news[:1]:
        # debugging
        print(f"doc Type: {type(doc)}")
        # print(doc)
        #prompt = ChatPromptTemplate.from_messages([
        #    ("system", "you are a professional newswirter. summrize belowed content in 3 sentences"),
        #    ("user","{input}")])
        messages = [{"role":"system","content":"you are a professional newswirter. summrize belowed content in 3 sentences"},
                {"role":"user","content":"{doc.page_content}"}
        ]


        #chain = prompt | llm | output_parser
        #aa = chain.invoke({"input":doc.page_content})
        print(llm.invoke(messages))
        #summaries.append(Document(page_content=summary))
        # print(summary)

        gc.collect()

    string_representation = dumps(summaries, pretty=True)
    with open("database/summaries.json", "w") as fp:
        json.dump(string_representation, fp)

    return



if __name__=='__main__':
    dotenv.load_dotenv()
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    with open("database/news.json", "r") as fp:
        news = loads(json.load(fp))
    # clean_news = cleanup(preferred_news)
    print(news[0].page_content)
    print('-------------')
    summaries = summarize(news)
    print(type(summaries))
    print(summaries)
