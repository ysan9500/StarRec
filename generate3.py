import dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoTokenizer
import transformers
import torch

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

def summarize(news):

    output_parser = StrOutputParser()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True)

    model = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=3500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})


    template = """
                Write a summary of the following text delimited by triple backticks.
                Return your response which covers the key points of the text within 30 words.
                ```{text}```
                SUMMARY:
            """

    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summaries = []

    for doc in news:
        text = doc.page_content

    # text = '''
    # vHorses have a funny way of keeping us on our toes, don’t they? Just when you think things are going one way, we constantly have to pivot another way.Our last update saw the big horse Mr Nobility getting ready to display his progress at the LongRun open house. Unfortunately, he had other plans. The day before the open house he had blood and mucus coming out of his nose. The next day, the morning of the open house, more mucus, coughing, and lots of blood pouring out of his nose. The veterinarian came out for an emergency endoscopy to see what was going on, and unfortunately didn’t really find much other than a very irritated soft palate and a lot of bloody mucus in the throat. Needless to say, we did not ride at the open house.\t\tMr Nobility naps in the weirdest places. (Courtesy of Lauren Millet-Simpson)\tSince we are getting down to the wire as such, we really needed to start going off property and experimenting with different adventures. My coach booked a schooling time at a local show barn for myself and her other student who is going to the Makeover for an off-property ride.The week leading up to going off property was one of the most frustrating weeks for us in a while.
    # '''
        summary = llm_chain.run(text)
        print(type(summary))
        summaries.append(summary)

        
    string_representation = dumps(summaries, pretty=True)
    with open("database/summaries.json", "w") as fp:
        json.dump(string_representation, fp)
    return


if __name__=='__main__':
    dotenv.load_dotenv()
    with open("database/news.json", "r") as fp:
        news = loads(json.load(fp))
    summarize(news)