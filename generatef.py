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
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=0.7,
        ),
        device=0,
        batch_size=1,
    )

    summaries = []



    #print(f"Type of news: {type(news)}")
    for doc in news[2:3]:
        # debugging
        # print(doc)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "you are a professional newswirter. summrize belowed content in 3 sentences"),
            ("user","{input}")])
        #messages = [{"role":"system","content":"you are a professional news writter. summerize belowed news within 3 words. "},
#                {"role":"user","content":"vHorses have a funny way of keeping us on our toes, don’t they? Just when you think things are going one way, we constantly have to pivot another way.Our last update saw the big horse Mr Nobility getting ready to display his progress at the LongRun open house. Unfortunately, he had other plans. The day before the open house he had blood and mucus coming out of his nose. The next day, the morning of the open house, more mucus, coughing, and lots of blood pouring out of his nose. The veterinarian came out for an emergency endoscopy to see what was going on, and unfortunately didn’t really find much other than a very irritated soft palate and a lot of bloody mucus in the throat. Needless to say, we did not ride at the open house.\t\tMr Nobility naps in the weirdest places. (Courtesy of Lauren Millet-Simpson)\tSince we are getting down to the wire as such, we really needed to start going off property and experimenting with different adventures. My coach booked a schooling time at a local show barn for myself and her other student who is going to the Makeover for an off-property ride.The week leading up to going off property was one of the most frustrating weeks for us in a while."}
#        ]

#        print(messages)
        chain = prompt | llm | output_parser
        aa = chain.invoke({"input":doc.page_content})
        print('---',aa)
        #print(llm.invoke(messages))

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
    #print(news[0].page_content)
    #print('-------------')
    summaries = summarize(news)
    #print(type(summaries))
    #print(summaries)
