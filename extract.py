# Indexing: Load -> Split -> Store
import pandas as pd
import shutil
import tempfile
import urllib.request
from langchain_community.document_loaders import BSHTMLLoader

def get_links():
    news_excel = 'news.xlsx'
    news_dict = pd.read_excel(news_excel, sheet_name=None)
    keywords = news_dict.keys()
    for keyword in keywords:
        links = news_dict[keyword]['link']
        print(links)
    

# Get HTML with "urllib"


with urllib.request.urlopen('http://python.org/') as response:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        shutil.copyfileobj(response, tmp_file)

with open(tmp_file.name) as html:
    pass

# Load HTML with "BeautifulSoup"


loader = BSHTMLLoader(tmp_file.name)
data = loader.load()

#print(data)

if __name__ == "__main__":
    get_links()