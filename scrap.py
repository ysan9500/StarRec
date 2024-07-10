import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from openpyxl import load_workbook
from datetime import datetime
import time
def get_keywords(day_name):
    if day_name == "Monday":
        keywords = ["Micro LED", "wearable device", "curved shape display", "VR display", "AR display", "MR display"]
    elif day_name == "Tuesday":
        keywords = ["RADAR sensor", "LIDAR sensor", "vision sensor", "wireless charging", "teraHertz bandwidth", "satellite communication antenna", "high index lense"]
    elif day_name == "Wednesday":
        keywords = ["solid eletrolyte", "electrolysis and clean hydrogen", "flow battery", "solid oxide fuel cell", "perovskite PV"]
    elif day_name == "Thursday":
        keywords = ["ceramic filter and fine dust particle removal", "carbon dioxide capture", "hydrogen storage and absorbent"]
    elif day_name == "Friday":
        keywords = ["2D packaging", "2.5D packaging", "3D packaging", "synthetic quarts and blankmask", "cell culture and bio filter"]
    return keywords
def getNewsData(keyword:str, num:int):
    D_name = datetime.today()
    start_date = datetime(D_name.year,D_name.month,D_name.day)
    start_date = str(start_date)[:10]
    if (D_name.day-7) < 1 :
        end_date = datetime(D_name.year,D_name.month-1,D_name.day-7+30)
    else:
        end_date = datetime(D_name.year,D_name.month,D_name.day-7)
    end_date = str(end_date)[:10]
    cd_min = start_date[6:7] + '/' + start_date[8:10] + '/' + start_date[:4]
    cd_max = end_date[6:7] + '/' + end_date[8:10] + '/' + end_date[:4]
    tbs = f'cdr:1,cd_min:{cd_min},cd_max:{cd_max}'
    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }
    response = requests.get(
        "https://www.google.com/search?q="+keyword+"&gl=us&tbm=nws&tbs="+tbs+"&num="+str(num), headers=headers
    )
    soup = BeautifulSoup(response.content, "html.parser")
    news_results = []
    for el in soup.select("div.SoaBEf"):
        news_results.append(
            {
                "link": el.find("a")["href"],
                #"title": el.select_one("div.MBeuO").get_text(),
                "title": el.select_one("div.n0jPhd").get_text(),
                "snippet": el.select_one(".GI74Re").get_text(),
                "date": el.select_one(".LfVVr").get_text(),
                "source": el.select_one(".NUnG9d span").get_text()
            }
        )
    return json.dumps(news_results, indent=2)
def json_to_df(results, keywords):
    for i in range(len(results)):
        if i == 0:
            with pd.ExcelWriter('news.xlsx', engine='openpyxl', mode='w') as writer:
                workBook = writer.book
                df = pd.DataFrame(json.loads(results[i]))
                df.to_excel(writer, sheet_name = keywords[i], index = False )
        else:
            with pd.ExcelWriter('news.xlsx', engine='openpyxl', mode='a') as writer:
                workBook = writer.book
                df = pd.DataFrame(json.loads(results[i]))
                df.to_excel(writer, sheet_name = keywords[i], index = False )
if __name__ == "__main__":
    day_name = datetime.today().strftime("%A")
    keywords = get_keywords(day_name)
    results = []
    #keywords = ['wireless charging','wearable device', 'Curved shape display',
    #            'VR display', 'AR display', 'MR display', 'Tera Hertz bandwidth',
    #            'flow battery', 'carbon dioxide capture', 'Satellite communication antenna',
    #            'high index lense', 'hydrogen storage', 'solid oxide fuel cell', 'perovskite PV',
    #            '2.5D glass packaging or 3D glass packaging', 'Blankmask synthetic and quartz',
    #            'solid electrolyte', 'hydrolysis clean', 'Radar sensor', 'Lidar sensor', 'vision sensor',
    #            'bio filter', 'hydrogen storage sorbent','fine particle removal']
    num = 30
    for keyword in keywords:
        lists = getNewsData(keyword, num)
        results.append(lists)
    json_to_df(results, keywords)