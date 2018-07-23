from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO

scrapeCount = 500
outSize = 100, 100
scraped = 23
currentPage = 3
SCRAPE_URL = "https://avatars.alphacoders.com/by_category/3?page="

import pathlib
pathlib.Path('dataset/anime').mkdir(parents=True, exist_ok=True) 

while (scraped <= scrapeCount):
    print("Parsing page " + str(currentPage))
    r = requests.get(SCRAPE_URL+str(currentPage))
    soup = BeautifulSoup(r.text)
    pics = soup.findAll("img", {"class": "avatar-thumb"})
    for pic in pics:
        src = pic['src']
        if ((src.endswith('.jpg') or src.endswith('.png')) and scraped <= scrapeCount):
            print(src)
            response = requests.get(src)
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img.thumbnail(outSize, Image.ANTIALIAS)
            img.save("dataset/anime/anime_"+str(scraped)+".jpg")
            scraped += 1
    if (scraped <= scrapeCount):
        currentPage += 1
        # print(pic['src'])