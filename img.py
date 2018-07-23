from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO

outSize = 100, 100
pics = []
count = 1
import pathlib
for filepath in pathlib.Path("dataset/09").glob('**/*'):
    pics.append(filepath.absolute())
pathlib.Path('dataset/faces').mkdir(parents=True, exist_ok=True)


for facepath in pics:
    img = Image.open(facepath)
    w, h = img.size
    if (w>100 and h> 100):
        img = img.convert('RGB')
        img.thumbnail(outSize, Image.ANTIALIAS)
        img.save("dataset/faces/face_"+str(count)+".jpg")
        count += 1
