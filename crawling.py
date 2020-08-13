from urllib.request import urlopen # url 읽기
from urllib.request import urlretrieve # url의 이미지 저장
from urllib.parse import quote_plus # 한글 키워드 입력
from bs4 import BeautifulSoup # 크롤링 필수템
from selenium import webdriver # webdriver 여는 데 필요한 selenium
import os # 디렉토리 접근 사용 때문에

# txt 파일 안 포켓몬 읽기 (없어도 됨)
f = open("C:/Users/jsl11/crawling/pokemon.txt", "r", encoding="utf-8")

search = []
while True:
    line = f.readline()
    if not line:
        break
    search.append(line.strip())  # '\n' 제거

f.close()

for se in range(len(search)):
    word = search[se]

    url = f'https://www.google.com/search?q={quote_plus(word)}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjRo8Wp7IXrAhWvyosBHak0B_8Q_AUoAXoECBkQAw&biw=965&bih=879'
    driver = webdriver.Chrome("C:/Users/chromedriver.exe")
    driver.get(url)

    for i in range(500):
        driver.execute_script("window.scrollBy(0, 10000)")

    html = driver.page_source
    soup = BeautifulSoup(html)
    img = soup.select('.rg_i.Q4LuWd')  # image 찾기

    n = 0
    imgurl = []

    for i in img:
        try:
            imgurl.append(i.attrs["src"])
        except KeyError:
            imgurl.append(i.attrs["data-src"])

    path = "C:/Users/jsl11/crawling/" + word + '/'
    os.mkdir(path)  # 포켓몬 폴더 생성

    for i in imgurl:
        urlretrieve(i, path + "/" + word + str(n + 1) + ".jpg")
        n += 1
        if n == 300:
            break

    driver.close()
