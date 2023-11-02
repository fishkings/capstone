# 구글 이미지 크롤링
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from webdriver_manager.chrome import ChromeDriverManager

import time
import csv
import pyautogui
import urllib.request  # 이미지 저장 모듈
import os

import requests
keyword = pyautogui.prompt("검색어를 입력하세요.")

chrome_options = Options()
chrome_options.add_experimental_option("detach",True)

service = Service(executable_path=ChromeDriverManager().install())  # Chrome driver 자동 업데이트
driver = webdriver.Chrome(service=service, options=chrome_options)
# driver.maximize_window() #화면 최대화

driver.get(f"https://www.google.com/search?q={keyword}&sca_esv=578808334&hl=ko&tbm=isch&sxsrf=AM9HkKmhDhqVXleLCdu9w87OULuggo1A3w:1698928634577&source=lnms&sa=X&ved=2ahUKEwic0YyuqqWCAxVRdfUHHTMmCjIQ_AUoAXoECAIQAw&biw=1920&bih=945&dpr=1")
driver.implicitly_wait(4)



# #  무한 스크롤
before_h = driver.execute_script("return window.scrollY")

while True:
    # 맨 아래로 스크롤 내린다
    driver.find_element(By.CSS_SELECTOR,"body").send_keys(Keys.END)
    time.sleep(1)
    after_h = driver.execute_script("return window.scrollY")

    if after_h == before_h:
        try:
             driver.find_element(By.CSS_SELECTOR,"input.LZ4I").click()  # 결과 더보기 클릭
        except:
            break
    before_h = after_h



imgs = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")

for idx,img in enumerate(imgs):
    # click intercept error가 발생했을 때 직접 javascript 코드로 제어 (img.click() 말고)
    driver.execute_script('arguments[0].click();',img)
    time.sleep(.1)

    # 이미지 URL 가져오기
    try:
        img_url = driver.find_element(By.CSS_SELECTOR, ".sFlh5c.pT0Scc.iPVvYb").get_attribute('src')
    except :
        continue

    file_path = r"C:\Users\kcjer\OneDrive\바탕 화면\크롤링공부사진"

    #HTTP Error 403 : Forbidden 에러 발생시
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent','Mozila/5.0')]
    urllib.request.install_opener(opener)

    # 이미지 다운로드
    try:
        print(idx)
        urllib.request.urlretrieve(img_url,f"{file_path}/{idx}.jpg")
    except:
        print("실패")
        pass
