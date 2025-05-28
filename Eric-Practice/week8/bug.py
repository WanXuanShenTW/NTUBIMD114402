import requests
from bs4 import BeautifulSoup

# 目標新聞網址
url = 'https://tw.news.yahoo.com/%E5%A4%A7%E9%9B%B7%E9%9B%A8%E8%A5%B2%E6%B7%A1%E6%B0%B4-%E5%8C%97%E6%8D%B7%E8%BB%8A%E5%BB%82%E8%AE%8A-%E6%B0%B4%E6%BF%82%E6%B4%9E-%E4%B9%98%E5%AE%A2%E5%82%BB%E7%9C%BC-20%E5%B9%B4%E9%A6%96%E8%A6%8B-040843538.html'

# 設定 User-Agent 模擬瀏覽器行為
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 發送 GET 請求
response = requests.get(url, headers=headers)

# 檢查請求是否成功
if response.status_code == 200:
    # 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')
else:
    print(f"無法取得網頁內容，狀態碼：{response.status_code}")

# 提取標題
title_tag = soup.find('h1')
if title_tag:
    title = title_tag.text.strip()
    print("標題：", title)
else:
    print("無法找到標題")

# 提取內文
content_tags = soup.find_all('p')
content = ''
for tag in content_tags:
    content += tag.text.strip() + '\n'

if content:
    print("內容：", content)
else:
    print("無法找到內文")
