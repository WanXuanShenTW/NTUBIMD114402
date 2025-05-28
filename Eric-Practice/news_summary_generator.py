from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_daily_summary():
    prompt = (
        "請用繁體中文列出今日三則適合台灣長者收聽的重要新聞，"
        "內容需清楚易懂，避免使用艱深詞彙。"
        "主題可包含：本區天氣、健康保健、食安、社會關懷、生活資訊等。"
        "每則新聞請用一段落呈現，方便轉換為語音播放。"
        "每個段落間請座分隔。"
        "請不要加入開場白或結尾語，直接列出新聞標題和內容。"
        "也不需要說1. 2. 3. etc."
        "把長者改成爺爺奶奶們，讓他們聽得懂。"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一位台灣新聞摘要專家，現在要講內容給老人家聽。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7
    )
    summary = response.choices[0].message.content.strip()
    return summary
