import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_daily_summary():
    """
    生成每日新聞摘要（非同步版本）。
    """
    prompt = (
        "爺爺奶奶，早安。以下是今日三則適合台灣長者收聽的重要新聞。\n"
        "請用繁體中文清楚易懂地描述，避免艱深詞彙，主題可包含天氣、健康、食安、生活資訊等。\n"
        "每則新聞請用一段落呈現，段落間空一行，適合轉為語音播放。\n"
        "不需要列出編號，例如 '1.', '2.' 等。\n"
        "也不需要過度強調長者身份，可以自然稱呼為『您』。"
    )

    try:
        response = await client.chat.completions.acreate(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一位台灣新聞播報員，善於口語化講解新聞內容給長輩聽。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        raise RuntimeError(f"生成新聞摘要失敗：{e}")