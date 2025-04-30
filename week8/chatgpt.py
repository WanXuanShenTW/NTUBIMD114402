from openai import OpenAI

client = OpenAI(api_key='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')
import bug


def generate_summary(text):
    prompt = f"""請將以下有趣的新聞內容轉換為簡短、適合語音朗讀的口語化摘要（控制在50字以內），將今日改成發生日期：
{text}
"""
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content.strip()

summary = generate_summary(bug.content)
print("摘要：", summary)
