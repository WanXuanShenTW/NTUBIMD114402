from openai import OpenAI

client = OpenAI(api_key=key)
from flask import Flask, request, jsonify
import os

key = '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'

# 初始化Flask應用程式
app = Flask(__name__)

# 從環境變數中獲取 OpenAI API 密鑰，避免將密鑰公開

# 生成文本的API端點
@app.route('/generate', methods=['POST'])
def generate_text():
    # 從請求中獲取用戶輸入的文本
    data = request.json
    prompt = data.get("prompt", "")

    # 使用OpenAI API調用GPT-4模型生成文本
    try:
        response = client.chat.completions.create(model="gpt-4",  # 使用 GPT-4 模型或 gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "none"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,  # 設定生成文本的最大長度
        temperature=0.7)

        # 獲取生成的文本
        generated_text = response.choices[0].message.content.strip()

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 啟動Flask伺服器
if __name__ == '__main__':
    app.run(debug=True)
