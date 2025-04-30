from openai import OpenAI

client = OpenAI(api_key="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")

PROMPT = "漫天星河，兩個人在觀星"

response = client.images.generate(prompt=PROMPT,
n=1,
# 尺寸部分是固定的，256 x 256、512 x 512、1024 x 1024
size="512x512")

print(response.data[0].url)