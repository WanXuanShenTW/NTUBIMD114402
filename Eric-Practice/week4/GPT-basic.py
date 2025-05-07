from openai import OpenAI

client = OpenAI(api_key=key)

key="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"


a = input("assistant:")
b = input("user:")

response = client.chat.completions.create(model="gpt-4o",
messages=[
  {"role": "system", "content": '請說繁體中文'},
  {"role": "assistant", "content": a},
  {"role": "user", "content": b}
])

print(response.choices[0].message.content)
