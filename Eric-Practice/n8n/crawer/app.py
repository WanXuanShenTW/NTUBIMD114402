from fastapi import FastAPI, Request
from news import GoogleNews
import json

app = FastAPI()

@app.post("/search")
async def search(request: Request):
    body = await request.body()
    body = json.loads(body)
    keyword = body.get("keyword", "")

    googlenews = GoogleNews()
    googlenews.search(keyword)
    result = googlenews.gettext()

    print(result)

    return {"results": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)