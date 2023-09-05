import random

from fastapi import FastAPI, Request
import uvicorn, json, datetime


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')
    system = json_post_list.get('system')
    docs = json_post_list.get('docs')
    history = json_post_list.get('history')
    ischat = json_post_list.get('ischat')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response = str(random.randint(0,100))
    history += [(query, response)]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + query + '", response:"' + repr(response) + '"'
    # print(log)
    # torch_gc()
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    # uvicorn.run(app, host='127.0.0.1', port=8000, workers=1)
