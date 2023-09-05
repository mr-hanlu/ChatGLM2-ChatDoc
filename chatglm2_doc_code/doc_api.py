from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

tokenizer = AutoTokenizer.from_pretrained("chatglm2_6b", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm2_6b", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2_6b", num_gpus=2)
model = model.eval()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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
    # response, history = model.chat(tokenizer,
    #                                prompt,
    #                                history=history,
    #                                max_length=max_length if max_length else 2048,
    #                                top_p=top_p if top_p else 0.7,
    #                                temperature=temperature if temperature else 0.95)

    # for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
    #                                                             return_past_key_values=True,
    #                                                             max_length=max_length, top_p=top_p,
    #                                                             temperature=temperature):
    response, history = model.chat_docs(tokenizer, query, docs, system=system,
                                        history=history, ischat=ischat,
                                        max_length=max_length if max_length else 2048,
                                        top_p=top_p if top_p else 0.7,
                                        temperature=temperature if temperature else 0.95)
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
    torch_gc()
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    # uvicorn.run(app, host='127.0.0.1', port=8000, workers=1)
