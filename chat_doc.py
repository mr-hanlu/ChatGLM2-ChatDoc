import gradio as gr
import os
import time
import fitz
from PIL import Image
import numpy as np
import cv2
import requests
import json
from read_doc import chorma_loader


class ReadDocument():
    def __init__(self):
        self.chormadb = None
        self.collection = None

    def show_img(self, file_path):
        document = fitz.open(file_path)
        out_img = []
        for i in range(len(document)):
            page = document[i]
            picture = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            image = Image.frombytes("RGB", (picture.width, picture.height), picture.samples)
            img = np.uint8(image)
            out_img.append(img)
        # out_img = np.concatenate(out_img, axis=0)
        return out_img

    def choose_loader(self, check_type, doc_type, file, client_path, chunk_size, chunk_overlap):

        # 创建或加载客户端
        self.chormadb = chorma_loader(client_path)

        if check_type == "url":
            ids, documents, metadatas = self.chormadb.read_url(file, chunk_size, chunk_overlap)
            collection_name = file.split("/")[-1]
            img = [np.zeros((500, 500, 3), dtype=np.uint8)]

        else:
            file_path = file.name
            print(file_path)
            file_name = os.path.basename(file_path)
            collection_name = file_name.split(".")[0]

            # if file_name.split(".")[-1] == "pdf":
            if doc_type == "pdf":
                ids, documents, metadatas = self.chormadb.read_pdf(file_path, chunk_size, chunk_overlap)

                img = self.show_img(file_path)
            else:
                return None
        # 加载或创建数据库，数据库名称为文件名
        self.collection = self.chormadb.load_or_create_collection(collection_name, ids, documents, metadatas)

        return img

    # 格式化显示对话
    def parse_text(self, text):
        """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split('`')
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f'<br></code></pre>'
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", "\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>" + line
        text = "".join(lines)
        return text

    def search_doc(self, query, k, fetch_k, ischat):
        if ischat == "chat":
            return "", []

        docs = self.collection.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

        docs_list = []
        show_doc = ""
        for doc in docs:
            show_doc += "===================================================\n"
            show_doc += doc.page_content
            docs_list.append(doc.page_content)
            show_doc += "\n\n"

        return show_doc, docs_list

    def predict_doc(self, query, chatbot, system, docs_list, history, ischat, max_length, top_p, temperature):
        print(ischat)
        if ischat == "doc":
            ischat = False
        else:
            ischat = True

        chatbot.append((self.parse_text(query), ""))
        print(ischat)

        response, history = self.get_request(query, docs_list, system, history, ischat, max_length, top_p, temperature)

        chatbot[-1] = (self.parse_text(query), self.parse_text(response))

        # 最后是清空input控件
        return chatbot, history, ""

    def get_request(self, query, docs, system, history, ischat, max_length, top_p, temperature):
        request_url = 'http://127.0.0.1:8000'
        # request_url = 'http://192.168.0.118:8000'

        # 传入界面修改的参数
        data_json = json.dumps({
            "query": query,
            "system": system,
            "docs": docs,
            "history": history,
            "ischat": ischat,
            "max_length": max_length,
            "top_p": top_p,
            "temperature": temperature
        })

        r = requests.post(url=request_url, data=data_json,
                          headers={'Connection': 'close'}).json()
        # headers={'Content-Type': 'application/json'}).json()
        print(r)

        response = r["response"]
        history = r["history"]
        status = r["status"]
        time = r["time"]

        return response, history


# 选择对话的是文档还是url
def change_up_doc(check_type):
    if check_type == "url":
        return gr.update(visible=False), gr.update(visible=True)
    elif check_type == "document":  # 其它格式文件
        return gr.update(visible=True), gr.update(visible=False)


# TODO 有问题，未解决，切换文档类型后怎么只能选择上传这种类型的文档
# 选择对话文档的类型
def change_doc_type(doc_type):
    if doc_type == "pdf":
        return gr.UploadButton("上传文档📁", file_types=[".pdf"])
    elif doc_type == "txt":
        # return gr.update(file_types=[".txt"])
        # gr.Dropdown.update(choices=[".txt"], value=".txt", visible=True)
        # gr.UploadButton.update(file_types=[".txt"])
        return gr.UploadButton("上传文档📁", file_types=[".txt"])


def hide_doc(ischat, history):
    # 更新，由chat变为doc，下面同理
    if ischat == "doc":
        history_dict["chat"] = history
        chatbot = history_dict["doc"]
        history = history_dict["doc"]
        return gr.update(visible=True), history, chatbot
    elif ischat == "chat":
        history_dict["doc"] = history
        chatbot = history_dict["chat"]
        history = history_dict["chat"]
        return gr.update(visible=False), history, chatbot


history_dict = {"chat": [], "doc": []}
read_doc = ReadDocument()
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM2-6B With Document</h1>""")
    ischat = gr.Radio(choices=["doc", "chat"], value="doc", label="chat with document or chat")
    with gr.Row():
        with gr.Column(visible=True) as doc_setting:
            # 显示加载pdf的设置
            with gr.Accordion(label="document setting", open=False):
                chunk_size = gr.Slider(label="chunk_size", minimum=800, maximum=1500, value=1000, step=100)
                chunk_overlap = gr.Slider(label="chunk_overlap", minimum=50, maximum=200, value=150, step=10)
                k = gr.Slider(label="k", minimum=1, maximum=5, value=2, step=1)
                fetch_k = gr.Slider(label="fetch_k", minimum=3, maximum=8, value=5, step=1)
            client_path = gr.Textbox(label="save_chromadb", value="./chroma")

            # 选择文档
            with gr.Row():
                with gr.Row():
                    check_type = gr.Radio(choices=["document", "url"], value="document",
                                          label="select the document type")
                # 上传文档
                with gr.Row(visible=True) as up_doc:
                    doc_type = gr.Dropdown(["pdf", "txt"], label="doc type", info="select the document type",
                                           value="pdf")
                    btn_up = gr.UploadButton("上传文档📁", file_types=[".pdf"])
                # 上传url
                with gr.Row(visible=False) as up_url:
                    with gr.Column():
                        url_input = gr.Textbox(label="url_link", value="")
                        url_send = gr.Button(value="Send Url")
                # 更新
                check_type.change(fn=change_up_doc, inputs=check_type, outputs=[up_doc, up_url])
                doc_type.change(fn=change_doc_type, inputs=doc_type, outputs=[btn_up])

            docBox = gr.Textbox(label="link_doc", container=True, lines=6)
            # show_pdf = gr.Image(label="show_doc")
            # 画廊
            show_doc = gr.Gallery(label="show_doc", height=700)

        with gr.Column() as chat_bot:
            chatbot = gr.Chatbot([], elem_id="chatbot", avatar_images=("../image/person.png", "../image/robot.png"))
            with gr.Accordion(label="model setting", open=False):
                system = gr.Textbox(label="System message", lines=2, value="")
                gr.Examples(examples=[
                    "假设你是一个读文档专家，我会给你几个文档做参考，我会给你一个问题，你需要根据这些文档，回答这个问题",
                    "假设你是一个资深程序员，我会问你一些代码相关的问题，你需要回答我的问题"], inputs=system)
                max_length = gr.Slider(label="max_length", minimum=2016, maximum=4032, value=2048, step=32)
                temperature = gr.Slider(label="temperature", minimum=0, maximum=1.0, value=0.95, step=0.05)
                top_p = gr.Slider(label="top_p", minimum=0, maximum=1.0, value=0.8, step=0.05)
            input = gr.Textbox(label="Input", container=True)
            btn_send = gr.Button(value="Send")

        # 不显示按钮，存储变量
        history = gr.State([])
        docs_list = gr.State([])

        # 使用返回空字符串的方式更新输入控件，succes前面执行成功继续执行触发
        btn_click = btn_send.click(read_doc.search_doc, [input, k, fetch_k, ischat], [docBox, docs_list]).success(
            fn=read_doc.predict_doc,
            inputs=[input, chatbot, system, docs_list, history, ischat, max_length, top_p, temperature],
            outputs=[chatbot, history, input],
            show_progress=True)

        input_msg = input.submit(read_doc.search_doc, [input, k, fetch_k, ischat], [docBox, docs_list]).success(
            fn=read_doc.predict_doc,
            inputs=[input, chatbot, system, docs_list, history, ischat, max_length, top_p, temperature],
            outputs=[chatbot, history, input],
            show_progress=True)

    # 切换对话还是文档后改变历史记录和chatbot显示的记录
    ischat.change(fn=hide_doc, inputs=[ischat, history], outputs=[doc_setting, history])

    # 上传文档，outputs定义了哪些组件会被这个函数的返回值更新
    btn_up.upload(fn=read_doc.choose_loader,
                  inputs=[check_type, doc_type, btn_up, client_path, chunk_size, chunk_overlap],
                  outputs=show_doc)

    url_send.click(fn=read_doc.choose_loader,
                   inputs=[check_type, doc_type, url_input, client_path, chunk_size, chunk_overlap],
                   outputs=show_doc)

    demo.queue()
    demo.launch()
    # demo.launch(share=True)
