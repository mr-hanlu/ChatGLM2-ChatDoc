import gradio as gr
import os
import time
import fitz
from PIL import Image
import numpy as np
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

    def choose_loader(self, check_type, file, client_path, chunk_size, chunk_overlap):
        # 创建或加载客户端
        self.chormadb = chorma_loader(client_path)

        if check_type == "url":
            ids, documents, metadatas, docstr = self.chormadb.read_url(file, chunk_size, chunk_overlap)
            collection_name = file.split("/")[-2] + "_" + file.split("/")[-1]
            img = None

        else:
            file_path = file.name
            print(file_path)
            file_name = os.path.basename(file_path)
            collection_name = file_name.split(".")[0]
            is_chinese = sum([True for ch in collection_name if u'\u4e00' <= ch <= u'\u9fff'])
            # 是中文，无法创建数据库名，用ASCII码转字符串
            if is_chinese:
                # 数据库名称为3-63的字符串
                collection_name = ("china_" + "_".join([str(ord(ch)) for ch in collection_name]))[0:62]
            doc_type = file_name.split(".")[-1]

            # if file_name.split(".")[-1] == "pdf":
            if doc_type == "pdf":
                ids, documents, metadatas, docstr = self.chormadb.read_pdf(file_path, chunk_size, chunk_overlap)

                img = self.show_img(file_path)
            elif doc_type in ["docx", "doc"]:
                ids, documents, metadatas, docstr = self.chormadb.read_docx(file_path, chunk_size, chunk_overlap)

                img = None

            elif doc_type == "csv":
                ids, documents, metadatas, docstr = self.chormadb.read_csv(file_path, chunk_size, chunk_overlap)

                img = None
            elif doc_type == "md":
                ids, documents, metadatas, docstr = self.chormadb.read_md(file_path, chunk_size, chunk_overlap)

                img = None
            elif doc_type == "txt":
                ids, documents, metadatas, docstr = self.chormadb.read_txt(file_path, chunk_size, chunk_overlap)

                img = None
            else:
                ids, documents, metadatas, docstr = self.chormadb.read_other(file_path, chunk_size, chunk_overlap)

                img = None
        # 加载或创建数据库，数据库名称为文件名
        self.collection = self.chormadb.load_or_create_collection(collection_name, ids, documents, metadatas)

        # return img
        if img is not None:
            return gr.update(value=img, visible=True), gr.update(value="", visible=False)
        else:
            img = [np.zeros((10, 10, 3), dtype=np.uint8)]
            return gr.update(value=img, visible=False), gr.update(value=docstr, visible=True)

    # 格式化显示对话
    def parse_text(self, text):
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
        if self.collection is None:
            raise gr.Error("没有加载文档...")

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
        if ischat == "doc":
            ischat = False
        else:
            ischat = True

        chatbot.append((self.parse_text(query), ""))

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


# 文档的类实例化
read_doc = ReadDocument()


# 选择对话的是文档还是url
def change_up_doc(check_type):
    if check_type == "url":
        return gr.update(visible=False), gr.update(visible=True)
    elif check_type == "document":  # 其它格式文件
        return gr.update(visible=True), gr.update(visible=False)


# 显示/隐藏读取文档
def hide_doc(ischat, history_dict_list):
    # 更新，由chat变为doc，下面同理
    if ischat == "doc":
        index = 0
    else:  # "chat"
        index = 1
    history_dict = history_dict_list[index]
    choices_list = list(history_dict.keys())
    if not choices_list:
        choices_list.append("")

    return gr.update(visible=not index), gr.update(choices=choices_list, value=choices_list[0])


# 改变bot的显示记录
def changeBot(all_chat_name, history_dict_list, ischat):
    # 没有这个bot
    if all_chat_name == "":
        return [], []

    if ischat == "doc":
        index = 0
    else:
        index = 1
    history_dict = history_dict_list[index]
    history = history_dict[all_chat_name]["history"]
    system = history_dict[all_chat_name]["system"]
    chatbot = []
    for query, response in history:
        chatbot.append((read_doc.parse_text(query), read_doc.parse_text(response)))
    return chatbot, history, system


def addHistory(history, all_chat_name, history_dict_list, ischat, system):
    if ischat == "doc":
        index = 0
    else:
        index = 1
    # 没有创建bot默认创建newbot
    if all_chat_name == "":
        all_chat_name = "newbot"
        botTemplate = {"ischat": bool(index), "history": [], "system": ""}
        history_dict_list[index][all_chat_name] = botTemplate
    history_dict_list[index][all_chat_name]["history"] = history
    history_dict_list[index][all_chat_name]["system"] = system
    # TODO 数据保存的最佳位置
    print(history_dict_list)
    return gr.update(choices=list(history_dict_list[index].keys()), value=all_chat_name), history_dict_list, ""


# 创建一个聊天bot
def createBot(newChatName, history_dict_list, ischat):
    # 每个bot的模板
    botTemplate = {"ischat": True, "history": [], "system": ""}
    if ischat == "doc":
        index = 0
    else:
        index = 1
    history_dict = history_dict_list[index]
    history_dict_list[index] = history_dict
    chat_name_list = list(history_dict.keys())
    # 新加bot名为空
    if newChatName == "":
        gr.Warning("bot name 不能为空...")
        return gr.update(), history_dict, ""
    # 如果存在就不创建了
    if newChatName in chat_name_list:
        gr.Warning("该聊天已存在,跳转到该聊天...")
        return gr.update(choices=list(history_dict.keys()), value=newChatName), history_dict_list, ""
    history_dict[newChatName] = botTemplate
    history_dict[newChatName]["ischat"] = True if ischat == "chat" else False
    history_dict_list[index] = history_dict
    # print(history_dict_list)
    # 最后的空字符串是清空输入名字的框
    return gr.update(choices=list(history_dict.keys()), value=newChatName), history_dict_list, ""


# 删除聊天
def deleteBot(all_chat_name, history_dict_list, ischat):
    if ischat == "doc":
        index = 0
    else:
        index = 1
    history_dict = history_dict_list[index]
    chat_name_list = list(history_dict.keys())
    deleteChatName = all_chat_name
    chat_name_list.remove(deleteChatName)
    history_dict.pop(deleteChatName)
    history_dict_list[index] = history_dict
    return gr.update(choices=chat_name_list, value=chat_name_list[0]), history_dict_list, ""


# TODO 1.一个棘手的问题，怎么改变选择上传文件的后缀类型。2.一个不懂的问题，上下文如果超过长度会怎么样。3.找一些好的默认prompt


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM2-6B With Document</h1>""")
    # 增加一个聊天就增加一个历史记录的键
    # 增加聊天通过改变下拉框的choices和value实现

    # 第一个字典是doc,第二个字典是chat
    history_dict_list = gr.State([{}, {}])
    # ischat = gr.Radio(choices=["doc", "chat"], value="doc", label="chat with document or chat")
    with gr.Row():
        # 用来创建新bot
        with gr.Column(scale=1):
            ischat = gr.Radio(choices=["doc", "chat"], value="doc", label="chat with doc or chat", interactive=True)
            chat_name = gr.Textbox(label="bot name", value="ChatGLM2-6B")  # 对话的名字
            with gr.Row():
                create = gr.Button(value="create new bot")  # 确定按钮
                delete = gr.Button(value="delete now bot")  # 删除按钮
            all_chat_name = gr.Dropdown([""], label="all chat name", value="", interactive=True)

            create_btn = create.click(fn=createBot, inputs=[chat_name, history_dict_list, ischat],
                                      outputs=[all_chat_name, history_dict_list, chat_name])
            delete.click(fn=deleteBot, inputs=[all_chat_name, history_dict_list, ischat],
                         outputs=[all_chat_name, history_dict_list, chat_name])

        with gr.Column(scale=4, visible=True) as doc_setting:
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
                    # TODO 没法选择上传文件的后缀类型
                    btn_up = gr.UploadButton("上传文档📁", file_types=[".pdf", ".csv", ".docx", ".doc", ".md", ".txt"])
                # 上传url
                with gr.Row(visible=False) as up_url:
                    with gr.Column():
                        url_input = gr.Textbox(label="url_link", value="")
                        url_send = gr.Button(value="Send Url")
                # 更新
                check_type.change(fn=change_up_doc, inputs=check_type, outputs=[up_doc, up_url])

            docBox = gr.Textbox(label="link_doc", container=True, lines=6)
            # show_pdf = gr.Image(label="show_doc")
            # 画廊
            show_doc_img = gr.Gallery(label="show_doc", height=700, visible=True)
            show_doc = gr.Textbox(label="document", container=True, lines=10, visible=False)
            # show_doc = gr.Markdown(label="document", container=True, lines=10, visible=False)

        with gr.Column(scale=4) as chat_bot:
            chatbot = gr.Chatbot([], elem_id="chatbot", avatar_images=("./image/person.png", "./image/robot.png"))
            with gr.Accordion(label="model setting", open=False):
                system = gr.Textbox(label="System message", lines=2, value="", interactive=True)
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

        # 使用返回空字符串的方式更新输入控件，success前面执行成功继续执行触发
        btn_click = btn_send.click(read_doc.search_doc, [input, k, fetch_k, ischat], [docBox, docs_list]).success(
            fn=read_doc.predict_doc,
            inputs=[input, chatbot, system, docs_list, history, ischat, max_length, top_p, temperature],
            outputs=[chatbot, history, input],
            show_progress=True).success(fn=addHistory,  # 更新history_dict里的history
                                        inputs=[history, all_chat_name, history_dict_list, ischat, system],
                                        outputs=[all_chat_name, history_dict_list, chat_name])

        input_msg = input.submit(read_doc.search_doc, [input, k, fetch_k, ischat], [docBox, docs_list]).success(
            fn=read_doc.predict_doc,
            inputs=[input, chatbot, system, docs_list, history, ischat, max_length, top_p, temperature],
            outputs=[chatbot, history, input],
            show_progress=True).success(fn=addHistory,  # 更新history_dict里的history
                                        inputs=[history, all_chat_name, history_dict_list, ischat, system],
                                        outputs=[all_chat_name, history_dict_list, chat_name])

        # 切换对话还是文档后改变历史记录和chatbot显示的记录
    ischat.change(fn=hide_doc, inputs=[ischat, history_dict_list], outputs=[doc_setting, all_chat_name])
    # 选择不同的对话，显示不同的bot
    all_chat_name.change(fn=changeBot, inputs=[all_chat_name, history_dict_list, ischat],
                         outputs=[chatbot, history, system])

    # 上传文档，outputs定义了哪些组件会被这个函数的返回值更新
    btn_up.upload(fn=read_doc.choose_loader,
                  inputs=[check_type, btn_up, client_path, chunk_size, chunk_overlap],
                  outputs=[show_doc_img, show_doc])

    url_send.click(fn=read_doc.choose_loader,
                   inputs=[check_type, url_input, client_path, chunk_size, chunk_overlap],
                   outputs=[show_doc_img, show_doc])

    demo.queue()
    demo.launch()
    # demo.launch(share=True)
