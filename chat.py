import gradio as gr
import os
import time
import fitz
from PIL import Image
import cv2
import numpy as np
from read_doc import chorma_loader


class ReadPdf():
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
        out_img = np.concatenate(out_img, axis=0)
        return out_img

    def choose_loader(self, file, client_path, chunk_size, chunk_overlap):
        file_path = file.name
        print(file_path)
        file_name = os.path.basename(file_path)
        if file_name.split(".")[-1] == "pdf":
            # 创建或加载客户端
            # global chormadb
            self.chormadb = chorma_loader(client_path)
            ids, documents, metadatas = self.chormadb.read_pdf(file_path, chunk_size, chunk_overlap)
            # 加载或创建数据库
            # global collection
            self.collection = self.chormadb.load_or_create_collection(file_name.split(".")[0], ids, documents,
                                                                      metadatas)
            img = self.show_img(file_path)
        else:
            self.collection = None
            img = None

        return img

    def add_text(self, history, query, k, fetch_k, temperature, system):
        self.temperature = temperature
        self.system = system
        docs = self.collection.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        show_doc = ""
        for doc in docs:
            show_doc += "===================================================\n"
            show_doc += doc.page_content
            show_doc += "\n\n"
        history = history + [(query, None)]
        return history, gr.update(value="", interactive=False), show_doc

    def bot(self, history):
        print(history)
        query = history[-1][0]
        response = "That's cool!"
        history[-1][1] = ""
        for character in response:
            history[-1][1] += character
            time.sleep(0.05)
            yield history


read_pdf = ReadPdf()
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # 显示加载pdf的设置
            with gr.Accordion(label="pdf setting", open=False):
                chunk_size = gr.Slider(label="chunk_size", minimum=800, maximum=1500, value=1000, step=100)
                chunk_overlap = gr.Slider(label="chunk_overlap", minimum=50, maximum=200, value=150, step=10)
                k = gr.Slider(label="k", minimum=1, maximum=5, value=2, step=1)
                fetch_k = gr.Slider(label="fetch_k", minimum=3, maximum=8, value=5, step=1)
            client_path = gr.Textbox(label="save_chromadb", value="./chroma")
            btn_up = gr.UploadButton("上传文档📁", file_types=[".pdf"])
            docBox = gr.Textbox(label="link_pdf", container=True, lines=6)
            show_pdf = gr.Image(label="show_pdf")

        with gr.Column():
            chatbot = gr.Chatbot([], elem_id="chatbot", avatar_images=(("../image/person.png"), ("../image/robot.png")))
            with gr.Accordion(label="Advanced options", open=False):
                system = gr.Textbox(label="System message", lines=2, value="")
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1.0, value=0.7, step=0.1)
            txt = gr.Textbox(label="Input", container=True)
            btn_send = gr.Button(value="Send")

        txt_msg = txt.submit(read_pdf.add_text, [chatbot, txt, k, fetch_k, temperature, system], [chatbot, txt, docBox],
                             queue=False).then(read_pdf.bot, chatbot, chatbot)
        # Click the button to send the text
        btn_click = btn_send.click(read_pdf.add_text, [chatbot, txt, k, fetch_k, temperature, system],
                                   [chatbot, txt, docBox], queue=False).then(read_pdf.bot, chatbot, chatbot)

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    btn_click.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    # 上传 pdf，outputs定义了哪些组件会被这个函数的返回值更新
    btn_up.upload(fn=read_pdf.choose_loader, inputs=[btn_up, client_path, chunk_size, chunk_overlap],
                  outputs=show_pdf)

demo.queue()
demo.launch(share=True)
