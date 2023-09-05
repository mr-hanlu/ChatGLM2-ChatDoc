from typing import List, Tuple
import torch
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, LogitsProcessorList, \
    InvalidScoreLogitsProcessor


class ChatGlmDoc(ChatGLMForConditionalGeneration):
    def build_inputs_docs(self, tokenizer, query: str, system: str, docs: list, history: list, ischat: bool):
        def build_doc_prompt(query, system, docs):
            prompt = ""
            prompt += f"系统：{system}\n\n"
            for i, doc in enumerate(docs):
                prompt += f"文档{i + 1}：{doc}\n\n"
            prompt += f"问：{query}\n\n答："
            return prompt

        def build_chat_prompt(query, system, history):
            prompt = ""
            prompt += f"系统：{system}\n\n"
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
            prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            return prompt

        if not ischat:
            prompt = build_doc_prompt(query, system=system, docs=docs)
        else:
            prompt = build_chat_prompt(query, system=system, history=history)
        print("prompt:", prompt)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    @torch.inference_mode()
    def chat_docs(self, tokenizer, query: str, docs: list[str], system: str, history: List[Tuple[str, str]] = None,
                  ischat: bool = False, max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8,
                  logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs_docs(tokenizer, query, system=system, docs=docs, history=history, ischat=ischat)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history
