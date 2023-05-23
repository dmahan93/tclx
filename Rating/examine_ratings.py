import json
import gradio as gr
import numpy as np


if __name__ == '__main__':
    with open("../testing.json") as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: x['tot_score'] if x['helpful'] else -x['tot_score'])

    def next_data(index):
        index = int(index) + 1
        return data[index]['text'], index, data[index]['pos'], data[index]['neg'], data[index]['rev_neg'], data[index]['rev_pos'], data[index]['tot_score'], data[index]['helpful']


    with gr.Blocks() as demo:
        prompt = gr.Textbox(value=data[0]['text'], label="prompt")
        with gr.Row():
            index = gr.Textbox(value="0", label="index")
            next = gr.Button("Next")
        with gr.Row():
            pos = gr.Textbox(value=data[0]['pos'], label="pos")
            neg = gr.Textbox(value=data[0]['neg'], label="neg")
            rev_neg = gr.Textbox(value=data[0]['rev_neg'], label="rev_neg")
            rev_pos = gr.Textbox(value=data[0]['rev_pos'], label="rev_pos")
            tot_score = gr.Textbox(value=data[0]['tot_score'], label="tot_score")
            helpful = gr.Textbox(value=data[0]['helpful'], label="helpful")
        next.click(
            fn=next_data,
            inputs=index,
            outputs=[prompt, index, pos, neg, rev_neg, rev_pos, tot_score, helpful]
        )
    demo.launch()