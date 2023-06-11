import openai
import datasets
from system_messages import get_system_prompt_for_flan2021, \
    get_system_prompt_for_niv2, \
    get_system_prompt_for_cot, \
    get_system_prompt_for_t0
import json
import tqdm


def get_continuation_chatgpt(system: str, user: str) -> str:
    messages = list()
    if system != "":
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    ).choices[0].message.content


if __name__ == '__main__':
    cot = iter(datasets.load_dataset("conceptofmind/cot_submix_original", split="train", streaming=True))
    cot_outputs = list()
    stream = tqdm.tqdm(cot, total=15)
    for data in stream:
        if data['template_type'] != 'zs_opt':
            continue
        question = data['inputs']
        system_prompt = get_system_prompt_for_cot()
        cot_outputs.append({
            "question": question,
            "system_prompt": system_prompt,
            "answer": get_continuation_chatgpt(system_prompt, question),
            "real": data['targets']
        })
        stream.update(len(cot_outputs))
        if len(cot_outputs) >= 15:
            break
    niv = iter(datasets.load_dataset("conceptofmind/niv2_submix_original", split="train", streaming=True))
    niv_outputs = list()
    stream = tqdm.tqdm(niv, total=44)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        question = data['inputs']
        system_prompt = get_system_prompt_for_niv2()
        niv_outputs.append({
            "question": question,
            "system_prompt": system_prompt,
            "answer": get_continuation_chatgpt(system_prompt, question),
            "real": data['targets']
        })
        stream.update(len(niv_outputs))
        if len(niv_outputs) >= 44:
            break
    with open("cot_outputs.json", "w") as f:
        json.dump(cot_outputs, f)
    with open("niv_outputs.json", "w") as f:
        json.dump(niv_outputs, f)
    flan = iter(datasets.load_dataset("conceptofmind/flan2021_submix_original", split="train", streaming=True))
    flan_outputs = list()
    while len(flan_outputs) < 250:
        data = next(flan)
        if "zs" not in data['template_type']:
            continue
        question = data['inputs']
        # Need to figure out multiple choice
        system_prompt = get_system_prompt_for_flan2021(False)
        flan_outputs.append({
            "question": question,
            "system_prompt": system_prompt,
            "answer": get_continuation_chatgpt(system_prompt, question),
            "real": data['targets']
        })
    t0 = iter(datasets.load_dataset("conceptofmind/t0_submix_original", split="train", streaming=True))
    t0_outputs = list()
    while len(t0_outputs) < 200:
        data = next(t0)
        if "zs" not in data['template_type']:
            continue
        question = data['inputs']
        system_prompt = get_system_prompt_for_t0()
        t0_outputs.append({
            "question": question,
            "system_prompt": system_prompt,
            "answer": get_continuation_chatgpt(system_prompt, question),
            "real": data['targets']
        })