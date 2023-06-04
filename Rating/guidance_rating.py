import lmql
from datasets import load_dataset
import tqdm
import json
from transformers import AutoTokenizer
import guidance
import argparse
import torch

@torch.no_grad()
def combine_ratings(passage: str):
    rating = guidance('''### Human: Here is a passage, question, answer, and critique, afterwards you will be asked on a question based on this entire passage:

{{passage}}

Question: Please select between one of the below two choices based on if the critique is helpful to correct the answer.
Judge as a professional and harsh critique rater.
Trivial critiques should be false.
Critiques that are already in the answer should be false.
Critiques that add new information to the answer should be true.
Critiques that restate information in the answer should be false.
Critiques that are not supported by anything in the answer should be false.
Critiques that are not supported by anything in the question should be false.
Critiques that are gibberish should be false.
Answers that are No answer with a critique should be true.
Answers with information not included in the article and called out in the critique should be false.
Questions that are unsupported by anything within the text and called out in the critique should be true.

Choice A: {{option_a}}
Choice B: {{option_b}}
### Assistant: {{initial_instruct_passage}}{{#select "answer" logprobs='logprobs'}} A{{or}} B{{/select}}''')

    rev_rating = guidance('''### Human: Here is a passage, question, answer, and critique, afterwards you will be asked on a question based on this entire passage:

{{passage}}

Question: Please select between one of the below two choices based on if the critique is harmful to correct the answer.
Judge as a professional and harsh critique rater.
Trivial critiques should be true.
Critiques that are already in the answer should be true.
Critiques that add new information to the answer should be false.
Critiques that restate information in the answer should be true.
Critiques that are not supported by anything in the answer should be true.
Critiques that are not supported by anything in the question should be true.
Critiques that are gibberish should be true.
Answers that are No answer with a critique should be false.
Answers with information not included in the article and called out in the critique should be true.
Questions that are unsupported by anything within the text and called out in the critique should be false.

Choice A: {{option_a}}
Choice B: {{option_b}}
### Assistant: {{initial_instruct_passage}}{{#select "answer" logprobs='logprobs'}} A{{or}} B{{/select}}''')
    option_a = "true"
    option_b = "false"
    initial_instruct = f"Between choice A: {option_a} and choice B: {option_b} as a professional and harsh critique rater based on the  pick choice"
    reversed_initial_instruct = f"Between choice A: {option_b} and choice B: {option_a} as a professional and harsh critique rater I'm forced to pick choice"
    pos_rate = rating(passage=passage, option_a=option_a, option_b=option_b, initial_instruct_passage=initial_instruct)
    neg_rate = rating(passage=passage, option_a=option_b, option_b=option_a, initial_instruct_passage=initial_instruct)
    rev_pos_rate = rev_rating(passage=passage, option_a=option_a, option_b=option_b, initial_instruct_passage=reversed_initial_instruct)
    rev_neg_rate = rev_rating(passage=passage, option_a=option_b, option_b=option_a, initial_instruct_passage=reversed_initial_instruct)
    pos_rate = [pos_rate['logprobs'][' A'], pos_rate['logprobs'][' B']]
    neg_rate = [neg_rate['logprobs'][' A'], neg_rate['logprobs'][' B']]
    rev_pos_rate = [rev_pos_rate['logprobs'][' A'], rev_pos_rate['logprobs'][' B']]
    rev_neg_rate = [rev_neg_rate['logprobs'][' A'], rev_neg_rate['logprobs'][' B']]
    return passage, (
            (pos_rate[0] - pos_rate[1]) +
            (neg_rate[1] - neg_rate[0]) +
            (rev_pos_rate[1] - rev_pos_rate[0]) +
            (rev_neg_rate[0] - rev_neg_rate[1])
    ), (pos_rate[0] - pos_rate[1]), (neg_rate[1] - neg_rate[0]), (rev_pos_rate[1] - rev_pos_rate[0]), (rev_neg_rate[0] - rev_neg_rate[1])

def main():
    parser = argparse.ArgumentParser(description="test some prompt rating")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_devices", type=int, default=1)
    args = parser.parse_args()
    # guidance.llm = guidance.llms.Transformers("CarperAI/stable-vicuna-13b-fp16", device=args.device, torch_dtype=torch.float16)
    dataset_without_tokens = load_dataset("dmayhem93/self-critiquing-helpful-rate")
    tokenizer = AutoTokenizer.from_pretrained("CarperAI/stable-vicuna-13b-fp16")
    def tokenization(example):
        test_data = tokenizer(example["prompt"])
        return test_data
    dataset = dataset_without_tokens.map(tokenization, batched=True)
    return_values = list()
    returns = list()
    for i, item in tqdm.tqdm(enumerate(dataset['train'])):
        if i % args.num_devices != args.device:
            continue
        if len(item['input_ids']) < 1800:
            return_values.append(combine_ratings(item['prompt']))
    with open('ratings.json', 'w') as f:
        json.dump(return_values, f, indent=4)

if __name__ == '__main__':
    main()