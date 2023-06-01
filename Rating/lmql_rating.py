import lmql
import asyncio
from datasets import load_dataset
import tqdm
import json


@lmql.query
async def rating(passage, option_a, option_b, initial_instruct_passage):
    '''
    argmax
        """### Human: Here is a passage, question, answer, and critique, afterwards you will be asked on a question based on this entire passage:
        {passage}

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

        Choice A: {option_a}
        Choice B: {option_b}
        ### Assistant: {initial_instruct_passage}[CLASSIFICATION]"""
    from
        "CarperAI/stable-vicuna-13b-fp16"
    distribution
        CLASSIFICATION in [" A", " B"]
    '''


@lmql.query
async def rev_rating(passage, option_a, option_b, initial_instruct_passage):
    '''
    argmax
        """### Human: Here is a passage, question, answer, and critique, afterwards you will be asked on a question based on this entire passage:
        {passage}

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

        Choice A: {option_a}
        Choice B: {option_b}
        ### Assistant: {initial_instruct_passage}[CLASSIFICATION]"""
    from
        "CarperAI/stable-vicuna-13b-fp16"
    distribution
        CLASSIFICATION in [" A", " B"]
    '''


async def combine_ratings(passage: str):
    option_a = "true"
    option_b = "false"
    initial_instruct = f"Between choice A: {option_a} and choice B: {option_b} as a professional and harsh critique rater based on the  pick choice"
    reversed_initial_instruct = f"Between choice A: {option_b} and choice B: {option_a} as a professional and harsh critique rater I'm forced to pick choice"
    pos_rate = rating(passage, option_a, option_b, initial_instruct)
    neg_rate = rating(passage, option_b, option_a, initial_instruct)
    rev_pos_rate = rev_rating(passage, option_a, option_b, reversed_initial_instruct)
    rev_neg_rate = rev_rating(passage, option_b, option_a, reversed_initial_instruct)
    pos_rate = (await pos_rate).variables['log P(CLASSIFICATION)']
    neg_rate = (await neg_rate).variables['log P(CLASSIFICATION)']
    rev_pos_rate = (await rev_pos_rate).variables['log P(CLASSIFICATION)']
    rev_neg_rate = (await rev_neg_rate).variables['log P(CLASSIFICATION)']
    return passage, (
            (pos_rate[0][0] - pos_rate[0][1]) +
            (neg_rate[0][1] - neg_rate[0][0]) +
            (rev_pos_rate[0][1] - rev_pos_rate[0][0]) +
            (rev_neg_rate[0][0] - rev_neg_rate[0][1])
    ), (pos_rate[0][0] - pos_rate[0][1]), (neg_rate[0][1] - neg_rate[0][0]), (rev_pos_rate[0][1] - rev_pos_rate[0][0]), (rev_neg_rate[0][0] - rev_neg_rate[0][1])

async def main():
    dataset = load_dataset("dmayhem93/self-critiquing-helpful-rate")
    return_values = list()
    for i, item in tqdm.tqdm(enumerate(dataset['train'])):
        return_values.append(combine_ratings(item['prompt']))
    return_values = await asyncio.gather(*return_values)
    with open('ratings.json', 'w') as f:
        json.dump(return_values, f, indent=4)

if __name__ == '__main__':
    asyncio.run(main())