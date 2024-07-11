import json
import pandas as pd
from datasets import load_dataset
from openai import OpenAI # type: ignore

DEEPINFRA_API_KEY = "DEADB33F"


openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

PROMPT_START = 'Determine the age and gender of ONLY the person who wrote text below. Do NOT include any other people mentioned in the text. The text is: "'
PROMPT_END = """"
Respond ONLY in JSON format, such as this: {"age": 21, "gender": "M", "reason": "The person mentioned they were 21 years old."}

If the age and gender cannot be determined, respond EXACTLY with {"age": -1, "gender": "UNK", "reason": "The person does not mention their own age."}
Respond ONLY with JSON or you WILL get punished.

Example of edge case:
"I like this girl who is 18F." -> {"age": -1, "gender": "UNK", "reason": "The person does not mention their own age."}
ALWAYS give an exact number. If the person mentions being in their 20s, respond with the exact age, such as 20.
NEVER include any other information in your response. Only include the age and gender of the person who wrote the text, not any other people mentioned in the text.
"""

def get_gender_age(post: str):
    prompt = PROMPT_START + post + PROMPT_END
    chat_completion = openai.chat.completions.create(
        model="microsoft/Phi-3-medium-4k-instruct",
        #model="google/gemma-1.1-7b-it",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.2,
        response_format={"type": "json"}
    )


    resp = chat_completion.choices[0].message.content
    if not resp:
        return {"age": -1, "gender": "UNK", "reason": "-"}
    try:
        resp = json.loads(resp)
        if not "age" in resp or not "gender" in resp:
            return {"age": -1, "gender": "UNK", "reason": "-"}
        if not "reason" in resp:
            resp["reason"] = "-"

        
    except json.JSONDecodeError:
        print(resp)
        print("Error parsing response.. Returning -1, UNK")
        return {"age": -1, "gender": "UNK", "reason": "-"}  
    return resp

from tqdm import tqdm

def label_test_data():
    df = pd.read_csv("./check_age_gender_regex.csv")
    for i, row in tqdm(df.iterrows()):
        post = row["text"]
        
        resp = get_gender_age(post)
        df.loc[i, "age"] = resp["age"]
        df.loc[i, "gender"] = resp["gender"]

    df.to_csv("./check_age_gender_phi3.csv", index=False)

import concurrent.futures
def label_data():
    ds = load_dataset("dim/tldr_17_50k")
    total_len = len(ds["train"])

    num_workers = 32
    chunk_size = total_len // num_workers
    chunk_idxs = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    chunk_idxs[-1] = (chunk_idxs[-1][0], total_len)
    
    def label_chunk(start, end):
        chunk_to_save = {
            "content": [],
            "summary": [],
            "age": [],
            "gender": [],
            "subreddit": []
        }

        for i in tqdm(range(start, end)):
            datum = ds["train"][i]
            post_for_det = datum["normalizedBody"]
            resp = get_gender_age(post_for_det)
            if resp["age"] == -1 or resp["gender"] == "UNK":
                continue
            
            chunk_to_save["content"].append(datum["content"])
            chunk_to_save["summary"].append(datum["summary"])
            chunk_to_save["subreddit"].append(datum["subreddit"])
            chunk_to_save["age"].append(resp["age"])
            chunk_to_save["gender"].append(resp["gender"])
            
            if i % 200 == 0:
                df = pd.DataFrame(chunk_to_save)
                df.to_csv(f"./labelled_tldr_{start}_{end}.csv", index=False)



        df = pd.DataFrame(chunk_to_save)
        df.to_csv(f"./labelled_tldr_{start}_{end}.csv", index=False)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda x: label_chunk(x[0], x[1]), chunk_idxs)
    



if __name__ == "__main__":
    label_data()