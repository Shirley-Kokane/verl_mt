from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )

def calculate_bleu(completions):
    self_bleu_scores = [0]*len(completions)
    for i in range(len(completions)):
        references = completions[:i] + completions[i+1:]
        score = sentence_bleu(references, completions[i])
        self_bleu_scores[i] = score
    
    return 1 - np.mean(self_bleu_scores).item()

def calculate_diversity(completions, cache_score):
    prompt = "You are an LLM Judge that is given a list of completions and needs to judge the diversity of the completions. Please provide a score for each completion between 0 to 5 in the following format: <score>3</score>\n\n" 
    prompt += "The completions are from the same prompt and are generated by the same model. The completions are in the following format: \n\n"
    for i, completion in enumerate(completions):
        prompt += f"Completion {i+1}: {completion}\n\n"
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0
            )
            response = response.choices[0].message.content
            scores = response.split("\n")
            scores = [int(score.split("<score>")[1].split("</score>")[0]) for score in scores]
            return np.mean(scores).item()/5
        except:
            continue
    return cache_score