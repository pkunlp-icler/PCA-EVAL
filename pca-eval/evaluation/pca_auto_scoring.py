import openai
import argparse
import json
from tqdm import tqdm
import requests
import concurrent.futures
import time

MAX_RETRY=2000
CONCURRENT_CONNECTIONS=2

parser = argparse.ArgumentParser()
parser.add_argument("--meta_data", type=str, default='data/v1.0/Open-World Game/meta_data.json')
parser.add_argument('--model_output', type=str, default='results/chatgpt_holmes_outputs/Open-World Game.json')
parser.add_argument("--openai_key", type=str, default="sk-xxxx")
parser.add_argument("--output_path", type=str, default="results/chatgpt_holmes_PCA_scores/Open-World Game.json")
parser.add_argument("--temperture", type=float, default=0)

args = parser.parse_args()

headers = {
    'Authorization': f'Bearer {args.openai_key}',
    'Content-Type': 'application/json',
}

template = """[Question]: {question}
[Action Choices]: {actions}

[Agent Answer]: {model_output}

[Correct Action]: {true_action}
[Key Concepts]: {key_concept}
[Reference Reasoning Process]: {reason}

[System]
We would like you to acess the agent's performance in the multimodal reasoning task about {domain}.
In this task, the agent is given an image, a [Question] and several candidate [Action Choices], and is asked to give an [Agent Answer] for the [Question].
The [Agent Answer] encapsulates the agent's precption of the image's [Key Concepts], the agent's cognition reasoning process and the final selected action.

We request you to give three types of scores for the agent's [Agent Answer] in comparison to the given [Key Concepts], [Reference Reasoning Process] and [Correct Action]:
1. action score: If the selected action in the [Agent Answer] matches that of the [Correct Action], the action score is 1; otherwise, it is 0.
2. perception score: This score evaluates the model's capability to perceive and interpret observations. It is contingent on whether the [Agent Answer] includes any of the [Key Concepts] of the instance. If it accurately describes any one of the [Key Concepts], the score is 1; otherwise, it is 0.
3. cognition score: This score gauges the model's ability to reason, comprehend, and make informed decisions based on perceived input data and world knowledge. If the reasoning process in the [Agent Answer] aligns with the [Reference Reasoning Process], the score is 1; otherwise, it is 0.
Please note that there are only scores of 0 and 1.

You should carefully compare the [Agent Answer] with the [Correct Action], [Key Concepts] and [Reference Reasoning Process] to give the your accessment.
You need first give your accessment evidences and then the scores. 

Your output MUST contains 6 lines with the following format:
action accessment evidence: <accessment evidences here>
action score: <score here>
perception accessment evidence: <accessment evidences here>
perception score: <score here>
cognition accessment evidence: <accessment evidences here>
cognition score: <score here>"""


def parse_scores(generated_text):
    generated_text = generated_text.split('\n')
    generated_text = [x for x in generated_text if x != '']
    action_score = float(generated_text[1].split(':')[-1])
    perception_score = float(generated_text[3].split(':')[-1])
    cognition_score = float(generated_text[5].split(':')[-1])
    return action_score, perception_score, cognition_score


def eval_entry(entry):
    meta_data = entry['meta_data']
    model_output = entry['model_output']
    domain = meta_data['domain']
    question = meta_data["question"]
    action_list = [f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(meta_data["actions"])]
    actions =  " ".join(action_list)
    true_action = action_list[meta_data['answer_index']]
    reason = meta_data["reason"]
    key_concept = meta_data["key_concept"]
    prompt = template.format(question=question, 
                             actions=actions, 
                             domain=domain, 
                             model_output=model_output, 
                             true_action=true_action, 
                             key_concept=key_concept, 
                             reason=reason)
    
    for i in range(MAX_RETRY):
        try:
            request = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": args.temperture,
            }
            
            while True:
                try:
                    response = requests.post('https://api.openai.com/v1/chat/completions',
                                    headers=headers,
                                    data=json.dumps(request),
                                    timeout=20)
                    response = json.loads(response.text)
                    review = response['choices'][0]['message']['content']
                    action_score, perception_score, cognition_score = parse_scores(review)
                    break 
                except Exception as e:
                    print(e)
                    request["temperature"] = 0.7
            time.sleep(1)
            return {"review": review,
                    "action_score": action_score,
                    "perception_score": perception_score,
                    "cognition_score": cognition_score,
                    "prompt": prompt
            }
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(20)
        raise RuntimeError(f"Failed after {MAX_RETRY} retries.")
    

if __name__ == '__main__':
    model_outputs = json.load(open(args.model_output, 'r'))
    meta_datas = json.load(open(args.meta_data, 'r'))
    data = []
    # assert len(model_outputs) == len(meta_datas)
    
    for meta_data, model_output in zip(meta_datas, model_outputs):
        assert meta_data['index'] == model_output['index']
        data.append({
            "question_id": model_output['index'],
            "meta_data": meta_data,
            "model_output": model_output['model_output']
        })
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_CONNECTIONS) as executor:
        future2entry = {executor.submit(eval_entry, entry): entry for entry in data}
        for future in tqdm(concurrent.futures.as_completed(future2entry), total=len(data)):
            entry = future2entry[future]
            result = future.result()
            entry.update(result) 
            del entry['meta_data']
            print(entry) 
    
    average_action_score = sum([d['action_score'] for d in data]) / len(data)
    average_perception_score = sum([d['perception_score'] for d in data]) / len(data)
    average_cognition_score = sum([d['cognition_score'] for d in data]) / len(data)
    
    data = [{"average_action_score": average_action_score, 
            "average_perception_score": average_perception_score,
            "average_cognition_score": average_cognition_score
            } ] + data
    json.dump(obj=data, fp=open(args.output_path, 'w'), indent=4)
