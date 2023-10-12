import openai
import argparse
import json
from tqdm import tqdm
import requests
import concurrent.futures
import time

MAX_RETRY=2000
CONCURRENT_CONNECTIONS=4

parser = argparse.ArgumentParser()
parser.add_argument("--meta_data", type=str, default='v10_97_data_with_output.json')
parser.add_argument('--model_output', type=str, default='chatgpt_Autonomous_Driving_output.json')
parser.add_argument("--openai_key", type=str, default="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")
parser.add_argument("--output_path", type=str, default="output.json")
args = parser.parse_args()

headers = {
    'Authorization': f'Bearer {args.openai_key}',
    'Content-Type': 'application/json',
}

start_template = """I will give you some standards now.
action score: If model_output and true_answer are the same, the score is 1, otherwise it is 0.
perception score: measures the model's ability to accurately perceive and interpret the observation. It is computed based on whether the agent's output image description includes the key concept of the instance. If accurately describes the key concept the score is 1, otherwise is 0.
cognition score: assesses the model's ability to reason, comprehend, and make informed decisions based on the perceived input data and world knowledge. If the reasoning process is correct the score is 1 , otherwise is 0.
Please note that there are only scores of 0 and 1, if not completely wrong then the score is 1

Now you are given an example:

Standard answer:
Question: You are a driving assistant. What should you do if you want to drive forward safely?
Choices: A. Reduce speed B. Speed up C. Turn around
object_detection: \{car\:0, human\:0, bicycle\:0\}
caption: curvy mountain road
weather: clear
ocr: None
reason: Based on the image, The car is traveling on a very curvy mountain road. So you need to slow down
Key concept: curvy mountain road
Answer: A. Reduce speed


Answers to be evaluated:
    Based on the information from the APIs, here is the analysis:
- There are no detected traffic signs in the image (\"detect_traffic_sign() = ['None']\").
- There are no detected cars, people, or bicycles in the image (\"object_detection() = {'car': 0, 'people': 0, 'bicycle': 0}\").\n- The weather is clear (\"weather_detection() = clear\").
Given that there are no traffic signs, no obstacles detected, and clear weather, it is safe to drive forward. Therefore, the recommended action would be to maintain the current speed or speed up if necessary (\"B. Speed up\").
Please note that without directly viewing the image, the analysis is based solely on the information provided by the APIs. It is important to prioritize safety while driving and consider other contextual factors that may not be captured in the provided information.

action description: since the correct answer is A. Reduce speed and assistant give the incorrect answer B. Speed up. So the action score is 0.
action score: 0 
perception description: since assistant didn't use caption() API to understand the car is on a curvey mountain road, so the perception score is 0.
perception score: 0
cognition description: since assistant didn't give correct reasoning for the correct answer. So the cognition score is 0.
cognition score: 0

Now you are provided a Standard answer and Answers to be evaluated. You need to provide the description and scores for Answers to be evaluated.

Standard answer:

"""


def parse_scores(generated_text):
    generated_text = generated_text.split('\n')
    try:
        action_score = float(generated_text[1].split(':')[-1])
        perception_score = float(generated_text[3].split(':')[-1])
        cognition_score = float(generated_text[5].split(':')[-1])
    except:
        action_score = -1
        perception_score = -1
        cognition_score = -1
    return action_score, perception_score, cognition_score


def eval_entry(entry):
    global start_template
    meta_data = entry['meta_data']
    model_output = entry['model_output']
    
    
    questions = meta_data["questions"]
    caption = meta_data["caption"]
    weather_detection = meta_data["weather_detection"]
    ocr = meta_data["ocr"]
    choices = meta_data["choices"]
    true_answer = meta_data["answer"]
    reason = meta_data["reason"]
    KeyConcept = meta_data["KeyConcept"]
    object_detection = meta_data["object_detection"]
    evaluate_answer = meta_data["evaluate_answer"]
        
    for i in range(MAX_RETRY):
        try:
            start_template += f"Question: {questions}\nChoices: {choices}\n"
            start_template += f"object_detection: {object_detection}\n"
            start_template += f"caption: {caption}\nweather: {weather_detection}\n"
            start_template += f"ocr: {ocr}\n"
            start_template += f"reason: {reason}\n"
            start_template += f"Key concept: {KeyConcept}\n"
            
            
            start_template += f"Answers to be evaluated:\n\nassistent: {evaluate_answer}\n\n"
            start_template += f"true_answer:{true_answer}\n\n"
            start_template += f"model_output: {model_output}\n\n"
            
            start_template += """Do not give the answers for the first one, Your output MUST contains 6 lines with the following format:
action description: <description>
action score: <score>
perception description: <description>
perception score: <score>
cognition description: <description>
cognition score: <score>"""
            request = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": start_template
                    }
                ]
            }
            
            while True:
                try:
                    response = requests.post('https://api.openai.com/v1/chat/completions',
                                    headers=headers,
                                    data=json.dumps(request))
                    response = json.loads(response.text)
                    break
                except Exception as e:
                    print(e)
                    continue
            
            if "choices" not in response:
                action_score = 0
                perception_score = 0
                cognition_score = 0
                review = "None"
            else: 
                review = response['choices'][0]['message']['content']
                action_score, perception_score, cognition_score = parse_scores(review)
                
            return {"review": review,
                    "action_score": action_score,
                    "perception_score": perception_score,
                    "cognition_score": cognition_score,
            }
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(2)
        raise RuntimeError(f"Failed after {MAX_RETRY} retries.")
    

if __name__ == '__main__':
    model_outputs = json.load(open(args.model_output, 'r'))[:5]
    meta_datas = json.load(open(args.meta_data, 'r'))[:5]
    data = []
    assert len(model_outputs) == len(meta_datas)
    
    for meta_data, model_output in zip(meta_datas, model_outputs):
        # assert meta_data['idx'] == model_outputs['idx']
        data.append({
            "question_id": model_output['idx'],
            "meta_data": meta_data,
            "model_output": model_output['model_output']
        })
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_CONNECTIONS) as executor:
        future2entry = {executor.submit(eval_entry, entry): entry for entry in data}
        for future in tqdm(concurrent.futures.as_completed(future2entry), total=len(data)):
             entry = future2entry[future]
             result = future.result()
             entry.update(result)  
    
    for d in data:
        del d['meta_data']
    
    average_action_score = sum([d['action_score'] for d in data]) / len(data)
    average_perception_score = sum([d['perception_score'] for d in data]) / len(data)
    average_cognition_score = sum([d['cognition_score'] for d in data]) / len(data)
    
    data = [{"average_action_score": average_action_score, 
             "average_perception_score": average_perception_score,
             "average_cognition_score": average_cognition_score
            } ] + data
    json.dump(obj=data, fp=open(args.output_path, 'w'), indent=4)