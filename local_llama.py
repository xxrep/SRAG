from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn, json, datetime
import torch
import os
import argparse
from peft import PeftModel

def postprocess(response, instruction):
    messages = response.split(f"{instruction}")
    if not messages:
        raise ValueError("Invalid template for prompt. The template should include the term 'Response:'")
    return messages[-1]

def preprocess(instruction):
    formatted_prompt = (
                f"<|begin_of_text|>"
                f"### Instruction:\n{instruction}\n### Response:\n"
            )
    instruction = "".join(formatted_prompt)  
    return instruction

def llama_inference(inf_pipeline, instruction, generation_config_dict, temp=True):
    if temp:
        instruction = preprocess(instruction)
    response = postprocess(inf_pipeline(instruction, **generation_config_dict)[0]['generated_text'], instruction)
    return response


app = FastAPI()
# torch.cuda.set_device(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    config = json_post_list.get('config')
    response = model.run(prompt, config)
    # print(config)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer

class LocalLLM:
    def __init__(self, config):
        self.config = config
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model'], 
            device_map='auto',
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(config['model'])
        if os.path.exists(config["reason_path"]) and os.path.exists(config["refine_path"]):
            self.model = PeftModel.from_pretrained(base_model, config["reason_path"], adapter_name="reasoner")
            self.model.load_adapter(config["refine_path"], adapter_name="refiner")
            self.reason_tokenizer = AutoTokenizer.from_pretrained(config["reason_path"])
            self.refine_tokenizer = AutoTokenizer.from_pretrained(config["refine_path"])
            self.flag = True
        else:
            print("Using Base Model...")
            self.model = base_model
            self.flag = False

    def run(self, prompt, generation_config_dict):
        if self.flag:
            self.model.unmerge_adapter()
            print(generation_config_dict)
            role = generation_config_dict["role"]
            if role == "reasoner":
                print("Using Reason Model...")
                self.model.merge_adapter(adapter_names=["reasoner"])
                tokenizer = self.reason_tokenizer
                temp = True
            elif role == "refiner":
                print("Using Refine Model...")
                self.model.merge_adapter(adapter_names=["refiner"])
                tokenizer = self.refine_tokenizer
                temp = True
            else:
                print("Using Base Model...")
                tokenizer = self.base_tokenizer
                temp = False
        else:
            print("Using Base Model...")
            tokenizer = self.base_tokenizer
            temp = False
        inf_pipeline =  pipeline('text-generation', model=self.model, tokenizer=tokenizer, do_sample=config['do_sample'], temperature=config['temperature'], top_p=config['top_p'], max_new_tokens=config['max_new_tokens'])
        generation_config_dict['tokenizer'] = tokenizer
        del generation_config_dict["role"]
        response = llama_inference(inf_pipeline, prompt, generation_config_dict, temp)
        return response

def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval server with different settings")
    
    parser.add_argument('--model_path', type=str, default='./llm_models/Meta-Llama-3-8B-Instruct', help='Path to base LLM')
    parser.add_argument('--reason_checkpoint_path', type=str, default='./checkpoints_packing/reasoner/checkpoint-xxx', help='Path to reason checkpoint')
    parser.add_argument('--refine_checkpoint_path', type=str, default='./checkpoints_packing/refiner/checkpoint-xxx', help='Path to refine checkpoint')
    parser.add_argument('--port', type=int, default=1051)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = {
        'model': args.model_path,
        'reason_path': args.reason_checkpoint_path,
        'refine_path': args.refine_checkpoint_path,
        'max_new_tokens': 128,
        'temperature': 0.01,
        'do_sample': True,
        'top_p': 0.9,
        'stop_strings': ['\n\n'],
    }
    model = LocalLLM(config)
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)
    