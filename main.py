import sys, os
sys.path.append('..')
root  = '../root/'

import json
import joblib
import tqdm
import random
from datasets import load_dataset, Dataset
from utils import summarize_trial, convert_to_dict
# from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy
from agent import KnowTrace
import argparse

tqdm.auto.tqdm.enabled = False

def parse_args():
    parser = argparse.ArgumentParser(description="Run KnowTrace with different settings")
    
    parser.add_argument('--loop_num', type=int, default=1, help='Number of loops to run')
    parser.add_argument('--step_num', type=int, default=5, help='Number of inference step')
    parser.add_argument('--test_size', type=int, default=500, help='Size of test set')
    parser.add_argument('--val_size', type=int, default=100, help='Size of val set')
    parser.add_argument('--dataset_path', type=str, default='./data/hotpotqa.json', help='Path to the dataset file')
    parser.add_argument('--base_llm', type=str, default='LLaMA3-8B-Instruct', help='LLaMA3-8B-Instruct or gpt-3.5-turbo-instruct')
    parser.add_argument('--collect_data', type=bool, default=False, help='Collect data for self-taught finetuning')
    parser.add_argument('--exploration_path', type=str, default="./exploration.json")
    parser.add_argument('--completion_path', type=str, default="./completion.json")
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints.json")
    
    return parser.parse_args()


def main(args):
    with open(args.dataset_path, 'r') as fin:
        raw_dataset = json.load(fin)
    random.shuffle(raw_dataset)
    test_dataset = raw_dataset[:args.test_size]
    val_dataset = raw_dataset[args.test_size:args.test_size+args.val_size]
    train_dataset = raw_dataset[args.test_size+args.val_size:]

    test_dataset = Dataset.from_dict(convert_to_dict(test_dataset))
    val_dataset = Dataset.from_dict(convert_to_dict(val_dataset))
    train_dataset = Dataset.from_dict(convert_to_dict(train_dataset))

    if args.collect_data:
        agent_list = [KnowTrace(row['question'], row['answer'], args.base_llm, args.step_num, args.collect_data, args.exploration_path, args.completion_path) for row in train_dataset]
    else:
        agent_list = [KnowTrace(row['question'], row['answer'], args.base_llm, args.step_num, args.collect_data, args.exploration_path, args.completion_path) for row in test_dataset]
    
    trial = 0
    for _ in range(args.loop_num):
        for agent in [a for a in agent_list if not a.is_correct(False)]:
            print(f'------------------------------------------------------------------------')
            print(f'Question: {agent.question}')
            agent.run()
            print(f'Answer: {agent.key}')
            print(f'------------------------------------------------------------------------')
        trial += 1
        correct, incorrect = summarize_trial(agent_list)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
