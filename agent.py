import re, string, os
from typing import List, Union, Literal
from enum import Enum
from prompts import init_prompt_template, reason_prompt_template, refine_prompt_template, direct_reason_prompt_template
from exemplars import INIT_EXAMPLE, REASON_EXAMPLE, REFINE_EXAMPLE, DIRECT_EXAMPLE
import tqdm
import numpy as np
import openai
from langchain_core._api.deprecation import LangChainDeprecationWarning
import warnings
import requests
import json
import subprocess
from retriever.search import Retriever
import networkx as nx

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

tqdm.auto.tqdm.enabled = False

# bm25_retriever = Retriever()

class KnowTrace:
    def __init__(self,
                 question: str,
                 key: str,
                 base_llm: str = "LLaMA3-8B-Instruct",
                 max_steps: int = 6,
                 collect_data: bool = False,
                 exp_path: str = "./exploration.json",
                 com_path: str = "./completion.json",
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.collect_data = collect_data
        self.exp_path = exp_path
        self.com_path = com_path
        self.base_llm = base_llm

        self.reason_llama = LocalLLM(1051, {'role': 'reasoner', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]})
        self.refine_llama = LocalLLM(1051, {'role': 'refiner', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]})
        self.init_llama = LocalLLM(1051, {'role': 'base', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]})
        self.direct_llama = LocalLLM(1051, {'role': 'base', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]})

        self.bm25_retriever = Retriever(8001)
        self.total_s = 1

    def run(self) -> None:
        self.__reset_agent()
        while not self.is_finished():
            self.step()
    
    def step(self) -> None:
        if self.step_n == 1:
            print("KG Triplets: None")
            entities_with_gd = self.init()
            self.step_n += 1
        elif self.truncation or self.step_n == self.max_steps:
            if self.kg_paths == '':
                self.kg_paths = "None"
            print(f"KG Triplets:\n{self.kg_paths} |||")
            print("Truncating...")
            direct_out = self.direct()
            direct_out = remove_blank_lines(direct_out)
            print('-----------------------------')
            thought, self.answer = self.parse_thought_answer(direct_out)
            print(f"Thought: {thought}")
            print(f"Answer: {self.answer}")
            if self.is_correct():
                print('Answer is CORRECT')
                if self.collect_data:
                    self.sample_correct(thought, trunc=True)
            else: 
                print('Answer is INCORRECT')
            self.finished = True
            self.step_n += 1
            self.total_s += 1
            return
        else:
            print(f"KG Triplets:\n{self.kg_paths} |||")
            reason_out = self.reason()
            reason_out = remove_blank_lines(reason_out)
            print('*****************************')
            can_answer, thought, answer, what = self.parse_reason(reason_out)
            print(f"Whether the given knowledge triples are sufficient: {can_answer}")
            if can_answer.lower() == 'yes':
                thought = thought
                self.answer = answer
                print(f"Thought: {thought}")
                print(f"Answer: {answer}")
                if self.is_correct():
                    print('Answer is CORRECT')
                    if self.collect_data:
                        self.sample_correct(thought)
                else: 
                    print('Answer is INCORRECT')
                self.finished = True
                self.step_n += 1
                self.total_s += 1
                return
            elif can_answer.lower() == 'no':
                entities_with_gd = what
                self.step_n += 1
                if "none" in what.lower():
                    self.truncation = True
                    return
            else:
                self.truncation = True
                print('REASON ERROR')
                return
        entities_with_gd = self.process_gd(entities_with_gd)
        print(f"What to retrieve in next step:\n{entities_with_gd}")
        entities_to_search, gd_index_list = self.extract_entities_from_gd(entities_with_gd)
        entities_to_search, entities_with_gd_list = self.filter_searched(entities_to_search, entities_with_gd, gd_index_list)
        if len(entities_to_search) == 0:
            self.truncation = True
            return
        refine_out_set = set()
        for i, entity in enumerate(entities_to_search):
            egd = entities_with_gd_list[i]
            searched_docs = self.search_entities([egd])
            refine_out = self.refine(searched_docs, egd)
            refine_out = remove_blank_lines(refine_out)
            refine_out = refine_out.strip().split('\n')[0]
            refine_out = self.filter_invalid(refine_out, refine_out_set)
            if len(refine_out) > 0:
                refine_out_set.add(refine_out)
            self.construct_kg(refine_out, egd, searched_docs)
        refine_out = "\n".join(list(refine_out_set))
        if len(refine_out) == 0:
            self.truncation = True
            return
        else:
            print(f"Result: {refine_out}")
        self.kg_paths = self.extend_triples(refine_out, list(set(entities_to_search)))
    
    def sample_correct(self, thought, trunc=False):
        inits = self.match_init_entities()
        # print(inits)
        targets = self.match_target_entities(thought)
        # print(targets)
        # print(self.kg.edges())
        traced_subgraph = self.trace_evidence(inits, targets)
        # print(traced_subgraph.edges())
        self.from_kg_to_data(traced_subgraph)
        self.append_final_reason(thought, trunc)

    def filter_invalid(self, refine_out, refine_out_set):
        triple_list = refine_out.strip().split("\n")
        filtered_triples = []
        for triple in triple_list:
            if "none" in triple.lower() or "n/a" in triple.lower() or "no triples" in triple.lower() or "unknown" in triple.lower() or triple in filtered_triples or triple in refine_out_set:
                continue
            elif len(triple.strip("();").strip().split("; ")) != 3:
                continue
            else:
                filtered_triples.append(triple)
        return "\n".join(filtered_triples)
    
    def match_target_entities(self, thought):
        full_string = thought + " " + self.key
        entities = self.kg.nodes()
        entities = sorted(entities, key=len, reverse=True)
        targets = set()
        for entity in entities:
            if entity in full_string or self.key in entity:
                targets.add(entity)
                full_string = full_string.replace(entity, "")
            else:
                continue
        return list(targets)

    def match_init_entities(self):
        inits = set()
        init_gd = []
        for u, v, d in self.kg.edges(data=True):
            if d["memory"] == "None":
                init_gd.append(d["gd"])
        init_gd = "\n".join(init_gd)
        entities = self.kg.nodes()
        print(entities)
        print(init_gd)
        for entity in entities:
            if entity in init_gd:
                inits.add(entity)
        return list(inits)
    
    def trace_evidence(self, inits, targets):
        # Initialize an empty set to store the nodes and edges in the subgraph
        subgraph_nodes = set()
        subgraph_edges = set()

        undirected_kg = self.kg.to_undirected()

        # Use BFS to find all paths from each start node to the answer node
        for start_node in inits:
            for end_node in targets:
                if not nx.has_path(undirected_kg, source=start_node, target=end_node):
                    continue
                for path in nx.all_shortest_paths(undirected_kg, source=start_node, target=end_node):
                    # print(path)
                    subgraph_nodes.update(path)
                    # Collect edges along the path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        # Get the edge data (relation)
                        relation = self.kg.get_edge_data(u, v)
                        if relation is None:
                            u, v = path[i + 1], path[i]
                            relation = self.kg.get_edge_data(u, v)
                        for key in relation:
                            subgraph_edges.add((u, v, relation[key]['relation'], relation[key]['passage'], relation[key]['memory'], relation[key]['gd']))
        
        # Create the subgraph
        subgraph = nx.MultiDiGraph()
        subgraph.add_nodes_from(subgraph_nodes)
        subgraph.add_edges_from([(u, v, {"relation": relation, "passage": passage, "memory": memory, "gd": gd}) for u, v, relation, passage, memory, gd in subgraph_edges])
        
        return subgraph

    def find_path(self, graph, source, target):
        try:
            return nx.shortest_path(graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return []
            
    def construct_kg(self, triple_str, triple_gd, passage):
        triple_list = triple_str.strip().split("\n")
        edges = []
        for triple in triple_list:
            triple = triple.strip("();")
            triple_ele = triple.strip().split("; ")
            if len(triple_ele) == 3:
                head, relation, tail = triple_ele
                edges.append((head, tail, {"relation": relation, "passage": passage, "memory": self.kg_paths if len(self.kg_paths) != 0 else "None", "gd": triple_gd}))
            else:
                continue
        self.kg.add_edges_from(edges)
    
    def from_kg_to_data(self, traced_subgraph):
        reason_data = {}
        refine_data = {}
        for u, v, d in traced_subgraph.edges(data=True):
            relation = d["relation"]
            passage = d["passage"]
            memory = d["memory"]
            gd = d["gd"]
            if memory in reason_data.keys():
                reason_data[memory].add(gd)
            else:
                reason_data[memory] = set()
                reason_data[memory].add(gd)
            if gd in refine_data.keys():
                refine_data[gd]["triple"].add(f"({u}; {relation}; {v})")
            else:
                refine_data[gd] = {}
                refine_data[gd]["passage"] = passage
                refine_data[gd]["triple"] = set()
                refine_data[gd]["triple"].add(f"({u}; {relation}; {v})")
            # for reasoner, instruction: question, memory, response: gd
            # for refiner, instruction: gd, contexts, response: triples
        for memory in reason_data.keys():
            data_point = {}
            if memory.lower() == "none":
                data_point["type"] = "init"
            else:
                data_point["type"] = "process"
            data_point["question"] = self.question
            data_point["memory"] = memory
            data_point["response"] = "\n".join(reason_data[memory])
            save_data(data_point, self.exp_path)
        for gd in refine_data.keys():
            data_point = {}
            data_point["type"] = "refine"
            data_point["gd"] = gd
            data_point["passage"] = refine_data[gd]["passage"]
            data_point["response"] = "\n".join(refine_data[gd]["triple"])
            save_data(data_point, self.com_path)
    
    def append_final_reason(self, thought, trunc):
        if trunc:
            response = "Thought: " + thought.strip() + "\nAnswer: " + self.key.strip()
            data_point = {"type": "trunc", "question": self.question, "memory": self.kg_paths if len(self.kg_paths) != 0 else "None", "response": response}
        else:
            response = "Whether the given knowledge triples are sufficient for answering: Yes" + "\nThought: " + thought.strip() + "\nAnswer: " + self.key.strip()
            data_point = {"type": "final", "question": self.question, "memory": self.kg_paths if len(self.kg_paths) != 0 else "None", "response": response}
        save_data(data_point, self.exp_path)
    
    def process_gd(self, entities_with_gd):
        sentences = entities_with_gd.split('- ')
        formatted_sentences = [f"- {sentence.strip()}" for sentence in sentences if sentence.strip()]
        return "\n".join(formatted_sentences)
    
    def parse_thought_answer(self, direct_out):
        lines = direct_out.split('\n')
        thought = "None"
        answer = "None"
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:") and thought=="None":
                parse_line = line.strip().split("Thought: ")
                if len(parse_line) > 1:
                    thought = parse_line[1]
            elif line.startswith("Answer:") and answer=="None":
                parse_line = line.strip().split("Answer: ")
                if len(parse_line)>1:
                    answer = parse_line[1]
        return thought, answer
    
    def parse_reason(self, reason_out):
        lines = reason_out.split('\n')
        whether = "None"
        thought = "None"
        answer = "None"
        what = "None"
        for index, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Whether") and whether=="None":
                words = line.lower().strip().split()
                if "no" in words:
                    whether = "No"
                elif "yes" in words:
                    whether = "Yes"
            elif whether=="Yes" and line.startswith("Thought") and thought=="None":
                parse_line = line.strip().split("Thought: ")
                if len(parse_line)>1:
                    thought = parse_line[1]
            elif whether=="Yes" and line.startswith("Answer") and answer=="None":
                parse_line = line.strip().split("Answer: ")
                if len(parse_line)>1:
                    answer = parse_line[1]
            elif whether=="No" and line.startswith("Retrieval"):
                for i in range(index+1, len(lines)):
                    line_i = lines[i].strip()
                    if not line_i.startswith('- '):
                        break
                    else:
                        if what == "None":
                            what = line_i
                        else:
                            what += "\n"+line_i
                break
        return whether, thought, answer, what
    
    def filter_searched(self, entities_to_search, entities_with_gd, gd_index_list):
        ### 当前改为无效果
        gd_lines = entities_with_gd.strip().split("\n")
        entity_list = []
        gd_list = []
        # for i, entity in enumerate(entities_to_search):
        for i, gd_index in enumerate(gd_index_list):
            gd = gd_lines[gd_index]
            if gd in self.filtered:
                continue
            else:
                entity_list.append(entities_to_search[i])
                # gd_list.append(gd_lines[gd_index_list[i]])
                gd_list.append(gd)
                # gd_list.append(gd_lines[gd_index_list[i]].split(': ')[-1])
                # self.filtered.append(entity)
        # self.filtered += entity_list
        # return entity_list, '\n'.join(gd_list)
        return entity_list, gd_list
    
    def extend_triples(self, new_triples_str, anchor_entities):
        new_triples_list = new_triples_str.strip().split("\n")
        attach = []
        for triple in new_triples_list:
            if triple not in self.kg_paths:
                attach.append(triple)
        if len(attach) > 0:
            attach = "\n".join(attach)
            if len(self.kg_paths) > 0:
                new_path = self.kg_paths + "\n" + attach
            else:
                new_path = attach
        else:
            new_path = self.kg_paths
            self.truncation = True
        return new_path

    def extend_paths(self, new_triples_str, anchor_entities):
        new_triples_list = new_triples_str.strip().split("; ")
        used_triples = []
        path_list = self.kg_paths.strip().split("\n")
        entity2triples = {}
        entity2path = {}
        for entity in anchor_entities:
            if entity not in entity2triples.keys():
                entity2triples[entity] = []
            for triple in new_triples_list:
                if triple not in self.kg_paths and triple not in used_triples and entity in triple:
                    entity2triples[entity].append(triple)
                    used_triples.append(triple)
            if entity not in entity2path.keys():
                entity2path[entity] = []
            for path_id, path in enumerate(path_list):
                if entity in path:
                    entity2path[entity].append(path_id)
        for entity in anchor_entities:
            if len(entity2triples[entity]) == 0:
                continue
            triples_str = "; ".join(entity2triples[entity])
            if len(entity2path[entity]) == 0:
                path_list.append("- " + triples_str)
            else:
                for p_id in entity2path[entity]:
                    path_list[p_id] += "; " + triples_str
        total_new = []
        for triple in new_triples_list:
            if triple not in used_triples and triple not in self.kg_paths:
                total_new.append(triple)
        if len(total_new) != 0:
            triples_str = "; ".join(total_new)
            path_list.append("- " + triples_str)
        return "\n".join(path_list).strip()

    def construct_candidates(self, candidate_entities, entities_to_search, triple_entities):
        new_candidate = [item for item in triple_entities if item not in entities_to_search]
        return "; ".join(new_candidate)

    def extract_entities_from_gd(self, entities_with_gd):
        pattern = r'- (.*?):'
        lines = entities_with_gd.strip().split('\n')
        entity_names_str_list = []
        gd_index_list = []
        for index, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                entity_name = match.group(1)
                gd_index_list.append(index)
                entity_names_str_list.append(entity_name)
        print(entity_names_str_list)
        return entity_names_str_list, gd_index_list
    
    def search_entities(self, entities_to_search):
        entity_docs = []
        retriever_outs = self.bm25_retriever.retrieve(entities_to_search) # list
        for out in retriever_outs: # dict
            contexts = out['contexts'] # list
            for context in contexts: # dict
                title = context['title']
                text = context['text']
                doc = "Title: " + title + "\t" + "Text: " + text
                entity_docs.append(doc)
        output_docs = [f"- " + doc for i, doc in enumerate(entity_docs)]
        return '\n'.join(output_docs)
    
    def init(self):
        init_prompt = self._build_init_prompt()
        # return llm(init_prompt, model="gpt-3.5-turbo").strip('\n').strip()
        if self.base_llm == "gpt-3.5-turbo-instruct":
            return llm(init_prompt, model="gpt-3.5-turbo-instruct").strip('\n').strip()
        else:
            return self.init_llama.run(init_prompt).strip('\n').strip()
    
    def _build_init_prompt(self):
        init_prompt = init_prompt_template.format(
                        examples=INIT_EXAMPLE,
                        # examples="",
                        question = self.question)
        return init_prompt
    
    def direct(self):
        direct_prompt = self._build_direct_prompt()
        # return llm(direct_prompt, model="gpt-3.5-turbo").strip('\n').strip()
        if self.base_llm == "gpt-3.5-turbo-instruct":
            return llm(direct_prompt, model="gpt-3.5-turbo-instruct").strip('\n').strip()
        else:
            return self.direct_llama.run(direct_prompt).strip('\n').strip()
    
    def _build_direct_prompt(self):
        direct_prompt = direct_reason_prompt_template.format(
                        examples=DIRECT_EXAMPLE,
                        # examples="",
                        question = self.question,
                        paths = self.kg_paths)
        return direct_prompt
    
    def reason(self):
        reason_prompt = self._build_reason_prompt()
        # return llm(reason_prompt, model="gpt-3.5-turbo").strip('\n').strip()
        if self.base_llm == "gpt-3.5-turbo-instruct":
            return llm(reason_prompt, model="gpt-3.5-turbo-instruct").strip('\n').strip()
        else:
            return self.reason_llama.run(reason_prompt).strip('\n').strip()
    
    def _build_reason_prompt(self):
        if len(self.kg_paths) != 0:
            p = self.kg_paths
        else:
            p = "None"
        reason_prompt = reason_prompt_template.format(
                        examples=REASON_EXAMPLE,
                        # examples="",
                        question = self.question,
                        # paths = self.kg_paths
                        paths = p)
        return reason_prompt
    
    def refine(self, searched_contexts, entities_with_gd):
        refine_prompt = self._build_refine_prompt(searched_contexts, entities_with_gd)
        # return llm(refine_prompt, model="gpt-3.5-turbo").strip('\n').strip()
        if self.base_llm == "gpt-3.5-turbo-instruct":
            return llm(refine_prompt, model="gpt-3.5-turbo-instruct").strip('\n').strip()
        else:
            return self.refine_llama.run(refine_prompt).strip('\n').strip()
    
    def _build_refine_prompt(self, searched_contexts, entities_with_gd):
        refine_prompt = refine_prompt_template.format(
                        examples = REFINE_EXAMPLE,
                        docs = searched_contexts,
                        entities = entities_with_gd)
        return refine_prompt
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self, print_f1=True) -> bool:
        max_f1 = 0
        max_em = False
        if type(self.key) == list:
            for each_key in self.key:
                each_f1 = F1(self.answer, each_key)
                if each_f1 > max_f1:
                    max_f1 = each_f1
                if not max_em:
                    max_em = EM(self.answer, each_key)
        else:
            max_f1 = F1(self.answer, self.key)
            max_em = EM(self.answer, self.key)
        if print_f1:
            print("F1 Score:", max_f1)
        return max_em

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.kg_paths = ''
        self.filtered = []
        self.truncation = False
        self.kg = nx.MultiDiGraph()

def remove_blank_lines(text, prefix=None):
    lines = text.split('\n')
    non_blank_lines = [line for line in lines if line.strip() != '']
    if prefix is not None:
        for line_index, line in enumerate(non_blank_lines):
            if line.startswith(prefix):
                prefix_index = line_index
                break
        non_blank_lines = non_blank_lines[prefix_index:]        
    return '\n'.join(non_blank_lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def F1(answer, key):
    key = normalize_answer(key)
    answer = normalize_answer(answer)
    f1_score = calculate_f1_score(answer, key)
    return f1_score

def EM(answer, key):
    return normalize_answer(answer) == normalize_answer(key)

def calculate_f1_score(predicted_answer, true_answer):
    predicted_set = set(re.split(r'[ -]', predicted_answer))
    true_set = set(re.split(r'[ -]', true_answer))

    if len(predicted_set) == 0:
        return 0
    
    true_positives = len(predicted_set.intersection(true_set))
    
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(true_set)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def save_data(data_point, data_path):
    with open(data_path, "a") as file:
        json.dump(data_point, file)  # 将单个 JSON 对象写入文件
        file.write('\n')

def llm(prompt, model="gpt-3.5-turbo", stop=["\n"]):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=350,
        temperature=0
    )
    return completion.choices[0].message.content

class LocalLLM:
    def __init__(self, port, config, url="localhost"):
        self.config = config
        self.url = f"http://{url}:{port}"

    def send_request(self, prompt, config, url, headers=None):
        if headers is None:
            headers = {"Content-Type": "application/json"}
        payload = {'prompt': prompt, 'config': config}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_message = f"Error occurred with status code: {response.status_code}, response text: {response.text}"
                print(error_message)
                return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    def run(self, prompt):
        request_config = {
            'role': self.config['role'],
            'max_new_tokens': self.config['max_new_tokens'],
            'do_sample': self.config['do_sample'],
            'temperature': self.config['temperature'],
            'top_p': self.config['top_p'],
            'stop_strings': self.config['stop_strings'],
        }
        # print(request_config)
        response = self.send_request(prompt, request_config, self.url)
        return response
    