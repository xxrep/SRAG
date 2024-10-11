# KnowTrace
Explicit Knowledge Tracing for Structured Retrieval-Augmented Generation

## Requirements
- python == 3.9.19
- numpy == 1.26.4
- datasets == 2.20.0
- requests == 2.32.3
- peft == 0.9.0
- networkx == 3.2.1
- openai == 0.28.0
- beir == 2.0.0
- fastapi == 0.111.0
- uvicorn == 0.30.1
- torch == 2.3.0
- transformers == 4.42.3
- elasticsearch == 7.9.1

## Prepare Data and Retrieval Corpus
#### 1. Download a MHQA dataset such as [HotpotQA](https://hotpotqa.github.io), where each entry contains a question and an answer:
```
{"question": "<input question text>", "answer": "<target answer text>"}
```
Note that we focus on the challenging open-domain setting, and do not use the supporting context provided by the raw dataset.

#### 2. Download [Wikipedia](https://hotpotqa.github.io) corpus for retrieval, and then process:
```
python ./retriever/process_wiki.py --input_dir xx --output_path xx
```

## Prepare Retriever and LLM Server
#### 1. For BM25, install and start [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/8.15/targz.html) first.
```
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch # start the server
```

#### 2. Activate retriever among `[bm25, contriever, dpr]`:
```
python ./retriever/retrieval_server.py --data_name xx --corpus_path xx --retrieval_method xx --topk xx --port 8001
```

#### 3. Activate LLM server if using an open-source LLM such as [LLaMA3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct):
```
python ./local_llama.py --model_path xx --port 1051
```

## Inference of KnowTrace
```
python ./main.py --dataset_path xx --base_llm xx --step_num xx
```

## Self-Taught Finetuning of KnowTrace
#### 1. Collect high-quality rationales using knowledge backtracing mechanism:
```
python ./main.py --dataset_path xx --base_llm xx --step_num xx --collect_data True --exploration_path xx --completion_path xx
```

#### 2. Finetune base LLM on the collected rationales:
```
python ./finetune_peft.py --dataset_name xx --model_name xx --num_train_epochs xx --per_device_train_batch_size xx --gradient_accumulation_steps xx --learning_rate xx --lora_r xx --lora_alpha xx --lora_dropout xx --output_dir xx
```

## Acknowledgement
We refer to the code of [SuRe](https://github.com/bbuing9/ICLR24_SuRe) and [ReAct](https://github.com/ysymyth/ReAct). Thanks for their contributions.


