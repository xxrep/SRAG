import os
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)


from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from exemplars import REFINE_EXAMPLE, REASON_EXAMPLE, INIT_EXAMPLE, DIRECT_EXAMPLE
from prompts import refine_prompt_template, reason_prompt_template, init_prompt_template, direct_reason_prompt_template
from transformers import pipeline


torch.manual_seed(42)


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """


    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})


    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=1.0)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=4000)
    model_name: Optional[str] = field(
        default="./llm_models/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to finetune."
        }
    )
    dataset_name: Optional[str] = field(
        default="./completion.json",
        metadata={"help": "The preference dataset to use."},
    )


    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        # default="cosine_with_warmup",
        default="cosine",


        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=100, metadata={"help": "How many optimizer update steps to take"})
    warmup_steps: int = field(default=10, metadata={"help": "# of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=5, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./checkpoints_packing/reasoner",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )




parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ds = load_dataset('json', data_files=script_args.dataset_name)
print(ds)
total_size = len(ds['train'])
eval_size = 500
train_val = ds['train'].train_test_split(
    test_size=int(total_size-500), shuffle=False
)
eval_data = train_val['train'].shuffle()
print(eval_data)
print(eval_data[0])
train_data = train_val['test'].shuffle()
print(train_data)
print(train_data[-1])

def gen_batches_train(ds):
    for sample in iter(ds):
        if sample['type'] == 'init':
            # instruction = init_prompt_template.format(
            #                 examples=INIT_EXAMPLE,
            #                 question = sample['question'])
            # input_text = None
            # out_text = sample['response']
            continue
        elif sample['type'] == 'final':
            instruction = reason_prompt_template.format(
                                examples=REASON_EXAMPLE,
                                question = sample['question'],
                                paths = sample['memory'])
            # input_text = None
            out_text = sample['response']
        elif sample['type'] == 'trunc':
            continue
            # instruction = direct_reason_prompt_template.format(
            #                     examples=DIRECT_EXAMPLE,
            #                     question = sample['question'],
            #                     paths = sample['memory'])
            # input_text = None
            # out_text = sample['response']
        elif sample['type'] == 'process':
            instruction = reason_prompt_template.format(
                                examples=REASON_EXAMPLE,
                                question = sample['question'],
                                paths = sample['memory'])
            # input_text = None
            out_text = 'Whether the given knowledge triples are sufficient for answering: No' + '\nRetrieval Guidance:\n' + sample['response']
        # For refine
        elif sample['type'] == 'refine':
            instruction = refine_prompt_template.format(
                            examples = REFINE_EXAMPLE,
                            docs = sample['passage'],
                            entities = sample['gd'])
            # input_text = None
            out_text = sample['response']
        # formatted_prompt = None 
            
        # if input_text is None or input_text == "":
        formatted_prompt = (
            f"<|begin_of_text|>"
            f"### Instruction:\n{instruction}\n### Response:\n"
            f"{str(out_text)}<|end_of_text|>"
        )
        
        formatted_prompt = "".join(formatted_prompt)

        yield {'text': formatted_prompt}


def create_and_prepare_model(args):
    # compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    # commented qlora stuff 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True,
    )
    
    
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # target_modules=["query_key_value"], 
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=['q_proj', 'v_proj'],
    )


    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    return model, peft_config, tokenizer




training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    eval_steps=10,
    eval_strategy='steps',
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    # max_steps=script_args.max_steps,
    # num_train_epochs = 3,
    num_train_epochs = script_args.num_train_epochs,
    load_best_model_at_end = True,
    save_total_limit = 6,
    warmup_steps=script_args.warmup_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    # report_to=script_args.report_to,
)


model, peft_config, tokenizer = create_and_prepare_model(script_args)


train_gen = Dataset.from_generator(gen_batches_train, gen_kwargs={"ds": train_data})
eval_gen = Dataset.from_generator(gen_batches_train, gen_kwargs={"ds": eval_data})

tokenizer.padding_side = "right"

response_template = "### Response:\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    eval_dataset=eval_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
    data_collator=collator,
)


# llama_inference(trainer, tokenizer)

trainer.train()

# llama_inference(trainer, tokenizer)

# if script_args.merge_and_push:
output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# Free memory for merging weights
del model
torch.cuda.empty_cache()


model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()


output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)
