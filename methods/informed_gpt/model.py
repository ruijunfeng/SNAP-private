import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.json_utils import update_json

from methods.informed_gpt.config import InformedGPTConfig
from data_module import HelocDataModule

# prepare the data module
config = InformedGPTConfig()
data_module = HelocDataModule()
dataset = data_module.get_profile_dataset(data_module.val_indices + data_module.test_indices)

# create the output directory
output_dir = "results/informed_gpt"
os.makedirs(output_dir, exist_ok=True)

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name, 
    padding_side="left",
)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    dtype="auto",
    device_map="auto",
).eval()

choices = ["Good", "Bad"]
choice_ids = tokenizer(choices, add_special_tokens=False).input_ids

for i in trange(len(dataset)):
    # prepare the model input
    prompt = config.question_template.format(
        profile=dataset[i]["profiles"],
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([f"{text}{config.answer_template}"], return_tensors="pt").to(model.device)
    
    # constrained decoding on choices
    with torch.no_grad():
        outputs = model(**model_inputs)
    
    # get the logits and select the choice with highest logits
    logits = outputs.logits[0, -1, choice_ids].squeeze() # shape: (num_choices,)
    y_pred = torch.argmax(logits).item()
    y_proba = torch.softmax(logits, dim=0)[1].item() # probability of "Bad"
    
    # save the results
    update_json(f"{output_dir}/predictions.json", dataset[i]["indices"], {"y_pred": y_pred, "y_proba": y_proba, "logits": logits.tolist()})
