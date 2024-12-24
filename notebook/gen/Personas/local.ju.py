# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../../../models/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model_path)


# %%
def inference(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# %% [md]
## Generate instructions

# %%
PROMPT_TEMPLATE = (
    """Generate a prompt the persona below might ask to an AI assistant:\n"""
)

system_prompt = "You are an AI assistant expert at simulating user interactions."
example_persona = "A behavioral economist or social psychologist interested in exploring strategies for influencing human decision-making and behavior change."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": PROMPT_TEMPLATE + example_persona},
]
print(inference(messages, model, tokenizer))

# %% [md]
## Generate diverse text for pre-training and post-training

# %%
PROMPT_TEMPLATE = """Write a Quora post in the language, style, and personality of the following persona:\n"""

system_prompt = "You are an AI assistant specialized in writing posts for social media."
example_persona = "An economist specializing in education policy and research, likely with a background in quantitative analysis and public policy."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": PROMPT_TEMPLATE + example_persona},
]
print(inference(messages, model, tokenizer))

# %% [md]
## Generate persona-specific problems

# %%
PROMPT_TEMPLATE = """Create a challenging math problem with the following persona:\n"""

system_prompt = "You are an AI assistant specialized in creating diverse but specific math problems. Just answer with your problem."
example_persona = "A behavioral economist or social psychologist interested in exploring strategies for influencing human decision-making and behavior change."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": PROMPT_TEMPLATE + example_persona},
]
print(inference(messages, model, tokenizer))
