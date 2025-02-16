# %%
import os

from openai import AzureOpenAI

api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_ENDPOINT"]
deployment = os.environ["AZURE_DEPLOYMENT"]
api_version = os.environ["AZURE_API_VERSION"]


# %%
def init_client():
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{endpoint}/openai/deployments/{deployment}",
    )
    return client


client = init_client()


# %%
def inference(client, message):
    try:
        response = client.chat.completions.create(
            model=deployment, messages=message, max_tokens=2048
        )
        response = response.choices[0].message.content
    except Exception as e:
        response = str(e)
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
print(inference(client, messages))

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
print(inference(client, messages))

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
print(inference(client, messages))
