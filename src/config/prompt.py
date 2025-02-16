SYSTEM_PROMPT = {"General": "You are a helpful assistant."}

ALPACA_PROMPT = {
    "prompt_input": ("{instruction}\n{input}\n"),
    "prompt_no_input": ("{instruction}\n"),
}

SHAREGPT_PROMPT = {"prompt": ("{system}\n{input}\n")}

PROMPT_NO_PERSONA = """请生成几个高质量non-reasoning prompt，主要从以下几点来保证prompt的多样性和新颖性。
1. 任务类型：头脑风暴、创意写作、专业知识、意见咨询、决策等
2. 人设/提问视角：不同年龄段、不同职业、不同行业领域、不同经验程度等
3. 考察能力：专业能力、人情世故等。

要求：
1. prompt要保证丰富性，可以是从以下几个方面：情景设定、人设设定、任务设定等
2. prompt控制在一段文字，不要分段分点
3. 输出格式：每个prompt前使用数字标号，不同prompt用换行分割，以下为示例
[示例]
1. xxx
2. xxx

本次生成的prompt请围绕的主题：{topic}
"""

PROOMP_NO_PERSONA = """
请生成一个高质量non-reasoning prompt，主要从以下几点来保证prompt的多样性和新颖性。
1. 任务类型：头脑风暴、创意写作、专业知识、意见咨询、决策等
2. 人设：{persona} 这是你当前的人设，请根据人设信息给出符合人设的prompt
3. 考察能力：专业能力、人情世故等。

要求：
1. prompt要保证丰富性，可以是从以下几个方面：情景设定、人设设定、任务设定等
2. prompt控制在一段文字，不要分段分点，prompt要保证是中文
3. 输出格式：以下为示例
[示例]
xxx

本次生成的prompt请围绕的主题：{topic}
"""
