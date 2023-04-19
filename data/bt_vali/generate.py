import os
import openai
import datasets

def collect_data():
    """
    Send a prompt to GPT-3.5-turbo and collect data.

    :param prompt: The prompt to send to the API
    :param n: The number of responses to generate
    :param max_tokens: The maximum number of tokens for each response
    :return: A list of generated responses
    """

    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    responses = [choice.text.strip() for choice in completions.choices]
    return responses

messages = []

__system_prompt = '''
You are an expert at reading five responses to an instruction
'''
