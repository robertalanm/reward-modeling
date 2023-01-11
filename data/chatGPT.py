
from revChatGPT.ChatGPT import Chatbot
import requests
import pandas as pd

from datasets import load_dataset

from tqdm import tqdm
import pdb
import time



def create_chatbot(config):

    chatbot = Chatbot(config)

    return chatbot


def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

def get_dataset_df(dataset_name):
    dataset = get_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])
    return df


last_row = 2_545

dataset_df = get_dataset_df('Dahoas/sft-static')
output_df = pd.DataFrame(columns=['prompt', 'response'])
# pdb.set_trace()
# drop the first n rows
dataset_df = dataset_df.drop(dataset_df.index[:last_row])
dataset_df.reindex()
# pdb.set_trace()

current_account = 0

chatbot = create_chatbot(session_ids[current_account])

for i in range(len(dataset_df)):
    try:
        i = i + last_row
        prompt = dataset_df['prompt'][i]
        
        response = chatbot.ask(prompt)
        # time.sleep(3)

        output_df = output_df.append({'prompt': prompt, 'response': response['message']}, ignore_index=True)

        print('saved response for prompt: ', prompt)
        print('response: ', response['message'])
        output_df.to_csv('bpt_dataset.csv')
        time.sleep(3)
    except Exception as e:
        print('failed to save response for prompt: ', prompt)
        print(e)
        if current_account == 0:
            current_account = 1
            chatbot = create_chatbot(session_ids[current_account])
            print('switching to account 1')
            continue
        else:
            current_account = 0
            print('sleeping for 20 minutes')
            for i in tqdm(range(1200)):
                time.sleep(1)
            chatbot = create_chatbot(session_ids[current_account])
            continue