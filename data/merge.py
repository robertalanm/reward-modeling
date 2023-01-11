import pandas as pd
from datasets import Dataset
import pdb
# merge bpt_dataset.part1.csv and bpt_dataset.part2.csv and save as bpt_dataset.part3.csv
df1 = pd.read_csv('bpt_dataset.part1.csv')
df2 = pd.read_csv('bpt_dataset.part2.csv')
df3 = pd.read_csv('bpt_dataset.part3.csv')
df4 = pd.read_csv('bpt_dataset.part4.csv')
df5 = pd.read_csv('bpt_dataset.csv')

df = pd.concat([df1, df2, df3, df4, df5])
df = df.drop(columns=['Unnamed: 0'])

# to huggingface Dataset
dataset = Dataset.from_pandas(df)

# save as parqet
dataset.to_parquet('data.parquet')
pdb.set_trace()