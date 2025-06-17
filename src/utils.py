# Helper functions

import json
##  Json is used to read and write the data in Json formate ( we are fetching the instruction_dataset.json so we need to read this file.)


from datasets import DataSet
 ## it is used to load th raw json where we want to convert it into a hugging face-compatible formate.  using .map()

from transformer import AutoTokenizer

# AutoTokenizer automatically pick the correct tokenizer based on the model_name ex gpt2 ) different model need different tokenization stragegies, just by passing the name of model it auto matical take it. 
## handles spliting text into tokens(words.)
## Mapping the tokens to numerical Ids.
## Padding or truncating sequences. 
## addmg specially tokens like EOS CLS.. etc.


def load_custom_dataset(path):
    ## creating a function which is used to load the custom dataset from given file path. 
    with open(path, 'r') as f: 

        ## open the path in read model 
        data =json.load(f)
        ## json.load(f) read the contents and converts the JSON data into  Pyton list of dictionaries.

    dataset_dict ={'instruction': [], 'input': [], 'output': []}
    for item in data:
        dataset_dict['instruction'].append(item['instruction'])
        dataset_dict['input'].append(item['input'])
        dataset_dict['output'].append(item['output'])
    return Dataset.from_dict(dataset_dict)

def preproces_funcion(example, tokenizer):
    prompt =f"{example['instruction']}\nInput: {example['input']}\nOutput:"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)