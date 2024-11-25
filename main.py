import json
from tqdm import tqdm
from openai import OpenAI
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

api_key = 'your_api_key'
client = OpenAI(api_key = api_key)

with open("tokenizer.json", "r") as f:
    data = json.load(f) 
    
id_to_content = {token["id"]: token["content"] for token in data["added_tokens"]}     
id_to_content.update({value: key for key, value in data["model"]["vocab"].items()})
content_to_id = {v: k for k, v in id_to_content.items()}

def trans(tids, id_to_content):
    # Transform embedding to readable token
    matched_contents = [id_to_content[num] for num in tids]
    tokens = [token for token in matched_contents if token != '<pad>']
    output = ''.join(token.replace('Ġ', ' ') for token in tokens)
    return output

def keys_with_prefix(prefix):
    return {key:content_to_id[key] for key in content_to_id.keys() if prefix.startswith(key)}

def tokenize_string(input_str):
    tokens = []  
    while input_str:
        matching_tokens = [token for token in content_to_id.keys() if input_str.startswith(token)]
        
        if not matching_tokens:
            raise ValueError(f"無法找到匹配的 token, 剩餘字串: {input_str}")
        
        longest_token = max(matching_tokens, key=len)
        tokens.append(content_to_id[longest_token])  
        input_str = input_str[len(longest_token):]  
    
    return tokens

def reverse_trans(text, content_to_id):
    tokens = text.replace(' ', ' Ġ').replace("<sep>", " <sep> ")
    tokens = tokens.split()
    
    tids = []
    for token in tokens:
        tids += tokenize_string(token)
    return tids

def custom_split(statement):
    result = ['<s>']

    temp = ""
    for char in statement:
        if char.isalnum() or char == '_':  
            temp += char
        else:
            if temp:
                result.append(temp)  
                temp = ""
            if char.strip(): 
                result.append(char)

    if temp:  
        result.append(temp)

    result.append('</s>')
    return result

with open('preds.jsonl', 'r') as file:
    predict = file.readlines()
    

if __name__ == "__main__":
    from teco.model.subtokenizer_bpe import SubtokenizerBPE
    from transformers import AutoTokenizer
    import os
    from pathlib import Path
    
    pretrained_tokenizer = os.path.relpath(Path("/root/teco/_work/subtokenizer/codet5"), Path.cwd())
    tokenizer: SubtokenizerBPE = SubtokenizerBPE(AutoTokenizer.from_pretrained(pretrained_tokenizer))
    
    updated_data = []
    for index, line in enumerate(tqdm(predict, desc="Processing lines", unit="line")):
        data = json.loads(line)
        src_readable = trans(data['input_stids'], id_to_content)
            
        ans = []
        for i in range(10):
            topk = data["topk"][i]
            
            # Call chatgpt api
            completion = client.chat.completions.create(
              model="gpt-4-turbo",
              messages=[
                {"role": "assistant", "content": "just give exactly 1 next statement, you should be able to test the method under test as shortes as possible."},
                {"role": "user", "content": src_readable}
              ]
            )
            
            result = completion.choices[0].message.content
            pre_stmt = custom_split(result)

            topk['tids'] = tokenizer.toks2stids(pre_stmt)
            topk['toks'] = pre_stmt
            ans.append(topk)
        
        data["topk"] = ans  
        updated_data.append(json.dumps(data))
    
    with open('preds_gpt.jsonl', 'w') as file:
        for line in updated_data:
            file.write(line + '\n')