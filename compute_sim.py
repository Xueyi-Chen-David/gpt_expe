from teco.model.prediction import (
    PredictInputs,
    Prediction,
    PredictionDataModule,
    PredictionWriter,
    compute_similarity_metrics,
)

import json
import ast
from tqdm import tqdm

def load_gstmt():
  result = []
  
  with open("gold_stmts.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            parsed_list = ast.literal_eval(line)
            result.append(parsed_list)
  
  return result

def load_gpt_pre():
  res = []
  with open("preds_gpt_new.jsonl", "r") as f:
     res = f.readlines()
  return res

if __name__ == "__main__":
  res = load_gpt_pre()
  target = load_gstmt()
  
  total_res = []

  for i in tqdm(range(len(res)), desc="Compute similarity metrics", unit="line"):
    result = compute_similarity_metrics(target[i], [item["toks"] for item in json.loads(res[i])['topk']])
    total_res.append(result)
  
  sums = {}
  
  for entry in tqdm(total_res, desc="Average", unit="line"):
    for key, value in entry.items():
        sums[key] = sums.get(key, 0) + value

  averages = {key: sums[key] / len(sums) for key in sums}
  # print("---------------------------------------------")
  # print("This is fucking results: ")
  # print(averages)
    
    
  with open("total_metrics.json", "w", encoding="utf-8") as json_file:
    json.dump(averages, json_file, ensure_ascii=False, indent=4)