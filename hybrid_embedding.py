# !pip install --upgrade torch torchvision torchaudio
# !pip install -U peft transformers FlagEmbedding

import gzip, json
import time
import torch
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import os
from google.colab import drive

drive.mount('/content/drive')

Data_path = '/content/drive/MyDrive/544_data'

insights_file = '/content/drive/MyDrive/product_insights_nuextract_aggregated_A100_Final_4000_4.jsonl.gz' ##insights file
meta_file = os.path.join(Data_path, 'meta_Appliances.jsonl.gz') ##metadata file
output_embed_file = os.path.join(Data_path, 'embed_Appliances_bgem3_hybrid_x.jsonl.gz')

target_data = {}

with gzip.open(insights_file, 'rt', encoding='utf-8') as f:
  for line in tqdm(f, desc="reading insights file"):
      try:
        record = json.loads(line)
        pid = record.get("parent_asin")
        if pid:
          target_data[pid] = record.get("insights", {})
      except ValueError:
        continue

print(f"Found {len(target_data)} target products.")

products_meta_dic = {}
matches_found = 0

with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
    for line in tqdm(f, desc="Reading Metadata"):
        try:
            record = json.loads(line)
            #Can be asin or parent_asin
            pid = record.get("parent_asin") or record.get("asin")

            if pid in target_data:
                products_meta_dic[pid] = record.get('title', 'Unknown Product')
                matches_found += 1

                if matches_found >= len(target_data):
                    break
        except ValueError: continue

print(f"Metadata task completed. Matched Titles for {matches_found} out of {len(target_data)} products.")

#Build rich strings
rich_doc_buffer = []

for pid, insights in target_data.items():
    title = products_meta_dic.get(pid, "Unknown Appliance")

    #Sometimes the LLM returns a list instead of a dictionary, fixing that here
    if isinstance(insights, list) and len(insights) > 0:
        insights = insights[0]
    elif not isinstance(insights, dict):
        insights = {}

    #Features
    feat_list = insights.get('Key_Features', [])
    if not isinstance(feat_list, list): feat_list = []
    clean_features = []
    for x in feat_list:
        if x is not None:
            clean_features.append(str(x))
    features = ', '.join(clean_features)

    #Pros
    pros_list = insights.get('Pros', [])
    if not isinstance(pros_list, list): pros_list = []
    clean_pros = []
    for x in pros_list:
        if x is not None:
            clean_pros.append(str(x))
    pros = '; '.join(clean_pros)

    #Cons
    cons_list = insights.get('Cons', [])
    if not isinstance(cons_list, list): cons_list = []
    clean_cons = []
    for x in cons_list:
        if x is not None:
            clean_cons.append(str(x))
    cons = '; '.join(clean_cons)

    #Uses
    uses_list = insights.get('best_uses', [])
    if not isinstance(uses_list, list): uses_list = []
    clean_uses = []
    for x in uses_list:
        if x is not None:
            clean_uses.append(str(x))
    uses = '; '.join(clean_uses)

    rich_doc = f"Title: {title} | Features: {features} | Pros: {pros} | Cons: {cons} | Uses: {uses}"
    rich_doc_buffer.append((pid, rich_doc))

#embedding part
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')

#Batch size 12 fits in the the GPU A100 without crashing
batch_size = 12
print(f"Embedding {len(rich_doc_buffer)} documents")

with gzip.open(output_embed_file, "wt", encoding='utf-8') as f_out:
    for i in tqdm(range(0, len(rich_doc_buffer), batch_size), desc="Embedding Batches"):
        batch = rich_doc_buffer[i : i + batch_size]
        batch_pids = [b[0] for b in batch]
        batch_texts = [b[1] for b in batch]

        output = model.encode(
            batch_texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True
        )

        dense_vecs = output['dense_vecs']
        lexical_weights = output['lexical_weights']

        for j, pid in enumerate(batch_pids):

            #needed to fix the sparse weights format for JSON storage
            current_sparse = lexical_weights[j]
            new_sparse = {}
            for k, v in current_sparse.items():
                new_sparse[k] = float(v)

            record = {
                "parent_asin": pid,
                "rich_doc": batch_texts[j],
                "dense_embed": dense_vecs[j].tolist(),
                "sparse_embed": new_sparse
            }
            f_out.write(json.dumps(record) + "\n")

print(f"Hybrid embeddings saved to: {output_embed_file}")