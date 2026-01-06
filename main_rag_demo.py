# !pip install -U -q FlagEmbedding transformers langchain langchain-community langchain-huggingface chromadb bitsandbytes accelerate peft fast_json_repair nltk

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys
CODE_PATH = '/content/drive/MyDrive/544_data'
sys.path.append(CODE_PATH)

import gzip, json
import numpy as np
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from google.colab import drive
from rank import rank
from rag_colab_8 import RAG

Data_path = '/content/drive/MyDrive/544_data'
meta_file = os.path.join(Data_path, 'meta_Appliances.jsonl.gz')
hybrid_file = os.path.join(Data_path, 'embed_Appliances_bgem3_hybrid.jsonl.gz')

products = {}
reviews_dense = {}
reviews_sparse = {}
rich_docs = {}

with gzip.open(meta_file, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            curr_line = json.loads(line)
            p_id = curr_line["parent_asin"]
            cleaned_obj = {}
            for k, v in curr_line.items():
                if k != "parent_asin":
                    cleaned_obj[k] = v

            products[p_id] = cleaned_obj
        except Exception as e:
          print(f"Error encountered while loading metadata:{e}")
          continue


with gzip.open(hybrid_file, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            curr_line = json.loads(line)
            pid = curr_line["parent_asin"]

            reviews_dense[pid] = curr_line["dense_embed"]
            reviews_sparse[pid] = curr_line["sparse_embed"]

            #defaulting to empty string if not there
            if "rich_doc" in curr_line:
                rich_docs[pid] = curr_line["rich_doc"]
            else:
                rich_docs[pid] = ""
        except Exception as e:
          print(f"Error encountered while loading embeddings:{e}")
          continue

retriever = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def run_search(user_query):
    encoded_query = retriever.encode([user_query], return_dense=True, return_sparse=True)
    q_dense = encoded_query['dense_vecs'][0]
    q_sparse = encoded_query['lexical_weights'][0]

    all_pids = list(reviews_dense.keys())

    vectors_list = []
    for p in all_pids:
        vectors_list.append(reviews_dense[p])
    dense_matrix = np.array(vectors_list)

    dense_sims = cosine_similarity([q_dense], dense_matrix)[0]

    sparse_scores = []
    for pid in all_pids:
        doc_weights = reviews_sparse[pid]
        current_score = 0.0

        for token, query_weight in q_sparse.items():
            token_str = str(token)
            if token_str in doc_weights:
                current_score += query_weight * doc_weights[token_str]

        sparse_scores.append(current_score)

    def normalize_scores(arr):
        val_min = np.min(arr)
        val_max = np.max(arr)
        return (arr - val_min) / (val_max - val_min + 1e-9)

    norm_dense = normalize_scores(dense_sims)
    norm_sparse = normalize_scores(np.array(sparse_scores))

    #Weighted Sum
    hybrid_final = (0.5 * norm_dense) + (0.5 * norm_sparse)

    top_100_idx = np.argsort(hybrid_final)[::-1][:100]

    candidates = []
    for i in top_100_idx:
        candidates.append(all_pids[i])

    #Cross encoder reranking
    pairs_to_rank = []
    for pid in candidates:
        pairs_to_rank.append([user_query, rich_docs[pid]])

    ce_scores = reranker.compute_score(pairs_to_rank)

    if isinstance(ce_scores, list):
        ce_scores = np.array(ce_scores)

    ce_norm = normalize_scores(ce_scores)

    #using the rank script to get the final best products
    final_ranked = rank(
        product_ids=candidates,
        similarities=ce_norm,
        products=products,
        reviews=reviews_dense,
        query=user_query,
        top_k_candidates=100,
        final_k=5
    )
    return final_ranked


start_pid = list(products.keys())[0]
rag_engine = RAG(embedding_model=retriever, default_meta=products[start_pid])

while True:
    txt = input("\n Enter your product search query (or type 'exit'): > ")
    if txt.lower().strip() == 'exit':
        print("Successfully exited")
        break

    print("Searching for the products...")
    results = run_search(txt)

    print(f"\nTop 5 Results:")
    for idx, (pid, score) in enumerate(results):
        if pid in products:
            p_title = products[pid].get('title', 'No Title')
            print(f"{idx+1}. {p_title[:80]} (Score: {score:.3f})")

    #auto selecting rank 1
    best_pid = results[0][0]
    best_title = products[best_pid]['title']
    print(f"\nType any query related to product #1: {best_title[:30]}")

    rag_engine.change_product(products[best_pid])

    while True:
        q = input("Ask a question (or type 'back' / type 'rank N'):")
        if q.lower() == 'back':
            break
        #changing product and it's context
        if q.lower().startswith('rank '):
            try:
                parts = q.split()
                rank_idx = int(parts[1]) - 1

                if 0 <= rank_idx < len(results):
                    new_pid = results[rank_idx][0]
                    rag_engine.change_product(products[new_pid])
                    print(f"Switched context to product #{rank_idx+1}")
                    continue
                else:
                    print("Incorrect rank has been entered.")
            except:
                print("Usage: 'rank 2' to switch to 2nd product")
                pass


        ans = rag_engine.generate_answer(q)
        print(f"AI: {ans}")