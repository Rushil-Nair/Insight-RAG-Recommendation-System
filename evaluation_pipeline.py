import gzip, json
import random, os, gc
import torch
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import drive
from sklearn.metrics.pairwise import cosine_similarity
import FlagEmbedding
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from fast_json_repair import repair_json
import textwrap


drive.mount('/content/drive')

Data_path = '/content/drive/MyDrive/544_data'
hybrid_file = os.path.join(Data_path, 'embed_Appliances_bgem3_hybrid.jsonl.gz')

#Clearing vram after each model's work is done
def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

rich_docs = {}
reviews_dense = {}
reviews_sparse = {}

with gzip.open(hybrid_file, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line)
            pid = record["parent_asin"]

            if "rich_doc" in record:
                rich_docs[pid] = record["rich_doc"]

            reviews_dense[pid] = record["dense_embed"]
            reviews_sparse[pid] = record["sparse_embed"]
        except Exception as e:
            print(f"Error:{e}")
            continue

all_pids = list(rich_docs.keys())
print(f"Loaded {len(all_pids)} products.")

##Teacher model will generate synthethic data and answer them which will serve as reference answers
teacher_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        load_in_4bit=True
)

def generate_synthetic_data(num_samples=50):
    synthetic_data = []
    target_pids = random.sample(all_pids, num_samples)

    print(f"Generating {num_samples} Q&A pairs...")

    for pid in tqdm(target_pids):
        product_context = rich_docs[pid][:1500]
        prompt = f"""<|user|>
Based on the following product description, write a specific question a user might ask, and provide the correct factual answer based ONLY on the text.
Product: {product_context}

Output format JSON:
{{
  "question": "...",
  "answer": "..."
}}
<|assistant|>
"""
        inputs = teacher_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = teacher_model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        generated_text = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)

        parsed_obj = None

        try:
            parsed_obj = repair_json(generated_text, return_objects=True)
            if isinstance(parsed_obj, list):
                parsed_obj = parsed_obj[0]
        except:
            parsed_obj = None

        if not parsed_obj or 'question' not in parsed_obj:
            try:
                json_str = generated_text.split("{")[-1].split("}")[0]
                parsed_obj = json.loads("{" + json_str + "}")
            except:
                parsed_obj = None

        if parsed_obj and 'question' in parsed_obj:
            synthetic_data.append({
                "pid": pid,
                "context": product_context,
                "question": parsed_obj['question'],
                "ground_truth": parsed_obj['answer']
            })

    return synthetic_data

eval_dataset = generate_synthetic_data(num_samples=50)

del teacher_model, teacher_tokenizer
clear_gpu_memory()

##Search models loading
retriever_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

dense_vectors_list = []
for p in all_pids:
    dense_vectors_list.append(reviews_dense[p])
search_matrix = np.array(dense_vectors_list)

retrieval_hits = 0


for record in tqdm(eval_dataset):
    user_query = record['question']
    target_pid = record['pid']

    query_output = retriever_model.encode([user_query], return_dense=True, return_sparse=True)
    q_dense = query_output['dense_vecs'][0]
    q_sparse = query_output['lexical_weights'][0]

    dense_scores = cosine_similarity([q_dense], search_matrix)[0]
    top_200_idx = np.argsort(dense_scores)[::-1][:200]

    #Hybrid scoring
    candidates = []
    for idx in top_200_idx:
        p_id = all_pids[idx]

        sparse_score = 0.0
        doc_keywords = reviews_sparse[p_id]

        for word, weight in q_sparse.items():
            word_str = str(word)
            if word_str in doc_keywords:
                sparse_score += weight * doc_keywords[word_str]

        final_score = (0.5 * dense_scores[idx]) + (0.5 * sparse_score)
        candidates.append((p_id, final_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_100_pids = [x[0] for x in candidates[:100]]

    rerank_pairs = []
    for c_pid in top_100_pids:
        rerank_pairs.append([user_query, rich_docs[c_pid]])

    rerank_scores = reranker_model.compute_score(rerank_pairs)

    if not isinstance(rerank_scores, list):
         rerank_scores = rerank_scores.tolist()

    ranked_results = list(zip(top_100_pids, rerank_scores))
    ranked_results.sort(key=lambda x: x[1], reverse=True)

    final_top_5 = [item[0] for item in ranked_results[:5]]

    if target_pid in final_top_5:
        retrieval_hits += 1
        record['search_success'] = True
    else:
        record['search_success'] = False

recall_score = (retrieval_hits / len(eval_dataset)) * 100
print(f"Recall@5: {recall_score:.2f}%")

del retriever_model, reranker_model, search_matrix
clear_gpu_memory()


#Student models (the model we used in our rag) will be used to generate answers
student_model_name = "Qwen/Qwen2.5-7B-Instruct"

student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

for record in tqdm(eval_dataset):
    messages = [
        {"role": "system", "content": f"Answer the user's question based solely on the context provided. Be concise and factual.: {record['context']}"},
        {"role": "user", "content": record['question']}
    ]

    input_text = student_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = student_tokenizer(input_text, return_tensors="pt").to("cuda")

    generated_ids = student_model.generate(**model_inputs, max_new_tokens=150, do_sample=False)
    response_text = student_tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)

    record['student_answer'] = response_text.strip()

del student_model, student_tokenizer
clear_gpu_memory()

##Judge model
judge_model_name = "prometheus-eval/prometheus-7b-v2.0"
print(f"Loading Judge: {judge_model_name}")

judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
judge_model = AutoModelForCausalLM.from_pretrained(
    judge_model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    load_in_4bit=True
)

factuality_scores = []

##Adapted the prompt from Prometheus's HF page
for record in tqdm(eval_dataset):
    eval_prompt = f"""###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, and a reference answer that gets a score of 5.
    1. Read the instruction and input and reference answer carefully.
    2. Read the response and compare it to the reference answer.
    3. Score the response from 1 to 5 (5 is best) based on Faithfulness and Completeness.

    ###The Instruction:
    {record['question']}

    ###Response to evaluate:
    {record['student_answer']}

    ###Reference Answer (Score 5):
    {record['ground_truth']}

    ###Score:
    """

    inputs = judge_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    outputs = judge_model.generate(**inputs, max_new_tokens=500, do_sample=False)
    critique = judge_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    record['judge_critique'] = critique

    try:
        if "[RESULT]" in critique:
            parts = critique.split("[RESULT]")
            score = int(parts[1].strip())
        else:
            matches = re.findall(r'[Ss]core:\s*([1-5])', critique)
            if matches:
                score = int(matches[-1])
            else:
                matches = re.findall(r'\b[1-5]\b', critique)
                score = int(matches[-1]) if matches else 3
    except:
        score = 3

    factuality_scores.append(score)
    record['score'] = score

avg_score = sum(factuality_scores)/len(factuality_scores) if factuality_scores else 0

print(f"\nFINAL RESULTS ")
print(f"Recall@5: {recall_score:.1f}%")
print(f"Factuality Score: {avg_score:.2f}/5")


df = pd.DataFrame(eval_dataset)
df.to_csv("final_eval_results.csv", index=False)
print("Saved the CSV file.")