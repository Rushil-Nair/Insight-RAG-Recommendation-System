# !pip install transformers==4.44.2 accelerate==0.33.0

from google.colab import drive
import os
import json
import gzip
import torch
import gc
import re
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from fast_json_repair import repair_json
import flash_attn

drive.mount('/content/drive')

Data_path = '/content/drive/MyDrive/544_data'
Reviews_file = os.path.join(Data_path, 'Appliances.jsonl.gz')
output_file = '/content/drive/MyDrive/product_insights_nuextract_aggregated_A100_Final_4000_x.jsonl.gz'
target_product_count = 4000

def load_model():
    model_id = "numind/NuExtract-1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Schema for the model to follow
json_schema = """{"Key_Features":[],"Pros":[],"Cons":[],"best_uses":[]}"""

def json_cleaning(text):
    try:
        parsed_json = repair_json(text, return_objects=True)
        return parsed_json
    except Exception:
        return None

def product_aggregator(path):
    product_buffer = {}

    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for line in tqdm(file, desc="Scanning File"):

            if len(product_buffer) >= target_product_count:
                break
            try:
                line_data = json.loads(line)

                pid = line_data.get('parent_asin')
                if not pid:
                    pid = line_data.get('asin')

                text = line_data.get('text')
                if not text:
                    text = line_data.get('reviewText')

                rating = line_data.get('rating')

                if not pid or not text:
                    continue

                if pid not in product_buffer:
                    product_buffer[pid] = {"reviews": [], "ratings": []}

                product_buffer[pid]["reviews"].append(text)

                if rating:
                    product_buffer[pid]["ratings"].append(float(rating))

            except Exception as e:
                print(f"Error:{e}")
                continue

    #Process the buffer and yield one item at a time
    for pid, data in product_buffer.items():
        all_reviews = data["reviews"]
        combined_text = " | ".join(all_reviews)
        if len(combined_text) > 6000:
            combined_text = combined_text[:6000]

        rating_list = data["ratings"]
        if len(rating_list) > 0:
            avg_rating = sum(rating_list) / len(rating_list)
        else:
            avg_rating = 0

        yield { "id": pid, "text": combined_text, "rating": avg_rating }

def process_in_batches(generator, batch_size):
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    #Batch 24 was set as we are using A100 in colab pro
    batch_size = 24

    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load_model()

    with gzip.open(output_file, 'wt', encoding='utf-8') as out_file:

        product_stream = product_aggregator(Reviews_file)
        product_batches = process_in_batches(product_stream, batch_size)

        for batch in tqdm(product_batches, desc="Processing Batches", unit="batch"):
            prompts = []
            for item in batch:
                #NuExtract template format
                p_text = f"<|input|>\n### Template:\n{json_schema}\n### Text:\n{item['text']}\n\n<|output|>"
                prompts.append(p_text)

            tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")

            # Generate
            with torch.no_grad():
                outputs = model.generate(**tokenized_inputs,max_new_tokens=256,do_sample=False,repetition_penalty=1.2)

            decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            # Process outputs
            for i in range(len(decoded_texts)):
                full_text = decoded_texts[i]
                generated_part = ""
                try:
                    parts = full_text.split("<|output|>")
                    if len(parts) > 1:
                        generated_part = parts[-1]
                    else:
                        generated_part = full_text
                except:
                    generated_part = full_text

                generated_part = generated_part.replace("<|endoftext|>", "")
                insights = json_cleaning(generated_part)

                #handling edge cases where repair_json returns a list
                if isinstance(insights, list):
                    if len(insights) > 0:
                        insights = insights[0]
                    else:
                        insights = None

                if insights:
                    record = {
                        "parent_asin": batch[i]['id'],
                        "rating": batch[i]['rating'],
                        "insights": insights
                    }
                    out_file.write(json.dumps(record) + '\n')

            #Flush to disk in case the script crashes
            out_file.flush()

            #cleaning memory
            del tokenized_inputs, outputs, decoded_texts
            torch.cuda.empty_cache()
            gc.collect()

    print("Completed extraction task.")

if __name__ == "__main__":
    main()