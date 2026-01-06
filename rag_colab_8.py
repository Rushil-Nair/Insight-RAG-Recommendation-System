import torch
import json
import time
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

##bge-m3 wrapper code
class STEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        
    def embed_documents(self, texts):
        if hasattr(self.model, 'encode'):
            res = self.model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            if isinstance(res, dict):
                return res['dense_vecs'].tolist()
            return res.tolist()
        return []
    
    def embed_query(self, text):
        if hasattr(self.model, 'encode'):
            res = self.model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
            if isinstance(res, dict):
                return res['dense_vecs'][0].tolist()
            return res[0].tolist()
        return []

class RAG:
    def __init__(self, embedding_model, default_meta= None):
        self.emb_model = STEmbeddings(embedding_model)
        self.rag_chain = None
        
        # Load Model ONCE
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        
        ##4-bit quantization is needed for the model to fit in colab memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True
        )
        
        ##text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            repetition_penalty=1.1,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question asked based ONLY on the following product context:
            {context}
            
            Question: {question}
            Answer:"""
        )
        
        if default_meta:
            self.change_product(default_meta)
            
        print("RAG Model Loaded Successfully.")

    def change_product(self, new_meta):
        """
        Changing the context to a new product without reloading the model.
        """
        if not new_meta: 
            return

        cleaning_meta_data = {}
        for k, v in new_meta.items():
            if 'embed' not in k:
                cleaning_meta_data[k] = v
        product_text = json.dumps(cleaning_meta_data, indent=2)

        col_name = f"prod_{int(time.time())}"
        
        try:
            store = Chroma.from_texts([product_text], embedding=self.emb_model,collection_name=col_name)
            retriever = store.as_retriever(search_kwargs={"k": 1})
            
            self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            print(f"Error switching product: {e}")

    def generate_answer(self, question):
        if not self.rag_chain:
            return "Please select a product first as it is not selected."
        else:
            try:
                rag_model_response = self.rag_chain.invoke(question)
                return rag_model_response
            except Exception as e:
                rag_model_error = print(f"Error: {e}")
                return rag_model_error