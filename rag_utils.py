import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os
import uuid

class RAGHandler:
    def __init__(self, persist_directory="rag_db"):
        """RAG Handler başlatır. Embedding modeli ve Vektör DB yüklenir."""
        self.persist_directory = persist_directory
        
        # Embedding Model (Hafif ve hızlı)
        # Ilk calistirmada indirir.
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Chroma Client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Koleksiyon (Tablo gibi)
        self.collection = self.client.get_or_create_collection(name="mail_knowledge_base")

    def add_feedback(self, email_text, correct_summary, correct_category):
        """Kullanıcının düzelttiği veriyi veritabanına ekler."""
        doc_id = str(uuid.uuid4())
        
        # Metadata olarak saklayalım
        meta = {
            "category": correct_category,
            "summary": correct_summary,
            "source": "user_feedback"
        }
        
        # DB'ye ekle
        # Vektörü otomatik oluşturmazsa, manuel embed edebiliriz ama 
        # ChromaDB default embedding fonksiyonu kullanırsa otomatik yapar.
        # Biz kontrol bizde olsun diye manuel embed edip veriyoruz.
        embedding = self.embed_model.encode(email_text).tolist()
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[email_text],
            metadatas=[meta]
        )
        
        # Loglama / Dataset Yedekleme (JSONL)
        self.save_to_jsonl(email_text, correct_summary, correct_category)
        return True

    def get_relevant_context(self, email_text, k=3):
        """Benzer geçmiş mailleri ve sonuçlarını getirir."""
        try:
            embedding = self.embed_model.encode(email_text).tolist()
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k
            )
            
            context_str = ""
            if results['documents']:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                
                context_str = "--- BENZER GECMIS ORNEKLER (REFERANS) ---\n"
                for i, doc in enumerate(documents):
                    cat = metadatas[i].get("category", "Unknown")
                    summ = metadatas[i].get("summary", "")
                    # Prompt'u şişirmemek için mailin sadece başını alalım
                    short_mail = doc[:200].replace("\n", " ")
                    
                    context_str += f"ORNEK {i+1}: Mail: '{short_mail}...', DOGRU KATEGORI: {cat}, OZET: {summ}\n"
                context_str += "--- REFERANS BITIS ---\n"
                
            return context_str
        except Exception as e:
            print(f"RAG Error: {e}")
            return ""

    def save_to_jsonl(self, text, summary, category):
        """İleride Fine-Tune yapmak için veri biriktirir."""
        data = {
            "text": text,
            "response": {
                "summary": summary,
                "category": category
            }
        }
        with open("dataset_feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def get_stats(self):
        """DB istatistikleri"""
        return self.collection.count()
