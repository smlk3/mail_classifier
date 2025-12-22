import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import re

# Yeni Modüller
from rag_utils import RAGHandler
from mail_utils import MailHandler

# ==========================================
# 👇 MODEL ADRESİN 👇
MODEL_ID = "efegokcekli/mail-asistani-v1"
# ==========================================

st.set_page_config(page_title="☁️ AI Hibrit Mail Analizi", layout="wide")

# --- INITIALIZATION ---
if 'rag' not in st.session_state:
    st.session_state['rag'] = RAGHandler()

if 'mail_handler' not in st.session_state:
    st.session_state['mail_handler'] = MailHandler()

if 'analyzed_df' not in st.session_state:
    st.session_state['analyzed_df'] = None

if 'imap_connected' not in st.session_state:
    st.session_state['imap_connected'] = False

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model():
    base_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # Windows'ta bitsandbytes bazen sorun çıkarabilir.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
        model = PeftModel.from_pretrained(base_model, MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None

# --- YARDIMCI FONKSİYONLAR ---
def parse_json(text):
    if "[/INST]" in text: text = text.split("[/INST]")[1]
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: return None
    return None

def analyze_text(text, model, tokenizer, use_rag=True):
    """Tek bir metni analiz eder."""
    
    rag_context = ""
    if use_rag:
        rag_context = st.session_state['rag'].get_relevant_context(text)

    prompt = f"""[INST] Analyze the email. Return JSON.
    RULES:
    1. "summary": English summary.
    2. "category": [WORK, PERSONAL, PROMOTION, FINANCE, SPAM].
    3. "entities": Extract names/dates.

    {rag_context}

    Email: "{str(text)[:1500]}" [/INST]"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=350, do_sample=True, temperature=0.1)
        res = parse_json(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return res, rag_context if rag_context else None
    except Exception as e:
        st.error(f"Analiz hatası: {e}")
        return None, None

# --- ARAYÜZ ---
st.title("🧠 AI Mail Analiz Merkezi")

# --- SIDEBAR (Mail Ayarları) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    use_rag_chk = st.checkbox("RAG (Hafıza) Kullan", value=True, help="Geçmiş düzeltmelerden öğrenerek analiz yapar.")
    
    st.divider()
    st.subheader("📧 Mail Sunucusu")
    imap_server = st.text_input("IMAP Sunucusu", "imap.gmail.com")
    email_user = st.text_input("Email", "ornek@gmail.com")
    email_pass = st.text_input("Şifre (Uygulama Şifresi)", type="password")
    
    if st.button("Bağlan / Yenile"):
        success, msg = st.session_state['mail_handler'].connect(imap_server, email_user, email_pass)
        if success:
            st.session_state['imap_connected'] = True
            st.success("Bağlandı!")
        else:
            st.error(f"Hata: {msg}")
            
    if st.session_state['imap_connected']:
        st.success(f"🟢 Bağlı: {email_user}")
        if st.button("Çıkış Yap"):
            st.session_state['mail_handler'].logout()
            st.session_state['imap_connected'] = False
            st.rerun()

    st.divider()
    stats = st.session_state['rag'].get_stats()
    st.info(f"📚 Bilgi Bankası: {stats} kayıt")


# GPU Kontrolü
if not torch.cuda.is_available():
    st.error("❌ GPU bulunamadı! Bu uygulama CUDA destekli bir NVIDIA GPU gerektirir.")
    st.stop()

with st.spinner("Model Hazırlanıyor..."):
    model, tokenizer = load_model()

if model is None: st.stop()

# --- SEKMELER (TABS) ---
tab_inbox, tab_single, tab_batch = st.tabs(["📨 Gelen Kutusu", "✍️ Tekli Analiz", "📂 Dosya Analizi"])

# ==========================================
# 0. SEKME: GELEN KUTUSU
# ==========================================
with tab_inbox:
    st.header("Gelen Kutusu")
    if not st.session_state['imap_connected']:
        st.warning("Lütfen sol menüden mail sunucusuna bağlanın.")
    else:
        if st.button("🔄 Mailleri Çek"):
            with st.spinner("Mailler alınıyor..."):
                st.session_state['fetched_emails'] = st.session_state['mail_handler'].fetch_latest_emails(limit=5)
        
        if 'fetched_emails' in st.session_state and st.session_state['fetched_emails']:
            for i, email in enumerate(st.session_state['fetched_emails']):
                with st.expander(f"{email['subject']} - {email['sender']}"):
                    st.write(f"**Date:** {email['date']}")
                    st.text_area("İçerik", email['body'], height=100, key=f"mail_body_{i}")
                    
                    if st.button(f"Analiz Et #{i}", key=f"analyze_btn_{i}"):
                        st.session_state[f'active_analysis_{i}'] = True
                    
                    if st.session_state.get(f'active_analysis_{i}'):
                        with st.spinner("Analiz ediliyor..."):
                            res, context = analyze_text(email['body'], model, tokenizer, use_rag=use_rag_chk)
                        
                        if res:
                            st.success(f"Kategori: {res.get('category')}")
                            st.info(res.get('summary'))
                            
                            # Feedback
                            col_f1, col_f2 = st.columns(2)
                            if col_f1.button("👍 Doğru", key=f"feed_good_{i}"):
                                # Doğruysa da ekleyebiliriz (Reinforcement) ama şimdilik sadece yanlışı düzeltiyoruz
                                st.toast("Geri bildirim için teşekkürler!")
                                
                            if col_f2.button("👎 Yanlış / Düzelt", key=f"feed_bad_{i}"):
                                st.session_state[f'correction_mode_{i}'] = True
                                
                            if st.session_state.get(f'correction_mode_{i}'):
                                with st.form(key=f"correct_form_{i}"):
                                    c_cat = st.selectbox("Doğru Kategori", ["WORK", "PERSONAL", "PROMOTION", "FINANCE", "SPAM"])
                                    c_sum = st.text_area("Doğru Özet (Opsiyonel)", value=res.get('summary'))
                                    if st.form_submit_button("💾 Kaydet ve Öğren"):
                                        st.session_state['rag'].add_feedback(email['body'], c_sum, c_cat)
                                        st.success("Bilgi bankasına eklendi! Gelecekteki analizlerde bu dikkate alınacak.")
                                        st.session_state[f'correction_mode_{i}'] = False
                                        st.rerun()

# ==========================================
# 1. SEKME: TEKLİ ANALİZ (MANUEL)
# ==========================================
with tab_single:
    st.header("Hızlı Mail Analizi")
    c1, c2 = st.columns([2, 1])
    with c1:
        manual_body = st.text_area("Mail İçeriği:", height=300, key="manual_input")

    with c2:
        if st.button("✨ ANALİZ ET", type="primary", use_container_width=True):
            if manual_body:
                with st.spinner("Yapay zeka okuyor..."):
                    res, context = analyze_text(manual_body, model, tokenizer, use_rag=use_rag_chk)

                    if res:
                        st.divider()
                        if context:
                            with st.expander("📚 RAG Kullanılan Bağlam (Benzer Örnekler)"):
                                st.text(context)
                                
                        st.success("✅ Analiz Tamamlandı")
                        m1, m2 = st.columns(2)
                        m1.metric("Kategori", res.get("category", "-"))
                        m2.metric("Varlıklar", str(len(res.get("entities", {}))) + " Adet")
                        st.info(res.get("summary", "-"))

                        # Feedback Section for Manual
                        st.markdown("---")
                        st.write("Sonuç doğru mu?")
                        fc1, fc2 = st.columns(2)
                        
                        # Burada basit bir session state toggle mantığı gerekebilir ama
                        # basitlik adına şimdilik direkt göstermek yerine expander kullanalım
                        with st.expander("👎 Hatalıysa Düzelt"):
                            with st.form("manual_correction"):
                                corr_cat = st.selectbox("Doğru Kategori", ["WORK", "PERSONAL", "PROMOTION", "FINANCE", "SPAM"])
                                corr_sum = st.text_area("Doğru Özet", value=res.get("summary"))
                                if st.form_submit_button("Öğret"):
                                    st.session_state['rag'].add_feedback(manual_body, corr_sum, corr_cat)
                                    st.success("Öğrenildi!")

# ==========================================
# 2. SEKME: TOPLU ANALİZ (DOSYA)
# (Önceki kodun aynısı, sadece fonksiyon çağrısı updated)
# ==========================================
with tab_batch:
    st.header("Toplu Dosya İşleme")
    uploaded_file = st.file_uploader("Excel veya CSV Yükle", type=["csv", "xlsx"], key="file_upl")

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        col1, col2, col3 = st.columns(3)
        with col1: c_subj = st.selectbox("Konu:", df.columns, key="b_sub")
        with col2: c_send = st.selectbox("Gönderen:", df.columns, key="b_send")
        with col3: c_body = st.selectbox("İçerik:", df.columns, index=len(df.columns)-1, key="b_body")

        if st.button("🚀 DOSYAYI ANALİZ ET", type="primary"):
            progress_bar = st.progress(0)
            new_data = []
            total = len(df)

            for index, row in df.iterrows():
                progress_bar.progress((index + 1) / total)
                # Batch analizde de RAG kullanılabilir
                result, _ = analyze_text(row[c_body], model, tokenizer, use_rag=use_rag_chk)

                summary, category, entities = "Error", "Unknown", "{}"
                if result:
                    summary = result.get("summary", "-")
                    category = result.get("category", "-")
                    entities = str(result.get("entities", {}))

                new_data.append({
                    "Subject": row[c_subj],
                    "Sender": row[c_send],
                    "Summary": summary,
                    "Category": category,
                    "Entities": entities
                })

            st.session_state['analyzed_df'] = pd.DataFrame(new_data)
            st.success("Analiz Bitti")
            st.dataframe(st.session_state['analyzed_df'])

