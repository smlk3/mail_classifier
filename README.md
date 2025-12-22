# 📧 Local AI Email Assistant (Mistral-7B + RAG)

Bu proje, yerel makinenizde veya Google Colab üzerinde çalışabilen, **Mistral-7B** tabanlı bir yapay zeka e-posta asistanıdır. E-postalarınızı analiz eder, özetler ve kategorize eder. **RAG (Retrieval-Augmented Generation)** kullanarak, yaptığınız düzeltmelerden öğrenir.

## 🌟 Özellikler
*   **Gizlilik Odaklı:** Verileriniz 3. parti API'lere gitmez.
*   **Hafıza (RAG):** Yanlış analizleri düzelttiğinizde sistem öğrenir.
*   **Türkçe & İngilizce:** İki dilde de etkili çalışır.

## 🚀 Google Colab'da Çalıştırma (ÖNERİLEN)
Eğer güçlü bir GPU'nuz yoksa veya kurulumla uğraşmak istemiyorsanız:

1.  Bu projeyi GitHub'da açın.
2.  `mistral_colab.ipynb` dosyasına tıklayın.
3.  "Open in Colab" butonuna tıklayın (veya dosya içeriğini Colab'a kopyalayın).
4.  Gerekli alanları (Ngrok Token vb.) doldurup çalıştırın.

## 💻 Yerel Kurulum (Gelişmiş)
**Gereksinimler:** NVIDIA GPU (Min 6GB VRAM), Python 3.10+

1.  Repoyu klonlayın:
    ```bash
    git clone https://github.com/KULLANICI_ADI/REPO_ADI.git
    cd REPO_ADI
    ```

2.  Gereksinimleri kurun:
    ```bash
    pip install -r requirements.txt
    ```

3.  Uygulamayı başlatın:
    ```bash
    streamlit run app.py
    ```

## 🛠 Kullanılan Teknolojiler
*   **Model:** `mistralai/Mistral-7B-Instruct-v0.2` (4-bit Quantized)
*   **Arayüz:** Streamlit
*   **Vektör DB:** ChromaDB
