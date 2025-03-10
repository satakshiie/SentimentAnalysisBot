# 🌸 Sentiment-Aware Chatbot 🌸

A kind and emotionally supportive AI chatbot that uses sentiment analysis to respond empathetically to user messages.
an avid listener,a comfort giver-all in one.

## 💡 Features
- 🤗 Local LLM using TinyLLaMA
- 💬 Sentiment detection using TextBlob
- 🎨 Friendly web interface (Flask + HTML)
- 🧠 Responses tailored to user emotion

## 🧪 How It Works

1. User inputs a message.
2. TextBlob analyzes sentiment (positive, negative, or neutral).
3. Prompt is crafted and sent to the TinyLLaMA model.
4. Model generates an emotionally relevant response.
5. Displayed in a chat-like UI.

## 🛠️ Tech Stack

- **Frontend**: HTML/CSS (simple interface)
- **Backend**: Python (Flask)
- **LLM**: TinyLLaMA (local `.gguf` model)
- **NLP**: TextBlob for sentiment

## ⚠️ Note on Model File

To keep the repo light, the `.gguf` model file is **excluded via `.gitignore`.**  
You'll need to [download TinyLLaMA](https://huggingface.co/cg123/TinyLlama-1.1B-Chat-v1.0-GGUF) and place it in your `ai/` folder manually.

---

## 🚀 Running the App

```bash
flask run
or
python app.py (whatever ver of python you have)