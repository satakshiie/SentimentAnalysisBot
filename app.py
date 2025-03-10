from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from textblob import TextBlob

# Load LLM
llm = Llama(
    model_path="/Users/satakshi/ai/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    n_ctx=1024,
    n_threads=8,
    use_mlock=False
)

app = Flask(__name__)

def detect_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    sentiment = detect_sentiment(user_input.lower())

    # Optional shortcut: handle common greetings manually
    greetings = ["hi", "hello", "hey"]
    if user_input.strip().lower() in greetings:
        return jsonify({
            "response": "Hey there!🍓How are you feeling today?",
            "sentiment": sentiment
        }) 

    prompt = f"""
   You are a kind, emotionally supportive assistant.

Based only on the users message and detected sentiment, respond briefly with empathy and relevance. 
Do not assume more than what is shared.

   Message: "{user_input}"
   Sentiment: {sentiment}

   Response:"""
  
    output = llm(
    prompt,
    max_tokens=100,
    temperature=0.5,
    top_p=0.9,
    stop=["\n", "User:", "Message:"],
    echo=False
)
    reply = output["choices"][0]["text"].strip()
    return jsonify({  
    "response": reply,
    "sentiment": sentiment
    })

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.5,
        top_p=0.9,
        stop=["\n", "User:", "Message:"],
        echo=False
    )
    reply = output["choices"][0]["text"].strip()
    return jsonify({
        "response": reply,
        "sentiment": sentiment
    })
if __name__ == "__main__":
    app.run(debug=True)