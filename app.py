from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from textblob import TextBlob
import joblib

# Load LLM
llm = Llama(
    model_path="/Users/satakshi/ai/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    n_ctx=1024,
    n_threads=8,
    use_mlock=False
)


symptom_model = joblib.load("symptom_checker_model.pkl")
symptom_labels = joblib.load("symptom_labels.pkl")


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
# Example function to check predicted probabilities

def predict_symptoms(user_input, model, sentiment, threshold=0.4):
    # If sentiment is positive or neutral, don't detect symptoms
    if sentiment == "positive" or sentiment == "neutral":
        return []
    
    # Get predicted probabilities for each symptom
    probabilities = model.predict_proba([user_input])
    
    detected_symptoms = []
    # Loop through symptoms and check if their probability exceeds the threshold
    for i, prob in enumerate(probabilities[0]):
        if prob >= threshold:
            detected_symptoms.append(symptom_labels[i])
    
    return detected_symptoms
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Detect sentiment
    sentiment = detect_sentiment(user_input.lower())
    
    # Predict symptoms based on the sentiment
    symptoms = predict_symptoms(user_input, symptom_model, sentiment)
    
    # Respond based on the greeting (e.g., "hi", "hello", "hey")
    greetings = ["hi", "hello", "hey"]
    if user_input.strip().lower() in greetings:
        return jsonify({
            "response": "Hey there!🍓How are you feeling today?",
            "sentiment": sentiment,
            "symptoms": symptoms
        })
    
    # Prepare the prompt for the LLM (Large Language Model)
    prompt = f"""
    You are a kind, emotionally supportive assistant.
    Based only on the user's message and detected sentiment, respond briefly with empathy and relevance. 
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
    
    # Get the response from the model
    reply = output["choices"][0]["text"].strip()

    # Return the response with sentiment and detected symptoms
    return jsonify({
        "response": reply,
        "sentiment": sentiment,
        "symptoms": symptoms
    })

if __name__ == "__main__":
    app.run(debug=True)