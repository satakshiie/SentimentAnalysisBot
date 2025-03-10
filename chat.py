from llama_cpp import Llama
from textblob import TextBlob
import time

# Load your model (Mistral in this case)
llm = Llama(
    model_path="/Users/satakshi/ai/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    n_ctx=512,           # Reduced context for quicker eval
    n_threads=4,         # Use more threads (adjust based on your CPU)
    use_mlock=False      # Avoid memory lock issues
)
def detect_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

def thinking_animation():
    print("Bot is thinking", end="", flush=True)
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

print("🦙 Sentiment-Aware Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break

    sentiment = detect_sentiment(user_input)
    print(f"[ Sentiment detected: {sentiment} ]")

    thinking_animation()

    prompt = (
        f"You are a supportive and understanding assistant.\n"
        f"The user feels {sentiment}. Respond accordingly.\n"
        f"User: {user_input}\n"
        f"Assistant:"
    )

    output = llm(prompt, max_tokens=32, stop=["</s>"], echo=False)
    print("Bot:", output["choices"][0]["text"].strip(), "\n")