from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news_text = request.form.get("news", "").strip()

        if len(news_text.split()) < 5:
            prediction = "⚠ Please enter a complete news article."
        else:

            transformed_text = vectorizer.transform([news_text])

            result = model.predict(transformed_text)

            if result[0] == 1:
                prediction = "✅ REAL NEWS"
            else:
                prediction = "❌ FAKE NEWS"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)