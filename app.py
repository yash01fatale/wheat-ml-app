from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

models = {
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb"))
}

accuracies = pickle.load(open("models/accuracy.pkl", "rb"))

# Change labels if needed
label_map = {
    0: "🌾 Wheat Type A",
    1: "🌾 Wheat Type B"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    accuracy = None
    model_name = None

    if request.method == "POST":
        features = [
            float(request.form[f"f{i}"]) for i in range(1, 8)
        ]

        model_name = request.form["model"]
        model = models[model_name]

        result = model.predict([features])[0]
        prediction = label_map.get(result, result)
        accuracy = accuracies[model_name]

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy=accuracy,
        model_name=model_name
    )

if __name__ == "__main__":
    app.run()
