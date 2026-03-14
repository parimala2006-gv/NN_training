import os
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model = None

def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model("model/iris_model.h5")
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    model = get_model()

    prediction = model.predict(features)

    result = np.argmax(prediction)

    classes = ["Setosa", "Versicolor", "Virginica"]

    return render_template(
        "index.html",
        prediction_text="Predicted Flower: " + classes[result]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)