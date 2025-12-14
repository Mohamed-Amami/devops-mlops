from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI()

model = mlflow.sklearn.load_model("models:/IrisModel@production")

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}
