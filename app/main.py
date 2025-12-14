from fastapi import FastAPI, Body
import mlflow.sklearn

app = FastAPI()

model = mlflow.sklearn.load_model("models:/IrisModel@production")

@app.post("/predict")
def predict(data: list = Body(...)):
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}