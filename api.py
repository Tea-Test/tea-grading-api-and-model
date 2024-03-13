from fastapi import FastAPI, UploadFile, File
from predict import model_train
import subprocess

app = FastAPI()

model_data = None

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    global model_data
    if model_data is None:
        model_data = model_train()

    with open(f"./predict/{file.filename}", "wb+") as f:
        f.write(file.file.read())

    result = model_data(f"./predict/{file.filename}")
    print(result)
    names_dict = result[0].names
    probs = result[0].probs
    index = probs.top1
    return {"result": names_dict[index]}


def run_uvicorn():
    uvicorn_command = [
        "uvicorn",
        "api:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ]
    subprocess.run(uvicorn_command, check=True)

if __name__ == "__main__":
    run_uvicorn()
