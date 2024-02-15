from fastapi import FastAPI, File, UploadFile
import uvicorn
from starlette.responses import RedirectResponse
from keras.utils import get_file
from keras.models import load_model
import cv2
import numpy as np

app_desc = """<h2>Furniture Classification using CNN: UW</h2>"""

app = FastAPI(title='Furniture Classification using CNN: UW', description=app_desc)
@app.get('/')
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post('/predict/image')
async def predict_image(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin=image_link
    )
    img = cv2.imread(img_path)

    model = load_model("my_model.h5")

    res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # Converting to grayscale
    gray = np.array(gray)
    gray = gray.reshape(1, 128, 128, 1).astype('float32')
    gray = (gray - 127.5) / 127.5
    result = np.argmax(model.predict(gray))

    if result == 0:
        return {"message": "BED"}
    elif result == 1:
        return {"message": "CHAIR"}
    elif result == 2:
        return {"message": "SOFA"}
    else:
        return {"message": "MISTAKE!"}




if __name__ == "__main__":
    uvicorn.run(app,port=8080,host='0.0.0.0')

