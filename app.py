# install libraries ---
# pip install fastapi uvicorn 

# 1. Library imports
import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
import pickle

# 2. Create the app object
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. load the model
rgModel = pickle.load(open("lr.pkl", "rb"))

# 4. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get("/predictcharges")
def gePredictcharges(age:int,sex:int, bmi:float,children:int,smoker:int, region:int):
    prediction = rgModel.predict([[age,sex,bmi,children,smoker,region]])
    return {'charges': prediction[0]}

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0')
    
# uvicorn app:app --host 0.0.0.0 --port 80
# http://127.0.0.1/predictcharges?age=19&sex=1&bmi=27.900&children=0&smoker=0&region=3