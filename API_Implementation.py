# import sys
# import os

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to the virtual environment's site-packages directory
# venv_site_packages = os.path.join(script_dir, ".venv", "Lib", "site-packages")

# # Add the virtual environment's site-packages directory to sys.path
# sys.path.append(venv_site_packages)

import fastapi
import uvicorn
import ssl
import tensorflow
from fastapi import *
import os
import deepFaceFW

import PredictionModel
# import PredicNew
import emotionDetection
# import ageDetection
import uuid
# import pandas as pd

IMAGEDIR = "Images/"

app = FastAPI()


def GenderMapper(gender_):
    genders = ['m', 'f']
    if gender_ == 'female':
        return genders[1]
    else:
        return genders[0]


@app.get('/')
def hello_world():
    return "Hello World" # to test


@app.post("/demographicsImage")
async def create_upload_file(
    # file: UploadFile = File(...)
    ):
    # file.filename = f"{uuid.uuid4()}.jpg"
    # contents = await file.read()  

    # # example of how you can save the file
    # with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
    #     f.write(contents)

    # filepath = IMAGEDIR + file.filename
    filepath = IMAGEDIR + "A.jpg"

    # df = pd.DataFrame(columns = ['path'])
    # Add records to dataframe using the .loc function
    # df.loc[0] = [filepath] 
    # Result = ModelLoading.finalImageOutput(df)
    gender = PredictionModel.predict_gender(filepath)
    emotion = emotionDetection.predict_emotion(filepath)
    # age = ageDetection.predict_age()
    # res =  PredicNew.finalImageOutput()
    # Result = ModelLoading.finalImageOutput(contents)
    age = deepFaceFW.predict_age(filepath)
   
    # return age
    return {"Gender": gender, 
            "Age": "20", 
            "Emotion": emotion['main_emotion']
            }
    # return {"Gender": "Male", 
    #         "Age":"20", 
    #         "Emotion": "normal"
    #         }


