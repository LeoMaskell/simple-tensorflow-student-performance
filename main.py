import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

# load model - see ./model/modelv1.keras
filepath = 'main dependencies/model/modelV1.keras'
model = keras.saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)

print(model.summary())

def getDataAndEval():
    print('for each field you MUST input the exact option or a number if no options are given.')
    age = input('age: \n')
    gender = input('gender (male/female/other): \n')
    course = input('course (ba/bba/bca/b.com/b.sc/b.tech/diploma): \n')
    study_hours = input('study hours: \n')
    class_attendance = input('class attendace (0-100, float) \n')
    internet_access = input('internet access (yes/no): \n')
    sleep_hours = input('sleep hours: \n')
    sleep_quality = input('sleep quality (good/poor/average): \n')
    study_method = input('study method (coaching/group study/mixed/online videos/self-study): \n')
    facility_rating = input('facility rating (low/medium/high): \n')
    exam_dificulty = input('exam dificulty (easy/moderate/hard): \n')

    # turn into a 1 row df
    ds = pd.DataFrame([{
        'age': age,
        'gender': gender.strip() if isinstance(gender, str) else gender,
        'course': course.strip() if isinstance(course, str) else course,
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'internet_access': internet_access.strip() if isinstance(internet_access, str) else internet_access,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality.strip() if isinstance(sleep_quality, str) else sleep_quality,
        'study_method': study_method.strip() if isinstance(study_method, str) else study_method,
        'facility_rating': facility_rating.strip() if isinstance(facility_rating, str) else facility_rating,
        # NOTE: training column is 'exam_difficulty' (with 'ulty')
        'exam_difficulty': exam_dificulty.strip() if isinstance(exam_dificulty, str) else exam_dificulty
    }])

    # ensure the numeric stuff is all float values
    for col in ['age', 'study_hours', 'class_attendance', 'sleep_hours']:
        ds[col] = pd.to_numeric(ds[col], errors='coerce').astype(float)

    # make the dataset match the set required by the model
    input_fields = ['age','gender','course','study_hours','class_attendance','internet_access','sleep_hours','sleep_quality','study_method','facility_rating','exam_difficulty']
    train_df = pd.read_csv('main dependencies/dataset.csv', usecols=input_fields) # get the input columns
    train_columns = pd.get_dummies(train_df).columns

    input_df = pd.get_dummies(ds).reindex(columns=train_columns, fill_value=0)
    input_array = input_df.to_numpy(dtype=float)
    print(model.predict(input_array))

getDataAndEval()