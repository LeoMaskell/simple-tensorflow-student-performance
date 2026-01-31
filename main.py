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

    # build a single-row df with pandas
    row = pd.DataFrame([{
        'age': None,
        'gender': gender.strip() if isinstance(gender, str) else gender,
        'course': course.strip() if isinstance(course, str) else course,
        'study_hours': None,
        'class_attendance': None,
        'internet_access': internet_access.strip() if isinstance(internet_access, str) else internet_access,
        'sleep_hours': None,
        'sleep_quality': sleep_quality.strip() if isinstance(sleep_quality, str) else sleep_quality,
        'study_method': study_method.strip() if isinstance(study_method, str) else study_method,
        'facility_rating': facility_rating.strip() if isinstance(facility_rating, str) else facility_rating,
        # note: training column is 'exam_difficulty' (with 'ulty')
        'exam_difficulty': exam_dificulty.strip() if isinstance(exam_dificulty, str) else exam_dificulty
    }])

    # convert numeric values
    def _to_float(v, name):
        try:
            return float(v)
        except Exception:
            print(f"Warning: could not parse numeric value for {name}: {v!r}. Using 0.0")
            return 0.0

    row.loc[0, 'age'] = _to_float(age, 'age')
    row.loc[0, 'study_hours'] = _to_float(study_hours, 'study_hours')
    row.loc[0, 'class_attendance'] = _to_float(class_attendance, 'class_attendance')
    row.loc[0, 'sleep_hours'] = _to_float(sleep_hours, 'sleep_hours')

    # load training feature columns from the dataset and align it with the input values
    try:
        _train = pd.read_csv('main dependencies/dataset.csv')
        _train = _train.drop(['student_id','exam_score'], axis=1)
        _REF_COLS = pd.get_dummies(_train).columns
    except Exception as e:
        print('Warning: could not derive reference columns from dataset.csv:', e)
        _REF_COLS = None

    if _REF_COLS is not None:
        proc = pd.get_dummies(row)
        proc = proc.reindex(columns=_REF_COLS, fill_value=0)
        inputData = proc.values.astype(np.float32)
    else:
        # use numeric columns only as a fallback
        inputData = row.select_dtypes(include=[np.number]).values.astype(np.float32)

    print('\nInput array shape:', inputData.shape)
    print(inputData)

    # predict with the values
    preds = model.predict(inputData)
    print('\nPrediction:')
    print(preds)

getDataAndEval()