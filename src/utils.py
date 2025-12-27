import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

model = None

class PredictExamScore():
    def __init__(self):
        pass

    def predict_score(self, user_input_data):
        self.load_saved_model()
        self.data = user_input_data
        self.create_test_df()
        self.predict = model.predict(self.test_df.values)
        #print("Predicted Score is :",self.predict)
        return np.round(float(self.predict[0][0]),4)

    def create_test_df(self):
        with open("artifacts/feature_names.pkl", "rb") as f:
            feature = pickle.load(f)
        test_array = np.zeros((1,len(feature)))

        test_array[0,feature.index("age")] = float(self.data["age"])
        test_array[0,feature.index("study_hours")] = float(self.data["study_hours"])
        test_array[0,feature.index("class_attendance")] = float(self.data["class_attendance"])
        test_array[0,feature.index("sleep_hours")] = float(self.data["sleep_hours"])

        gender = f'gender_{self.data["gender"]}'
        gender_index = feature.index(gender)
        #print("gender_index", gender_index)
        test_array[0,gender_index] = 1
 
        course = f'course_{self.data["course"]}'
        course_index = feature.index(course)
        test_array[0,course_index] = 1

        internet_access = f'internet_access_{self.data["internet_access"]}'
        internet_access_index = feature.index(internet_access)
        test_array[0,internet_access_index] = 1

        sleep_quality = f'sleep_quality_{self.data["sleep_quality"]}'
        sleep_quality_index = feature.index(sleep_quality)
        test_array[0,sleep_quality_index] = 1

        study_method = f'study_method_{self.data["study_method"]}'
        study_method_index = feature.index(study_method)
        test_array[0,study_method_index] = 1

        facility_rating = f'facility_rating_{self.data["facility_rating"]}'
        facility_rating_index = feature.index(facility_rating)
        test_array[0,facility_rating_index] = 1

        exam_difficulty = f'exam_difficulty_{self.data["exam_difficulty"]}'
        exam_difficulty_index = feature.index(exam_difficulty)
        test_array[0,exam_difficulty_index] = 1

        #print("test_array", test_array)
        self.test_df = pd.DataFrame(test_array, columns = feature)

    def load_saved_model(self):
        global model
        filepath = os.path.join("artifacts", "AnnExamScoreModel.keras")
        if model is None:
            model = load_model(filepath)
        return model
