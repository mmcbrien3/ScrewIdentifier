from sklearn.externals import joblib
import parameter_extractor

# This is where the model is loaded from
MODEL_PATH = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\trained_model.sav"


def load_model():
    return joblib.load(MODEL_PATH)

# This function returns a prediction based on the inputted test values
# Test values must be a 2d array (i.e. [[actual values]])
# Current headers are: 'total_screw_length' 'body_screw_length' 'body_screw_width'
#  'head_screw_width' 'head_max_loc' 'thread_count'
def predict_from_values(test_values):
    trained_model = load_model()
    prediction = trained_model.predict(test_values)

    print(prediction)
    return prediction

# This function returns a prediction based on the inputted image
# It calls the get_parameters function to extract parameters from the image
def predict_from_image(image_loc):
    values = parameter_extractor.get_parameters(image_loc)
    predictionValues = [values[:-1]]
    filledImage = values[-1]
    return predict_from_values(predictionValues), filledImage


