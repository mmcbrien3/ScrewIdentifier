import os
import predictor
import shutil
cloudImageFolder = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\Cloud Image Folder"
piResultsFolder = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\Pi Results Folder"
predictionLocation = piResultsFolder + r"\\prediction.txt"
imageLocation = piResultsFolder + r"\\filled_image.bmp"
class CloudController(object):

    def __init__(self):
        while True:
            self.waitForImage()
            print("image found")
            prediction, filled_image = self.makePrediction()
            self.sendResults(prediction, filled_image)
            self.deleteImage()

    def waitForImage(self):

        while len(os.listdir(cloudImageFolder)) == 0:
            pass

    def makePrediction(self):
        prediction, filled_image = predictor.predict_from_image(cloudImageFolder + r"\\taken_image.jpg")
        return prediction, filled_image


    def sendResults(self, prediction, filled_image):
        #TODO: SSH prediction and .bmp to py
        shutil.copyfile(filled_image, piResultsFolder + r"\\filled_image.bmp")
        predictionFile = open(predictionLocation, "w")
        predictionFile.writelines((prediction))
        predictionFile.close()

    def deleteImage(self):
        for file in os.listdir(cloudImageFolder):
            os.remove(cloudImageFolder + r"\\" + file)


startingPoint = CloudController()