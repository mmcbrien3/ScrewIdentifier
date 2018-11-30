import parameter_extractor
import os, csv

IMAGE_SUPER_FOLDER = r"C:\Users\Valued Customer\Documents\Senior Design\Prototype 3 Image Database"
DATABASE_LOC = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\database3.csv"

for photo_folder in os.listdir(IMAGE_SUPER_FOLDER):
    print(photo_folder)
    for image in os.listdir(IMAGE_SUPER_FOLDER + "\\" + photo_folder):
        parameters = parameter_extractor.get_parameters(IMAGE_SUPER_FOLDER + "\\" + photo_folder + "\\" + image)[0:-1]
        print(parameters)
        parameters = ["%.5f" % a for a in parameters]
        parameters = [photo_folder] + parameters
        with open(DATABASE_LOC, "a", newline="") as database:
            csvwriter = csv.writer(database, delimiter=',')
            csvwriter.writerow(parameters)


