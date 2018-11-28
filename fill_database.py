import parameter_extractor
import os, csv

IMAGE_SUPER_FOLDER = r"C:\Users\Valued Customer\Documents\Senior Design\IMAGE_PROCESSING\image_database\prototype_2"
DATABASE_LOC = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\database2.csv"

for photo_folder in os.listdir(IMAGE_SUPER_FOLDER):
    print(photo_folder)
    for image in os.listdir(IMAGE_SUPER_FOLDER + "\\" + photo_folder):
        parameters = parameter_extractor.get_parameters(IMAGE_SUPER_FOLDER + "\\" + photo_folder + "\\" + image)
        print(parameters)
        parameters = ["%.5f" % a for a in parameters]
        parameters = [photo_folder] + parameters
        with open(DATABASE_LOC, "a", newline="") as database:
            csvwriter = csv.writer(database, delimiter=',')
            csvwriter.writerow(parameters)


