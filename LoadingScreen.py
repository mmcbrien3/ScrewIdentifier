import tkinter
import predictor
import threading, time
from PIL import ImageTk, Image

bgColor = "#444444"


class LoadingScreen(object):

    def __init__(self, root, width, height):
        self.root = root
        self.width = width
        self.height = height
        self.prediction = None
        self.filledImage = None

        self.takeImage()
        self.predictThread = threading.Thread(target=self.predict)
        self.predictThread.start()

    def takeImage(self):
        self.image = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\Classifier Test 15Nov2018\image4.jpg"

    def predict(self):
        self.prediction, self.filledImage = predictor.predict_from_image(self.image)

    def start(self):

        startTime = time.time()

        self.clearScreen()

        bufferingFrames = [tkinter.PhotoImage(file='buffering.gif',format = 'gif -index %i' %(i)) for i in range(8)]

        scaleWidth = bufferingFrames[0].width() / (self.width / 5)
        scaleHeight = bufferingFrames[0].height() / (self.height / 5)
        bufferingFrames = [frame.subsample(round(scaleWidth), round(scaleHeight)) for frame in bufferingFrames]

        bufferingFrameX = round(self.width / 2 - bufferingFrames[0].width() / 2)
        bufferingFrameY = round(self.height / 2 - bufferingFrames[0].height() / 2)

        bufferingFrameLabel = tkinter.Label(self.root,
                                            image=bufferingFrames[0],
                                            bg=bgColor)

        textFrameWidth = round(self.width / 5)
        textFrameHeight = round(self.height / 20)

        textFrame = tkinter.Frame(self.root,
                                  width=textFrameWidth,
                                  height=textFrameHeight,
                                  bg=bgColor)

        textFrameX = round(self.width / 2 - textFrameWidth / 2)
        textFrameY = round(bufferingFrameY + bufferingFrames[0].height() + textFrameHeight / 2)

        textFrame.pack_propagate(0)
        textLabel = tkinter.Label(textFrame,
                                  bg=bgColor,
                                  fg="white",
                                  text="Timer:  ")

        img = Image.open(self.image)
        imageWidth = round(self.width / 6)
        imageHeight = round(self.height / 6)

        img = img.resize((imageWidth, imageHeight), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        imageLabel = tkinter.Label(self.root, image=img)
        imageLabel.image = img

        imageX = round(self.width / 2 - imageWidth / 2)
        imageY = round(textFrameY + textFrameHeight + imageHeight / 3)

        def update(index, totalUpdates):
            frame = bufferingFrames[index]
            if index < 7:
                index += 1
            else:
                index = 0
            totalUpdates += 1
            if not self.predictThread.is_alive():
                self.startResultsScreen()
            else:
                bufferingFrameLabel.configure(image=frame)
                textLabel.configure(text="Timer: " +str(int(time.time()-startTime)))
                self.root.after(100, update, index, totalUpdates)

        bufferingFrameLabel.place(x=bufferingFrameX, y=bufferingFrameY)
        textFrame.place(x=textFrameX, y=textFrameY)
        textLabel.pack(fill="both", expand=1)
        imageLabel.place(x=imageX, y=imageY)

        self.root.after(0, update, 0, 0)

    def clearScreen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def startResultsScreen(self):
        import ResultsScreen
        self.resultsScreen = ResultsScreen.ResultsScreen(self.root, self.width, self.height, self.image, self.filledImage, self.prediction)
        self.resultsScreen.start()
