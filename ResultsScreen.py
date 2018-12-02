import tkinter
from PIL import ImageTk, Image
bgColor = "#444444"

stockImageFolder = r"C:\Users\Valued Customer\Documents\Senior Design\Result Images\\"
class ResultsScreen(object):

    def __init__(self, root, width, height, imageLoc, filledImage, results):
        self.root = root
        self.width = width
        self.height = height
        self.results = results
        self.image = imageLoc
        self.filledImage = filledImage
        self.stockImage = stockImageFolder + results[0] + ".jpg"
        print(filledImage)

    def start(self):
        self.clearScreen()

        restartButtonFrameWidth = round(self.width / 10)
        restartButtonFrameHeight = round(self.height / 20)

        restartButtonFrame = tkinter.Frame(self.root,
                                           bg=bgColor,
                                           width=restartButtonFrameWidth,
                                           height=restartButtonFrameHeight)

        restartButton = tkinter.Button(restartButtonFrame,
                                       bg="white",
                                       fg=bgColor,
                                       activebackground="gray",
                                       activeforeground="white",
                                       text="Restart",
                                       command=self.restartProcess)

        restartButtonFrame.place(x=0, y=0)
        restartButton.config(width=100, height=100)
        restartButton.place(relx=0.5, rely=0.5, anchor="center")

        resultFrameOne = ""
        if self.filledImage is not None:
            resultFrameOne = self.makeResultFrame(self.image, self.filledImage, self.stockImage, self.results)
        else:
            resultFrameOne = self.makeFailureFrame(self.results)

        xPosition = self.width/10
        xPosition += self.width / 3
        yPosition = self.height/3
        resultFrameOne.place(x=round(xPosition),
                             y=round(yPosition))

        self.root.mainloop()

    def makeFailureFrame(self, result):
        frame = tkinter.Frame(self.root,
                              width=round(self.width / 5),
                              height=round(self.height / 3),
                              bg=bgColor)

        textLabel = tkinter.Label(frame,
                              bg=bgColor,
                              fg="white",
                              text=result)

        textLabel.pack()
        return frame

    def makeResultFrame(self, capturedImage, filledImage, stockImage, resultData):
        frame = tkinter.Frame(self.root,
                              width=round(self.width/5),
                              height=round(self.height/3),
                              bg=bgColor)

        img = Image.open(capturedImage)
        imageWidth = round(self.width/5)
        imageHeight = round(self.height*2/15)

        img = img.resize((imageWidth, imageHeight), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        capturedImageLabel = tkinter.Label(frame, image=img)
        capturedImageLabel.image = img

        img = Image.open(filledImage)
        imageHeight = round(self.height * 2 / 15)
        imageWidth = round(img.width * (imageHeight / img.height))

        img = img.resize((imageWidth, imageHeight), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        filledImageLabel = tkinter.Label(frame, image=img)
        filledImageLabel.image = img

        img = Image.open(stockImage)
        imageHeight = round(self.height * 2 / 15)
        imageWidth = round(img.width * (imageHeight / img.height))

        img = img.resize((imageWidth, imageHeight), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        stockImageLabel = tkinter.Label(frame, image=img)
        stockImageLabel.image = img

        textLabel = tkinter.Label(frame,
                                  bg=bgColor,
                                  fg="white",
                                  text=self.formatResultText(resultData[0]))

        capturedImageLabel.pack()
        filledImageLabel.pack()
        stockImageLabel.pack()
        textLabel.pack()
        return frame

    def formatResultText(self, res):
        name = res[:res.index("#")]
        formattedName = name[0].capitalize()

        underscore = False
        for s in name[1:]:
            if s == "_":
                underscore = True
                formattedName += " "
            else:
                underscore = False
                if underscore:
                    formattedName += s.capitalize()
                else:
                    formattedName += s

        numberAndLength = res[res.index("#"):]

        number = numberAndLength[:numberAndLength.index("x")]
        length = numberAndLength[numberAndLength.index("x")+1:]

        fullInches =  length[:length.index("p")]
        inchesInHundredths = "0." + length[length.index("p")+1:]
        fractionInches = float.as_integer_ratio(float(inchesInHundredths)/100.0)
        fractionInches = str(fractionInches[0]) + "/" + str(fractionInches[1])

        formatted = number + " x "
        if not fullInches == "0":
            formatted += fullInches
        if not fractionInches == "0/1":
            formatted += "-" + fractionInches
        formatted += " in. " + formattedName

        return formatted

    def clearScreen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def restartProcess(self):
        import HomeScreen
        self.homeScreen = HomeScreen.HomeScreen(self.root, self.width, self.height)

        self.homeScreen.start()


