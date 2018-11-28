import tkinter
from PIL import ImageTk, Image
bgColor = "#444444"


class ResultsScreen(object):

    def __init__(self, root, width, height, imageLoc, filledImage, results):
        self.root = root
        self.width = width
        self.height = height
        self.results = results
        self.image = imageLoc
        self.filledImage = filledImage
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

        resultFrameOne = self.makeResultFrame(self.image, self.filledImage, self.results)

        xPosition = self.width/10
        xPosition += self.width / 3
        yPosition = self.height/3
        resultFrameOne.place(x=round(xPosition),
                             y=round(yPosition))

        self.root.mainloop()

    def makeResultFrame(self, capturedImage, filledImage, resultData):
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

        textLabel = tkinter.Label(frame,
                                  bg=bgColor,
                                  fg="white",
                                  text=resultData)

        capturedImageLabel.pack()
        filledImageLabel.pack()
        textLabel.pack()
        return frame


    def clearScreen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def restartProcess(self):
        import HomeScreen
        self.homeScreen = HomeScreen.HomeScreen(self.root, self.width, self.height)

        self.homeScreen.start()


