import tkinter, os
screenName = "Home Depot Object Identifier"

# The Home Depot Logo Color (255, 101, 1)
bgColor = "#444444"
bgImageFilename = "THD_logo.gif"


class HomeScreen(object):

    def __init__(self, root, width, height):
        self.root = root
        self.width = width
        self.height = height

    def start(self):
        self.clearScreen()

        thdImage = tkinter.PhotoImage(file=bgImageFilename)
        scaleWidth = thdImage.width() / (self.width / 5)
        scaleHeight = thdImage.height() / (self.height / 5)
        thdImage.subsample(round(scaleWidth), round(scaleHeight))

        thdImageX = round(self.width/2 - thdImage.width()/2)
        thdImageY = round(self.height/2 - thdImage.height()/2)

        thdStartButton = tkinter.Button(self.root,
                                text="start",
                                command=None, image=thdImage)

        textFrameWidth = round(self.width/5)
        textFrameHeight = round(self.height/20)

        textFrame = tkinter.Frame(self.root,
                                  width=textFrameWidth,
                                  height=textFrameHeight,
                                  bg=bgColor)

        textFrameX = round(self.width / 2 - textFrameWidth / 2)
        textFrameY = round(thdImageY + thdImage.height() + textFrameHeight / 2)

        textFrame.pack_propagate(0)
        loadingText = tkinter.Label(textFrame,
                                    text="Place Screw/Bolt in Container and then press \"Begin\"",
                                    fg="white",
                                    bg=bgColor)

        startButtonFrameWidth = round(self.width/5)
        startButtonFrameHeight = round(self.width/20)

        startButtonFrame = tkinter.Frame(self.root,
                                         bg=bgColor,
                                         width=startButtonFrameWidth,
                                         height=startButtonFrameHeight)

        startButton = tkinter.Button(startButtonFrame,
                                     text="Begin",
                                     command=self.process,
                                     bg="white",
                                     fg=bgColor,
                                     activebackground="gray",
                                     activeforeground="white")

        startButtonX, startButtonY = self.getCenteredPosition(startButtonFrame.winfo_width(), startButtonFrame.winfo_height())

        startButtonFrame.place(x=startButtonX-startButtonFrameWidth/2,
                               y=thdImageY+thdImage.height()+textFrameHeight+startButtonFrameHeight/2)

        startButton.config(width=100,height=100)
        startButton.place(relx=0.5, rely=0.5, anchor="center")

        textFrame.place(x=textFrameX,
                        y=textFrameY)

        loadingText.pack(fill="both",
                         expand=1)

        thdStartButton.place(x=thdImageX,
                             y=thdImageY)

        self.root.mainloop()

    def getCenteredPosition(self, width, height):
        xPos = round(self.width / 2 - width / 2)
        yPos = round(self.height / 2 - height / 2)
        return xPos, yPos


    def clearScreen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def process(self):
        import LoadingScreen
        self.loadingScreen = LoadingScreen.LoadingScreen(self.root, self.width, self.height)
        self.loadingScreen.start()



