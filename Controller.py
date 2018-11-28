import tkinter
import HomeScreen
bgColor = "#444444"


class Controller(object):

    def __init__(self):
        self.root = tkinter.Tk()
        self.width, self.height = self.set_geometry()
        self.set_bg_color()

        homeScreen = HomeScreen.HomeScreen(self.root,self.width,self.height)
        homeScreen.start()

    def set_geometry(self):
        screenWidth = self.root.winfo_screenwidth()
        screenHeight = self.root.winfo_screenheight()
        self.root.geometry(str(screenWidth) + "x" + str(screenHeight))
        return screenWidth, screenHeight

    def set_bg_color(self):
        self.root.configure(background=bgColor)


startingPoint = Controller()