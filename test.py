from cImage import *


def redPixel(pixel):
    intensity = pixel.getRed()
    aveRGB = intensity
    newPixel = Pixel(aveRGB, 0, 0)
    return newPixel


def makeRedScale(imageFile):
    oldImage = FileImage(imageFile)
    width = oldImage.getWidth()
    height = oldImage.getHeight()

    myImageWindow = ImageWin("Enhanced Red", width * 2, height)
    oldImage.draw(myImageWindow)
    newIm = EmptyImage(width, height)

    for row in range(height):
        for col in range(width):
            oldPixel = oldImage.getPixel(col, row)
            newPixel = redPixel(oldPixel)
            newIm.setPixel(col, row, newPixel)

    newIm.setPosition(width + 1, 0)
    newIm.draw(myImageWindow)
    myImageWindow.exitOnClick()


makeRedScale("butterfly.gif")