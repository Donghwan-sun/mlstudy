import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

im = Image.open("D:/logo.png")
size = (170, 34)
im.thumbnail(size)
im.save('test.png')
