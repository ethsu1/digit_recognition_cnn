#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 23:36:31 2017

@author: ethan
"""
from tkinter import *
from PIL import Image, ImageDraw
import io
import os, glob
from keras.models import model_from_json
import numpy as np
from scipy.misc import imread,imshow
from skimage.transform import resize
import cv2
import math
from scipy.ndimage.measurements import center_of_mass

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

b1 = 'up'
xval, yval = None, None 
white = (255, 255, 255)
black = (0, 0, 0)
root = Tk()
root.title("Handwriting Prediction")
root.geometry('240x240')
canvas_draw = Canvas(root, width=30, height=30)
image = Image.new("RGB",(28, 28),white)
canvas_draw.create_rectangle(0, 0, 30,30, width=1, fill='white')
draw = ImageDraw.Draw(image)
label = Label(root, text='temp')
label.pack(side=BOTTOM)
label.pack_forget()
predict = True
def main():
    
    canvas_draw.pack()
    canvas_draw.bind("<Motion>", motion)
    canvas_draw.bind("<ButtonPress-1>", b1down)
    canvas_draw.bind("<ButtonRelease-1>", b1up)
    frame = Frame(root)
    frame.pack()
    button_enter = Button(frame, text='Enter', command=save_image)
    button_enter.pack(side=RIGHT)
    button_clear=Button(frame,text='Clear',command=clear_image)
    button_clear.pack(side=LEFT)
    
    root.mainloop()
    
def save_image():
    global predict
    if predict:
        canvas_draw.postscript(file="image.ps", colormode='color')
        
        filename = "image.jpg"
        image.save(filename)

    
        x = imread('image.jpg',mode='L')
        x = np.invert(x)
        x = resize(x,(28,28))
        x = x.reshape(1,28,28)
        x = np.expand_dims(x,axis=0)
        out = loaded_model.predict(x)
        value = "Predicted: " + str(np.argmax(out,axis=1))
        global label 
        label = Label(root, text=value)
        label.pack()
        predict = False
        
    
    
    
def clear_image():
    global predict
    if not predict:
        canvas_draw.delete("line")
        global image 
        image = Image.new("RGB",(28, 28),white)
        global draw 
        draw = ImageDraw.Draw(image)
        global label
        label.pack_forget()
        predict=True
    

def b1down(event):
    global b1
    b1 = "down" 

def b1up(event):
    global b1, xval, yval
    b1 = "up"
    xval = None           
    yval = None
    
def motion(event):
    if b1 == "down":
        global xval, yval
        if xval is not None and yval is not None:
            event.widget.create_line(xval,yval,event.x,event.y,smooth=TRUE,width=2,
                                     tag='line')
    
            draw.line((xval, yval, event.x, event.y), black, width=2)
        xval = event.x
        yval = event.y

if __name__ == "__main__":
    main()
