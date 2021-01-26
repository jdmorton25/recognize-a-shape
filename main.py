import  numpy as np
from    tkinter import *
import  tkinter as tk
import  tkinter.ttk as ttk
from    tkinter import colorchooser
import  pathlib
import  cv2
import  os, io
import  tensorflow as tf
from    tensorflow.keras import Sequential
from    tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tqdm import tqdm
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model


from data import pic_size, train_samples, test_samples

x_padding = 20
y_padding = 15

button_width = 90
button_height = 50

canvas_width = 400
canvas_height = 400

output_width = 200
output_height = 50

label_width = 2*button_width + x_padding
label_height = 20

scale_width = 200
scale_height = 40

footer_width = canvas_width + scale_width + 3*x_padding
footer_height = label_height

width = canvas_width + scale_width + 3*x_padding
height = canvas_height + 2*y_padding + footer_height


def load_dataset(path, count):
    image_samles = list()
    image_labeles = list()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    items = ['triangle', 'rectangle', 'ellipse']
    for item in tqdm(items):
        for i in range(count):
            image_samles.append(cv2.imread(path + '{}{:04}.jpg'.format(item, i)))
            if item == 'triangle':
                image_labeles.append(0)
            elif item == 'rectangle':
                image_labeles.append(1)
            elif item == 'ellipse':
                image_labeles.append(2)
    return np.array(image_samles), np.array(image_labeles)

class Program:
    def __init__(self):
        self.old_x = None
        self.old_y = None
        self.penwidth = 8
        self.color = '#000000'
        self.image = Image.new(mode='RGB', size=(canvas_width, canvas_height), color=(255, 255, 255))
        self.imgdraw = ImageDraw.Draw(self.image)
        
        self.draw()
        
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=[64, 64, 3]))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        self.model.add(Dropout(rate=.5))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=3, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        
        self.model.summary()
        
        self.root.mainloop()
    
    def paint(self, current):
        if (self.old_x and self.old_y):
            self.canvas.create_line(self.old_x, self.old_y, current.x, current.y, width=self.penwidth, fill=self.color, capstyle=ROUND, smooth=True)
            self.imgdraw.line([self.old_x, self.old_y, current.x, current.y], fill=self.color, width=int(self.penwidth))
        self.old_x = current.x
        self.old_y = current.y

    def reset(self,e):
        self.old_x = None
        self.old_y = None
    
    def change_penwidth(self, width):
        self.penwidth = width
    
    def askcolor(self):
        color_code = colorchooser.askcolor(title ="Color")  
        self.color = color_code[1]
        color = hex(int('ffffff', 16) - int(self.color[1:3] + self.color[1:3] + self.color[1:3], 16))[2:]
        while (len(color) != 6):
            color = '0' + color
        self.button4.configure(bg=self.color, fg='#' + color)
    
    def draw(self):
        self.root = Tk()
        self.root.geometry('{}x{}'.format(width, height))
        self.root.resizable(False, False)
        self.root.configure(background='lightgray')
    
        self.frame = Frame(self.root, bg='lightgray').place(x=canvas_width + output_width + 3*x_padding, y=y_padding, width=width, height=height)
    
        self.canvas = Canvas(self.frame, bg='white', relief='groove', highlightthickness=1, highlightbackground="#000000")
        self.canvas.place(x=x_padding, y=y_padding, width=canvas_width, height=canvas_height)

        self.label0 = Label(self.frame, text='model options', bg='lightgray')
        self.label0.place(x=canvas_width + 2*x_padding, y=y_padding, width=label_width, height=label_height)

        self.button0 = Button(self.frame, text='train', bg='#cccccc', relief='groove', highlightthickness=1, command=lambda : self.train() )
        self.button0.place(x=canvas_width + 2*x_padding, y=2*y_padding + label_height, width=button_width, height=button_height)

        self.button1 = Button(self.frame, text='predict', bg='#cccccc', relief='groove', highlightthickness=1, command=lambda : self.predict(), state=tk.DISABLED)
        self.button1.place(x=canvas_width + 3*x_padding + button_width, y=2*y_padding + label_height, width=button_width, height=button_height)

        self.button2 = Button(self.frame, text='load', bg='#cccccc', relief='groove', highlightthickness=1, command=lambda : self.load() )
        self.button2.place(x=canvas_width + 2*x_padding, y=3*y_padding + label_height + button_height, width=button_width, height=button_height)

        self.button3 = Button(self.frame, text='save', bg='#cccccc', relief='groove', highlightthickness=1, command=lambda : self.save(), state=tk.DISABLED)
        self.button3.place(x=canvas_width + 3*x_padding + button_width, y=3*y_padding + label_height + button_height, width=button_width, height=button_height)

        self.label1 = Label(self.frame, text='pen width', bg='lightgray')
        self.label1.place(x=canvas_width + 2*x_padding, y=4*y_padding + label_height + 2*button_height, width=label_width, height=label_height)

        self.scale = Scale(self.frame, from_=8, to=24, orient=HORIZONTAL, bg='lightgray', highlightthickness=0, command=self.change_penwidth)
        self.scale.place(x=canvas_width + 2*x_padding, y=4*y_padding + label_height + 2*button_height + label_height, width=scale_width, height=scale_height)

        self.button4 = Button(self.frame, text='pen color', bg='#000000', fg='#ffffff', relief='groove', highlightthickness=1, command=lambda : self.askcolor() )
        self.button4.place(x=canvas_width + 2*x_padding, y=6*y_padding + label_height + 2*button_height + label_height + scale_height, width=button_width, height=button_height)

        self.button5 = Button(self.frame, text='clear canvas', bg='#cccccc', relief='groove', highlightthickness=1, command=lambda : self.clear() )
        self.button5.place(x=canvas_width + 3*x_padding + button_width, y=6*y_padding + label_height + 2*button_height + label_height + scale_height, width=button_width, height=button_height)

        self.label2 = Label(self.frame, text='none', bg='lightgray', relief="groove")
        self.label2.place(x=canvas_width + 2*x_padding, y=canvas_height + y_padding - output_height, width=output_width, height=output_height)

        self.footer = Frame(self.root, bg='#bbbbbb').place(x=0, y=height - footer_height, width=footer_width, height=footer_height)

    def clear(self):
        self.canvas.delete(ALL)
        self.image.paste( (255,255,255), [0, 0, self.image.size[0], self.image.size[1]])
        
    
    def save_image(self):
        self.image.save('pred.jpg')
        
    
    def train(self):
        (X_train, y_train), (X_test, y_test) = load_dataset(path='./data/train/', count=train_samples), load_dataset(path='./data/test/', count=test_samples)
        X_train = X_train/255
        X_test = X_test/255
        self.model.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1, shuffle=True, validation_data=(X_test, y_test))
        self.button1.configure(state=tk.NORMAL)
        self.button3.configure(state=tk.NORMAL)
    
    def predict(self):
        self.save_image()
        image = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pred.jpg'))
        image = cv2.resize(image, (pic_size, pic_size))
        cv2.imwrite(filename='pred.jpg', img=image)
        image = image / 255
        prediction = self.model.predict(image[np.newaxis])
        value = np.max(prediction[0])
        index = np.argmax(prediction[0])
        label = ''
        if index == 0:
            label = 'triangle'
        elif index == 1:
            label = 'rectangle'
        else:
            label = 'circle'
        self.label2.config(text='{} {:.2%}'.format(label, value))

    def save(self):
        self.model.save('model.h5')
    
    def load(self):
        self.model = load_model('model.h5')
        self.button1.configure(state=tk.NORMAL)
        self.button3.configure(state=tk.NORMAL)



program = Program()



