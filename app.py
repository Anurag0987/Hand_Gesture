from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
import cv2
import os
import numpy as np
import operator
from string import ascii_uppercase
import pyttsx3
path = os.getcwd()


class Application:
    
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.loaded_model = tf.keras.models.load_model("checkpoint_vgg16_28.4\weights-improvement-01.h5")
       
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.configure(bg="ivory2") # tan1, dark slate gray
        self.root.title("Sign language Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1250x650")
        self.panel = tk.Label(self.root,bg='ivory2',fg='red4')
        self.panel.place(x = 20, y = 10, width = 640, height = 640)
        self.panel2 = tk.Label(self.root,bg='ivory2',fg='red4') # initialize image panel
        self.panel2.place(x = 350, y = 90, width = 310, height = 310)
        self.T = tk.Label(self.root,bg='ivory2',fg='red4')
        self.T.place(x=45,y = 17)
        self.T.config(text = "Sign Language Converter",font=("Times",40,"bold"))



        self.panel3 = tk.Label(self.root) # Current SYmbol
        self.panel3.place(x = 970,y=50)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 690,y = 50)
        self.T1.config(text="Symbol :",font=("Courier",40,"bold"),bg='ivory2',fg='red4')
        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 950,y=110)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 690,y = 110)
        self.T2.config(text ="Word :",font=("Courier",40,"bold"),bg='ivory2',fg='red4')
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 1030,y=170)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 690,y = 170)
        self.T3.config(text ="Sentence :",font=("Courier",40,"bold"),bg='ivory2',fg='red4')
        
        self.str=""
        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"

        self.video_loop()
        def read(event):    
            engine = pyttsx3.init()
            engine.say(self.str)
            engine.runAndWait()
    
        # self.root.bind("r",read)
        button1  = tk.Button(self.root,text=" Read ",font=("Courier",15,"bold"),bg='red3',fg='white')
        button1.bind("<Button-1>", read)
        button1.place(x=690,y=300)

        def remove(event):
            self.word = self.word[:-1]
        self.root.bind("<BackSpace>",remove)

        def clear_sen():
            self.str = ""


        button2 = tk.Button(self.root,text=" clear ",font=("Courier",15,"bold"),bg='red3',fg='white',command=clear_sen)
        button2.place(x=790,y=300)

        def save():
            sent = self.str
            file = open("saved sentences.txt","a")
            file.write(f"{sent}\n")
            file.close()


        button3 = tk.Button(self.root,text=" Save Text ",font=("Courier",15,"bold"),bg='red3',fg='white',command=save)
        button3.place(x=910,y=300)
        

        

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            # cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            cv2image = cv2image[y1:y2, x1:x2]


            def rgba2rgb( rgba, background=(255,255,255) ):
                row, col, ch = rgba.shape

                if ch == 3:
                    return rgba

                assert ch == 4, 'RGBA image has 4 channels.'

                rgb = np.zeros( (row, col, 3), dtype='float32' )
                r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

                a = np.asarray( a, dtype='float32' ) / 255.0

                R, G, B = background

                rgb[:,:,0] = r * a + (1.0 - a) * R
                rgb[:,:,1] = g * a + (1.0 - a) * G
                rgb[:,:,2] = b * a + (1.0 - a) * B

                return np.asarray( rgb, dtype='uint8' )


            cv2image = rgba2rgb(cv2image)
            # print(cv2image.shape)
            # print(cv2image.ndim)
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            res = cv2.merge((res,res,res))
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol,font=("Courier",40,'bold'),bg='ivory2',fg='red4')
            self.panel4.config(text=self.word,font=("Courier",40,'bold'),bg='ivory2',fg='red4')
            self.panel5.config(text=self.str,font=("Courier",40,'bold'),bg='ivory2',fg='red4')

            predicts=self.word

            
                   
        self.root.after(30, self.video_loop)
    def predict(self,test_image):
        # print(test_image.ndim)
        # print(test_image.shape)
        test_image = cv2.resize(test_image, (224,224))
        # print(test_image.ndim)
        # print(test_image.shape)
        result = self.loaded_model.predict(test_image.reshape(1,224,224,3))

        prediction={}
        prediction['blank'] = result[0][0]
        index = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][index]
            index += 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        


        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if(self.ct[self.current_symbol] > 50):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 20):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    


    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()
  
print("Starting Application...")

pba = Application()
pba.root.mainloop()
