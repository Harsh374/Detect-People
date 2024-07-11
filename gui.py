import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load pre-trained models
face_cascade = cv2.CascadeClassifier("C:/Users/harsh/Desktop/Task_2_1/haarcasade_frontalface_default.xml")
gender_model = load_model("C:/Users/harsh/Desktop/Task_2_1/gender_model.h5")
age_model = load_model("C:/Users/harsh/Desktop/Task_2_1/age_model.h5")

# Define functions for gender and age prediction
def predict_gender(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.reshape(1, 64, 64, 3) / 255.0  # Normalize image
    gender_prob = gender_model.predict(face_img)
    print(f"Gender prediction probabilities: {gender_prob}")  # Debugging statement
    return 'Male' if gender_prob[0][0] > 0.5 else 'Female'

def predict_age(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.reshape(1, 64, 64, 3) / 255.0  # Normalize image
    age = age_model.predict(face_img)
    print(f"Age prediction: {age}")  # Debugging statement
    return int(age[0][0])

# Detect shirt color
def detect_shirt_color(image, face_rect):
    x, y, w, h = face_rect
    shirt_region = image[y+h:y+2*h, x:x+w]  # Adjust the region to cover the shirt
    average_color = np.mean(shirt_region, axis=(0, 1))
    print(f"Average shirt color: {average_color}")  # Debugging statement
    if np.all(average_color > [200, 200, 200]):
        return 'White'
    elif np.all(average_color < [50, 50, 50]):
        return 'Black'
    else:
        return 'Other'

def analyze_meeting(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        return f"Error: Unable to load image at path {image_path}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) < 2:
        return "Less than 2 people detected, feature not applied."

    results = []
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        gender = predict_gender(face_img)
        age = predict_age(face_img)
        shirt_color = detect_shirt_color(image, (x, y, w, h))

        if shirt_color == 'White':
            age = 23
        elif shirt_color == 'Black':
            gender = 'Child'

        results.append({'Gender': gender, 'Age': age, 'Shirt Color': shirt_color})

    return results

# GUI application
class GenderAgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender and Age Detection")
        self.root.geometry('800x600')
        self.root.configure(background='#CDCDCD')
        
        self.setup_gui()
        
    def setup_gui(self):
        heading = Label(self.root, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
        heading.configure(background="#CDCDCD", foreground="#364156")
        heading.pack()

        self.sign_image = Label(self.root)
        self.sign_image.pack(side='bottom', expand=True)

        self.label1 = Label(self.root, background="#CDCDCD", font=('arial', 15, "bold"))
        self.label2 = Label(self.root, background="#CDCDCD", font=('arial', 15, 'bold'))
        self.label1.pack(side="bottom", expand=True)
        self.label2.pack(side="bottom", expand=True)

        upload = Button(self.root, text="Upload an Image", command=self.upload_image, padx=10, pady=5)
        upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        upload.pack(side='bottom', pady=50)
        
    def show_detect_button(self, file_path):
        detect_b = Button(self.root, text="Detect Image", command=lambda: self.detect(file_path), padx=10, pady=5)
        detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        detect_b.place(relx=0.79, rely=0.46) 

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((self.root.winfo_width() / 2.25), (self.root.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label1.configure(text='')
            self.label2.configure(text='')
            self.show_detect_button(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image. {str(e)}")
            print(e)
        
    def detect(self, file_path):
        results = analyze_meeting(file_path)
        if isinstance(results, str):
            messagebox.showinfo("Detection Results", results)
        else:
            result_text = "\n".join([f"Gender: {res['Gender']}, Age: {res['Age']}, Shirt Color: {res['Shirt Color']}" for res in results])
            messagebox.showinfo("Detection Results", result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeDetectionApp(root)
    root.mainloop()
