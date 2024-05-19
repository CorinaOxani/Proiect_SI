import tkinter as tk
from tkinter import filedialog, Button, Label
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load the trained model to classify the sign
model = load_model('traffic_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
           2:'Speed limit (30km/h)',
           3:'Speed limit (50km/h)',
           4:'Speed limit (60km/h)',
           5:'Speed limit (70km/h)',
           6:'Speed limit (80km/h)',
           7:'End of speed limit (80km/h)',
           8:'Speed limit (100km/h)',
           9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing vehicle with a weight greater than 3.5 tons' }
# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#2D2D2D')

# Styling Variables
bg_color = "#2D2D2D"
text_color = "#FFFFFF"
button_color = "#FF5722"
font_type = "Verdana"

label = Label(top, background=bg_color, font=(font_type, 15, 'bold'), fg=text_color)
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path).convert('RGB')  # Asigură-te că imaginea este în format RGB
    image = image.resize((30, 30))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Adaugă o dimensiune batch pentru a obține forma (1, 30, 30, 3)
    
    pred = model.predict(image)  
    sign = classes[np.argmax(pred) + 1]  # Utilizează +1 pentru că indicii claselor încep de la 1 în dicționarul tău
    print(sign)
    label.configure(foreground=text_color, text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background=button_color, foreground='white', font=(font_type, 10, 'bold'), relief=tk.FLAT)
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(e)

upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background=button_color, foreground='white', font=(font_type, 10, 'bold'), relief=tk.FLAT)
upload.pack(side=tk.BOTTOM, pady=50)
sign_image.pack(side=tk.BOTTOM, expand=True)
label.pack(side=tk.BOTTOM, expand=True)
heading = Label(top, text="Check Traffic Sign", pady=20, font=(font_type, 20, 'bold'))
heading.configure(background=bg_color, foreground=text_color)
heading.pack()

top.mainloop()