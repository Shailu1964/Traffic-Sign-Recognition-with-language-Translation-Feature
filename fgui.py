import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
import pytesseract
from deep_translator import GoogleTranslator
from keras.models import load_model

# Load the trained model to classify signs
model = load_model('my_model.h5')

# Dictionary to label all traffic signs class
classes = {1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)', 4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)', 7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)', 9: 'Speed limit (120km/h)', 10: 'No passing',
           11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection', 13: 'Priority road',
           14: 'Yield', 15: 'Stop', 16: 'No vehicles', 17: 'Veh > 3.5 tons prohibited', 18: 'No entry',
           19: 'General caution', 20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve',
           23: 'Bumpy road', 24: 'Slippery road', 25: 'Road narrows on the right', 26: 'Road work',
           27: 'Traffic signals', 28: 'Pedestrians', 29: 'Children crossing', 30: 'Bicycles crossing',
           31: 'Beware of ice/snow', 32: 'Wild animals crossing', 33: 'End speed + passing limits',
           34: 'Turn right ahead', 35: 'Turn left ahead', 36: 'Ahead only', 37: 'Go straight or right',
           38: 'Go straight or left', 39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory',
           42: 'End of no passing', 43: 'End no passing veh > 3.5 tons'}

# Initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    # Convert the image to RGB format if it's not already in that format
    image = image.convert('RGB')
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)

    # Make predictions
    predictions = model.predict(image)

    # Get the class with the highest probability for each prediction
    predicted_classes = numpy.argmax(predictions, axis=1)

    sign = classes[predicted_classes[0] + 1]

    # Translate the sign text according to the selected language
    target_language = lang_option.get()
    if target_language != "english":
        try:
            sign = GoogleTranslator(source='english', target=target_language).translate(sign)
        except Exception as e:
            print("Translation error:", e)

    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def select_language(text):
    global lang_option
    lang_option = tk.StringVar(top)
    lang_option.set("english")  # default value
    lang_menu = OptionMenu(top, lang_option, "english", "spanish", "french", "german", "italian", "hindi", "marathi")
    lang_menu.place(relx=0.79, rely=0.53)

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
        select_language(file_path)  # Call select_language to display language options after selecting an image
    except Exception as e:
        print(e)

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()
