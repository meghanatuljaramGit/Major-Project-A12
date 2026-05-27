from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import hashlib

from tinydb import TinyDB, Query
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Reshape, LSTM

global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"

def uploadDataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")
 
def imageProcessing():
    text.delete('1.0', END)
    global X, Y, model_folder, filename

    X_file = os.path.join(model_folder, "X_compressed.npz")
    Y_file = os.path.join(model_folder, "Y.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)['X']  # Load from compressed .npz
        Y = np.load(Y_file)
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name + " " + os.path.join(root, directory[j]))
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(os.path.join(root, directory[j]))
                    img_resized = cv2.resize(img_array, (128, 128))
                    im2arr = np.array(img_resized).reshape(128, 128, 3)
                    X.append(im2arr)
                    Y.append(categories.index(name))
        
        X = np.asarray(X, dtype='float32') / 255.0
        Y = np.asarray(Y)
        
        np.savez_compressed(X_file, X=X)  # Save X as compressed .npz
        np.save(Y_file, Y)                # Save Y normally

    text.insert(END, 'Image Preprocessing Completed\n')


   

def Train_Test_split():
    global X,Y,x_train,x_test,y_train,y_test
    
        
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
    
    text.insert(END,"Total samples found in training dataset: "+str(x_train.shape)+"\n")
    text.insert(END,"Total samples found in testing dataset: "+str(x_test.shape)+"\n")


def calculateMetrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       
from sklearn.linear_model import Perceptron

def Existing_ML():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    num_samples_train, height, width, channels = x_train.shape
    num_samples_test, _, _, _ = x_test.shape
    x_train_flattened = x_train.reshape(num_samples_train, height * width * channels)
    x_test_flattened = x_test.reshape(num_samples_test, height * width * channels)

    model_filename = os.path.join(model_folder, "Perceptron_model.pkl")
    
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = Perceptron(max_iter=1, tol=None, random_state=42, shuffle=False)
        mlmodel.fit(x_train_flattened, y_train)
        joblib.dump(mlmodel, model_filename)
        print(f'Perceptron Model saved to {model_filename}')

    y_pred = mlmodel.predict(x_test_flattened)
    calculateMetrics("Existing Perceptron", y_pred, y_test)

def Existing_DecisionTree():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    # Flatten image data
    num_samples_train, height, width, channels = x_train.shape
    num_samples_test, _, _, _ = x_test.shape

    x_train_flattened = x_train.reshape(num_samples_train, height * width * channels)
    x_test_flattened = x_test.reshape(num_samples_test, height * width * channels)

    # Model path
    model_filename = os.path.join(model_folder, "DecisionTree_model.pkl")

    # Load or train model
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = DecisionTreeClassifier(
            criterion='gini',
            max_depth=None,
            random_state=42
        )
        mlmodel.fit(x_train_flattened, y_train)
        joblib.dump(mlmodel, model_filename)
        print(f'Decision Tree Model saved to {model_filename}')

    # Prediction
    y_pred = mlmodel.predict(x_test_flattened)

    # Performance metrics
    calculateMetrics("Existing Decision Tree", y_pred, y_test)


def DNN_Model():
    global x_train,x_test,y_train,y_test,model_folder,categories
    text.delete('1.0', END)
    
    y_train1 = to_categorical(y_train, num_classes=len(categories))  
    y_test1  = to_categorical(y_test, num_classes=len(categories))  
    
    Model_file = os.path.join(model_folder,    "Basic_DL_model.json")
    Model_weights = os.path.join(model_folder, "Basic_DL_model_weights.h5")
    Model_history = os.path.join(model_folder, "Basic_DL_history.pckl")
    num_classes = len(categories)

    if os.path.exists(Model_file):
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights(Model_weights)
        model._make_predict_function()   
        print(model.summary())
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
            acc = history['accuracy']
            acc = acc[4] * 100
    else:
        model = Sequential() 
        model.add(Flatten(input_shape=(128, 128, 3)))  
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(model.summary())
        hist = model.fit(x_train, y_train1, batch_size=16, epochs=5, validation_data=(x_test, y_test1), shuffle=True, verbose=2)
        model.save_weights(Model_weights)            
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)
        with open(Model_history, 'rb') as f:
            accuracy = pickle.load(f)
            acc = accuracy['accuracy']
            acc = acc[-1] * 100

    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test1 = np.argmax(y_test1, axis=1) 
    calculateMetrics("Existing NN", Y_pred_classes, y_test1)

def hybrid():  
    global X, Y, model_folder, categories, model, history
    text.delete('1.0', END)

    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  

    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]  
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    y_train = to_categorical(y_train, num_classes=len(categories))  
    y_test  = to_categorical(y_test, num_classes=len(categories))  

    Model_file = os.path.join(model_folder, "CNN_model.json")
    Model_weights = os.path.join(model_folder, "CNN_model_weights.h5")
    Model_history = os.path.join(model_folder, "CNN_history.pckl")
    num_classes = len(categories)

    if os.path.exists(Model_file):
    # Load model architecture
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)

        # Load model weights
        model.load_weights(Model_weights)

        # No need to call `_make_predict_function()` in TensorFlow 2.x
        # model._make_predict_function()  # Deprecated in TF 2.x

        print(model.summary())

        # Load model training history
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
            hist = history['accuracy']
            acc = hist[4] * 100 


    else:
        model = Sequential()
        model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Reshape((model.output_shape[1]*model.output_shape[2], model.output_shape[3])))  # Reshape for LSTM
        
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        hist = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), shuffle=True, verbose=2)

        model.save_weights(Model_weights)
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)

        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    calculateMetrics("CNN with RNN", y_pred, y_test)

def predict():
    global model, model_folder, categories
    categories=['Assault helicopter', 'Self-propelled artillery', 'Tank', 'Transport airplane', 'Transport helicopter']
    # Load model
    Model_file = os.path.join(model_folder, "CNN_model.json")
    Model_weights = os.path.join(model_folder, "CNN_model_weights.h5")

    with open(Model_file, "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(Model_weights)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Select and preprocess image
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (128, 128))
    im2arr = np.array(img).reshape(1, 128, 128, 3)
    test = im2arr.astype('float32') / 255.0

    # Predict
    y_pred = model.predict(test)
    predict_class = np.argmax(y_pred)

    # Display result
    img_display = cv2.resize(img, (500, 500))
    cv2.putText(img_display, 'Classified as : ' + categories[predict_class], 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Classified as : ' + categories[predict_class], img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def graph():
    global history

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation accuracy
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

def close():
    main.destroy()

db = TinyDB("users_db.json")
users_table = db.table("users")

def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            User = Query()
            if users_table.search((User.username == username) & (User.role == role)):
                messagebox.showerror("Error", f"{role} with this username already exists!")
                return

            users_table.insert({
                "username": username,
                "password": hashed_password,
                "role": role
            })

            messagebox.showinfo("Success", f"{role} Signup Successful!")
            signup_window.destroy()
            show_auth_buttons()

        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)


def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            User = Query()
            result = users_table.search(
                (User.username == username) &
                (User.password == hashed_password) &
                (User.role == role)
            )

            if result:
                messagebox.showinfo("Success", f"{role} Login Successful!")
                login_window.destroy()
                clear_buttons()
                if role == "Admin":
                    show_admin_buttons()
                elif role == "User":
                    show_user_buttons()
            else:
                messagebox.showerror("Error", "Invalid Credentials!")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)


def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button):
            widget.destroy()

    

import tkinter as tk

def show_admin_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    # Add ADMIN-specific buttons
    tk.Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=50, y=550)
    tk.Button(main, text="Preprocess Dataset", command=imageProcessing, font=font1).place(x=200, y=550)
    tk.Button(main, text="Train Test Splitting", command=Train_Test_split, font=font1).place(x=400, y=550)
    tk.Button(main, text="Train Perceptron", command=Existing_ML, font=font1).place(x=600, y=550)
    tk.Button(main, text="Train Decision Tree", command=Existing_DecisionTree, font=font1).place(x=800, y=550)
    tk.Button(main, text="Train DNN", command=DNN_Model, font=font1).place(x=1000, y=550)
    tk.Button(main, text="Proposed CNN with RNNLSTM", command=hybrid, font=font1).place(x=1200, y=550)

    tk.Button(main, text="Logout", command=show_auth_buttons, font=font1, bg="red").place(x=20, y=450)

def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

def show_user_buttons():
    font1 = ('times', 13, 'bold')

    # Clear USER-related buttons
    clear_buttons()
    # Add USER-specific buttons
    font1 = ('times', 13, 'bold')

    tk.Button(main, text="Prediction",
              command=predict,
              font=font1).place(x=200, y=550)

    tk.Button(main, text="Logout",
              command=show_user_buttons,
              font=font1,
              bg='red').place(x=20, y=500)

    tk.Button(main, text="Exit",
              command=close,
              font=font1).place(x=20, y=600)




    tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='red').place(x=100, y=100)
    tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='red').place(x=400, y=100)
    tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=700, y=100)
    tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=1000, y=100)

def close():
    main.destroy()



main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")


import os
import joblib
import tkinter as tk
from tkinter import filedialog, Text, ttk
from PIL import Image, ImageTk



# Load Background Image
bg_image = Image.open("background.jpg")  
bg_image = bg_image.resize((1380, 730), Image.LANCZOS)  
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(main, image=bg_photo)
bg_label.place(relwidth=1, relheight=1) 

# Configure title
font = ('times', 18, 'bold')
title_text = "Machine Learning for Tactical Decision Support: A Data-Driven Study of Military Scenarios"
title = tk.Label(main, text=title_text, bg='white', fg='black', font=font, height=3, width=120)
title.pack()





# Text area for displaying results or logs
text = tk.Text(main, height=15, width=70)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=180)
text.config(font= ('times', 12, 'bold'))

# ================= AUTHENTICATION BUTTONS =================
font_auth = ('times', 14, 'bold')

tk.Button(
    main,
    text="Admin Signup",
    command=lambda: signup("Admin"),
    font=font_auth,
    width=20,
    bg='red'
).place(x=150, y=100)

tk.Button(
    main,
    text="Admin Login",
    command=lambda: login("Admin"),
    font=font_auth,
    width=20,
    bg='LightPink'
).place(x=450, y=100)

tk.Button(
    main,
    text="User Signup",
    command=lambda: signup("User"),
    font=font_auth,
    width=20,
    bg='red'
).place(x=750, y=100)

tk.Button(
    main,
    text="User Login",
    command=lambda: login("User"),
    font=font_auth,
    width=20,
    bg='LightPink'
).place(x=1050, y=100)

def show_auth_buttons():
    clear_buttons()
    font_auth = ('times', 14, 'bold')

    tk.Button(main, text="Admin Signup",
              command=lambda: signup("Admin"),
              font=font_auth, width=20, bg='red').place(x=150, y=100)

    tk.Button(main, text="Admin Login",
              command=lambda: login("Admin"),
              font=font_auth, width=20, bg='LightPink').place(x=450, y=100)

    tk.Button(main, text="User Signup",
              command=lambda: signup("User"),
              font=font_auth, width=20, bg='red').place(x=750, y=100)

    tk.Button(main, text="User Login",
              command=lambda: login("User"),
              font=font_auth, width=20, bg='LightPink').place(x=1050, y=100)



main.config(bg='deep sky blue')


main.mainloop()


