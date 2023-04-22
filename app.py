import os

import joblib
import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from keras.models import load_model
model = load_model('model.h5')
plantmodel = joblib.load('plantmodel.joblib')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
plantpklmodel = load_model('keras_model.h5')


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import secrets


app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = secrets.token_hex(16)

@app.route("/")
def home():
    return render_template("index.html")


def botdiseaseresponse(image):
    # Open the image file using PIL
    img = Image.open(io.BytesIO(image.read()))
    img = img.resize((224, 224))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the pixel values
    img_array = img_array / 255.0

    # Reshape the array to add a batch dimension
    img_array = img_array.reshape((1, 224, 224, 3))
    predictions = plantpklmodel.predict(img_array)
    class_index = tf.argmax(predictions, axis=-1)[0]
    classes = {
        0: 'Tomato___healthy',
        1: 'Tomato___Tomato_mosaic_virus',
        2: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        3: 'Tomato___Target_Spot',
        4: 'Tomato___Spider_mites Two-spotted_spider_mite',
        5: 'Tomato___Septoria_leaf_spot',
        6: 'Tomato___Leaf_Mold',
        7: 'Tomato___Late_blight',
        8: 'Tomato___Early_blight',
        9: 'Tomato___Bacterial_spot',
        10: 'Strawberry___healthy',
        11: 'Strawberry___Leaf_scorch',
        12: 'Squash___Powdery_mildew',
        13: 'Soybean___healthy',
        14: 'Raspberry___healthy',
        15: 'Potato___healthy',
        16: 'Potato___Late_blight',
        17: 'Potato___Early_blight',
        18: 'Pepper,_bell___healthy',
        19: 'Pepper,_bell___Bacterial_spot',
        20: 'Peach___healthy',
        21: 'Peach___Bacterial_spot',
        22: 'Orange___Haunglongbing_(Citrus_greening)',
        23: 'Grape___healthy',
        24: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        25: 'Grape___Esca_(Black_Measles)',
        26: 'Grape___Black_rot',
        27: 'Corn_(maize)___healthy',
        28: 'Corn_(maize)___Northern_Leaf_Blight',
        29: 'Corn_(maize)___Common_rust_',
        30: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        31: 'Cherry_(including_sour)___healthy',
        32: 'Cherry_(including_sour)___Powdery_mildew',
        33: 'Blueberry___healthy',
        34: 'Apple___healthy',
        35: 'Apple___Cedar_apple_rust',
        36: 'Apple___Black_rot',
        37: 'Apple___Apple_scab'
    }

    predicted_class_name = classes[class_index.numpy()]

    # Print the predicted class name
    return predicted_class_name


@app.route("/chat" , methods=['POST'])
def get_bot_response():
    text = request.form.get('text')
    image = request.files.get('image')
    if image is not None:
        return botdiseaseresponse(image)
    if text is not None:
        if (text == '1'):
            session["plantrealtedtopic"] = True
            # plantrealtedtopic = 1
            return "Please specify the symptoms of the plant"
        if (session.get("plantrealtedtopic")):
            vectorizer = joblib.load('vectorizer1.joblib')

            vocab = vectorizer.vocabulary_
            vectorizer = CountVectorizer(stop_words='english', vocabulary=vocab)

            new_symptom_vec = vectorizer.transform([text])
            prediction = plantmodel.predict(new_symptom_vec)[0]
            session["plantrealtedtopic"] = False
            # return chatbot_response(userText)
            return "Your plant must be infected with disease: " + prediction
        else:
            return chatbot_response(text)
    return jsonify({"response": "Hello, I am the bot!"})





if __name__ == "__main__":
    app.run()