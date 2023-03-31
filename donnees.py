import streamlit as st
import pandas as pd
from io import StringIO
from html.parser import HTMLParser
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


@st.experimental_singleton
def load_dataset():
    # On charge le dataset d'entrainement
    X_train = pd.read_csv("X_train.csv", index_col=0)
    y_train = pd.read_csv("Y_train.csv",index_col=0)
    categories = pd.read_csv("categories.csv", sep="\t")

    categories_alphasort = categories.reset_index().astype(str).sort_values(by='prdtypecode').reset_index()
    categories_alphasort.drop(['index'], axis=1, inplace=True)
    categories_numbered = categories_alphasort.copy()
    categories_numbered.set_index('prdtypecode', inplace=True)
    categories_numbered.insert(0, 'number', range(0, len(categories)))
    categories_numbered.index = categories_numbered.index.map(int)
    
    X = pd.concat([X_train, y_train], axis=1)
    X["categorie"] = X.prdtypecode.map(categories_numbered.number)

    X["image"] = "image_" + X.imageid.astype(str) + "_product_" + X.productid.astype(str) + ".jpg"
    X["texte"] = X.designation + " " + X.description.fillna('').astype(str)
    X["orig_texte"] = X.designation + " " + X.description.fillna('').astype(str)
    #X.drop(["productid", "imageid", "prdtypecode", "designation", "description"], axis=1, inplace=True)
    X.drop_duplicates(subset="texte", inplace=True)
    #X["texte"] = X.texte.apply(strip_tags)
    
    return X_train, y_train, categories, X, categories_numbered, categories_alphasort




# Algorithme multimodal


from sklearn.linear_model import LogisticRegression
import pickle


@st.experimental_singleton
def load_logisitc_regression_model():
    filename = 'logistic_regression.sav'
    return pickle.load(open(filename, 'rb'))



from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer("\w+")

stop_words_fr = stopwords.words("french")
stop_words_en = stopwords.words("english")
stop_words_de = stopwords.words("german")
stop_words = stop_words_fr + stop_words_en + stop_words_de + [str(i) for i in range(0, 100)] + ["x"]
stop_words = {word:0 for word in stop_words} # Optimisation for faster filtering

def stop_words_filtering(string_list):
    return [ w for w in string_list if w not in stop_words ]

lemmatisation = WordNetLemmatizer()

vectorizer = pickle.load(open("vectorizer.sav", 'rb'))

def process_text(texte):
    # On enleve les tags html
    texte = strip_tags(texte)

    # On passe l'ensemble du texte en minuscule
    texte = " ".join(x.lower() for x in texte.split())
    # On lemmatise (on garde la racine des mots)
    texte = " ".join(lemmatisation.lemmatize(x) for x in texte.split())
    # On enleve les stop words
    texte = stop_words_filtering(tokenizer.tokenize(texte))

    vector = vectorizer.transform([" ".join(texte)]).astype('uint8')
    return texte, vector
    

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


@st.experimental_singleton
def load_vgg16_cnn_model(nb_of_classes):

    base_model = VGG16(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(layers.Lambda(preprocess_input, name='preprocessing', input_shape=(224, 224, 3)))
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(nb_of_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("checkpoint_vgg16_3")
    return model


def get_image(image):
  path = f"/image_train/{image}"
  image = tf.keras.utils.load_img(
    path,
    grayscale=False,
    color_mode='rgb',
    target_size=(224,224,3),
    interpolation='bilinear'
  )
  return tf.keras.preprocessing.image.img_to_array(image)
