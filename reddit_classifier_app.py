from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import praw
import pandas as pd
import numpy as np
import string
import spacy
import json
import pickle

app = Flask(__name__)
# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'fsdjgvs4it8ysfdjkvnsorut456'
Bootstrap(app)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True

# Connecting to PRAW
with open('config.json', 'r') as f:
    config = json.load(f)
reddit = praw.Reddit(client_id=config['app_client_id'],
                     client_secret=config['app_secret'],
                     user_agent="ChangeMeClient/0.1 by " + config['username'])

# Text preprocessing files and functions
maxlen = 60
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)           

def preprocess_text(inp_text):
   
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words
    table = str.maketrans('', '', string.punctuation)
    url_regex = r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))'

    text_df = pd.DataFrame([inp_text], columns=['orig_text'])
    text_df['clean_text'] = text_df['orig_text'].replace(to_replace=url_regex, value=' ', regex=True)
    text_df['clean_text'] = text_df['clean_text'].replace(to_replace="'", value='')
    text_df['clean_text'] = text_df['clean_text'].apply(lambda x: x.lower().translate(table))
    text_df['clean_text'] = text_df['clean_text'].apply(lambda text: ' '.join([word for word in text.split() if word not in all_stopwords]))
    text_df['clean_text'] = text_df['clean_text'].apply(lambda text: text.encode('ascii', 'ignore').decode('ascii'))
    return text_df['clean_text']       

#Model files
model_file = 'BLSTM_model.h5'
model = load_model(model_file)

encoder_file= 'model_classes.npy'
encoder = LabelEncoder()
encoder.classes_ = np.load(encoder_file, allow_pickle=True)

#Predictor function

def flair_predictor(url):
        submission = reddit.submission(url=url)
        comments_list = submission.comments.list()
        comments  = [comment.body for comment in comments_list[:min(5, len(comments_list))]]
        comments_text = ' '.join(comments)
        submission_title = submission.title
        if submission_title == None:
            submission_title = ''
        submission_title_text = submission.selftext
        if submission_title_text == None:
            submission_title_text = ''
        comments_text += ' ' + submission_title + ' ' + submission_title_text
        cleaned_comments_text = preprocess_text(comments_text).values[0]
        cleaned_comments_text_seq = tokenizer.texts_to_sequences([cleaned_comments_text])
        cleaned_comments_text_pad = sequence.pad_sequences(cleaned_comments_text_seq, maxlen=60)
        msg = encoder.inverse_transform([np.argmax(model.predict(cleaned_comments_text_pad))])[0]
        return msg

class IndexPage(FlaskForm):
    url = StringField('Enter submission link', validators=[Required()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
def predict_flair():
    msg = 'Error, flair could not be predicted'
    page = IndexPage()

    if page.validate_on_submit():
        url = page.url.data
        page.url.data = ""
        submission = reddit.submission(url=url)
        comments_list = submission.comments.list()
        comments  = [comment.body for comment in comments_list[:min(5, len(comments_list))]]
        comments_text = ' '.join(comments)
        submission_title = submission.title
        if submission_title == None:
            submission_title = ''
        submission_title_text = submission.selftext
        if submission_title_text == None:
            submission_title_text = ''
        comments_text += ' ' + submission_title + ' ' + submission_title_text
        cleaned_comments_text = preprocess_text(comments_text).values[0]
        cleaned_comments_text_seq = tokenizer.texts_to_sequences([cleaned_comments_text])
        cleaned_comments_text_pad = sequence.pad_sequences(cleaned_comments_text_seq, maxlen=60)
        #msg = encoder.inverse_transform([np.argmax(model.predict(cleaned_comments_text_pad))])[0]
        msg = flair_predictor(url)
    return render_template('index.html', form=page, message=msg)

@app.route('/automated_testing', methods=['POST'])
def get_request():
    file = request.files['upload_file']

    links = file.readlines()
    response = {}

    for link in links:
        link = link.decode("utf-8")
        response[link] = flair_predictor(link)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)