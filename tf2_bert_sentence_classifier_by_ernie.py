# INSTALL ernie https://github.com/labteral/ernie
!pip install ernie

# MOUNT GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/TF2')

from ernie import SentenceClassifier, Models

# READ CSV CONTENT
import pandas as pd
import numpy as np
data=pd.read_csv('input/All.csv',encoding='utf-8',header=None)
data = data.rename(columns = {0:'story'})
data = data.rename(columns = {1:'score'})
data['story'] = data['story'].replace(np.nan,'*')
data.isnull().any()

df = pd.DataFrame(data)

# SET CLASSIFIER CONFIG
classifier = SentenceClassifier(
    model_name="bert-base-chinese",
    max_length=128,
    labels_no=4
)
classifier.load_dataset(df, validation_split=0.2)

#FINE TUNE CLASSIFIER
classifier.fine_tune(
    epochs=10,
    learning_rate=3e-5,
    training_batch_size=32,
    validation_batch_size=64
)

# EXPORT MODEL
classifier.dump('model')

# READ MODEL
new_classifier = SentenceClassifier(model_path='./model')

# TEST PREDICT
text = "常吃高溫烹調之高蛋白質含量食物，恐致癌。應避免過量攝取肉類或加工食品，均衡飲食，才是健康的王道。"

probabilities = new_classifier.predict_one(text)

# GET PREDICT RESULT
probabilities

# GET PREDICT SCORE
probabilities.index(max(probabilities))