# Ts.Erkhembileg & B.Khaliun
# Natural Language Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

# Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

nltk.download('stopwords')

corpus = []
# iterate through dataset
for i in range(0, 1000):
    # preparing dataset to process
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    # removing stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Transform into vector
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# saving as a file
filename = 'trained_model.sav'
joblib.dump(classifier, filename)

# predict
y_pred = classifier.predict(X_test)

loaded_model = joblib.load(filename)
y_pred_file = loaded_model.predict(X_test)

# to present
cm = confusion_matrix(y_test, y_pred)
cm_file = confusion_matrix(y_test, y_pred_file)

print("Nagative review predicted", cm[0, 0], " (", cm[0, 0] * 100 / ( cm[0, 0] + cm[0, 1] ), "%)")
print("Positive review predicted", cm[1, 1], " (", cm[1, 1] * 100 / ( cm[1, 0] + cm[1, 1] ), "%)")

print("Overall ", (cm[0, 0] + cm[1, 1]) * 100 / ( cm[1, 0] + cm[1, 1] + cm[0, 0] + cm[0, 1] ))