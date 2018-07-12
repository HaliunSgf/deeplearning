# Ts.Erkhembileg & B.Khaliun
# Natural Language Processing

# Libraries duudaj baina
# too boddog lib
import numpy as np
# graphic ntr zurdag lib
import matplotlib.pyplot as plt
# data-aa too bolgoj teren deeree too boddog lib
import pandas as pd
# ene text-iig zadlah yanz bur bolgohod heregledeg lib
import re
# ene natural language toolkit geed heregtei lib
import nltk
# stopwords gedeg n 'the, and, a, an, in' geh met oguulberiig stop hiij baigaa ugnuud
from nltk.corpus import stopwords
# ugiig hyalbarshuuldag lib. zovhon yazguuriig n hadgalna gesen ug lde. Loved => love, Liked => like etc
from nltk.stem.porter import PorterStemmer
# ugiig too bolgoj baina gesen ug. vector luu horvuulne
from sklearn.feature_extraction.text import CountVectorizer
# dataset-ee train, test gej 2 huvaahad hereglene ene lib-iig
from sklearn.cross_validation import train_test_split
# za ene manai classification buyu undsen train hiih model
# yag code deeree heregleheer sain tailbarlaad ugnu
from sklearn.naive_bayes import GaussianNB
# train hiisen classification-oo file bolgoj hadgalj avahad hereglene
from sklearn.externals import joblib
# predict hiisen uzuuleltee hund oilgomjtoi haruulna
from sklearn.metrics import confusion_matrix

# Dataset oruulj irj baina
# ene tsv file-iig read_csv-eer duudaj bolno
# delimiter gedeg n name-ee salgaj bgaa gesen ug '\t' gej python seperator
# quoting = Control field quoting behavior per csv.QUOTE_* constants.
# er n ene yag yagaad baina gej bodoj baigaa code-oo ctrl+i daraad yu hiideg function gedgiig n harch bolno
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# dataset-ees avsan text-uudee train hiihed beldej baina baina
# nuguu stopword-uudaa download hiij com deeree hadgalj bn
nltk.download('stopwords')

# daraa ashiglana gej uzeed hooson massive uusgej baina
# yagaad corpus gej uu? machine learning deer hooson array-uudiig torloos n hamaarch yanz yanzar nerledeg
# coprus gedeg n text helberiin collection-uudiig heldeg FYI
corpus = []
# manai dataset 1000 oguulbertei baigaa uchiraas 0 - 1000 davtalt hiij baina
for i in range(0, 1000):
    # oguulberees usegnees busdiign hasch baina
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # capital baij magadgui geed lower hiij baina
    review = review.lower()
    # oguulberiig ugeer n zadalj baina
    review = review.split()
    # ugiig hyalbarshuulah gej baina
    ps = PorterStemmer()
    # oguulberiig zadalsan ugnuudiig stopwordoos shalgaad hasna. nuguu stopword ugnuudiig hasna gesen ug 
    # word-iig porterstemmer-eer hyalbarshuulaad ene loopiig arai hurdan ajiluulj baina gesen ug lde
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # tegeed haschihaad butsaad ugnuudee niiluuleed oguulber bolgoj baina
    review = ' '.join(review)
    # tegeed oguulberee corpus massive ruu hiij shine dataset uusgaj baina
    corpus.append(review)

# oguulberiig vector luu horvuulne
# eniig bag of words gej nerlej baigaa
# max_features gedeg n ugiin davtaltaar n yalgana gesen ug.
# ooroor helbel Like gedeg ug oguulberuuded olon orvol max_features dee deeguur bairand bairlana
# manai dataset-d niit 1565 ug baina. manai model 1500g n avahad bolno gej uzej baina
# heden ug baigaag harahdaa max_features-iig taaruulahguigeer countVecorizer-ee hooson function-oor unshuulchih
# za yaj vector boldoo bainaa gej uu? amarhan 1 x 1500 hemjeetei vector uusne
# oguulbert 3n yanziin ug baival 0 0 0 1 1 1 0 0 0 0... geh met
# i love this place gedeg oguulber baival i love place bolno tegeed 3n ugtei 3n gazar 1 1 1 busad n 0 0 0
# gehdee 1 ug oguulbert 2 baival 0 0 0 2 gej orno gesen ug. za oilgohgui bol utasdaad asuuj bolno
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
# dataset-ees shuud ene ereg sorog uu gedeg 1 0 utgiig n avch baina tegehgui bol ene oguulber yaj bn gedgiig n medehgui shude
y = dataset.iloc[:, 1].values

# za x, y -ee 2 huvaaj baina training bolon test. gej 2 huvaana
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Naive Bayes gedeg model 
# https://www.youtube.com/watch?v=EGKeC2S44Rs
# https://www.youtube.com/watch?v=M59h7CFUwPU
# za sorry commentoor tailbarlaj chadsangui ene 2 bichleg oilguulchih baih
# nuguu boon math-uudiig ingeed 2 line code hiichij bgan
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# classifier-aa file bolgoj hadgalj avch baina
filename = 'trained_model.sav'
joblib.dump(classifier, filename)

# surgasan modelooroo predict hiij uzey
y_pred = classifier.predict(X_test)

# file bolgoj hadgalsan modelooroo predict hiij uzey
loaded_model = joblib.load(filename)
y_pred_file = loaded_model.predict(X_test)

# predict hiisen value-aa hund haruulahad oilgomjtoi bolgoj baina
cm = confusion_matrix(y_test, y_pred)
# file-aar predict hiij uzej baina
cm_file = confusion_matrix(y_test, y_pred_file)

print("Nagative review predicted", cm[0, 0], " (", cm[0, 0] * 100 / ( cm[0, 0] + cm[0, 1] ), "%)")
print("Positive review predicted", cm[1, 1], " (", cm[1, 1] * 100 / ( cm[1, 0] + cm[1, 1] ), "%)")

print("Overall ", (cm[0, 0] + cm[1, 1]) * 100 / ( cm[1, 0] + cm[1, 1] + cm[0, 0] + cm[0, 1] ))