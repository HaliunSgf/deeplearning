# Natural Language Processing
> Ts.Erkhembileg & B.Khaliun

Natural Language Processing нь Хиймэл оюун ухааны нэг салбар бөгөөд компьютер-т хүний хэлийг ойлгох, сурах боломж олгосон. Энэ нь хэрэглэгч компьютер-тайгаа харьцахдаа заавал artificial languages болох Java, C гэх мэт хэлнүүдээр биш өөрийн хэлийг ашиглах боломжтой гэсэн үг юм. 

Энэхүү ажлаар рестораны санал хүсэлтүүдийг эерэг сөрөг байгааг таних AI хийхийг зорилоо. Уг даалгаврыг сонгосон шалтгаан нь бидэнд дээр дурдсан Natural Language Processing-н боломжууд нь маш сонирхолтой санагдсан. Мөн уг ажлыг цааш өргөжүүлэн Монгол хэлний NLP-г сайжруулах, хэрэглэгчид болон байгууллагуудын хоорондын харилцааг хялбарчлах, цаг хугацаа хэмнэх гэх мэт олон төрлөөр ашиглах боломжтой гэж бодож байна.

------------------

## Онцлох хэсгүүд

Үгийг хялбаршуулахдаа stopwords-уудыг хассанаар илүү хурдан ажиллуулах боломжтой болсон
```python
    ps = PorterStemmer()
    # removing stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
```

Манай өгөгдөл 1565 үгтэй учир bag of words үүсгэхдээ max_features-г 1500-аар авахад болно гэж үзсэн.
```python
    cv = CountVectorizer(max_features = 1500)
```

Сургалтын болон тестийн өгөгдлийг 80:20 харьцаагаар сонгосон
```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
```

Ангилал хийхдээ Naive Bayes моделыг ашигласан
```python
    from sklearn.naive_bayes import GaussianNB
    ...
    classifier = GaussianNB()
```

------------------

## Ашигласан сангууд
numpy (Math)

matplotlib.pyplot (Graphics)

pandas (Data import)

nltk (Natural language toolkit)

- stopwords
- PorterStemmer

sklearn
- CountVectorizer
- train_test_split
- GaussianNB (Classification)
- joblib (Classification into file)
- confusion_matrix (Presentation)

##  
