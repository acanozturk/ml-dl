###################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # Naive-Bayes kullanacağız
import re # Regular Expression. Pattern search etmek için kullanlılan özel textler
import nltk # Natural Language ToolKit
from nltk.corpus import stopwords # Stopwordler corpus klasörüne indirilir. Buradan import ederiz
import nltk as nlp
from sklearn.feature_extraction.text import CountVectorizer # Bag of words için kullancağımız metod
nltk.download("stopwords") 
nltk.download('punkt')
nltk.download('wordnet')
###################################################################

###################################################################
data = pd.read_csv(r"genderdata.csv", encoding ="latin1") # Datada latin harfleri olduğu için belirtmemiz lazım

data = pd.concat([data.gender, data.description], axis = 1) # Sadece gender ve description kullanacağımız için bunları aldık

data.dropna(axis = 0, inplace = True) # Datadaki "nan" ları bulur ve satırı datadan kaldırır

data.gender = [1 if each == "female" else 0 for each in data.gender] # String datayı kadın ise 1 erkek ise 0 olarak belirledik
###################################################################
'''
# DATA CLEANING #
1. re.sub: Eğer textte harf dışında karaketer varsa bunu boşlukla değştirir. ^ koyarsak bunları dahil etme anlamında gelir. ^a-zA-Z, a'dan z'ye büyük küçük harf tara ama bunları değiştirme anlamına gelir
2. description.lower(): Küçük harfe çevirir.
'''
'''
# STOP-WORDS(IRRELEVANT WORDS) # 
1. description = description.split(): String grubunu kelime kelime ayırarak karşılaştırma için bir listede tutar
2. Split yerine word.tokenize() metodunu da kullanabiliriz. Tokenize kesme işareti vs. karakterleri kullanarak kelimeleri ayırabilirken split sadece boşluğa göre ayırır.
   Örneğin shouldn't kelimesini split ile "shouldn't" tokenize ile "should" ve "n't" olarak görürüz.
3. [word for word in description if not word in set(stopwords.words("english"))]: İngilizce gereksiz kelimeleri listeden çıkarır
'''
'''
# LEMMATATIZATION: Kelime köklerini bulma # 
1. nlp.WordNetLemmatizer(): Kelimelerin köklerini bulmaya yarar
2. [lemma.lemmatize(word) for word in description]: Her kelimeyi tek tek gezerek kelime köklerini bulur
3. description = " ".join(description): tüm kelimeleri boşluk kullanarak birleştirir
'''
'''
# BAG OF WORDS # 
1. max_features: Tüm kelimeler içinden, belirtilen sayı kadar en çok kullanlan kelimeleri seçmeye yarar.
2. stop_words,lowercase,tokenizer burada kullanabiliriz.
3. sparse_matrix: 0-1 lerden oluşan, her cümle için satır satır oluşturulan kelime var-yok matrisi
'''
description_list = []

for description in data.description:
#################################################################### 
# DATA CLEANING #
    description = re.sub("[^a-zA-Z]", " ", description) 
    description = description.lower()
###################################################################
    
    
#################################################################### 
# STOP-WORDS(IRRELEVANT WORDS) # 
    description = nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))]
###################################################################

#################################################################### 
# LEMMATATIZATION: Kelime köklerini bulma # 
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description] 
    
    description = " ".join(description)
    
    description_list.append(description) 
    
###################################################################       
    
#################################################################### 
# BAG OF WORDS #   
max_features = 3500
count_vectorizer = CountVectorizer(max_features = max_features)
    
x = sparse_matrix = count_vectorizer.fit_transform(description_list).toarray() # Bunlar bizim featurelarımız(x'lerimiz)

y = data.iloc[:,0].values # Gender labelları. Featureları çıkartmıştık şimdi labelları da çıkartmış olduk   

#print("Most used {} words: {}".format(max_features, count_vectorizer.get_feature_names())) 
###################################################################

#################################################################### 
# TEXT CLASSIFICATION #   
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)

gender_classifier = GaussianNB()
gender_classifier.fit(x_train,y_train)

y_predict = gender_classifier.predict(x_test).reshape(-1,1)


print("Accuracy:",gender_classifier.score(y_predict,y_test))
###################################################################
















    


