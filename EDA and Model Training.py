import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re
import os
os.makedirs('Models', exist_ok=True)
df = pd.read_csv(r"C:\Users\admin\Downloads\amazon_alexa.tsv" , delimiter = '\t', quoting = 3)
#print(df.head(10))
#print(df['rating'])
#print(df.columns)
#print(df.isnull().sum())
#print(df['verified_reviews'].isnull == True)
df.dropna(inplace=True)
#print(df.shape)
#print(df.iloc[10]['verified_reviews'])
#print(len(df.iloc[10]['verified_reviews']))
#print(df['rating'].value_counts())
df['rating'].value_counts().plot.bar(color = 'red')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
#plt.show()
#print(round(df['rating'].value_counts()/df.shape[0]*100,2))
fig = plt.figure(figsize=(7,7))
colors = ('red', 'green', 'blue','orange','yellow')
wp = {'linewidth':1, "edgecolor":'black'}
tags = df['rating'].value_counts()/df.shape[0]
explode=(0.1,0.1,0.1,0.1,0.1)
tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of rating')
#plt.show()

#print(df['feedback'].value_counts())
#print(df[df['feedback'] == 0].iloc[1]['verified_reviews'])
#print(df[df['feedback'] == 1].iloc[1]['verified_reviews'])
df['feedback'].value_counts().plot.bar(color = 'blue')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
#plt.show()
#print(round(df['feedback'].value_counts()/df.shape[0]*100,2))

cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(df.verified_reviews)

reviews = " ".join([review for review in df['verified_reviews']])
wc = WordCloud(background_color='white', max_words=50)
plt.figure(figsize=(10, 10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
#plt.show()

neg_reviews = " ".join([review for review in df[df['feedback'] == 0]['verified_reviews']])
neg_reviews = neg_reviews.lower().split()

pos_reviews = " ".join([review for review in df[df['feedback'] == 1]['verified_reviews']])
pos_reviews = pos_reviews.lower().split()

unique_negative = [x for x in neg_reviews if x not in pos_reviews]
unique_negative = " ".join(unique_negative)

unique_positive = [x for x in pos_reviews if x not in neg_reviews]
unique_positive = " ".join(unique_positive)

wc = WordCloud(background_color='white', max_words=50)
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('Wordcloud for negative reviews', fontsize=10)
plt.axis('off')
#plt.show()

wc = WordCloud(background_color='white', max_words=50)
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('Wordcloud for positive reviews', fontsize=10)
plt.axis('off')
#plt.show()

corpus = []
stemmer = PorterStemmer()
for i in range(0, df.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', df.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)
  #print(corpus)

cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()
y = df['feedback'].values
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))
#print(X.shape)
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))

model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)
#print(model_rf.score(X_train_scl, y_train))
#print(model_rf.score(X_test_scl, y_test))

y_preds = model_rf.predict(X_test_scl)
cm = confusion_matrix(y_test, y_preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
cm_display.plot()
#plt.show()

accuracies = cross_val_score(estimator = model_rf, X = X_train_scl, y = y_train, cv = 10)

#print(accuracies.mean())
#print(accuracies.std())
params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}
cv_object = StratifiedKFold(n_splits = 2)
grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_scl, y_train.ravel())
#print(grid_search.best_params_)

model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)
#print(model_xgb.score(X_train_scl, y_train))
#print(model_xgb.score(X_test_scl, y_test))
y_preds = model_xgb.predict(X_test)
cm = confusion_matrix(y_test, y_preds)
#print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
cm_display.plot()
#plt.show()
pickle.dump(model_xgb, open('Models/model_xgb.pkl', 'wb'))

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)
#print(model_dt.score(X_train_scl, y_train))
#print(model_dt.score(X_test_scl, y_test))
y_preds = model_dt.predict(X_test)
cm = confusion_matrix(y_test, y_preds)
#print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
cm_display.plot()
#plt.show()
pickle.dump(model_dt, open('Models/model_dt.pkl', 'wb'))