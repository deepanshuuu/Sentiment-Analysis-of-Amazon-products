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
