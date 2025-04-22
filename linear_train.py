import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
df = pd.read_csv('email_phishing_data.csv')
features = [
  'num_words','num_unique_words','num_stopwords','num_links',
  'num_unique_domains','num_email_addresses','num_spelling_errors','num_urgent_keywords'
]
df[features] = MinMaxScaler().fit_transform(df[features])
X = df.drop(['label'], axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
model_cv = LogisticRegressionCV(cv=10, max_iter=50000)
model_cv.fit(X_train, y_train)
y_pred = model_cv.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(pd.Series(model_cv.coef_[0], index=features).sort_values())
print(Counter(y_train), Counter(y_test))
print("Baseline:", max(Counter(y_test).values())/len(y_test))
