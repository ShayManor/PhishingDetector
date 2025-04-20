import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
df = pd.read_csv('email_phishing_data.csv')
df['num_words_norm'] = scalar.fit_transform(df[['num_words']])
df['num_unique_words_norm'] = scalar.fit_transform(df[['num_unique_words']])
df['num_stopwords_norm'] = scalar.fit_transform(df[['num_words']])
df['num_links_norm'] = scalar.fit_transform(df[['num_links']])
df['num_unique_domains_norm'] = scalar.fit_transform(df[['num_unique_domains']])
df['num_email_addresses_norm'] = scalar.fit_transform(df[['num_email_addresses']])
df['num_spelling_errors_norm'] = scalar.fit_transform(df[['num_spelling_errors']])
df['num_urgent_keywords_norm'] = scalar.fit_transform(df[['num_urgent_keywords']])
X = df.drop(['label', 'num_words', 'num_unique_words', 'num_stopwords', 'num_links', 'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 'num_urgent_keywords'], axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
model = LogisticRegression(multi_class='ovr', max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
from collections import Counter
print(Counter(y_train), Counter(y_test))
print("Baseline:", max(Counter(y_test).values())/len(y_test))
