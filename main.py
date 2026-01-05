

import pandas as pd
# splits data
from sklearn.model_selection import train_test_split
# Converts text into numbers
from sklearn.feature_extraction.text import TfidfVectorizer
# Niave_Bayes =spam classifier
from sklearn.naive_bayes import MultinomialNB
# Checks performance
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


df = pd.read_csv(r"spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Converts label into numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

x = df['message']
y = df['label']

# Split data (Train and Test)
X_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Converts text into numbers
vectorizer = TfidfVectorizer(
    stop_words='english', ngram_range=(1, 2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(x_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Test the model

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", accuracy)

while True:
    user_input = input("Enter to check or type quit to exit: ")

    if user_input.lower() == 'quit':
        break
    text_tfidf = vectorizer.transform([user_input])
    result = model.predict(text_tfidf)

    if result[0] == 1:
        print("Spam")
    else:
        print("Not Spam")

    # print(check_spam("Congratulations ! You have won a lottery"))
    # print(check_spam("Hello. How are you?"))
    # print(check_spam("Congratulations! You have won a free prize. Claim now"))
    # print(check_spam("Win cash now! Call this number immediately"))
