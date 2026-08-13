"""
SMS Spam Detection

Rewritten from the original notebook (SMS_spam_detection.ipynb) as a standalone script.

Two bugs from the original notebook are fixed here:

1. LEAKAGE: the original code oversampled the minority (spam) class BEFORE
   splitting into train/test. This let duplicated spam messages appear in
   both sets, inflating the reported accuracy. Fix: split first, then
   oversample only the training set.

2. JOIN BUG: the original corpus-building step used ''.join(lemm_words)
   instead of ' '.join(lemm_words), which mashed all words in a message
   into a single unreadable token with no spaces. Fix: join with a space.

3. The original predict_spam() helper used a regex that stripped LETTERS
   instead of keeping them (re.sub(pattern="[a-zA-Z]", repl='', ...)),
   which is backwards from the cleaning used everywhere else. Fixed to
   reuse the same clean_message() function as training.
"""

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_data(path: str = "SpamCollectionSMS.txt") -> pd.DataFrame:
    """Load the SMS Spam Collection dataset (tab-separated, no header).

    Pass an explicit path if the dataset isn't in the current working
    directory, e.g. train_and_evaluate(data_path=r"D:\path\to\SpamCollectionSMS.txt")

    Use a raw string (r"...") or forward slashes for Windows paths so
    backslash sequences aren't misread as escape characters.
    """
    df = pd.read_csv(path, sep="\t", names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def clean_message(text: str) -> str:
    """Lowercase, strip non-letters, remove stopwords, lemmatize."""
    message = re.sub(pattern="[^a-zA-Z]", repl=" ", string=text)
    message = message.lower()
    words = message.split()
    filtered_words = [w for w in words if w not in STOP_WORDS]
    lemm_words = [LEMMATIZER.lemmatize(w) for w in filtered_words]
    return " ".join(lemm_words)  # FIXED: was ''.join()


def oversample_minority(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Duplicate minority-class (spam) rows to roughly balance classes.

    Should only ever be called on a TRAINING set, after the train/test
    split, to avoid leaking duplicated rows into the test set.
    """
    only_spam = df[df[label_col] == 1]
    if only_spam.empty:
        return df

    count = int((df.shape[0] - only_spam.shape[0]) / only_spam.shape[0])
    oversampled = df.copy()
    for _ in range(max(count - 1, 0)):
        oversampled = pd.concat([oversampled, only_spam])
    return oversampled.reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional EDA-style features from the original notebook (word count,
    presence of currency symbols, presence of digits). Not used by the
    TF-IDF models below, kept here for parity with the original analysis.
    """
    df = df.copy()
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))

    currency_symbols = ["€", "$", "¥", "£", "₹"]
    df["contains_currency_symbols"] = df["message"].apply(
        lambda msg: int(any(sym in msg for sym in currency_symbols))
    )
    df["contains_num"] = df["message"].apply(
        lambda msg: int(any(ch.isdigit() for ch in msg))
    )
    return df


def train_and_evaluate(data_path: str = "SpamCollectionSMS.txt"):
    dataset = load_data(data_path)
    dataset = add_engineered_features(dataset)

    # --- Split FIRST, on the original (non-oversampled) data ---
    train_df, test_df = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=dataset["label"]
    )

    # --- Oversample only the training set ---
    train_df = oversample_minority(train_df)

    # --- Clean text ---
    train_df["clean_message"] = train_df["message"].apply(clean_message)
    test_df["clean_message"] = test_df["message"].apply(clean_message)

    # --- TF-IDF: fit on train only, transform test with the same vectorizer ---
    tfidf = TfidfVectorizer(max_features=500)
    x_train = tfidf.fit_transform(train_df["clean_message"]).toarray()
    x_test = tfidf.transform(test_df["clean_message"]).toarray()

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("=" * 50)
    print("Multinomial Naive Bayes")
    print("=" * 50)
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_pred_mnb = mnb.predict(x_test)
    print(classification_report(y_test, y_pred_mnb))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_mnb))

    print("\n" + "=" * 50)
    print("Decision Tree")
    print("=" * 50)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    print(classification_report(y_test, y_pred_dt))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_dt))

    def predict_spam(sms: str) -> int:
        """Predict whether a single SMS message is spam (1) or ham (0)."""
        message = clean_message(sms)  # FIXED: reuse the correct cleaning fn
        vec = tfidf.transform([message]).toarray()
        return int(dt.predict(vec)[0])

    return {
        "tfidf": tfidf,
        "mnb": mnb,
        "dt": dt,
        "predict_spam": predict_spam,
    }


if __name__ == "__main__":
    train_and_evaluate()
