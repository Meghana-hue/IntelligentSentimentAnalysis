import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (only the first time)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))

# Keep important negation words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")

# Keep contraction parts that affect sentiment
stop_words.discard("don't")
stop_words.discard("didn't")
stop_words.discard("won't")
stop_words.discard("isn't")
stop_words.discard("wasn't")
stop_words.discard("aren't")
stop_words.discard("couldn't")
stop_words.discard("shouldn't")
# Keep negation words for sentiment analysis
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)