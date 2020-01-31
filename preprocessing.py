from imports import *
from parameters import *
from data_management import load_data


# clean the articles


def clean_articles(articles, stopwords):

    cleaned_articles = []

    for article in articles:

        # remove html content
        review_text = BeautifulSoup(article).get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # change into small letters all characters
        review_text = review_text.lower()

        # remove all stopwords
        for word in stopwords:
            token = " " + word + " "
            review_text = review_text.replace(token, " ")
            review_text = review_text.replace(" ", " ")

        cleaned_articles.append(review_text)

    return cleaned_articles


# function to split article into train,validation and test


def split_articles(articles, train_portion, test_size):

    train_size = int(len(articles) * train_portion)

    train_articles = articles[0:train_size]

    other_articles = articles[train_size:]

    validation_articles = other_articles[test_size:]

    test_articles = other_articles[0:test_size]

    return train_articles, validation_articles, test_articles


def split_labels(labels, train_portion, test_size):

    train_size = int(len(labels) * train_portion)

    train_labels = labels[0:train_size]

    other_labels = labels[train_size:]

    validation_labels = other_labels[test_size:]

    test_labels = other_labels[0:test_size]

    return train_labels, validation_labels, test_labels


# tokenize the articles


def articles_tokenizer(train_articles, vocab_size, oov_tok):

    articles_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    articles_tokenizer.fit_on_texts(train_articles)

    return articles_tokenizer


# tokenize the labels


def labels_tokenizer(labels):
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    return label_tokenizer


# convert text into sequences
def sequences_converter(tokenizer, articles, max_length, padding_type, trunc_type):

    sequences = tokenizer.texts_to_sequences(articles)

    # padding the sequences
    padded = pad_sequences(
        sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )

    return padded


if __name__ == "__main__":

    # load data
    my_articles, my_labels = load_data(data_path=DATA_PATH, file_name=FILENAME)

    # clearn the data
    my_articles = clean_articles(articles=my_articles, stopwords=STOPWORDS)

    # split articles
    train_articles, validation_articles, test_articles = split_articles(
        articles=my_articles, train_portion=TRAIN_PORTION, test_size=TEST_SIZE
    )

    # split labels
    train_labels, validation_labels, test_labels = split_labels(
        labels=my_labels, train_portion=TRAIN_PORTION, test_size=TEST_SIZE
    )

    print("Train Articles:", len(train_articles))
    print("validation Articles:", len(validation_articles))
    print("Test Articles:", len(test_articles))
    print("-------------------")
    print("Train Labels:", len(train_labels))
    print("validation Labels:", len(validation_labels))
    print("Test Labels:", len(test_labels))
