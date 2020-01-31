from imports import *
from parameters import *
from data_management import *
from preprocessing import *


def load_model(model_name):

    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model(os.path.join(MODEL_PATH, model_name))

    return model


if __name__ == "__main__":

    # load data
    my_articles, my_labels = load_data(data_path=DATA_PATH, file_name=FILENAME)

    # clearn the data
    my_articles = clean_articles(articles=my_articles, stopwords=STOPWORDS)

    # split articles
    _, _, test_articles = split_articles(
        articles=my_articles, train_portion=TRAIN_PORTION, test_size=TEST_SIZE
    )

    # split labels
    _, _, test_labels = split_labels(
        labels=my_labels, train_portion=TRAIN_PORTION, test_size=TEST_SIZE
    )

    # tokenize articles

    tokenizer = articles_tokenizer(
        train_articles=my_articles, vocab_size=VOCAB_SIZE, oov_tok=OOV_TOK
    )

    # convet test rticles into sequences

    test_padded = sequences_converter(
        tokenizer=tokenizer,
        articles=test_articles,
        max_length=MAX_LENGTH,
        padding_type=PADDING_TYPE,
        trunc_type=TRUNC_TYPE,
    )

    # tokenize labels

    label_tokenizer = labels_tokenizer(labels=my_labels)
    test_label_seq = np.array(label_tokenizer.texts_to_sequences(test_labels))

    # load the model
    model = load_model("lstm_model.h5")

    # test the model with test padded
    loss, acc = model.evaluate(test_padded, test_label_seq, verbose=2)

    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
