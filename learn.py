from imports import *
from parameters import *
from data_management import *
from preprocessing import *

# create the LSTM model


def lstm_model(vocab_size, embedding_dim):

    # design the lstm model
    model = tf.keras.Sequential(
        [
            # Add an Embedding layer expecting input vocab of size 5000,
            # and output embedding dimension of size 64 we set at the top
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
            # use ReLU in place of tanh function since they are very good alternatives
            # of each other.
            tf.keras.layers.Dense(embedding_dim, activation="relu"),
            # Add a Dense layer with 6 units and softmax activation.
            # When we have multiple outputs, softmax convert outputs layers
            # into a probability distribution.
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )

    return model


# train the model

def train(
    model,
    num_epochs,
    train_padded,
    train_label_seq,
    validation_padded,
    validation_label_seq,
    verbose_num,
):

    # set loss, optimizer and metric
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    history = model.fit(
        train_padded,
        train_label_seq,
        epochs=num_epochs,
        validation_data=(validation_padded, validation_label_seq),
        verbose=verbose_num,
    )

    # save the model
    model.save(os.path.join(MODEL_PATH, "lstm_model.h5"))

    return history


# plot graphs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel("Train_" + string)
    plt.legend(["Train_" + string, "val_" + string])
    plt.show()


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

    # tokenize articles

    tokenizer = articles_tokenizer(
        train_articles=my_articles, vocab_size=VOCAB_SIZE, oov_tok=OOV_TOK
    )

    # convet articles into sequences

    train_padded = sequences_converter(
        tokenizer=tokenizer,
        articles=train_articles,
        max_length=MAX_LENGTH,
        padding_type=PADDING_TYPE,
        trunc_type=TRUNC_TYPE,
    )

    validation_padded = sequences_converter(
        tokenizer=tokenizer,
        articles=validation_articles,
        max_length=MAX_LENGTH,
        padding_type=PADDING_TYPE,
        trunc_type=TRUNC_TYPE,
    )

    # tokenize labels

    label_tokenizer = labels_tokenizer(labels=my_labels)

    train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))

    validation_label_seq = np.array(
        label_tokenizer.texts_to_sequences(validation_labels)
    )

    # train the model

    model = lstm_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

    history = train(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_padded=train_padded,
        train_label_seq=train_label_seq,
        validation_padded=validation_padded,
        validation_label_seq=validation_label_seq,
        verbose_num=2,
    )

    # plot graphs
    plot_graphs(history, "accuracy")

    print("------Training End-------")



