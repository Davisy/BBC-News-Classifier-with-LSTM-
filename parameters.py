from imports import *

# set parameters

STOPWORDS = set(stopwords.words("english"))

VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
MAX_LENGTH = 200
TRUNC_TYPE = "post"
PADDING_TYPE = "post"
OOV_TOK = "<OOV>"
TRAIN_PORTION = 0.8
TEST_SIZE = 100
FILENAME = "bbc_text.csv"
DATA_PATH = "data/"
MODEL_PATH = "models/"
NUM_EPOCHS = 4
