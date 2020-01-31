from imports import *
from parameters import DATA_PATH, FILENAME


def load_data(data_path, file_name):
    # load the dataset and remove all stopwords
    articles = []
    labels = []

    with open(os.path.join(data_path, file_name), "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            labels.append(row[0])
            articles.append(row[1])

    return articles, labels


if __name__ == "__main__":

	# load dataset
    articles, labels = load_data(
        data_path=DATA_PATH, file_name=FILENAME
    )
    
    print("labels length:", len(labels))
    print("articles length:", len(articles))
    print("First News:", articles[:1])
    print("Labels:", list(set(labels)))
