# this is where we will use both of our .py files
# to train the model

# TO-DO
# TRAIN / TEST
# CROSS-VALIDATION
# ENSEMBLE

from sklearn.model_selection import train_test_split
import preprocessing as pp

stop_words = pp.get_nltk_data()
x, y = pp.csv_to_dataframe(stop_words)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)