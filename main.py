from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Dropout, concatenate, multiply, Input
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend


def read_user_id():
    with open('./input.txt', 'r') as f:
        return [l.strip().split(',') for l in  f.readlines()]


def write_output(prediction):
    with open('./output.txt', 'w') as f:
        for pred in prediction:
            f.write(pred+"\n")


def build_MLP(n_users, n_items):

    # Build model
    dim_embedding_user = 50
    dim_embedding_item = 50

    ## item embedding
    item_input= Input(shape=[1], name='item')
    item_embedding = Embedding(n_items + 1, dim_embedding_item, name='Item-Embedding')(item_input)
    item_vec = Flatten(name='Item-Flatten')(item_embedding)
    item_vec = Dropout(0.2)(item_vec)

    ## user embedding
    user_input = Input(shape=[1], name='User')
    user_embedding = Embedding(n_users + 1, dim_embedding_user, name ='User-Embedding')(user_input)
    user_vec = Flatten(name ='User-Flatten')(user_embedding)
    user_vec = Dropout(0.2)(user_vec)

    ## concatenate flattened values
    concat = concatenate([item_vec, user_vec])
    concat_dropout = Dropout(0.2)(concat)

    ## add dense layer (can try more)
    dense_1 = Dense(50, name ='Dense1', activation='relu')(concat)
    dropout_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(20, activation="relu", name = "Dense2")(dropout_1)
    dropout_2 = Dropout(0.2)(dense_2)
    dense_3 = Dense(10, activation="relu", name = "Dense3")(dropout_2)
    dropout_3 = Dropout(0.2)(dense_3)

    ## define output (can try sigmoid instead of relu)
    result = Dense(1, activation ='relu',name ='Activation')(dropout_3)

    ## define model with 2 inputs and 1 output
    return Model(inputs=[user_input, item_input], outputs=result, name="MLP")



def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


if __name__ == "__main__":
    df_train = pd.read_csv('data/ratings_train.csv', usecols = ['userId', 'movieId', 'rating'])
    df_valid = pd.read_csv('data/ratings_vali.csv', usecols = ['userId', 'movieId', 'rating'])

    # prepare matix shape
    total_user = np.concatenate((df_train.userId.unique(), df_valid.userId.unique()))
    total_item = np.concatenate((df_train.movieId.unique(), df_valid.movieId.unique()))
    n_users, n_items = max(total_user), max(total_item)

    ## define model
    recommender = build_MLP(n_users, n_items)
    # recommender.summary()

    # compile model
    opt_adam = Adam(lr = 0.002)
    recommender.compile(optimizer=Adam(lr = 0.002),
            loss= ['mse'], metrics=['accuracy', rmse ])


    ## fit model
    track_training = recommender.fit([df_train['userId'], df_train['movieId']],
                                    df_train['rating'],
                                    batch_size = 256,
                                    validation_data=(
                                        [df_valid['userId'], df_valid['movieId']],
                                        df_valid['rating']),
                                    validation_split = 0.005,
                                    epochs = 10,
                                    verbose = 0)
    # store model weights
    TRAINED_PARAM = 'param.data'
    recommender.save_weights(TRAINED_PARAM)
#     recommender.load_weights(TRAINED_PARAM)

    # predict requests
    inputs = read_user_id()
    predictions = []
    for user, movie in inputs:
        target = [[int(user)],[int(movie)]]
        predict = recommender.predict(target)[0][0]
        predict = round(predict, 9)
        predictions.append('{},{},{}'.format(user, movie, str(predict)))
    write_output(predictions)
