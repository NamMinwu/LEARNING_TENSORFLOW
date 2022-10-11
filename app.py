import tensorflow as tf
import pandas as pd
import numpy as np
import os

# checkpoint_path="training_1/cp.ckpt"
#using CSV
def preprocess():
    data = pd.read_csv('gpascore.csv')
    # print(data.isnull().sum()) 빈부분 확인
    data = data.dropna()
    y_data = data['admit'].values
    x_data = [ ]

    for i, rows in data.iterrows():
        x_data.append([ rows['gre'], rows['gpa'], rows['rank'] ])

    return x_data, y_data
#useful fuction

# print(data['gpa'])->그 부분만 가져옴
# print(data['gpa'].min())
# print(data['gpa'].count())->갯수파악
# data.fillna()빈 부분 추가


#Making model
def create_model():
    model = tf.keras.models.Sequential([  
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    #model compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

#Making checkpoint
def create_checkpoint(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     period=5,
                                                     verbose=1)
    return cp_callback, checkpoint_dir


x_data, y_data = preprocess()
checkpoint, checkpointDir = create_checkpoint("training_1/cp.ckpt")
model = create_model()

#fit(입력, 출력, 반복)->학습을 해주세요
# model.fit(np.array(x_data), 
#           np.array(y_data), 
#           callbacks=[checkpoint],
#           epochs=1000)
os.listdir(checkpointDir)

#predit
predict = model.predict([[750, 3.70, 3],[400, 2.2, 1]])
