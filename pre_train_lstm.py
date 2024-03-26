
'''
    Pre-training dnn with simulated dataset
    generated by the single-scattering RTM

    @author Zhu dong
    @data 2023.6.27
'''


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from model import VegeBackscatCoefLstm
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def loadData(x_train_path, y_train_path, x_val_path, y_val_path):

    temp_train_x = h5py.File(x_train_path)
    temp_train_y = h5py.File(y_train_path)
    train_x = np.array(temp_train_x['train_x'])
    train_y = np.array(temp_train_y['train_y'])

    train_x = train_x.reshape((train_x.shape[0], 9, 1))
    train_y = train_y.reshape((train_y.shape[0], 1))

    temp_val_x = h5py.File(x_val_path)
    temp_val_y = h5py.File(y_val_path)
    val_x = np.array(temp_val_x['val_x'])
    val_y = np.array(temp_val_y['val_y'])

    val_x = val_x.reshape((val_x.shape[0], 9, 1))
    val_y = val_y.reshape((val_y.shape[0], 1))

    return train_x, train_y, val_x, val_y

# flag = 'S'

flag = 'C'

x_train_path = '1027/data/pretrainData/' + flag + '/train_x.mat'
y_train_path = '1027/data/pretrainData/' + flag + '/train_y.mat'

x_val_path = '1027/data/pretrainData/' + flag + '/val_x.mat'
y_val_path = '1027/data/pretrainData/' + flag + '/val_y.mat'

train_x, train_y, val_x, val_y = loadData(x_train_path, y_train_path, x_val_path, y_val_path)

# normalize
train_y = train_y / -30
val_y = val_y / -30

## ------ Model -------##
model = VegeBackscatCoefLstm()

model.compile(loss='mse',
              optimizer=Adam(lr=0.001),
              metrics=[metrics.mse])

batch = 32
epochs = 500
checkPoint = ModelCheckpoint('1027/model/pretrain/' + flag + '/lstm_SGD_1027.hdf5', monitor='loss', save_best_only=True, period=1)
history = model.fit(train_x, train_y, batch_size=batch, epochs=epochs, shuffle=True,
                    verbose=1, validation_data=(val_x, val_y), callbacks=checkPoint)

# name = 'history/pre_train_1/'
# pd.DataFrame(history.history).to_csv(name + flag + '_lstm_SGD_705.csv', index=False)




fig = plt.figure()
loss = history.history['loss']
val_loss = history.history['val_loss']
ep = range(len(loss))
plt.plot(ep, loss, 'g', label='Train loss')
# plt.plot(ep, val_loss, 'b', label='Validation loss')
plt.show()






