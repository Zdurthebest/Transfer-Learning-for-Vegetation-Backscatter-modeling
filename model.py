
'''

    Two baseline networks
    namely:
    1. DNN
    2. LSTM

    Note that, although the layers are equal for both baseline networks, the total
    number of the parameters of LSTM are larger than DNN, due to the unique structure.

    @author Zhudong
    @date   2023.5.24

'''


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, ReLU


def VegeBackscatCoefDnn():
    inputs = Input(shape=(1, 9))
    outputs = Dense(32, kernel_initializer='uniform')(inputs)
    outputs = ReLU()(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(64, kernel_initializer='uniform')(outputs)
    outputs = ReLU()(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(128, kernel_initializer='uniform')(outputs)
    outputs = ReLU()(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(1)(outputs)
    outputs = ReLU()(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="DNN")
    model.summary()
    return model


def VegeBackscatCoefLstm():
    input = Input(shape=(9, 1))
    out = LSTM(32, return_sequences=True)(input)
    out = ReLU()(out)
    out = Dropout(0.2)(out)
    out = LSTM(64, return_sequences=True)(out)
    out = ReLU()(out)
    out = Dropout(0.2)(out)
    out = LSTM(128, return_sequences=False)(out)
    out = ReLU()(out)
    out = Dropout(0.2)(out)
    out = Dense(1)(out)
    model = Model(inputs=input, outputs=out, name='Lstm')
    model.summary()
    return model



if __name__ == "__main__":
    model1 = VegeBackscatCoefDnn()
    model2 = VegeBackscatCoefLstm()

