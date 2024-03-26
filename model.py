
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
    model = VegeBackscatCoefLstm()

