import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adadelta,Adam
from tensorflow.keras import regularizers
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
from tensorflow.random import set_seed
from numpy.random import seed
import pickle


#define model parameters
tf.keras.backend.set_epsilon(1e-9)
cIntInputSeqLen=384
cIntOutputSeqLen=128
cIntProcFeatures=1
cIntHiddenNode=64
cIntMaskValue=0
seed(3)
set_seed(3)


#define customloss
class CustomLoss:
    @staticmethod
    def RMSE(y_true, y_pred):
        return kb.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))

    @staticmethod
    def MaskedRMSE(y_true, y_pred):
        isMask = kb.equal(y_true, 0)
        isMask = kb.all(isMask, axis=-1)
        isMask = kb.cast(isMask, dtype=kb.floatx())
        isMask = 1 - isMask
        isMask = kb.reshape(isMask, tf.shape(y_true))
        masked_squared_error = kb.square(isMask * (y_true - y_pred))
        masked_mse = kb.sum(masked_squared_error, axis=-1) / (kb.sum(isMask, axis=-1) + kb.epsilon())
        return kb.sqrt(masked_mse)

    @staticmethod
    def MaskedMSE(y_true, y_pred):
        isMask = kb.equal(y_true, 0)
        isMask = kb.all(isMask, axis=-1)
        isMask = kb.cast(isMask, dtype=kb.floatx())
        isMask = 1 - isMask
        isMask = kb.reshape(isMask, tf.shape(y_true))
        masked_squared_error = kb.square(isMask * (y_true - y_pred))
        masked_mse = kb.sum(masked_squared_error, axis=-1) / (kb.sum(isMask, axis=-1) + kb.epsilon())
        return masked_mse

    @staticmethod
    def MaskedMAE(y_true, y_pred):
        isMask = kb.equal(y_true, 0)
        isMask = kb.all(isMask, axis=-1)
        isMask = kb.cast(isMask, dtype=kb.floatx())
        isMask = 1 - isMask
        isMask = kb.reshape(isMask, tf.shape(y_true))
        masked_AE = kb.abs(isMask * (y_true - y_pred))
        masked_mae = kb.sum(masked_AE, axis=-1) / (kb.sum(isMask, axis=-1) + kb.epsilon())
        return masked_mae

    # numpy function wrapper
    @staticmethod
    @tf.function
    def MaskedMAPE(y_true, y_pred):
        return tf.py_function(CustomLoss.numpyMaskedMAPE, (y_true, y_pred), tf.double)

    @staticmethod
    def numpyMaskedMAPE(y_true, y_pred):
        MapeLst = list()
        Mape_mean = list()
        # for elm_t,elm_p in zip(y_true,y_pred):
        elm_t0 = y_true[0]
        elm_p0 = y_pred[0]
        for batch_t, batch_p in zip(elm_t0, elm_p0):
            true = batch_t[0:np.count_nonzero(batch_t)]
            pred = batch_p[0:np.count_nonzero(batch_t)]
            mape_batch = np.mean(np.abs((pred - true) / true) * 100)
            MapeLst.append(mape_batch)
        Mape_mean.append(np.mean(MapeLst))
        elm_t1 = y_true[1]
        elm_p1 = y_pred[1]
        for batch_t, batch_p in zip(elm_t1, elm_p1):
            true = batch_t[0:np.count_nonzero(batch_t)]
            pred = batch_p[0:np.count_nonzero(batch_t)]
            mape_batch = np.mean(np.abs((pred - true) / true) * 100)
            MapeLst.append(mape_batch)
        Mape_mean.append(np.mean(MapeLst))

        return np.array(Mape_mean, dtype=float)

# Generates samples of data
def BuildSeqs(Cap, IR):
    # declare list for input capacity and input IR as ls1 and ls2
    # declare list for output capacity and output IR as ls3 and ls4
    ls1, ls2, ls3, ls4 = list(), list(), list(), list()
    for SelectCap, SelectIR in zip(Cap, IR):
        if (len(SelectIR) < len(SelectCap)):
            SelectCap = SelectCap[0:len(SelectIR)]
        elif (len(SelectCap) < len(SelectIR)):
            SelectIR = SelectIR[0:len(SelectCap)]
        SelectIR = SelectIR / 0.04 * 100
        SelectCap = SelectCap / 1.85 * 100
        x_lst = []
        x_lst2 = []
        y_lst = []
        y_lst2 = []
        for i in range(20, len(SelectIR) - 20, 1):
            splitPos = i
            inputSeq = SelectCap[0:splitPos]
            x_lst.append(inputSeq.reshape(-1, 1))
            inputSeq2 = SelectIR[0:splitPos]
            x_lst2.append(inputSeq2.reshape(-1, 1))
            OutputSeq = SelectCap[splitPos - 1::4].tolist()
            y_lst.append(OutputSeq)
            OutputSeq2 = SelectIR[splitPos - 1::4].tolist()
            y_lst2.append(OutputSeq2)
        # zero padding
        Proc_X = tf.keras.preprocessing.sequence.pad_sequences(x_lst, maxlen=cIntInputSeqLen, dtype='float64',
                                                               padding='pre', value=0)
        Proc_X2 = tf.keras.preprocessing.sequence.pad_sequences(x_lst2, maxlen=cIntInputSeqLen, dtype='float64',
                                                                padding='pre', value=0)
        Proc_Y1 = tf.keras.preprocessing.sequence.pad_sequences(y_lst, maxlen=cIntOutputSeqLen, dtype='float64',
                                                                padding='post', value=0)
        Proc_Y2 = tf.keras.preprocessing.sequence.pad_sequences(y_lst2, maxlen=cIntOutputSeqLen, dtype='float64',
                                                                padding='post', value=0)
        Proc_X = Proc_X.reshape(-1, cIntInputSeqLen, cIntProcFeatures)
        Proc_X2 = Proc_X2.reshape(-1, cIntInputSeqLen, cIntProcFeatures)
        Proc_Y1 = Proc_Y1.reshape(-1, cIntOutputSeqLen, cIntProcFeatures)
        Proc_Y2 = Proc_Y2.reshape(-1, cIntOutputSeqLen, cIntProcFeatures)
        for a, b, c, d in zip(Proc_X, Proc_X2, Proc_Y1, Proc_Y2):
            ls1.append(a)
            ls2.append(b)
            ls3.append(c)
            ls4.append(d)

    return (np.array(ls1, dtype=np.float), np.array(ls2, dtype=np.float)), (
    np.array(ls3, dtype=np.float), np.array(ls4, dtype=np.float))


#Model structure

#model structure definition
#input part
InputCap=tf.keras.layers.Input(shape=(cIntInputSeqLen,cIntProcFeatures))
MaskedInputCap=tf.keras.layers.Masking(mask_value=cIntMaskValue)(InputCap)
InputIR=keras.layers.Input(shape=(cIntInputSeqLen,cIntProcFeatures))
MaskedInputIR=tf.keras.layers.Masking(mask_value=cIntMaskValue)(InputIR)
CombInput = keras.layers.Concatenate(axis=-1)([MaskedInputCap, MaskedInputIR])
#encoder part
EncCtext=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(CombInput)
EncCtext=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(EncCtext)
EncCtext=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(EncCtext)
EncCtextOut=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=False))(EncCtext)
CombCtext=tf.keras.layers.RepeatVector(cIntOutputSeqLen)(EncCtextOut)
#decoder part for capacity
Dec1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(CombCtext)
Dec1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(Dec1)
Dec1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(Dec1)
Dec1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(Dec1)
Dec1=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(cIntHiddenNode*2, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))(Dec1)
Dec1=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(cIntHiddenNode/2, activation="relu"))(Dec1)
DecOutCap=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="relu"))(Dec1)
#decoder part for IR
Dec2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(CombCtext)
Dec2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(Dec2)
Dec2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(Dec2)
Dec2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cIntHiddenNode,return_sequences=True))(Dec2)
Dec2=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(cIntHiddenNode*2, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))(Dec2)
Dec2=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(cIntHiddenNode/2, activation="relu"))(Dec2)
DecOutIR=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="relu"))(Dec2)

model=tf.keras.Model(inputs=[InputCap,InputIR],outputs=[DecOutCap,DecOutIR])

#load training data
trCap=pickle.load(open("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/trCap.p","rb"))
trIR=pickle.load(open("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/trIR.p","rb"))
vaCap=pickle.load(open("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/vaCap.p","rb"))
vaIR=pickle.load(open("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/vaIR.p","rb"))


#generate training data
x0,y0=BuildSeqs(trCap,trIR)
x1,y1=BuildSeqs(vaCap,vaIR)


#generate testdata
teCap=pickle.load(open("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/teCap.p","rb"))
teIR=pickle.load(open("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/teIR.p","rb"))
x2, y2=BuildSeqs(teCap,teIR)


# EQUAL WEIGHTS  (model1)

tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir='/theia/scratch/brussel/102/vsc10208/MTL BATTERY/tensor', histogram_freq=5, write_graph=True)


#train model for stage 2
checkpoint_path0 = "/theia/scratch/brussel/102/vsc10208/MTL BATTERY/hdf51"
checkpoint_path1 = "/theia/scratch/brussel/102/vsc10208/MTL BATTERY/hdf52"
checkpoint_path2 = "/theia/scratch/brussel/102/vsc10208/MTL BATTERY/hdf53"
#model.load_weights('capir/weight1_best.hdf5')
#define checkpoint and early stopping callback
cp_callback0 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path0, save_weights_only=True,verbose=1,save_freq='epoch')
cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path1, save_weights_only=True,verbose=1,save_freq='epoch')
cp_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path2, save_weights_only=True,verbose=1,save_freq='epoch')
callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=32,restore_best_weights=True)

#define model parameters
tf.keras.backend.set_epsilon(1e-9)
cIntInputSeqLen=384
cIntOutputSeqLen=128
cIntProcFeatures=1
cIntHiddenNode=64
cIntMaskValue=0
#laod model stage 1
for w1 in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]:
  model1=keras.models.load_model("/theia/scratch/brussel/102/vsc10208/MTL BATTERY/2CapIR2_stg1_new_seed0.h5", compile=False)
  #defreeze all weights for stage 2
  for idx in range(len(model1.layers)):
    model1.layers[idx ].trainable=True
  #train model for stage 2
  #model.load_weights('capir/weight1_best.hdf5')
  model1.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),loss=CustomLoss.MaskedMAE,metrics=[CustomLoss.MaskedMAE],loss_weights=([w1,2-w1]))
  #model.load_weights('capir/weight2_best.hdf5')
  model1.fit(x0,y0,batch_size=512,epochs=150,verbose=1,validation_data=(x1,y1),shuffle=True,callbacks=[cp_callback2, callback2])
  #save model
  y_pred_1=model1.predict(x2)
  results_1=CustomLoss.numpyMaskedMAPE(y2, y_pred_1)
  print("task weight" + str(w1) + "result")
  print(results_1)

