import copy
from enum import Enum, unique

import torch

from Requirements import *
from RandomScaleCrop import *

@unique
class DataName(Enum): #supported datasets
    Battery_Aachen="Battery_Aachen"
    Battery_MIT="Battery_MIT"



class Dataset():
    def __init__(self,  number_of_features: object, NTask: object) -> object:

        self.NFeat=number_of_features
        self.NTask=NTask


class Battery(Dataset):

    def __init__(self, number_of_features=384, ntask=2, MIT=False):
        Dataset.__init__(self, number_of_features, ntask)
        self.MIT=MIT

    def Sequence_Builder(self, capacity_sequence, resistance_sequence, val=False): #cap en IR are arrays containing sequences for each battery cell. For training, Cap contains 29 arrays corresponding to 29 cells. Each of these arrays contains a variable sequence length.

        if self.MIT==False or val==True:
            self.splitpos=20
        else:
            self.splitpos=1000

        X, Y = list(), list()  # create empty lists that will collect all windowed and padded sequences
        X_length, Y_length = list(), list()
        i = 0
        for capseries, irseries in zip(capacity_sequence, resistance_sequence):  # select capacity and Ir serie of one cell

        # First step is to make sure the capacity and resistance serie of each cell or equal in length. This is needed because we treat both of them as a feature of the same serie.
            if len(capseries) < len(irseries):  # if capseries shorter, then ir is reduced to this size
                irseries = irseries[0:len(capseries)]
            elif len(capseries) > len(irseries):
                capseries = capseries[0:len(irseries)]

            irseries = irseries / 0.04 * 100
            capseries = capseries / 1.85 * 100

        # Create X AND Y sequences FOR capacity AND resistance. Apply windowing.

            if i == 0:
                x_cap = []
                y_cap = []

                x_ir = []
                y_ir = []
                i += 1

            for ind in range(self.splitpos, len(irseries) - self.splitpos, 20):  # increasing window for Xcap and Xir, decreasing window for ycap and yir. X[0]=
                splitPos = ind

                x_cap.append(torch.tensor(capseries[0:splitPos]).reshape(-1, 1))  # reshape adds dim such that we can conc later
                x_ir.append(torch.tensor(irseries[0:splitPos]).reshape(-1, 1))

                y_cap.append(torch.tensor(capseries[splitPos - 1::4].reshape(-1,
                                                                         1)))  # from split pos to end but skip every four points (reduces number of features)
                y_ir.append(torch.tensor(irseries[splitPos - 1::4].reshape(-1, 1)))

        # Time for padding =>

        Padded_x_cap = torch.nn.utils.rnn.pad_sequence(x_cap, batch_first=True,padding_value=0)  # gives back tensor of shape (#sequences, #max_seq_length, 1)
        Padded_x_ir = torch.nn.utils.rnn.pad_sequence(x_ir, batch_first=True, padding_value=0)

        Padded_y_cap = torch.nn.utils.rnn.pad_sequence(y_cap, batch_first=True, padding_value=0)
        Padded_y_ir = torch.nn.utils.rnn.pad_sequence(y_ir, batch_first=True, padding_value=0)

    # add sequences of this cell to the final list of sequences

        X = torch.cat((Padded_x_cap, Padded_x_ir), dim=-1)
        X_length = Padded_x_cap.size()[1]  # save max seq length (len tensor, needed for packing later)

    # print(Padded_y_cap.size())
    # print(Padded_y_ir.size())
        Y = torch.cat((Padded_y_cap, Padded_y_ir), dim=-1)
        Y_length = Padded_y_cap.size()[1]

        data = torch.utils.data.TensorDataset(X, Y)

        if self.MIT==True or val==True:
            return X, Y, X_length, Y_length
        else:
            return data, X_length, Y_length  # X and Y tensors with shape: (# sequences, sequencelenght (idem for every sequence thanks to padding), #features=2 equal to number of tasks cap and ir). X_length and Y_lenght are equal to the sequence lenght (used for padding, i.e., max seq length)

    def train_val_test_split_seq(self, cap_data, res_data, test_size=0.15,
                                 random_state=33):  # test size is the percentage = valsize and testsize so trianingsze is 1-2*testsize

        train_size=1-test_size*2
        training_length = int((1 - test_size * 2) * len(cap_data))
        val_length = int(test_size * len(cap_data))
        test_length = int(test_size * len(cap_data))

        index = list(range(0, len(cap_data)))
        random.Random(random_state).shuffle(index)

        training_cap_seq = [cap_data[i] for i in index[0:training_length]]
        training_ir_seq = [res_data[i] for i in index[0:training_length]]
        val_cap_seq = [cap_data[i] for i in index[training_length:training_length + val_length]]
        val_ir_seq = [res_data[i] for i in index[training_length:training_length + val_length]]
        test_cap_seq = [cap_data[i] for i in
                        index[training_length + val_length:training_length + val_length + test_length]]
        test_ir_seq = [res_data[i] for i in
                       index[training_length + val_length:training_length + val_length + test_length]]

        return training_cap_seq, training_ir_seq, val_cap_seq, val_ir_seq, test_cap_seq, test_ir_seq

