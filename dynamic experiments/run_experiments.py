import numpy as np
import torch.utils.data

from Datasets import *
from Requirements import *
from MTL_MODEL_OPT import *
from Experiment_Fit import *
from Dynamic_weighting import *
from MultiTaskLoader import *
## **Hyperparameter Configuration**"

hyperparameter_configuration= {
          'method': 'grid', #grid, random, bayes
          'metric': {
            'name': 'none',
            'goal': 'minimize'
          },
          'early_terminate': {
            'type': 'hyperband',
            'min_iter': 256,
            's': 2,
            'eta': 32
          },
          'parameters': {
              'Task_Weighting_strategy': {
                  'values': [AlgType.Unif.value] #supported weighting algorithms see datasets.py
              },
              'Dataset': {
                  'values': [DataName.Battery_Aachen.value] #supported datanames see datasets.py
              },
              'Number_of_Tasks': {
                  'values': [2]#[numTask]
              },
              'input_dimension': {
                  'values': [10]#[25]#[xDim]
              },
              'output_dimension_task1': {
                  'values': [1]#[10]#[yDim] #dim task 1
              },
              'output_dimension_task2': {
                  'values': [1]#[10]#[yDim] #dim task 2
              },
              'Epochs': {
                  'values': [50] #np.arange(minLimEpoch, maxLimEpoch, 1).tolist()
              },
              'Batch_Size': {
                  'values': [32]#[256] #np.arange(minLimBatchsize, maxLimBatchsize, 1).tolist()
              },
              'val_Batch_size': {
                  'values': [32]
              },
              'Number_of_Shared_Layers': {
                  'values': [4] #3, np.arange(minLimNumSharedLayer, maxLimNumSharedLayer, 1).tolist() 3
              },
              'Dim_of_Shared_Layers': {
                  'values': [48] #np.arange(minLimDimSharedLayer, maxLimDimSharedLayer, 2).tolist()
              },
              'Number_of_Task_Layers': {
                  'values': [4] #[2] #np.arange(minLimNumTaskLayer, maxLimNumTaskLayer, 1).tolist()
              },
              'Dim_Task_Layers': {
                  'values': [34] #np.arange(minLimDimTaskLayer, maxLimDimTaskLayer, 1).tolist()
              },
              'Optimizer': {
                  'values': ["sgd"] #['adam', 'sgd']#
              },
              "beta_1_backbone": {
                  'values': [0.9] #[0.9]'beta1' for adam, but 'momentum' value for sgd optimizer
              },
              "beta_2_backbone": {
                  'values': [0.99]
              },
              'Learning_Weight': {
                  'values': [1e-2]#1e-6, 1e-5, 1e-4, 1e-3, [5e-2, 5e-3, 5e-4]#
              },
              "onlymain": {
                  'values': [True]
              },
              "noise": {
                  'values':   [0]
              },
              "random_seed": {
                  'values': [33]
              },
              "Regression":{
                  'values': [True] #Regression, Classification => changes the used loss function
              },
              "UNI":{
                  'values': [True]
              },
          }
      }






def run_experiment():
    with wandb.init() as run:
        configProj = wandb.config  # hyperparameter_configuration["parameters"]
        if torch.cuda.is_available():
            configProj["device"] = torch.device("cuda:0")
        else:
            configProj["device"] = torch.device("cpu")
        print(configProj["device"])
        if configProj["Task_Weighting_strategy"]==AlgType.CAgrad.value:
            configProj["Alpha"]=1
            configProj["Rescale"]=1
        elif configProj["Task_Weighting_strategy"]==AlgType.Olaux.value:
            configProj["gradspeed"]=0.4
        


        elif configProj["Dataset"]==DataName.Battery_Aachen.value:

            trCap = pickle.load(
                open('C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_aachen/trCap.p', "rb"))
            trIR = pickle.load(
                open('C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_aachen/trIR.p', "rb"))
            vaCap = pickle.load(
                open('C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_aachen/vaCap.p', "rb"))
            vaIR = pickle.load(
                open('C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_aachen/vaIR.p', "rb"))
            teCap = pickle.load(
                open("C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_aachen/teCap.p", "rb"))
            teIR = pickle.load(
                open("C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_aachen/teIR.p", "rb"))

            data = Battery(MIT=False)
            traindata, xtrlength, ytrlength = data.Sequence_Builder(trCap, trIR)
            valdata, vallengthx, vallengthy = data.Sequence_Builder(vaCap, vaIR)
            testdata, lengx, lengy = data.Sequence_Builder(teCap, teIR)

            trainloader=torch.utils.data.DataLoader(traindata, batch_size=configProj["Batch_Size"])
            valloader=torch.utils.data.DataLoader(valdata, batch_size=vallengthx) #vallengthx
            testloader=torch.utils.data.DataLoader(testdata, batch_size=lengx)  #lengx

            for batch in testloader:
                testfeatures, testlabels=batch
            Fit_MTL = Fit_MTL_Optimization(configProj)
            weights = Fit_MTL.Fit_NYU(trainloader, valloader, testfeatures, testlabels)

        elif configProj["Dataset"]==DataName.Battery_MIT.value:
            number_of_cells = 46 * 3
            number_of_tasks = 2

            cap_data = []
            res_data = []

            for Batch in ["Batch1"]:  # , "Batch2", "Batch3"]:
                for cellnumber in range(0, 48):
                    if Batch == "Batch1":
                        if cellnumber == 8 or cellnumber == 10 or cellnumber == 12 or cellnumber == 13 or cellnumber == 22 or cellnumber == 46 or cellnumber == 47:
                            continue
                        else:
                            data = pd.read_csv("C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_MIT/result/result/" + str(Batch) + "/b1c" + str(cellnumber) + ".csv")
                    elif Batch == "Batch2":
                        if cellnumber == 7 or cellnumber == 8 or cellnumber == 9 or cellnumber == 15 or cellnumber == 16:
                            continue
                        else:
                            data = pd.read_csv("C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_MIT/result/result" + str(Batch) + "/b2c" + str(cellnumber) + ".csv")
                    elif Batch == "Batch3":
                        if cellnumber == 2 or cellnumber == 23 or cellnumber == 32 or cellnumber == 37 or cellnumber == 42 or cellnumber == 43 or cellnumber == 46 or cellnumber == 47:
                            continue
                        else:
                            data = pd.read_csv("C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/battery_MIT/result/result" + str(Batch) + "/b3c" + str(cellnumber) + ".csv")

                    cap_data.append(torch.tensor(data["discharge_capacity"]))
                    res_data.append(torch.tensor(data["Resistance"]))


            data=Battery(MIT=True)
            #print(len(cap_data))
            trcap, trir, valcap, valir, testcap, testir = data.train_val_test_split_seq(cap_data, res_data, test_size=0.15)
            X_tr, Y_tr, Xlen, ylen = data.Sequence_Builder(trcap, trir)
            X_val, Y_val, Xlenv, ylenv = data.Sequence_Builder(valcap, valir, val=True)
            X_test, Y_test, Xlent, ylent = data.Sequence_Builder(testcap, testir, val=True)


            trainloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, Y_tr), batch_size=configProj["Batch_Size"])
            valloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, Y_val), batch_size=3) #vallengthx
            testloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=3)  #lengx

            for batch in testloader:
                testfeatures, testlabels = batch
            Fit_MTL = Fit_MTL_Optimization(configProj)
            weights = Fit_MTL.Fit_NYU(trainloader, valloader, testfeatures, testlabels)



        return weights




wandb.login()
sweep_id = wandb.sweep(hyperparameter_configuration, project="MTL_CODEBASE_CHECK")
wandb.agent(sweep_id, function=run_experiment)
run_experiment()





