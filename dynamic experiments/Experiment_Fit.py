import torch

from Requirements import *
from MTL_MODEL_OPT import *
from Dynamic_weighting import *
from Datasets import *


class Fit_MTL_Optimization():

    def __init__(self, configuration):

        self.config = configuration
        self.Learning_Alg=Global_MTL(configProj=self.config)
        # add dataset
        # add multi task model
        # Detect Dynamic Weighting Algorithm
        if self.config["Task_Weighting_strategy"] == AlgType.Random.value:
            self.dynamic_alg_Type = Random(self.Learning_Alg)
        elif self.config["Task_Weighting_strategy"] == AlgType.PCGrad.value:
            self.dynamic_alg_Type = PCGrad(self.Learning_Alg)
        elif self.config["Task_Weighting_strategy"] == AlgType.CAgrad.value:
            self.dynamic_alg_Type = CAgrad(self.Learning_Alg, alpha=self.config["Alpha"], rescale=self.config["Rescale"])
        elif self.config["Task_Weighting_strategy"] == AlgType.Unif.value:
            self.dynamic_alg_Type = Unif(self.Learning_Alg)
        elif self.config["Task_Weighting_strategy"] == AlgType.Olaux.value:
            self.dynamic_alg_Type = Olaux(self.Learning_Alg, gradspeed=self.config["gradspeed"])

    def Fit_NYU(self, traindataloader, valdata_x, valdata_y, testdata_x, testdata_y, verbose=0):

        Batch_Size = self.config["Batch_Size"]
        Epoch = self.config["Epochs"]

        indx = 0
        print("startfit")
        for epoch in range(0, Epoch):

            history = dict()
            history['Epoch'] = epoch
            batchnr=0


            for batch in traindataloader:
                
                    features, labels = batch
            
           
                    batchnr += 1
                    self.dynamic_alg_Type.train(features, labels)

                    result_train_Batch = self.Learning_Alg.Evaluate(features, labels, needgrad=False, weighting=False,
                                                                    type="both", task=-1)

                    if self.config["Dataset"]==DataName.Battery_Aachen.value:
                        result_val = self.Learning_Alg.Evaluate(valdata_x, valdata_y, needgrad=False, weighting=False,
                                                            type="both", task=-1, val=True)
                        result_test = self.Learning_Alg.Evaluate(testdata_x, testdata_y, needgrad=False, weighting=False,
                                                             type="both", task=-1, test=True)
                    else:
                        result_val = self.Learning_Alg.Evaluate(valdata_x, valdata_y, needgrad=False, weighting=False,
                                                                type="both", task=-1)
                        result_test = self.Learning_Alg.Evaluate(testdata_x, testdata_y, needgrad=False,
                                                                 weighting=False,
                                                                 type="both", task=-1)


                    # explicitly log everything to wandb

                    # we save all losses and configurations to wandb. create an empty dictionary first
                    history["Batch"] = indx
                    Task_Weights = self.Learning_Alg.weight.detach().clone()
                    
                    
                    for ti in range(0, self.config["Number_of_Tasks"]):
                        total_task_weight = 0
                        if self.Learning_Alg.dynamic_alg== AlgType.Olaux or self.Learning_Alg.dynamic_alg==AlgType.PCGrad:
                          Task_weights=self.dynamic_alg_Type.dot_prod
                          history["Gradient Dot"]=Task_weights
                        else:
                          history["Task_weight" + str(ti)] = Task_Weights[ti]  # save task weights
                        # save all kind of losses and metrics
                        history['trainLoss_t' + str(ti)] = result_train_Batch["TaskLoss"]["task" + str(ti)].detach().clone()
                        history['trainMetric_' + str(ti)] = result_val['taskMetric']["task" + str(ti)].detach().clone()
                        history['valLoss_t' + str(ti)] = result_test['TaskLoss']["task" + str(ti)].detach().clone()
                        history['valMetric_t' + str(ti)] = result_val['taskMetric']["task" + str(ti)].detach().clone()
                        history['testLoss_t' + str(ti)] = result_test['TaskLoss']["task" + str(ti)].detach().clone()
                        history['testMetric_t' + str(ti)] = result_test['taskMetric']["task" + str(ti)].detach().clone()
                    
                    if epoch==self.config["Epochs"]-1:
                        for ti in range(0, self.config["Number_of_Tasks"]):
                            ypred["task"+str(task)]
                            history["test_labels"]=testdata_y[:,:, ti]
                            history["test_prediction"]=ypred["task"+str(ti)]

                    # send history to wandb
                    wandb.log(history)

                    if verbose == 1 and indx % 32 == 0:  # print temporary results
                        print("\n{}::Epoch-{}/{}, Batch-{}/{}: {}".format(epoch, indx, history))

        return
