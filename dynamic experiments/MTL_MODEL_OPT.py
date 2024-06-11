import torch.nn

from Requirements import *
from Datasets import *
from Backbones import *
from Custom_Losses import *
from Dynamic_weighting import *
from Custom_Metrics import *

class Global_MTL():
    def __init__(self, configProj, **kwargs):

        # general
        self.config = configProj
        self.seed = configProj["random_seed"]
        self.device = torch.device("cpu")  # configProj["device"]
        self.input_dim = configProj["input_dimension"]
        self.output_dim_1 = configProj["output_dimension_task1"]
        self.output_dim_2 = configProj["output_dimension_task2"]
        self.dtype = torch.float
        # shared hyperparam
        self.NSharedL = configProj["Number_of_Shared_Layers"]
        self.NSharedDim = configProj["Dim_of_Shared_Layers"]

        # task specific hyperparam
        self.NTask = configProj["Number_of_Tasks"]
        self.NTaskL = configProj["Number_of_Task_Layers"]
        self.Taskdim = configProj["Dim_Task_Layers"]

        ##  Weighting Method
        if configProj["Task_Weighting_strategy"] == AlgType.Random.value:
            self.dynamic_alg = AlgType.Random
        elif configProj["Task_Weighting_strategy"] == AlgType.PCGrad.value:
            self.dynamic_alg = AlgType.PCGrad
        elif configProj["Task_Weighting_strategy"] == AlgType.CAgrad.value:
            self.dynamic_alg = AlgType.CAgrad
        elif configProj["Task_Weighting_strategy"] == AlgType.Unif.value:
            self.dynamic_alg = AlgType.Unif
        elif configProj['Task_Weighting_strategy']==AlgType.Olaux.value:
            self.dynamic_alg=AlgType.Olaux
            self.gradSpeed_N=configProj["gradspeed"]
    
        ## initialize task weights
        if self.dynamic_alg == AlgType.Random:
            randomw = torch.randn(self.NTask, dtype=torch.float32)
            self.weight = torch.tensor(np.exp(randomw) / sum(np.exp(randomw)), dtype=torch.float, requires_grad=False)

        elif self.dynamic_alg==AlgType.PCGrad or self.dynamic_alg == AlgType.CAgrad or self.dynamic_alg == AlgType.Unif:
            self.weight = torch.tensor(np.ones(self.NTask, dtype=np.float32) * [0.5,0.5], dtype=torch.float,
                                       requires_grad=False)
       
    
        elif self.dynamic_alg == AlgType.Olaux:
            self.gradSpeed_N = 2

            self.weight = torch.tensor(np.ones(self.NTask, dtype=np.float32), dtype=self.dtype, requires_grad=False)
            self.weightOlaux = torch.tensor(np.ones(self.NTask - 1, dtype=np.float32), dtype=self.dtype,
                                            requires_grad=True)

            if self.config['Optimizer'] == OptType.Sgd.value:
                self.optTaskWeight = torch.optim.SGD([self.weightOlaux],
                                                     lr=self.config["Learning_Weight"] * self.gradSpeed_N,
                                                     momentum=self.config["beta_1_backbone"])
            else:
                self.optTaskWeight = torch.optim.Adam([self.weightOlaux],
                                                      lr=self.config["Learning_Weight"] * self.gradSpeed_N,
                                                      betas=(
                                                      self.config["beta_1_backbone"], self.config["beta_2_backbone"]))

        self.weight.to(self.device)

        elif configProj["Dataset"]==DataName.Battery_Aachen.value:
            self.backbone=Seq2SeqMTL(MIT=False)
            self.backbone.to(self.device)

        elif configProj["Dataset"]==DataName.Battery_MIT.value:
            self.backbone=Seq2SeqMTL(MIT=True)
            self.backbone.to(self.device)



        ##  optimizer for the backbone model (SGD or ADAM)
        if self.config["Optimizer"] == OptType.Sgd.value:
            self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=self.config["Learning_Weight"],
                                             momentum=self.config["beta_1_backbone"])

        else:
            self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.config["Learning_Weight"],
                                              betas=(self.config["beta_1_backbone"], self.config["beta_2_backbone"]))

        # init Losses
        self.loss = []
        self.metric = []
        self.metricElmtwise = []
        self.lossElmt = []
        

       
        if configProj["Dataset"]==DataName.Battery_Aachen.value or configProj["Dataset"]==DataName.Battery_MIT.value:
            for ti in range(0, self.NTask):
                self.loss.append(MaskedMae())
                self.lossElmt.append(MaskedMae(reduction='none'))
                self.metric.append(MaskedMape()) #watchoutnotdiff

        # Prediction Method of the algorithm

    def setInvalid(self, yin, repv=1 - 6):
        # replace nan with finite values
        # return torch.where(torch.isnan(yin), torch.mul(repv, torch.ones_like(yin)), yin)

        # replace non-finite with finite values
        return torch.where(torch.isfinite(yin), yin, torch.mul(repv, torch.ones_like(yin)))

    def predict(self, x, task=-1, train=False, keepfinite=True,  val=False, test=False):
        if train == True:
            self.backbone.train()  # still need to train the model before making predictions
        else:
            self.backbone.eval()  # some kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.

        if keepfinite == True:

            if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                ypred=self.backbone(x, task, val, test)
            else:
                ypred = self.backbone(x.to(self.device), task)  # calls the forward function defined in backbone model

            if task == -1:
                for i in range(0, self.NTask):
                    ypred["task" + str(i)] = self.setInvalid(ypred["task" + str(i)], 0)
            else:
                ypred["task" + str(task)] = self.setInvalid(ypred["task" + str(task)], 0)
        else:
            return self.backbone(x.to(self.device), task)
        return ypred

    def process_preds(self, preds): #additional step to be done with predictions from MTAN
        img_size = (288, 384)
        preds = F.interpolate(preds, img_size, mode='bilinear', align_corners=True)
        return preds

    # Define Loss calculator

    def GetLoss(self, ytrue, ypred, weighting=False, elmt=False, task=-1,
                Dynamic_alg=None):  # weighting indicates if the task weights are uniform or defined by the weighting algorithms (most of the time weighting=false if losses compute for first backward pass, and weighting true for update). For the moment, assume algorithms all lead to elmt=FALSE so parameter not really requiered.
        NObs = self.config["Batch_Size"]  # number of observations to put in the loss
        TaskLoss = dict()  # save the tasklosses
        TotLoss = 0  # save the total loss
        # ytrue=torch.tensor(ytrue,dtype=torch.float32) #make sure ground truth is also saved as a torch tensor, not as an array
        if task == -1:  # then compute loss for all tasks + total loss
            for ti in range(0, self.NTask):
                if self.config["Dataset"]==DataName.NYUv2.value:
                    if ti == 0:
                        ytrue[ti] = ytrue["segmentation"]
                    elif ti == 1:
                        ytrue[ti] = ytrue["depth"]
                    elif ti == 2:
                        ytrue[ti] = ytrue["normal"]

                    ypred["task" + str(ti)] = self.process_preds(ypred["task" + str(ti)]) #for nyu, the predictions should be processed before feeding to loss


                if weighting == False:  # tot loss is equal to unweighted sum of task losses
                    if elmt == False:  #loss is averaged over entire batch
                        if self.config["Dataset"] == DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                            Loss= self.loss[ti].compute_loss(ypred["task"+str(ti)].squeeze(), ytrue[:,:,ti].to(self.device))
                            TotLoss = TotLoss + Loss  # no weighting, losses are simply added 

                    else:
                        if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                            Loss=self.loss[ti].compute_loss(ypred["task"+str(ti)].squeeze(), ytrue[:,:,ti].to(self.device))
                            TotLoss = TotLoss + torch.sum(Loss) / Loss.shape[0]  # the elmtwise losses are summed over and this sum is added to total loss
                        
                            

                else:  # totloss is equal to a weighted sum of task losses
                    if elmt == True:
                        if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                            elos=self.lossElmt[ti].compute_loss(ypred["task"+str(ti)].squeeze(), torch.tensor(ytrue[:,:,ti].to(self.device), dtype=torch.float32))
                        
                            Loss = torch.dot(self.weight[ti, :elos.shape[0]], elos) / elos.shape[
                                0]  # instancelevel weights assigned to each sample
                            TotLoss = TotLoss + Loss

                    else:
                        if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                            Loss=self.loss[ti].compute_loss(ypred["task"+str(ti)].squeeze(), ytrue[:,:,ti].to(self.device))
                            TotLoss = TotLoss + self.weight[ti] * Loss

                TaskLoss["task" + str(ti)] = Loss  # save task loss
        else:  # compute only one taskloss
            ti = task
            
            if weighting == False:  # tot loss is equal to unweighted sum of task losses
                if elmt == False:

                    if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                        Loss=self.loss[ti].compute_loss(ypred["task"+str(ti)].squeeze(), ytrue[:,:,ti].to(self.device))
                        TotLoss = TotLoss + torch.sum(Loss)
            

                else:
                    if self.config["Dataset"] == DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                        Loss = self.loss[ti].compute_loss(ypred["task" + str(ti)].squeeze(),
                                                          ytrue[:, :, ti].to(self.device))
                        TotLoss = TotLoss + torch.sum(Loss)/Loss.shape[0]

            else:  # totloss is equal to a weighted sum of task losses
                if elmt == True 
                
                    if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                        elos=self.lossElmt[ti].compute_loss(ypred["task"+str(ti)].squeeze(), ytrue[:,:,ti].to(self.device))
                        Loss=torch.dot(self.weight[ti, elos.shape[0]], elos)/elos.shape[0] #instancelevel weights assigned to each sample
                        TotLoss = TotLoss + Loss
                    else:
                        elos = self.lossElmt[ti](ypred["task" + str(ti)],
                                             torch.tensor(ytrue[ti], dtype=torch.float32).to(self.device))
                        Loss = torch.dot(self.weight[ti, :elos.shape[0]], elos) / elos.shape[
                        0]  # instancelevel weights assigned to each sample
                        TotLoss = TotLoss + Loss
                else:
                   

                    if self.config["Dataset"]==DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                        Loss = self.loss[ti].compute_loss(ypred["task" + str(ti)].squeeze(),
                                                          ytrue[:, :, ti].to(self.device))
                        TotLoss = TotLoss + self.weight[ti] * Loss  # weighted task losses (not sample level)
                    else:
                        Loss = self.loss[ti](ypred["task" + str(ti)],
                                         torch.tensor(ytrue[ti], dtype=torch.float32).to(self.device))
                        TotLoss = TotLoss + self.weight[ti] * Loss  # weighted task losses (not sample level)
            TaskLoss["task" + str(ti)] = Loss  # save task loss

        return TaskLoss, TotLoss

    # Def Metric calculator (for now just assume loss and metric are the same)

    def GetMetric(self, ytrue, ypred, weighting=False, elmt=False, task=-1,
                  Dynamic_alg=None):  # weighting indicates if the task weights are uniform or defined by the weighting algorithms (most of the time weighting=false if losses compute for first backward pass, and weighting true for update). For the moment, assume algorithms all lead to elmt=FALSE so parameter not really requiered.
        NObs = self.config["Batch_Size"]  # number of observations to put in the loss
        TaskMetric = dict()  # save the tasklosses
        TotMetric = 0  # save the total loss
        
        if task == -1:  # then compute loss for all tasks + total loss
            for ti in range(0, self.NTask):
    
                if weighting == False:  # tot loss is equal to unweighted sum of task losses
                    Metric=self.metric[ti].compute_loss(ypred["task" + str(ti)].squeeze(),ytrue[:, :, ti].to(self.device))
                    TotMetric = TotMetric + Metric  # no weighting, losses are simply added
                else:  # totloss is equal to a weighted sum of task losses
                    if self.config["Dataset"] == DataName.NYUv2.value:
                        self.metric[ti].update_fun(ypred["task" + str(ti)], torch.tensor(ytrue[ti], dtype=torch.float32).to(self.device))
                        Metric = self.metric[ti].score_fun()
                    else:
                        Metric = self.metric[ti].compute_loss(ypred["task" + str(ti)].squeeze(),
                                                          ytrue[:, :, ti].to(self.device))
                    TotMetric = TotMetric + self.weight[ti] * Metric  # weighted task losses (not sample level)

                TaskMetric["task" + str(ti)] = Metric  # save task loss
    
            if weighting == False:  # tot loss is equal to unweighted sum of task losses
                if elmt == False:
            
                    Metric = self.metric[ti].compute_loss(ypred["task" + str(ti)].squeeze(),
                                                              ytrue[:, :, ti].to(self.device))
                    TotMetric = TotMetric + torch.sum(Metric)  # no weighting, losses are simply added
                else:
                
                    Metric = self.metric[ti].compute_loss(ypred["task" + str(ti)].squeeze(),
                                                              ytrue[:, :, ti].to(self.device))

                    TotMetric = TotMetric + torch.sum(Metric) / Metric.shape[
                        0]  # the elmtwise losses are summed over and this sum is added to total loss
            else:  # totloss is equal to a weighted sum of task losses
    
                    Metric=self.metric[ti].compute_loss(ypred["task" + str(ti)].squeeze(),
                                                 ytrue[:, :, ti].to(self.device))

                TotMetric = TotMetric + self.weight[ti] * Metric  # weighted task losses (not sample level)
            TaskMetric["task" + str(ti)] = Metric  # save task loss

        return TaskMetric, TotMetric


    # Define Evaluate method for the algorithm (from x to tot and task losses)

    def Evaluate(self, x, ytrue, needgrad=True, weighting=False, type='both', elmt=False, task=-1, Dynamic_alg=None, val=False, test=False):
        NObs = self.config["Batch_Size"]
        x = torch.tensor(x.to(torch.float32))


        if needgrad == True:  # basically say: remember all the transformations that will be applied on x until loss is obtained because it will be needed to compute the gradients (his is not needed if we are evaluating on test set for example)
            if self.config["Dataset"] == DataName.Battery_Aachen.value:
                ypred = self.predict(x, task, train=True, val=val, test=test)  # training=True (because we need grad it is likely that we are still in training phase, keepfinite=true, almost always)
            else:
                ypred = self.predict(x, task, True, True)  # training=True (because we need grad it is likely that we are still in training phase, keepfinite=true, almost always)

            result = dict()

            if type == 'loss':
                taskLos, totLos = self.GetLoss(ytrue, ypred, weighting, elmt, task)
                result["TaskLoss"] = taskLos
                result["Total_Loss"] = totLos

            elif type == 'metric':  # still assuming the metric and the lossfunction are the same
                if  self.config["Dataset"]==DataName.Battery_MIT.value or self.config["Dataset"]==DataName.Battery_Aachen.value:
                    taskMetric, totMetric=self.GetMetric(ytrue,ypred, weighting, elmt, task)
                else:
                    taskMetric, totMetric = self.GetLoss(ytrue, ypred, weighting, elmt, task)

                result["taskMetric"] = taskMetric
                result["totMetric"] = totMetric


            else:  # both
                taskLos, totLos = self.GetLoss(ytrue, ypred, weighting, elmt, task)
                result["TaskLoss"] = taskLos
                result["Total_Loss"] = totLos
                if  self.config["Dataset"]==DataName.Battery_MIT.value or self.config["Dataset"]==DataName.Battery_Aachen.value:
                    taskMetric, totMetric = self.GetMetric(ytrue, ypred, weighting, elmt, task)
                else:
                    taskMetric, totMetric = self.GetLoss(ytrue, ypred, weighting, elmt, task)
                result["taskMetric"] = taskMetric
                result["totMetric"] = totMetric
            return result

        else:  # if we do not need the gradients of the losses (for example because we are evaluating generalization performance of a trained model)
            with torch.no_grad():
                if self.config["Dataset"] == DataName.Battery_Aachen.value or self.config["Dataset"]==DataName.Battery_MIT.value:
                    ypred = self.predict(x, task, train=False, val=val,
                                         test=test)  # we do not need gradients so training=false.
                else:
                    ypred = self.predict(x, task, False, True)  # we do not need gradients so training=false.
                result = dict()

                if type == 'loss':
                    taskLos, totLos = self.GetLoss(ytrue, ypred, weighting, elmt, task)
                    result["TaskLoss"] = taskLos
                    result["Total_Loss"] = totLos

                elif type == 'metric':  # still assuming the metric and the lossfunction are the same
                    if self.config[
                        "Dataset"] == DataName.Battery_MIT.value or self.config["Dataset"] == DataName.Battery_Aachen.value:
                        taskMetric, totMetric = self.GetMetric(ytrue, ypred, weighting, elmt, task)
                    else:
                        taskMetric, totMetric = self.GetLoss(ytrue, ypred, weighting, elmt, task)

                    result["taskMetric"] = taskMetric
                    result["totMetric"] = totMetric

                elif type=='both':
                    taskLos, totLos = self.GetLoss(ytrue, ypred, weighting, elmt, task)
                    result["TaskLoss"] = taskLos
                    result["Total_Loss"] = totLos
                    if self.config[
                        "Dataset"] == DataName.Battery_MIT.value or self.config["Dataset"] == DataName.Battery_Aachen.value:
                        
                        taskMetric, totMetric = self.GetMetric(ytrue, ypred, weighting, elmt, task)
                    else:
                        taskMetric, totMetric = self.GetLoss(ytrue, ypred, weighting, elmt, task)
                    result["taskMetric"] = taskMetric
                    result["totMetric"] = totMetric
            return result

    # Define train method of the algorithm

    def train(self, xin, yout, alg_dynamic=None):  # forward pass, compute gradient, backward pass

        # forward pass
        result_Batch = self.Evaluate(xin, yout, needgrad=True, weighting=True, type="loss", elmt=False,
                                     Dynamic_alg=alg_dynamic)

        # reset gradients (to avoid acumulation to save memory)
        self.optimizer.zero_grad()

        # backward pass
        result_Batch['Total_Loss'].backward()

        # parameter update of the backbone
        self.optimizer.step()






