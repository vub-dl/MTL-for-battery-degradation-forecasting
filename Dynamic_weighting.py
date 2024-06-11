from enum import Enum, unique
from Requirements import *
import copy

@unique


class AlgType(Enum): #supported algorithms
    Unif = 'unif'
    SLGrad = 'slgrad' #new sample level weighting algorithm
    Olaux = 'olaux'
    Gcosim = 'gcosim' #gradient cosine similarity
    Gnorm = 'gnorm'
    Random= 'random'
    PCGrad='pcgrad'
    CAgrad='cagrad'
    IMTL='IMTL'
    Nash='Nash'

@unique
class OptType(Enum): #optimizer types
    Adam = 'adam'
    Sgd = 'sgd'

@unique
class NormType(Enum):
    Nun = 'none' #No-norm
    WM = 'wm' #multiply by weight
    ND = 'nd' #divide by norm
    NDWM = 'ndwm' #divide by norm and multiply by weight

@unique
class RunType(Enum): #optimizer types
    SimpleSweep = 'ssweep_run' #simple sweep run
    KfoldCv = 'kfcv_run' #k-fold cross-validation (sweem or not sweep),
    Bare = 'simp_run' #run without sweep and/or corss-validation


class Dynamic_Weighting():
    def __init__(self, model):
        self.mdl=model

    def getGradsTaskLayers(self, task):
        grad = []
        for name, param in self.mdl.backbone.named_parameters():
            if "shared" not in name:
                if "Task." + str(task) in name:
                    grad.append(param.grad.requires_grad_(True))
                    #return grad.append(param.grad.requires_grad_(True))
        return grad

    def getGradsLastLayerKernel(self, taskId):
            ntmp = "TaskLayers." + str(taskId) + "." + str(2 * self.mdl.backbone.nTaskL - 2) + '.weight'
            for name, param in self.mdl.backbone.named_parameters():
                if ntmp in name:
                    # print(name)
                    return torch.reshape(param.grad.cpu().detach(), (-1,))

            return 0

    def _compute_grad_dim(self,shared=True):
        self.grad_index = []
        for name, param in self.mdl.backbone.named_parameters():
            if shared==True:
                if "shared" in name:
                    self.grad_index.append(param.data.numel())
                    self.grad_dim = sum(self.grad_index)
            else:
                if "Task" in name:
                    self.grad_index.append(param.data.numel())
                    self.grad_dim = sum(self.grad_index)


    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for name, param in self.mdl.backbone.named_parameters():
            if "shared" in name:
                if param.grad is not None:
                    beg = 0 if count == 0 else sum(self.grad_index[:count])

                    end = sum(self.grad_index[:(count + 1)])

                    grad[beg:end] = param.grad.data.view(-1)

                count += 1
        return grad

    def _reset_grad(self, new_grads):
        count = 0
        for name, param in self.mdl.backbone.named_parameters():
            if "shared" in name:
                if param.grad is not None:
                    beg = 0 if count == 0 else sum(self.grad_index[:count])
                    end = sum(self.grad_index[:(count + 1)])
                    param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                count += 1

    def getGradsModelParam(self, vecshape=True):
        grad = []
        for name, param in self.mdl.backbone.named_parameters():
            # if "shared" in name or "taskLayers."+str(task)+"." in name:
            if "shared" in name:
                if vecshape == True:
                    # grad.append(torch.reshape(param.grad.cpu().detach(), (-1,)))
                    grad.append(torch.reshape(param.grad, (-1,)))  # .requires_grad_(True))
                else:
                    # grad.append(param.grad.cpu().detach()
                    grad.append(param.grad.requires_grad_(True))

        return grad

    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters.
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn + 1) != self.task_num else False
                    self.rep[task].backward(batch_weight[tn] * per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    '''
    def getGradsTaskLayers(self, tasklbase, normg=False):
        tgrads = dict()
        for name, param in self.mdl.backbone.named_parameters():
            if tasklbase in name:
                if normg == False:
                    tgrads[name] = param.grad.cpu().detach()
                else:
                    grd = param.grad.cpu().detach()
                    tgrads[name] = torch.mul(1.0 / torch.norm(torch.reshape(grd, (-1,))), grd)

        return tgrads

    '''


class Unif(Dynamic_Weighting):
    def __init__(self, model):
        Dynamic_Weighting.__init__(self, model)

    def train(self, xin, yout):
        self.mdl.train(xin, yout, AlgType.Unif)


class Random(Dynamic_Weighting):
    def __init__(self, model):
        Dynamic_Weighting.__init__(self, model)

    def train(self, xin, yout):
      randomw= torch.randn(self.mdl.NTask, dtype=torch.float32)
      self.mdl.weight=torch.tensor(np.exp(randomw)/sum(np.exp(randomw)), dtype=torch.float, requires_grad=False)
      self.mdl.train(xin, yout, AlgType.Random)
      return


class Gnorm(Dynamic_Weighting):
    def __init__(self, mdl):
        Dynamic_Weighting.__init__(self, mdl)

    def train(self, xin, yin):
        ##WARNING: Gnorm is observed to become numerically unstable if the learning-rate is too high (e.g., 1e-1)

        ######## train model parameters: layers weights and biases #############################
        self.mdl.train(xin, yin, AlgType.Gnorm)

        # calculate gradient of each task wrt kernel of last layer of the task
        taskLossGrads = dict()
        for ti in range(0, self.mdl.NTask):
            taskLossGrads["task" + str(ti)] = self.getGradsLastLayerKernel(ti)

        resBatch = self.mdl.Evaluate(xin, yin, needgrad=False, weighting=True, type='loss', elmt=False, task=-1,
                                     Dynamic_alg=AlgType.Gnorm)
        taskLos = []
        for ti in range(0, self.mdl.NTask):
            taskLos.append(resBatch['TaskLoss']["task" + str(ti)].detach().item())
        taskLos = np.array(taskLos)
        # if ibatch == 0:
        # self.initTaskLos = taskLos

        ############ Task-Weight Computations ##################################################
        # 1-step: reset gradients
        self.mdl.optimizer.zero_grad()

        # 2-step: loss function computation
        # print("Li: {}, L0: {}".format(resBatch['taskLoss'], self.initTaskLos))
        taskGradsNorm = []
        for i in range(0, self.mdl.NTask):
            gradsi = taskLossGrads["task" + str(i)]
            taskGradsNorm.append(torch.norm(gradsi).detach().item())
        taskGradsNorm = torch.tensor(taskGradsNorm, dtype=self.mdl.dtype, device=self.mdl.device, requires_grad=True)

        self.initTaskLos = 0
        Gw = torch.dot(self.mdl.weight, taskGradsNorm)  # Gw
        GwBar = torch.mul(1.0 / self.mdl.NTask, Gw)  # GwOverBar
        epsilon = 1e-12
        Ltilda = taskLos / (epsilon + self.initTaskLos)  # Ltilda
        r = torch.tensor(Ltilda / (epsilon + np.mean(Ltilda)), dtype=self.mdl.dtype,
                         device=self.mdl.device)  # r
        Lgrad = torch.sum(torch.abs(Gw - GwBar * (r ** self.mdl.alpha)))  # Lgrad

        # 3-step: perform a backward pass.
        Lgrad.backward()

        # 4-step: update the weights
        self.mdl.optimizer.step()

        with torch.no_grad():
            # in some runs, observed numerical instability, when weights become negative, can be avoid by taking absolute value
            self.mdl.weight.abs_()

            # renormalize weight
            weight = self.mdl.weight.cpu().detach().numpy()
            scaleFact = self.mdl.NTask / np.sum(weight)

            # set new values
            self.mdl.weight.mul_(scaleFact)

        return


class CAgrad(Dynamic_Weighting):
    def __init__(self, mdl, alpha=1, rescale=1):
        Dynamic_Weighting.__init__(self, mdl)
        self.alpha = alpha
        self.rescale = rescale



    def train(self, xin, yout):

        calpha, rescale = self.alpha, self.rescale

        self._compute_grad_dim()

        resBatch = self.mdl.Evaluate(xin, yout, needgrad=True, weighting=True, type='loss', elmt=False, task=-1,
                                     Dynamic_alg=self.mdl.dynamic_alg)

        # reset gradients
        self.mdl.optimizer.zero_grad()
        # backward pass.
        pc_grad = dict()
        grads = torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)

        for task in range(self.mdl.NTask):
            taskloss = resBatch['TaskLoss']
            taskloss['task' + str(task)].backward(retain_graph=True)
            grads[task] = self._grad2vec()

        GG = torch.matmul(grads, grads.t()).cpu()  # [num_tasks, num_tasks]

        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.mdl.NTask) / self.mdl.NTask
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()

        c = (calpha * g0_norm + 1e-8).item()

        def objfn(x):

            return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1)) + c * np.sqrt(
                x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.mdl.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1 + calpha ** 2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError('No support rescale type {}'.format(rescale))   
        self._reset_grad(new_grads)
        self.mdl.optimizer.step()
        return


class PCGrad(Dynamic_Weighting):

    def __init__(self, mdl):
        Dynamic_Weighting.__init__(self,mdl)

    # PCGRad only changes the gradients with respect to the shared layers

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        self._compute_grad_dim()
        if not rep_grad:
            grads = torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)
            for task in range(self.mdl.NTask):
                if mode == 'backward':
                    losses[task].sum().backward(retain_graph=True) if (task + 1) != self.mdl.NTask else losses[
                        task].backward()
                    grads[task] = torch.tensor(self.getGradsModelParam(self.mdl))
                else:
                    raise ValueError('No support {} mode for gradient computation')
                    self.mdl.optimizer.zero_grad()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.mdl.NTask, *self.rep.size()).to(self.mdl.device)
        # print("ok")
        return grads

    def train(self, xin, yout):
        # standaard forward pass
        self._compute_grad_dim()
        resBatch = self.mdl.Evaluate(xin, yout, needgrad=True, weighting=True, type='loss', elmt=False, task=-1,
                                     Dynamic_alg=self.mdl.dynamic_alg)
        # reset gradients
        self.mdl.optimizer.zero_grad()
        # backward pass.
        pc_grad = dict()
        grads = torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)
        for task in range(self.mdl.NTask):
            taskloss = resBatch['TaskLoss']
            taskloss['task' + str(task)].backward(retain_graph=True)
            grads[task] = self._grad2vec()
        pc_grad = grads.clone()
        # at this point we collected the gradients. Nowe we need to adapt them
        for task_i in range(self.mdl.NTask):
            task_ind = list(range(self.mdl.NTask))
            random.shuffle(task_ind)
            for task_j in task_ind:
                g_ij = torch.dot(pc_grad[task_i], grads[task_j])
                self.dot_prod=g_ij
                if g_ij < 0:
                    pc_grad[task_i] -= (g_ij * grads[task_i]) / (grads[task_j].norm().pow(2))
                    # pc_grad["task"+str(task_i)]) -= 1
        new_grads = pc_grad.sum(0)
        self._reset_grad(new_grads)
        self.mdl.optimizer.step()
        # if g_ij <0:
        # print(grads[task_i])
        return g_ij
        # for task_j in task_ind:


class Olaux(Dynamic_Weighting):
    def __init__(self, mdl, gradspeed=0.4):
        Dynamic_Weighting.__init__(self, mdl)
        self.gradSpeed_buf = np.zeros((self.mdl.NTask - 1), dtype=float)
        self.gradSpeed_j = gradspeed

    def train(self, xin, yin):
        ######## train model parameters: layers weights and biases #############################
        self.mdl.train(xin, yin, AlgType.Olaux)

        ######## Calculate task-loss gradient wrt shared layer parameters ####################
        taskLossGrads = dict()
        for ti in range(0, self.mdl.NTask):

            self.mdl.backbone.zero_grad()
            resBatch = self.mdl.Evaluate(xin, yin, needgrad=True, weighting=False, type='loss', elmt=False, task=ti,
                                         Dynamic_alg=AlgType.Olaux)
            resBatch["TaskLoss"]["task" + str(ti)].backward()

            if ti == 0:
                grdMain = self.getGradsModelParam(vecshape=True)
            else:
                grdAux = self.getGradsModelParam( vecshape=True)
                if len(grdMain) <= len(grdAux):
                    numel = len(grdMain)
                else:
                    numel = len(grdAux)
                prod = 0
                for k in range(0, numel):
                    prod = prod + torch.dot(grdMain[k], grdAux[k]).detach().item()

                self.gradSpeed_buf[ti - 1] = self.gradSpeed_buf[ti - 1] + prod

        self.gradSpeed_j = self.gradSpeed_j + 1
        self.dot_prod= prod/numel
        if self.gradSpeed_j == self.mdl.gradSpeed_N:
            # 1-step: reset gradients
            self.mdl.optTaskWeight.zero_grad()

            # 2-step: set grad
            # print("{}, {}".format(self.weightOlaux.grad = torch.tensor(self.gradSpeed_buf, dtype=dtype)))
            self.gradSpeed_buf = torch.tensor(self.gradSpeed_buf, dtype=self.mdl.dtype)
            self.gradSpeed_buf.mul_(-self.mdl.config["Learning_Weight"])
            self.mdl.weightOlaux.grad = self.gradSpeed_buf
            self.mdl.weightOlaux_grad=self.mdl.weight_Olaux.grad

            # print("\t{}".format(self.mdl.weightOlaux.grad))

            # 34tep: update the weights
            self.mdl.optTaskWeight.step()

            self.gradSpeed_buf = np.zeros((self.mdl.NTask - 1), dtype=float)
            self.gradSpeed_j = 0

            # print("w: {}".format(self.mdl.weightOlaux.detach().numpy()))
            # weightOlaux = np.maximum(self.mdl.weightOlaux.detach().numpy(),
            #                         np.zeros(self.mdl.weightOlaux.detach().numpy().shape))
            weightOlaux = self.mdl.weightOlaux.cpu().detach().numpy()

            weight = self.mdl.weight.cpu().numpy()
            weight[1:self.mdl.NTask] = weightOlaux[0:len(weightOlaux)]

            self.mdl.weight = torch.tensor(weight, dtype=self.mdl.dtype, device=self.mdl.device, requires_grad=False)

            # print("{}, {}".format(self.mdl.weightOlaux.grad.detach().numpy(), weightOlaux))

        # self.model.zero_grad()
        return


class SLGrad(Dynamic_Weighting):
    def __init__(self, mdl, metric_inc=False):
        Dynamic_Weighting.__init__(self,mdl)
        self.metric_inc=metric_inc
    def comptInProd(self, v1, v2, lr, metric_incr=True):
        # print(v1)
        # print(v2)
        if len(v1) <= len(v2):
            numel = len(v1)
        else:
            numel = len(v2)

        # print('i-{}, v1: {}, {}\tv2: {}, {}'.format(0, v1[0].requires_grad, v1[0].grad_fn, v2[0].requires_grad, v2[0].grad_fn))
        # print("dotprod")
        prd = torch.dot(v1[0], v2[0])
        # print("welok")
        for i in range(1, numel):
            # print('i-{}, v1: {}, {}\tv2: {}, {}'.format(i, v1[i].requires_grad, v1[i].grad_fn, v2[i].requires_grad, v2[i].grad_fn))
            prd = prd + torch.dot(v1[i], v2[i])  # .cpu().detach().item()
            # prd = prd + torch.dot(v1[i], torch.mul(1.0/torch.norm(v2[i]).detach().item(), v2[i])).detach().item()
        if metric_incr == True:
            return -prd
        else:
            return prd

    def train(self, xtrain, ytrain, xval, yval):
        # **1** Look-ahead update for the current batch *********************************************
        # copy weights and stuf
        # print("allesok")
        # self.mdl.backbone.load_state_dict(self.mdl.backbone.state_dict())
        # self.mdl.backbone.load_state_dict(self.mdl.optimizer.state_dict())

        self.mdl.train(xtrain, ytrain, AlgType.SLGrad)  # if hashtagged then approximated

        # print("trainModelStd ok")
        # **2** Now compute loss and metric functions (including observation weights) ****************
        #      and there corresponding gradients with respect to model parameters    ****************

        # forward pass on the validation set
        resBatch = self.mdl.Evaluate(xval, yval, needgrad=True, weighting=False, type='loss', elmt=False, task=0,
                                     Dynamic_alg=AlgType.SLGrad)
        #print("forward pass on the validation set ok")
        # reset gradientsh
        self.mdl.optimizer.zero_grad()
        # backward pass.
        resBatch["TaskLoss"]["task" + str(0)].backward()
        # don't update model paramters, instead get grads
        gradMetric = copy.deepcopy(self.getGradsModelParam())

        for tid in range(0, self.mdl.NTask):
            # pass on training batch
            # print(tid)
            # print("hierpass")
            resBatch = self.mdl.Evaluate(xtrain, ytrain, needgrad=True, weighting=False, type='loss', elmt=True,
                                         task=tid, Dynamic_alg=AlgType.SLGrad)

            # bsize = self.mdl.configProj["batchSize"];

            bsize = resBatch['TaskLoss']["task" + str(tid)].shape[0]

            weight = self.mdl.weight.detach()
            for ei in range(0, bsize):
                # print("ei: {}, loss: {}".format(ei, resBatch['totLoss'].size()))
                # reset gradients
                self.mdl.optimizer.zero_grad()
                # backward pass.
                # print(bsize)

                if tid == self.mdl.NTask - 1 and ei == bsize - 1:
                    resBatch['TaskLoss']["task" + str(tid)][ei].backward()  #watch out we use the loss not metric here (becaus emetric not diff voor nyu

                else:
                    resBatch['TaskLoss']["task" + str(tid)][ei].backward(retain_graph=True)
                    #print("aha")
                    #print(resBatch['TaskLoss']["task" + str(tid)][ei])
                # don't update model paramters, instead get grads
                # print("gradi")
                gradLoss = copy.deepcopy(self.getGradsModelParam(vecshape=True))
                # print(gradMetric)
                # print("oefieiei")
                # compute gain on the validation
                deltaM = - self.comptInProd(gradLoss, gradMetric, self.mdl.config["Learning_Weight"], self.metric_inc)
                # self.weight[tid, ei] -= self.mdl.configProj["lrWeight"] * deltaM
                # print("indendelta?")
                self.mdl.weight[tid, ei] = -deltaM
                # print("ofdeweight?")
                # print('pw: {}, deltaM: {}, nw: {}'.format(weight[tid, ei], deltaM, self.weight[tid, ei]))

        # bound weight >= 0 and sum(weight) >= 0 and sum(weight) <= 1
        weight = self.mdl.weight.detach()
        weight = torch.clamp(weight, min=0)
        norm_c = torch.sum(weight)
        if norm_c != 0:
            weight = weight / norm_c

        self.mdl.weight.data = weight
        # print('weight: {}\n'.format(self.mdl.weight))

        # **3** Final update after restoring model & optim state **********************************************
        # self.mdl.backbone.load_state_dict(self.mdl.model_copy.state_dict())
        # self.mdl.optimizer.load_state_dict(self.mdl.optim_copy.state_dict())

        self.mdl.train(xtrain, ytrain, AlgType.SLGrad)
        # print("allemaalok")

        return


class Gcosim(Dynamic_Weighting):
    def __init__(self, mdl):
        Dynamic_Weighting.__init__(self,mdl)



    def getNormalizedGrads(self, grads, weight, normType=None):
        assert (normType == NormType.Nun.value or
                normType == NormType.WM.value or
                normType == NormType.ND.value or
                normType == NormType.NDWM.value)

        if normType == NormType.Nun.value:
            return grads
        elif normType == NormType.WM.value:
            return [torch.mul(weight, grd) for grd in grads]
        elif normType == NormType.ND.value:
            return [torch.mul(float(1) / torch.norm(torch.reshape(grd, (-1,))), grd) for grd in grads]
        elif normType == NormType.NDWM.value:
            return [torch.mul(weight / torch.norm(torch.reshape(grd, (-1,))), grd) for grd in grads]
        else:
            return [torch.mul(weight, grd) for grd in grads]


    def getGcosim(self, gradsMain, weightMain, gradsAux, weightAux, normType=NormType.Nun.value):

        #normMain = [torch.norm(torch.reshape(grad, (-1,))) for grad in gradsMain]
        #normAux = [torch.norm(torch.reshape(grad, (-1,))) for grad in gradsAux]

        normMain= [torch.norm(torch.reshape(grad, (-1,))) for grad in gradsMain]
        normAux = [torch.norm(torch.reshape(grad, (-1,))) for grad in gradsAux]


        # simGradMag = []
        cosTheta = 0
        totGrad = []

        for i in range(0, len(gradsMain)):
            # simGradMag.append((2*normMain[i]*normAux[i])/(normMain[0]**2+normAux[i]**2))

            xvec = torch.reshape(gradsMain[i], (-1,))
            yvec = torch.reshape(gradsAux[i], (-1,))
            dprod = torch.dot(xvec, yvec)

        #ossim=torch.nn.CosineSimilarity(dim=-1)
        #cosTheta= 1-cossim(torch.tensor(normMain), torch.tensor(normAux))
            cosTheta = cosTheta + dprod /(normMain[i] * normAux[i])


        if cosTheta >= 0: #>=
            normGradsMain = self.getNormalizedGrads(gradsMain, weightMain, normType)
            normProjGradsAux = self.getNormalizedGrads(gradsAux, weightAux, normType)
            totGrad = []
            for i in range(0, len(gradsMain)):
                totGrad.append(normGradsMain[i] + normProjGradsAux[i])
        else:
            totGrad = self.getNormalizedGrads(gradsMain, weightMain, normType)

        return totGrad

    def train(self, xin, yin):

        gradTaskL = dict()
        self._compute_grad_dim()
        totGradSharedL = []
        self.mdl.backbone.zero_grad()
        for ti in range(0, self.mdl.NTask):
            resBatch = self.mdl.Evaluate(xin, yin, needgrad=True, weighting=False, type='loss', elmt=False, task=ti,
                                          Dynamic_alg=AlgType.Gcosim)
            resBatch["TaskLoss"]["task" + str(ti)].backward(retain_graph=True)
            gradTaskL["task"+str(ti)] = self.getGradsTaskLayers(ti) #compute grad with respect to task specific layers. We do not adapt these gradients, just keep them
            #gradTaskL[baseNameTaskL]=torch.mul(1.0 / torch.norm(torch.reshape(gradTaskL[baseNameTaskL], (-1,))), gradTaskL[baseNameTaskL])
            #print(gradTaskL)
            if ti > 0:
                gradAux = self.getGradsModelParam(False) #compute gradient of aux tasj witg respect to task specific layers
                resGrad = self.getGcosim(gradMain, self.mdl.weight[0], gradAux, self.mdl.weight[ti], NormType.Nun.value) # gives back total gradient wrt shared layers ( main + aux- ) if cossimn pos, otherwise only main
                if ti < self.mdl.NTask:
                   totGradSharedL = resGrad
                else:
                    for j in range(0, len(totGradSharedL)):
                        totGradSharedL[j] = totGradSharedL[j] + resGrad[j]
            else:
                gradMain = self.getGradsModelParam(False)  #compute gradients of main task with respect to shared layers


        for j in range(0, len(totGradSharedL)):
            grd = totGradSharedL[j]
            totGradSharedL[j] = torch.mul(1.0 / torch.norm(torch.reshape(grd, (-1,))), grd)

        #self.mdl.optimizer.zero_grad()
        j = 0

        for name, param in self.mdl.backbone.named_parameters(): #change the values of the gradients with respect to shared layers based on those that yield pos cosine sim.
            if "shared" in name:
                # print("{}, {}, {}".format(len(totGradSharedL), param.grad.size(), totGradSharedL[j].size()))
                param.grad = totGradSharedL[j].to(self.mdl.device)
                j = j + 1

        # update parameters with custom shared gradients (that take into account aux gradient or not depending on cosine similarity)
        self.mdl.optimizer.step()

        return

class IMTL(Dynamic_Weighting):

    def __init__(self, mdl):
        Dynamic_Weighting.__init__(self, mdl)
        self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.mdl.NTask, device=self.mdl.device))

    def train(self, xin, yout):
        self._compute_grad_dim()
        # reset gradients
        self.mdl.optimizer.zero_grad()
        grads = torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)
        unit_grads= torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)
        for ti in range(0, self.mdl.NTask):
            resBatch = self.mdl.Evaluate(xin, yout, needgrad=True, weighting=True, type='loss', elmt=False, task=ti,
                                     Dynamic_alg=self.mdl.dynamic_alg)
            losses=resBatch["TaskLoss"]["task" + str(ti)]*self.loss_scale[ti].exp()-self.loss_scale[ti]
            losses.backward(retain_graph=True)
            grads[ti] = self._grad2vec()
            unit_grads[ti]=grads[ti]/torch.norm(grads, p=2)

        D=grads[0:1].repeat(self.mdl.NTask-1, 1) - grads[1:]
        U=unit_grads[0:1].repeat(self.mdl.NTask-1, 1)-unit_grads[1:]

        alpha = torch.matmul(torch.matmul(grads[0], U.t()), torch.inverse(torch.matmul(D, U.t())))
        alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)

        new_grads = sum([alpha[i] * grads[i] for i in range(self.mdl.NTask)])
        self._reset_grad(new_grads)
        self.mdl.optimizer.step()

        return

class Nash(Dynamic_Weighting):
    def __init__(self, mdl, update_weights_every=1, max_iter=20, max_norm=1):
        Dynamic_Weighting.__init__(self, mdl)
        self.update_rate=update_weights_every
        self.optim_iter=max_iter
        self.max_norm=max_norm
        self.step = 0
        self.prvs_alpha_param = None
        self.init_gtg = np.eye(self.mdl.NTask)
        self.prvs_alpha = np.ones(self.mdl.NTask, dtype=np.float32)
        self.normalization_factor = np.ones((1,))



    def _stop_criteria(self, gtg, alpha_t):
        return (
                (self.alpha_param.value is None)
                or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                or (
                        np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                        < 1e-6
                )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.mdl.NTask,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.mdl.NTask,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.mdl.NTask, self.mdl.NTask), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.mdl.NTask):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)


    def train(self, xin, yout):

        if self.step==0:
            self._init_optim_problem()
        if (self.step % self.update_rate) == 0:
            self.step += 1

            self._compute_grad_dim()
            self.mdl.backbone.zero_grad()
            resBatch = self.mdl.Evaluate(xin, yout, needgrad=True, weighting=True, type='loss', elmt=False, task=-1,
                                     Dynamic_alg=self.mdl.dynamic_alg)
            grads = torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)
            sharedparam =torch.zeros(self.mdl.NTask, self.grad_dim).to(self.mdl.device)
            print(self.grad_dim)
            for ti in range(0, self.mdl.NTask):
                resBatch["TaskLoss"]["task" + str(ti)].backward(retain_graph=True)
                count=0
                for name, param in self.mdl.backbone.named_parameters():
# use official code instead: https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py
                    if "shared" in name:
                        count += 1
                        sharedparam[ti][count]=param
                    g=torch.autograd.grad(resBatch["TaskLoss"]["task" + str(ti)], sharedparam, retain_graph=True)

            print(grads.size())
            #grad= torch.cat([g.view(-1) for g in grads])
            print(grads)
            GTG = torch.mm(grads, grads.t()) #make matrix
            print(GTG)
            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            print(self.normalization_factor)
            GTG = GTG / self.normalization_factor.item()
            print(GTG)
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())

        else:
            self.step += 1
            alpha = self.prvs_alpha

        alpha = torch.from_numpy(alpha).to(torch.float32).to(self.mdl.device)
        for ti in range(0, self.mdl.NTask):
            torch.sum(alpha * resBatch["TaskLoss"]["task" + str(ti)]).backward() #backward pass with new alpha


        if self.max_norm > 0:

            torch.nn.utils.clip_grad_norm_(self.getGradsModelParam(), self.max_norm)

        return




