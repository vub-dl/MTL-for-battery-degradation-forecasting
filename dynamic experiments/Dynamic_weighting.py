from enum import Enum, unique
from Requirements import *
import copy

@unique


class AlgType(Enum): #supported algorithms
    Unif = 'unif'
    Olaux = 'olaux'
    Random= 'random'
    PCGrad='pcgrad'
    CAgrad='cagrad'

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


