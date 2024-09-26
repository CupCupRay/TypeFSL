import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot
    

    def forward(self, XS, YS, XQ, YQ, mask=None):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        O_YQ = YQ.clone().detach()

        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta

        std_pred = pred.clone().detach()

        if mask is not None:

            temp = pred.clone().detach()

            for i in range(len(mask)):

                for j in range(len(mask[i])):

                    if mask[i][j] == 0:

                        temp[i][j] = -99.9

            pred = temp

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        if mask is not None:
            error = list()
            # Index to number
            label_in = dict()
            N_YQ = YQ.clone().detach()
            for i in range(len(N_YQ)):
                label_in[int(N_YQ[i].item())] = O_YQ[i].item()
            pred_ans = torch.argmax(pred, dim=1)
            for i in range(len(N_YQ)):
                if N_YQ[i] != pred_ans[i]:
                    error.append((label_in[N_YQ[i].item()], label_in[pred_ans[i].item()]))

            std_acc = BASE.compute_acc(std_pred, YQ)

            """
            # Test filter's error
            std_ans = torch.argmax(std_pred, dim=1) == YQ
            ans = torch.argmax(pred, dim=1) == YQ
            for i in range(len(std_ans)):
                if std_ans[i] == True and ans[i] == False:
                    print(std_ans, ans)
                    print(std_pred, pred)
                    raise ValueError('Filter lead to an error, check the Mask!')
                    """
            return [std_acc, acc], loss, error

        return acc, loss
