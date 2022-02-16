
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from utils import calc_weight
from eval_utils import evaluate, evaluate_ari
import copy

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class Trainer_moETM(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None
        self.best_decoder = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_NLL(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

            nll_mod1 = (-recon_log_mod1 * x_mod1).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * x_mod2).sum(-1).mean()


        return nll_mod1.item(), nll_mod2.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moETM_OT(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, classifier, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.classifier = classifier
        self.optimizer = optimizer


    def train(self, x_mod1, x_mod2, celltype, celltype_prop, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)
        toogle_grad(self.classifier, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()
        self.classifier.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()

        logits = self.classifier(mu)
        cls_loss = F.cross_entropy(logits, celltype)

        protocol = self.classifier.classifier.weight.data.clone()

        domain_loss = self.domain_loss(protocol, mu, celltype_prop)

        Loss = nll_mod1 + nll_mod2 + cls_loss + 0.1 * domain_loss + KL_weight * KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item(), domain_loss.item(), cls_loss.item()

    def domain_loss(self, protocol, feature, celltype_prop):
        sim_mat = torch.matmul(protocol, feature.T)

        log_prior = torch.log(celltype_prop)
        new_logits = sim_mat/1+log_prior

        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/1, dim=1)

        cost_mat = self.pairwise_cosine_dist(protocol, feature)
        source_loss = (0.5*cost_mat*s_dist).sum(0).mean()
        target_loss = ((0.5*cost_mat*t_dist).sum(1)*celltype_prop.squeeze(1)).sum()

        loss = source_loss + target_loss

        return loss

    def pairwise_cosine_dist(selfself, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

        return 1-torch.matmul(x,y.T)

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu.to('cpu'))
        return out

    def get_NLL(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod1, recon_log_mod2 = self.decoder(Theta)

            nll_mod1 = (-recon_log_mod1 * x_mod1).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * x_mod2).sum(-1).mean()


        return nll_mod1.item(), nll_mod2.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_cobolt(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer


    def train(self, x_mod1, x_mod2, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu.to('cpu'))
        return out

    def get_NLL(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod1, recon_log_mod2 = self.decoder(Theta)

            nll_mod1 = (-recon_log_mod1 * x_mod1).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * x_mod2).sum(-1).mean()


        return nll_mod1.item(), nll_mod2.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

def Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs):
    LIST = list(np.arange(0, train_num))

    X_mod1, X_mod2, batch_index = Train_set
    test_X_mod1, test_X_mod2, batch_index_test, test_adate = Test_set

    nll = 1e100
    best_ari = 0


    for epoch in range(Total_epoch):
        Loss_all = 0
        NLL_all_mod1 = 0
        NLL_all_mod2 = 0
        KL_all = 0

        tstart = time.time()

        np.random.shuffle(LIST)
        KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-3)

        trainer.encoder_mod1.cuda()
        trainer.encoder_mod2.cuda()
        trainer.decoder.cuda()

        for iteration in range(train_num // batch_size):
            x_minibatch_mod1_T = X_mod1[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            x_minibatch_mod2_T = X_mod2[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            batch_minibatch_T = batch_index[LIST[iteration * batch_size: (iteration + 1) * batch_size]]

            loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

            Loss_all += loss
            NLL_all_mod1 += nll_mod1
            NLL_all_mod2 += nll_mod2
            KL_all += kl

        tend = time.time()



        #if test_nll<nll:
        if epoch % 10 == 0:

            test_nll = 0
            ts_bs = 2000
            for ii in range(int(np.ceil(test_X_mod1.shape[0] / ts_bs))):
                nll_mod1_test, nll_mod2_test = trainer.get_NLL(test_X_mod1[ii * ts_bs:(ii + 1) * ts_bs].to('cuda'),
                                                               test_X_mod2[ii * ts_bs:(ii + 1) * ts_bs].to('cuda'),
                                                               batch_index_test[ii * ts_bs:(ii + 1) * ts_bs].to('cuda'))
                test_nll += (nll_mod1_test + nll_mod2_test)

            #nll = test_nll
            #trainer.best_encoder_mod1 = copy.deepcopy(trainer.encoder_mod1.eval()).to('cpu')
            #trainer.best_encoder_mod2 = copy.deepcopy(trainer.encoder_mod2.eval()).to('cpu')
            #trainer.best_decoder = copy.deepcopy(trainer.decoder.eval()).to('cpu')
            trainer.encoder_mod1.to('cpu')
            trainer.encoder_mod2.to('cpu')
            trainer.decoder.to('cpu')

            print('[epoch %0d finished time %4f], Loss=%.4f, Train_NLL_m1=%.4f, Train_NLL_m2=%.4f, KL=%.4f, Test_NLL_m1=%.4f, Test_NLL_m2=%.4f, Test_NLL_all=%.4f' %
              (epoch, tend - tstart, Loss_all, NLL_all_mod1, NLL_all_mod2, KL_all, nll_mod1_test, nll_mod2_test, test_nll))

            embed = trainer.get_embed(test_X_mod1, test_X_mod2)
            test_adate.obsm.update(embed)
            result = evaluate(adata=test_adate, n_epoch=epoch, return_fig=True, **Eval_kwargs)
            print('epoch %0d, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f, Batch_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f, Batch_ebm=%.4f' %
                (epoch, result['ari'], result['nmi'], result['asw'], result['asw_2'], result['k_bet'], result['batch_asw'],
                 result['batch_graph_score'], result['ebm']))

            if best_ari<result['ari']:
                Result = result
                best_ari = result['ari']

    return Result

def Train_moETM_OT(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs):
    LIST = list(np.arange(0, train_num))

    X_mod1, X_mod2, cell_type, cell_type_prop = Train_set
    test_X_mod1, test_X_mod2, batch_index_test, test_adate = Test_set

    for epoch in range(Total_epoch):
        Loss_all = 0
        NLL_all_mod1 = 0
        NLL_all_mod2 = 0
        KL_all = 0
        Domain_loss_all = 0
        Cls_loss_all = 0

        tstart = time.time()

        np.random.shuffle(LIST)
        KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-3)

        for iteration in range(train_num // batch_size):
            x_minibatch_mod1_T = X_mod1[LIST[iteration * batch_size: (iteration + 1) * batch_size], :]
            x_minibatch_mod2_T = X_mod2[LIST[iteration * batch_size: (iteration + 1) * batch_size], :]
            celltype_minibatch_T = cell_type[LIST[iteration * batch_size: (iteration + 1) * batch_size]]

            loss, nll_mod1, nll_mod2, kl, domain_loss, cls_loss = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, celltype_minibatch_T, cell_type_prop, KL_weight)

            Loss_all += loss
            NLL_all_mod1 += nll_mod1
            NLL_all_mod2 += nll_mod2
            KL_all += kl
            Domain_loss_all += domain_loss
            Cls_loss_all += cls_loss

        tend = time.time()

        if epoch % 1 == 0:
            embed = trainer.get_embed(test_X_mod1, test_X_mod2)
            nll_mod1_test, nll_mod2_test = trainer.get_NLL(test_X_mod1, test_X_mod2)
            test_adate.obsm.update(embed)
            result = evaluate(adata=test_adate, n_epoch=epoch, return_fig=True, **Eval_kwargs)
            print('[epoch %0d finished time %4f], Loss=%.4f, NLL_mod1=%.4f, NLL_adt=%.4f, KL=%.4f, NLL_mod1=%.4f, NLL_adt=%.4f' % (epoch, tend - tstart, Loss_all, NLL_all_mod1, NLL_all_mod2, KL_all, nll_mod1_test, nll_mod2_test))
            print('Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f' % (result['ari'], result['nmi'], result['asw'], result['k_bet'], result['batch_asw'], result['batch_graph_score']))

    embed = trainer.get_embed(test_X_mod1, test_X_mod2)
    nll_mod1_test, nll_mod2_test = trainer.get_NLL(test_X_mod1, test_X_mod2)
    return embed, nll_mod1_test, nll_mod2_test

def Train_cobolt(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs):
    LIST = list(np.arange(0, train_num))

    X_mod1, X_mod2, batch_index = Train_set
    test_X_mod1, test_X_mod2, batch_index_test, test_adate = Test_set

    for epoch in range(Total_epoch):
        Loss_all = 0
        NLL_all_mod1 = 0
        NLL_all_mod2 = 0
        KL_all = 0

        tstart = time.time()

        np.random.shuffle(LIST)
        KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-3)

        for iteration in range(train_num // batch_size):
            x_minibatch_mod1_T = X_mod1[LIST[iteration * batch_size: (iteration + 1) * batch_size], :]
            x_minibatch_mod2_T = X_mod2[LIST[iteration * batch_size: (iteration + 1) * batch_size], :]

            loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, KL_weight)

            Loss_all += loss
            NLL_all_mod1 += nll_mod1
            NLL_all_mod2 += nll_mod2
            KL_all += kl

        tend = time.time()

        if epoch % 100 == 0:
            embed = trainer.get_embed(test_X_mod1, test_X_mod2)
            nll_mod1_test, nll_mod2_test = trainer.get_NLL(test_X_mod1, test_X_mod2)
            test_adate.obsm.update(embed)
            result = evaluate(adata=test_adate, n_epoch=epoch, return_fig=True, **Eval_kwargs)
            print('[epoch %0d finished time %4f], Loss=%.4f, NLL_mod1=%.4f, NLL_adt=%.4f, KL=%.4f, NLL_mod1=%.4f, NLL_adt=%.4f' % (epoch, tend - tstart, Loss_all, NLL_all_mod1, NLL_all_mod2, KL_all, nll_mod1_test, nll_mod2_test))
            print('Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f' % (result['ari'], result['nmi'], result['asw'], result['k_bet'], result['batch_asw'], result['batch_graph_score']))

    embed = trainer.get_embed(test_X_mod1, test_X_mod2)
    nll_mod1_test, nll_mod2_test = trainer.get_NLL(test_X_mod1, test_X_mod2)
    return embed, nll_mod1_test, nll_mod2_test