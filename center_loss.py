import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True,wt_var=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.u_tri_ind = np.triu_indices(num_classes,1)
        self.th_min=10
        # self.cos_th=np.cos(.95*2*np.pi/num_classes) # 10 --> num of classes
        self.cos_th=np.cos(.9*2*np.pi/num_classes) # 10 --> num of classes

        self.th_max=100
        self.register_buffer('wt_var',wt_var)
        self.m=.5
        self.cluster_margin=self.m*np.sqrt(2)*(1-np.cos(np.pi/10))
        self.euclid_cluster_size=.8

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
        # if self.use_gpu:
        #     self.centers=nn.Parameter(torch.randn(self.num_classes, self.feat_dim).renorm(2,1,1e-1).mul(1e1).cuda())
        # else: 
        #     self.centers=nn.Parameter(torch.randn(self.num_classes, self.feat_dim).renorm(2,1,1e-1).mul(1e1))

            
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        # x=F.normalize(x)
        # xx=F.normalize(self.centers)
        xx=self.centers
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(xx, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, xx.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#.type(torch.FloatTensor)
 
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist=distmat[mask].clamp(min=1e-12, max=1e+12)
        loss = dist.mean()
        # loss = 1*torch.relu(dist-self.euclid_cluster_size).mean()
        # return loss

        # *** soft cluster margin ***
        # rad=self.centers.norm(2,1).mean()
        # soft_clusters_margin=rad*self.cluster_margin
        # loss=torch.relu(dist-soft_clusters_margin).mean()
        # return loss
        # ***************dot product***********************#
        xx=F.normalize(xx)
        dists=torch.matmul(xx,xx.transpose(1,0))
        loss_inner_prod=torch.sum(torch.relu(10*(dists[self.u_tri_ind]-self.cos_th)))
        return loss.clamp(min=1e-6,max=1e2)+loss_inner_prod

        # ***************dot product***********************#
    
    
        # ***************distance in Euclidean Space***********************#
        # xx=self.centers 
        # x_norm = (xx**2).sum(1).view(-1, 1)
        # y_t = torch.transpose(xx, 0, 1)
        # y_norm = x_norm.view(1, -1)
    
        # dist_c = x_norm + y_norm - 2.0 * torch.mm(xx, y_t)
        # dist_c = (dist_c - torch.diag(dist_c.diag())).clamp(min=1e-12, max=1e+12)
        # dist_c = dist_c[self.u_tri_ind]
        # loss_cnt_dis=torch.sum(torch.relu(self.th_min-dist_c)) #+ 0.2*torch.sum(torch.relu(dist_c-self.th_max))
#         loss_cnt_dis=torch.sum(torch.abs(self.th_min-dist_c)) #+ 0.2*torch.sum(torch.relu(dist_c-self.th_max))
    
        # return loss+loss_cnt_dis+self.wt_var*dist_c.std()
        # return loss+loss_cnt_dis+ dist_c.std()
        # return loss+loss_cnt_dis+loss_inner_prod
        # ***************distance in Euclidean Space***********************#

    
        #         pdb.set_trace()
#         dist = []
#         for i in range(batch_size):
#             value = distmat[i][mask[i]]
#             value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
#             dist.append(value)
#         dist = torch.cat(dist)
#         loss = dist.mean()
        
        
       
        



        
 

        
        
        
        

#         # compute the variance of within-class distances
#         labs=labels[:,0]
#         d=[]
#         for ii in classes:
#             tmp=dist[labs.eq(ii)].clamp(min=1e-12, max=1e+12)
#             if tmp.nelement() > 1: d.append(torch.var(tmp)) #d.append(torch.mean(torch.abs(1-tmp)))


#         dd=torch.stack(d)
#         loss_var=dd.var()#torch.mean(torch.abs(1-dd))#
# #         print('center loss',(loss),'var loss',loss_var)
#         return .8*loss+.0*loss_var
        