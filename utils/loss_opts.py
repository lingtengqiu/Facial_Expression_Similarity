import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Triple loss function
margins={
'ONE_CLASS_TRIPLET':0.1,
'TWO_CLASS_TRIPLET':0.2,
'THREE_CLASS_TRIPLET':0.3,
}
def triple(x1,x2,y1,sigm):
    triple_1 = torch.sum((x1 - x2) ** 2) - torch.sum((x1 - y1) ** 2) + sigm
    triple_2 = torch.sum((x1 - x2) ** 2) - torch.sum((x2 - y1) ** 2) + sigm
    return torch.max(torch.zeros(1).cuda(),triple_1)+torch.max(torch.zeros(1).cuda(),triple_2)
def triple_loss(inputs, labels, sigm = 1e-6, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    loss_item =0.
    cnt =0
    for ind,target in enumerate(labels):
        if(target == 1):
            loss = triple(inputs[1][ind],inputs[2][ind],inputs[0][ind],sigm)
        elif (target == 2):
            loss = triple(inputs[0][ind],inputs[2][ind],inputs[1][ind],sigm)
        else:
            loss = triple(inputs[0][ind], inputs[1][ind], inputs[2][ind],sigm)
        if loss.item() != 0:
            cnt+=1
        loss_item += (loss)
    if size_average :
        loss_item/=cnt
    return loss_item
def batch_triple_loss(inputs,labels, types,size_average=True):
    #according to paper
    emb1 = torch.matmul(inputs,torch.transpose(inputs,0,1))
    sq = torch.diag(emb1)
    matrix = (sq[:,None] - 2*emb1+sq[None,:])
    #batch and batch
    #matrix[i,j,k] = aij -aik
    matrix = matrix[:,:,None] - matrix[:,None,:]
    batch_size = matrix.shape[0]//3
    labels = labels.numpy()
    masks_one = np.zeros(matrix.shape,dtype=np.bool)
    masks_two = masks_one.copy()
    for i,label in enumerate(labels):
        if(label ==1):
            # print(label)
            # print(batch_size*1+i,batch_size*2+i,i)
            # print(batch_size*2+i,batch_size*1+i,i)
            masks_one[batch_size*1+i,batch_size*2+i,i] = True
            masks_two[batch_size*2+i,batch_size*1+i,i] = True
        elif (label == 2):
            # print(label)
            # print(i, batch_size * 2 + i, batch_size*1+i)
            # print(i, batch_size * 2 + i, batch_size*1+i)
            masks_one[i, batch_size * 2 + i, batch_size*1+i] = True
            masks_two[batch_size * 2 + i, i, batch_size*1+i] = True
        elif (label == 3):
            # print(label)
            # print(batch_size + i, i, batch_size * 2 +i)
            # print(i, batch_size * 1 + i, batch_size * 2 +i)
            masks_one[batch_size + i, i, batch_size * 2 +i] = True
            masks_two[i, batch_size * 1 + i, batch_size * 2 +i] = True
    masks_one = torch.from_numpy(masks_one)
    masks_two = torch.from_numpy(masks_two)
    margin_matrix = [margins[i] for i in types]
    margin_matrix = torch.from_numpy(np.asarray(margin_matrix)).float().cuda()
    loss1 = matrix[masks_one]+margin_matrix
    loss2 = matrix[masks_two]+margin_matrix
    loss1[loss1<0]=0
    loss2[loss2<0] = 0
    loss = loss1+loss2
    return 1-torch.nonzero(loss).shape[0]/batch_size,torch.mean(loss)
def batch_triple_loss_acc(inputs,labels, types,size_average=True):
    #according to paper
    emb1 = torch.matmul(inputs,torch.transpose(inputs,0,1))
    sq = torch.diag(emb1)
    matrix = (sq[:,None] - 2*emb1+sq[None,:])
    #batch and batch
    #matrix[i,j,k] = aij -aik
    matrix = matrix[:,:,None] - matrix[:,None,:]
    batch_size = matrix.shape[0]//3
    labels = labels.numpy()
    masks_one = np.zeros(matrix.shape,dtype=np.bool)
    masks_two = masks_one.copy()
    for i,label in enumerate(labels):
        if(label ==1):
            masks_one[batch_size*1+i,batch_size*2+i,i] = True
            masks_two[batch_size*2+i,batch_size*1+i,i] = True
        elif (label == 2):
            masks_one[i, batch_size * 2 + i, batch_size*1+i] = True
            masks_two[batch_size * 2 + i, i, batch_size*1+i] = True
        elif (label == 3):
            masks_one[batch_size + i, i, batch_size * 2 +i] = True
            masks_two[i, batch_size + i, batch_size * 2 +i] = True
    masks_one = torch.from_numpy(masks_one)
    masks_two = torch.from_numpy(masks_two)
    # margin_matrix = torch.from_numpy(np.asarray(margin_matrix)).float().cuda()
    loss1 = matrix[masks_one]
    loss2 = matrix[masks_two]
    loss1[loss1 == 0]=1.
    loss1[loss1<0]=0
    loss2[loss2<0] = 0
    loss = loss1+loss2
    return torch.nonzero(loss).shape[0],loss




