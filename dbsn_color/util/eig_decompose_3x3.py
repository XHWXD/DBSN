import torch
import math
from .arccos_self import LLTMFunction

# A: N*3*3, 
def eigs_vec_comp(A, A_eig):
    p1 = A[:,0,1].pow(2) + A[:,0,2].pow(2) + A[:,1,2].pow(2)
    
    diag_matrix_flag = p1==0
    A_diag = A.clone()
    A_diag = A_diag[diag_matrix_flag]
    A_non_diag = A.clone()
    A_non_diag = A_non_diag[~diag_matrix_flag]
    #p1_diag = p1[diag_matrix_flag]
    p1_non_diag = p1.clone()
    p1_non_diag = p1_non_diag[~diag_matrix_flag]
    # for eigenvectos
    A_v = torch.zeros(3,3,device='cuda').repeat(A.shape[0],1,1)
    
    if A_diag.shape[0]>0:
        v_diag = torch.eye(3,device='cuda').repeat(A_diag.shape[0],1,1) 
        A_v[diag_matrix_flag]=v_diag
               
    A_non_diag = A.clone()
    A_non_diag = A_non_diag[~diag_matrix_flag]
    A_eig_non_diag = A_eig.clone()
    A_eig_non_diag = A_eig_non_diag[~diag_matrix_flag]
    if A_eig_non_diag.shape[0]>0:
        I_matrix = torch.eye(3,device='cuda').repeat(A_non_diag.shape[0],1,1)
        # for those non-diagonal matrix
        lamb1 = A_eig_non_diag[:,0,0].unsqueeze(1).view(A_eig_non_diag.shape[0],1,1).repeat(1,3,3)  # N*3*3
        A_lamb1 = A_non_diag - lamb1*I_matrix
        lamb2 = A_eig_non_diag[:,1,1].unsqueeze(1).view(A_eig_non_diag.shape[0],1,1).repeat(1,3,3)
        A_lamb2 = A_non_diag - lamb2*I_matrix
        lamb3 = A_eig_non_diag[:,2,2].unsqueeze(1).view(A_eig_non_diag.shape[0],1,1).repeat(1,3,3)
        A_lamb3 = A_non_diag - lamb3*I_matrix

        v1_m = A_lamb2 @ A_lamb3
        v1 = v1_m.sum(2)
        v1_norm = (v1 / (v1.pow(2).sum(1).sqrt().repeat(3,1).t())).unsqueeze(2)
        v2_m = A_lamb1 @ A_lamb3
        v2 = (v2_m).sum(2)
        v2_norm = (v2 / (v2.pow(2).sum(1).sqrt().repeat(3,1).t())).unsqueeze(2)
        v3_m = A_lamb1 @ A_lamb2
        v3 = (v3_m).sum(2)
        v3_norm = (v3 / (v3.pow(2).sum(1).sqrt().repeat(3,1).t())).unsqueeze(2)
        #
        v_non_diag = torch.cat((v1_norm,v2_norm),2)
        v_non_diag = torch.cat((v_non_diag,v3_norm),2)
        #
        A_v[~diag_matrix_flag]=v_non_diag
    
    return A_v


# A: N*3*3
def eigs_comp(A):
    p1 = A[:,0,1].pow(2) + A[:,0,2].pow(2) + A[:,1,2].pow(2)
    
    diag_matrix_flag = p1==0
    A_diag = A.clone()
    A_diag = A_diag[diag_matrix_flag]
    A_non_diag = A.clone()
    A_non_diag = A_non_diag[~diag_matrix_flag]
    #p1_diag = p1[diag_matrix_flag]
    p1_non_diag = p1.clone()
    p1_non_diag = p1_non_diag[~diag_matrix_flag]

    eig1_diag = torch.zeros(A_diag.shape[0],1, device='cuda')
    eig2_diag = torch.zeros(A_diag.shape[0],1, device='cuda')
    eig3_diag = torch.zeros(A_diag.shape[0],1, device='cuda')
    if A_diag.shape[0]>0:
        # for those diagonal matrix
        eig1_diag_tmp = A_diag[:,0,0].unsqueeze(1)
        eig2_diag_tmp = A_diag [:,1,1].unsqueeze(1)
        eig3_diag_tmp = A_diag[:,2,2].unsqueeze(1)
        EIG = torch.cat((eig1_diag_tmp,eig2_diag_tmp),1)
        EIG = torch.cat((EIG,eig3_diag_tmp),1)
        EIG, ind = EIG.sort(1)
        eig1_diag = EIG[:,2].unsqueeze(1)
        eig2_diag = EIG[:,1].unsqueeze(1)
        eig3_diag = EIG[:,0].unsqueeze(1)
        
    eig1_non_diag = torch.zeros(A_non_diag.shape[0],1, device='cuda')
    eig2_non_diag = torch.zeros(A_non_diag.shape[0],1, device='cuda')
    eig3_non_diag = torch.zeros(A_non_diag.shape[0],1, device='cuda') 
    if A_non_diag.shape[0]>0:
        # for those non-diagonal matrix
        tr_A = (A_non_diag[:,0,0]+A_non_diag[:,1,1]+A_non_diag[:,2,2])/3               # trace(A) is the sum of all diagonal values
        p2 = (A_non_diag[:,0,0] - tr_A).pow(2) + (A_non_diag[:,1,1] - tr_A).pow(2) + (A_non_diag[:,2,2]- tr_A).pow(2) + 2 * p1_non_diag
        p3 = torch.sqrt(p2/6)
        I_matrix = torch.eye(3,device='cuda').repeat(A_non_diag.shape[0],1,1)
        tmp_tr_A = tr_A.view(A_non_diag.shape[0],1,1).repeat(1,3,3) 
        tmp_p = p3.view(A_non_diag.shape[0],1,1).repeat(1,3,3)
        B = (1 / tmp_p) * (A_non_diag - tmp_tr_A * I_matrix)    # I is the identity matrix
        tmp_det_B = torch.det(B)/2
            
        #grads1={}
        #def save_grad1(name):
        #    def hook(grad):
        #        grads1[name]=grad
        #    return hook
        #tmp_det_B.register_hook(save_grad1('tmp_det_B'))  
    
        # In exact arithmetic for a symmetric matrix  -1 <= tmp <= 1
        # but computation error can leave it slightly outside this range.
        pi_tmp = LLTMFunction.apply(tmp_det_B)
            
        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1_non_diag = tr_A + 2 * p3 * torch.cos(pi_tmp)
        eig3_non_diag = tr_A + 2 * p3 * torch.cos(pi_tmp + (2*math.pi/3))
        eig2_non_diag = 3 * tr_A - eig1_non_diag - eig3_non_diag     # since trace(A) = eig1 + eig2 + eig3
        
        eig1_non_diag = eig1_non_diag.unsqueeze(1)
        eig2_non_diag = eig2_non_diag.unsqueeze(1)
        eig3_non_diag = eig3_non_diag.unsqueeze(1)
    
    eig1 = torch.zeros(A.shape[0],1, device='cuda')
    eig1[diag_matrix_flag] = eig1_diag
    eig1[~diag_matrix_flag] = eig1_non_diag
    eig2 = torch.zeros(A.shape[0],1, device='cuda')
    eig2[diag_matrix_flag]=eig2_diag
    eig2[~diag_matrix_flag]=eig2_non_diag
    eig3 = torch.zeros(A.shape[0],1, device='cuda')
    eig3[diag_matrix_flag]=eig3_diag
    eig3[~diag_matrix_flag]=eig3_non_diag
    
    A_eig = torch.zeros(A.shape,device='cuda')
    A_eig[:,0,0]=eig1.squeeze()
    A_eig[:,1,1]=eig2.squeeze()
    A_eig[:,2,2]=eig3.squeeze()   
    
    return A_eig