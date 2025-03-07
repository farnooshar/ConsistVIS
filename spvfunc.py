import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class SPV(nn.Module):
    def __init__(self):
        super(SPV, self).__init__()
        self.maxvalue = 1000000;
    
    def normalize_matrix_value(self,mat,min_val,max_val):
        if max_val - min_val == 0:
            return np.zeros_like(mat)  # To avoid division by zero, return zeros if the matrix is constant
        return (mat - min_val) / (max_val - min_val)

    # Function to calculate L2 distance (Euclidean distance) between two normalized matrices
    def normalized_l2_distance_(self,mat1, mat2,min_val,max_val):
        # Normalize both matrices to the range [0, 1]
        norm_mat1 = self.normalize_matrix_value(mat1,min_val,max_val)
        norm_mat2 = self.normalize_matrix_value(mat2,min_val,max_val)

        # Compute the L2 distance between the normalized matrices
        distance = np.linalg.norm(norm_mat1 - norm_mat2)

        return distance


    def support_vector_distance(self,ln,support_vectors_input,minn,maxn,T):

        fraction = min(np.min(ln),3)
        
        spv=[]
        for t in range(len(ln)):
           if ln[t]!=self.maxvalue:
            #print('len(support_vectors_input[t]) --> ',ln[t],len(support_vectors_input[t]))
            spv.append(support_vectors_input[t][np.random.choice(support_vectors_input[t].shape[0], size=fraction, replace=False)])

        # Function to calculate the pairwise Euclidean distance between two matrices

        maxn = np.max(maxn)
        minn = np.max(minn)

        if len(spv)>1 and (maxn-minn)!=0:

          # Calculate pairwise distances between the 3 matrices
          distances = np.zeros((len(spv), len(spv)))  # 3x3 matrix to store distances

          for i in range(len(spv)):
              for j in range(i + 1, len(spv)):
                  #print(len(spv[i]),len(spv[j]))
                  distances[i, j] = self.normalized_l2_distance_(spv[i], spv[j],minn,maxn)
                  distances[j, i] = distances[i, j]  # Symmetric matrix


          # Extract the upper triangular part of the matrix (excluding the diagonal)
          upper_triangular = np.triu(distances, k=1)

          # Get non-zero elements from the upper triangular part
          non_zero_elements = upper_triangular[upper_triangular > 0]

          # Calculate the average of the non-zero elements
          average_upper_non_zero = np.mean(non_zero_elements) if non_zero_elements.size > 0 else 0

          average_upper_non_zero = average_upper_non_zero
        else:
          average_upper_non_zero = 10;

        return average_upper_non_zero


    def score_outside(self,clusters,K,flattened,T):
        support_vectors = []
        ln=[]; maxn=[];minn=[]
        ouside=[]

        
        for k in range(K):
            # Create a binary label: 1 for the current cluster, 0 for all other clusters
            y = np.where(clusters == k, 1, 0)

            
            if len(y)<=10 or len(np.unique(y))<=1:
                continue
                
            # Train a binary SVM classifier (one-vs-rest)
            svm = SVC(kernel='linear')
            svm.fit(flattened, y)

            
            # Extract the support vectors that belong to the current cluster (y == 1)
            cluster_sv = svm.support_vectors_[y[svm.support_] == 1]

            #print('ZZZZZZZZZZZ outside',len(cluster_sv),len(y))
                  
            # Add all support vectors for this cluster to the list
            support_vectors.append(cluster_sv)

            maxn.append(np.max(cluster_sv))
            minn.append(np.min(cluster_sv))

            if len(cluster_sv)==0:
                ln.append(self.maxvalue)
            else:
                ln.append(len(cluster_sv))


        if len(ln)==0:
            ouside = 0.0
        else:  
            ouside = self.support_vector_distance(ln,support_vectors,minn,maxn,T)
        
        return ouside
    
    def score_inside(self,clusters,K,flattened,keeptime,T):
        
    # Step 2: One-vs-Rest SVM to find all support vectors for each cluster
        
        inside=[]

        for k in range(K):
            # Create a binary label: 1 for the current cluster, 0 for all other clusters
            #y = np.where(clusters == k, 1, 0)

            kps = np.where(clusters == k)
            flattened_k = flattened[kps]
            keeptime_k  = keeptime[kps]
            support_vectors_inside = []
            ln=[];maxn=[];minn=[]
            for t in range(T):

              yt = np.where(keeptime_k==t,1,0)
            
              #print('len(np.unique(yt))',len(np.unique(yt)))
              if len(yt)<=10 or len(np.unique(yt))<=1:
                continue

              svm = SVC(kernel='linear')
              svm.fit(flattened_k, yt)


              cluster_sv = svm.support_vectors_[yt[svm.support_] == 1]
              support_vectors_inside.append(cluster_sv)

              maxn.append(np.max(cluster_sv))
              minn.append(np.min(cluster_sv))

              if len(cluster_sv)==0:
                ln.append(self.maxvalue)
              else:
                ln.append(len(cluster_sv))

            if len(ln)==0:
              inside.append(10);
              continue

            inside.append(self.support_vector_distance(ln,support_vectors_inside,minn,maxn,T))


        inside_score = np.mean(inside)

        return inside_score
    
    def forward(self, tensor, KK):
        
        
        ############################## SIL ##############################
            #KK=[];
            #for gkey in targets:
            #    #print('gkey ',gkey['labels'])
            #    KK.append(gkey['labels'].size()[0])
                
        ############################## SIL ##############################

        
        #print('KKKKKKKKKKKK',KK)
        
        
        B, C, T, H, W = tensor.shape
        #select_random_c = np.random.choice(C, size=50, replace=False)
        
        #print('select_random_c',select_random_c,len(select_random_c))
        
        #tensor = tensor[:,select_random_c]
        #print('tensor shape',tensor.shape)
            
        #B, C, T, H, W = tensor.shape
        
        #print('tensor shape',tensor.shape)
        
          # Calculate the new height and width
        new_H = int(H // 4)
        new_W = int(W // 4)

        tensor = torch.nn.functional.interpolate(tensor.view(B * T, C, H, W), size=(new_H, new_W), mode='bilinear', align_corners=False).view(B, C, T, new_H, new_W)
        
        ##################### SELECT RANDOM POINT ################
        flat_tensor = tensor.view(B, C, T, -1)
        # Select 130 random indices
        num_pixels = 130
        random_indices = torch.randperm(new_H * new_W)[:num_pixels]
        # Gather the selected pixels across the flattened height and width
        sampled_pixels = flat_tensor[..., random_indices]
        # Reshape to B*T*C*26*5
        tensor = sampled_pixels.view(B, C, T, 26, 5)
        ##########################################################
        #print('tensor shape',tensor.shape)
        
        keeptime = torch.zeros((1, 1, T, 26, 5), dtype=torch.float32)

        #print('keeptime shape',keeptime.shape)

            
        for t in range(0,T):
            keeptime[:,:, t, :, :] = t  # For each t, set the entire H, W grid to the value t

        keeptime = keeptime[0].permute(1, 2, 3, 0).reshape(-1).detach().cpu().numpy()

        #print('tensor shape',tensor.shape)
        
        score = 0.0
        
        try:
            for i in range(B):
                # Flatten T, H, W dimensions, keeping C as the feature dimension
                flattened = tensor[i].permute(1, 2, 3, 0).reshape(-1, C).detach().cpu().numpy()  # Convert to numpy array
                #try:
                    # Perform K-means clustering
                #print('flattened',flattened.shape,keeptime.shape)
                kmeans = KMeans(n_clusters=KK[i])
                clusters = kmeans.fit_predict(flattened)


                inside  = self.score_inside(clusters,KK[i],flattened,keeptime,T)/10
                if KK[i]>1:
                    outside = self.score_outside(clusters,KK[i],flattened,T)/10
                else:
                    outside = 0.0;



                score = score + (inside + max((1-outside),0.0))/2
                #print('SPV ->' , inside,outside,score,max((1-outside),0.0))
        except:
            pass
        
        # Average the loss over the batch
        score = 1 + (score / B)
        return torch.tensor(score, requires_grad=True, device=tensor.device)
        