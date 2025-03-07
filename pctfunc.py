import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import keyact
import time

class PCTKernelLayer(nn.Module):
    def __init__(self, image_size):
        super(PCTKernelLayer, self).__init__()
        self.image_size = image_size

    def forward(self, image, list_nl):

        response =[]
        x = torch.linspace(1, self.image_size, self.image_size).to(image.device)
        y = torch.linspace(1, self.image_size, self.image_size).to(image.device)
        X, Y = torch.meshgrid(x, y)

        # Center the grid
        c = (1 + self.image_size) / 2
        X = (X - c) / (c - 1)
        Y = (Y - c) / (c - 1)

        # Compute polar coordinates
        R = torch.sqrt(X**2 + Y**2)
        Theta = torch.atan2(Y, X)

        # Mask for R <= 1 (unit circle)
        mask = (R <= 1).float()

        for nl in list_nl:
          n,l = nl
          # Compute the PCT kernel (magnitude and phase)
          amplitude = mask * torch.cos(np.pi * n * R**2)  # Amplitude component
          phase = mask * torch.exp(-1j * l * Theta)       # Phase component

          # Combine to get complex result
          complex_result = amplitude * phase  # Complex representation

          # Use magnitude for rotation invariance
          magnitude = torch.abs(complex_result)

          # Perform convolution with the magnitude
          #image =   # Ensure input has the right shape (batch_size, channels, H, W)
          kernel = magnitude.unsqueeze(0).unsqueeze(1)  # Shape (1, 1, H, W) to match for conv2d

          #print('kernel',kernel.size())
          kernel = kernel.repeat(3, 1, 1, 1)
          #print('kernel',kernel.size())
          #print('kernel rep',kernel.size(),n,l)
            
          #plt.imshow(kernel[0][0].detach().numpy())

          #plt.figure()
          convolved_image = F.conv2d(image, kernel, padding='same',groups=3)

          #print('convolved_image',convolved_image.size(),n,l)
          #plt.imshow(convolved_image[0][0].detach().numpy())
          response.append(convolved_image)
          #plt.figure()

        return response

# Subnetwork to output n and l from input
class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply Sigmoid activation to get values between 0 and 1, then scale to [1, 4]
        x = torch.sigmoid(x) * 3 + 1  # Scales to range [1, 4]
        
        return x  # Output values are now between 1 and 4

# Subnetwork to output power
class PowNet(nn.Module):
    def __init__(self):
        super(PowNet, self).__init__()
        self.fc1 = nn.Linear(18, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply Sigmoid activation to get values between 0 and 1, then scale to [1, 3]
        x = torch.sigmoid(x) * 2 + 1  # Scales to range [1, 3]
        
        return x  # Output values are now between 1 and 3
    
# Main model
class MainModel(nn.Module):
    def __init__(self, image_size):
        super(MainModel, self).__init__()
        self.subnet = SubNet()  # Subnet to get n and l
        self.pct_layer = PCTKernelLayer(image_size)  # PCT layer to apply PCT kernel

    def forward(self, input_data, image):
        n_l_output = self.subnet(input_data)  # Process 16x1 input to get n and l

        n1 = n_l_output[:, 0].unsqueeze(1)
        l1 = n_l_output[:, 1].unsqueeze(1)

        n2 = n_l_output[:, 2].unsqueeze(1)
        l2 = n_l_output[:, 3].unsqueeze(1)

        #n3 = n_l_output[:, 4].unsqueeze(1)
        #l3 = n_l_output[:, 5].unsqueeze(1)

        convolved_image = self.pct_layer(image, [[n1,l1],[n2,l2]])
        return convolved_image
    
class PCT(nn.Module):
    def __init__(self):
        super(PCT, self).__init__()
        self.maxvalue = 1000000;
    
    def norm(self,image):
        normimage =  (image - np.min(image))/(np.max(image)-np.min(image))
        return normimage;

    '''
    def targetpair(self,targets,H,W):
        
        zr = np.zeros((H,W)).astype('uint8')
        MaskI=[];MaskO=[];
        for Batch in targets:
          Masks = Batch['masks'].detach().cpu().numpy()
          I,T,H,W = Masks.shape

          for t in range(T):
            zrt = zr.copy()
            for i in range(I):
              ips = np.where(Masks[i,t]!=0)
              zrt[ips]=i+1;

            MaskI.append(zrt)
            MaskO.append(np.sum(targets[0]['masks'].detach().cpu().numpy()[:,t],axis=0))

        return 
    '''
    
    def keypoint_func(self,img1,img2):

        #gpath1 = path1.replace('AnnTrue','JPEGImages').replace('.png','.jpg')
        #img1 = cv2.imread(gpath1)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        #gpath2 = path2.replace('AnnTrue','JPEGImages').replace('.png','.jpg')
        #img2 = cv2.imread(gpath2)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=2000)

        # Detect keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)


        # Match features using FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)  # KD-Tree algorithm for FLANN
        search_params = dict(checks=50)  # Number of checks to perform
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        matched_indices1 = set()

        # Store only good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
                matched_indices1.add(m.queryIdx)

        # Check if there are enough good matches
        #if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        unmatched_keypoints1 = [kp for idx, kp in enumerate(keypoints1) if idx not in matched_indices1]

        # Check if there are enough good matches
        #if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        w = img1.shape[1]; h = img1.shape[0];
        transformed_keypoints=[]

        H=[]
        #print('good_matches',len(good_matches),len(keypoints1))
        if len(good_matches) > 10:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if len(unmatched_keypoints1)>0:
                
                # Extract the (x, y) coordinates from the KeyPoint objects
                keypoint_coords = np.array([kp.pt for kp in unmatched_keypoints1], dtype=np.float32)

                # Reshape the array to match the input format of cv2.perspectiveTransform
                # It needs to be of shape (n_points, 1, 2)
                try:
                    keypoint_coords = keypoint_coords.reshape(-1, 1, 2)

                    # Apply the perspective warp to the keypoints using the homography matrix
                    transformed_keypoints = cv2.perspectiveTransform(keypoint_coords, H)

                    # Reshape the result back to (n_points, 2)
                    transformed_keypoints = transformed_keypoints.reshape(-1, 2)
                except:
                    transformed_keypoints=[];
                    pass

            '''
            matches_mask = mask.ravel().tolist()

            # Draw matches with inliers highlighted
            img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                          matchColor=(0, 255, 0), singlePointColor=None,
                                          matchesMask=matches_mask, flags=2)

            # Draw matches with inliers highlighted
            img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                          matchColor=(0, 255, 0), singlePointColor=None, flags=2)

            # Show the result
            plt.figure(figsize=(15, 8))
            plt.imshow(img_matches)
            plt.title('Good Matches with RANSAC')
            plt.axis('off')
            plt.show()
            '''

        #return {'src':src_pts,'des':dst_pts}
        return (src_pts,dst_pts),keypoints1,transformed_keypoints,keypoints2,H,unmatched_keypoints1

    def scale(self,targets):
        ratio_scale=[];stat=[]
        for Batch in targets:
          Masks = Batch['masks'].detach().cpu().numpy()
          I,T,H,W = Masks.shape
          cmax = []
          for i in range(0,I):
            cmax.append(len(np.where(Masks[i]>0)[0]))
          
          if len(cmax)!=0:
              if np.max(cmax)!=0:
                  ratio_scale.append(np.sum(cmax/np.max(cmax)))
                  stat.append(1)
              else:
                ratio_scale.append(1)
                stat.append(-1)
          else:
            ratio_scale.append(1)
            stat.append(-1)
            
        return torch.tensor(ratio_scale).float().cuda(),stat
    
    def warp(self,hi,wi,H):
        size = (wi,hi)
        one = np.ones((hi,wi))
        if len(H)!=0:
            one = cv2.warpPerspective(one, H, size)
        return one
    
    def warp_branch(self,input_tensor):

        conv2d = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(2, 2)).cuda()
        conv_1x1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1, 1), stride=(2, 2)).cuda()   
        conv_output = conv_1x1(conv2d(torch.tensor(input_tensor).unsqueeze(0).unsqueeze(0).float().cuda()))
        global_pooling = torch.mean(conv_output, dim=(2,3))
        
        return global_pooling

    def feat_branch(self,input_tensor,C):
        
        conv3d_1x1 = nn.Conv3d(in_channels=C, out_channels=16, kernel_size=(1, 1, 1), stride=(1, 1, 1)).cuda()
        conv_output = conv3d_1x1(input_tensor)
        global_pooling = torch.mean(conv_output, dim=(2, 3, 4))  # Output shape will be (B, 16)
        
        return global_pooling

    def forward(self, images,src_masks,outputs,targets):
        
        #print('images',images.shape)
        #print('src_masks',src_masks.shape)
        #print('outputs',outputs['pred_masks'].shape)
        
        pz=2
        
        kernelpct = MainModel(image_size=4).cuda()
        
        Bi, Ci, H, W = images.shape
        B, C, T, Hf, Wf = outputs['pred_masks'].shape
        
        #start_time = time.time()

        up   = self.feat_branch(outputs['pred_masks'],C)
        pownet = PowNet().cuda()
        number_instances = torch.tensor([len(targets[0]['labels'])/6,len(targets[1]['labels'])/6]).float().cuda()
        
        scalevalue,stat = self.scale(targets)
        
        power = pownet(torch.cat((up,number_instances.unsqueeze(1),scalevalue.unsqueeze(1)),axis=1))
        
        #print("--- %s 11111 seconds ---" % (time.time() - start_time))
        
        #print('number_instances',number_instances.shape)
        #print('scale',self.scale(targets).shape)
        
        
        self.keyact = keyact.KeyAct()
        
        #print('up',up.shape)
        #print('power',power.shape,power)
        
        #print(Asd)
        lossbatch=0
        MaskO=[]
        Be=-1;Ri=0;Qi=0;Si=0;
        for b in range(0,Bi,3):
            #print('b',b,b+1,b+2)
            
            Be+=1;
            if stat[Be]==-1:
                continue
            
            Masks = targets[Be]['masks'].detach().cpu().numpy()
            I,T,_,_ = Masks.shape
            Qi= Ri + I
            
            for t in range(T):
                MaskO.append(np.sum(targets[Be]['masks'].detach().cpu().numpy()[:,t],axis=0))
    
            
            #print('src_masks',src_masks.shape)
            #print('Ri,Qi',I,Ri,Qi)
            #print('--------------------------')
                  
            img1 = (self.norm(images[b].permute(1,2,0).detach().cpu().numpy())*255).astype('uint8')
            img2 = (self.norm(images[b+1].permute(1,2,0).detach().cpu().numpy())*255).astype('uint8')
            img3 = (self.norm(images[b+2].permute(1,2,0).detach().cpu().numpy())*255).astype('uint8')

            
            keys__x,keypoints1__x,transformed_keypoints__x,keypoints2__x,Hx,unmatched_keypoints__x = self.keypoint_func(img1,img2)
            keys__y,keypoints2__y,transformed_keypoints__y,keypoints3__y,Hy,unmatched_keypoints__y = self.keypoint_func(img2,img3)
            keys__z,keypoints3__z,transformed_keypoints__z,keypoints1__z,Hz,unmatched_keypoints__z = self.keypoint_func(img3,img1)
            
            warped__x = self.warp(H,W,Hx); warped__y = self.warp(H,W,Hy); warped__z = self.warp(H,W,Hz)
            down__x = self.warp_branch(warped__x); down__y = self.warp_branch(warped__y); down__z = self.warp_branch(warped__z);
            #print('down__x',down__x.shape)
            
            #print("--- %s keypoint detection seconds ---" % (time.time() - start_time))
            
            #start_time = time.time()
            input_kernel__x = torch.cat((up[Be:Be+1],down__x[0:1]),axis=1)
            input_kernel__y = torch.cat((up[Be:Be+1],down__y[0:1]),axis=1)
            input_kernel__z = torch.cat((up[Be:Be+1],down__z[0:1]),axis=1)
            
            #print('input_kernel__x',input_kernel__x.shape)
            
            response__x = torch.cat(kernelpct(input_kernel__x, src_masks[Ri:Qi]))
            response__y = torch.cat(kernelpct(input_kernel__y, src_masks[Ri:Qi]))
            response__z = torch.cat(kernelpct(input_kernel__z, src_masks[Ri:Qi]))
            
            #print("--- %s kernel est seconds ---" % (time.time() - start_time))
            
            #start_time = time.time()
                
            losskey_x,losskey_y,losskey_z,plus = self.keyact([keys__x,keypoints1__x,transformed_keypoints__x,keypoints2__x,unmatched_keypoints__x],[keys__y,keypoints2__y,transformed_keypoints__y,keypoints3__y,unmatched_keypoints__y],[keys__z,keypoints3__z,transformed_keypoints__z,keypoints1__z,unmatched_keypoints__z],response__x,response__y,response__z,Masks,MaskO,H,W,Hf,Wf,I,pz,power[Be])
            
            
            #print("--- %s key detail ---" % (time.time() - start_time))
            
            #start_time = time.time()
                
            loss_base = self.keyact.forward_base(Masks,I,keypoints1__x,keypoints2__x,keypoints3__y,response__x,H,W,Hf,Wf,plus,pz,power[Be])
            
            #print("--- %s key base ---" % (time.time() - start_time))

                
            lossbatch =  lossbatch + (loss_base + losskey_x+losskey_y+losskey_z)/4
            
            #print('loss_det',losskey_x,losskey_y,losskey_z,plus)
            #print('loss_base',loss_base)
            
            Ri=Qi

            
            #import pickle
            #with open('oooo.obj', 'wb') as fp:
            #    pickle.dump([keys__x,[r.pt for r in keypoints1__x],transformed_keypoints__x,[r.pt for r in keypoints2__x],Hx,response__x,src_masks.detach().cpu().numpy(),targets,img1,img2,img3,Masks,MaskO], fp)
            
            Si+=1;
           
            
            #print(Asd)
            
            #break
        
        return lossbatch/Si