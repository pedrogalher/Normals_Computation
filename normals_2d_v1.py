# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:58:32 2019

@author: zz19101
"""

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
import math
from math import pi

from normal_tree_class import*


def normals_2d_v1(void_data, k):
    
    """This function calculate the normal of a point taking into account the points 
    surrounding it by calculating containing them and performing a 
    PCA (Principal Component Analysis)
    
    INPUT: 
        ellipsoid_data: M x 2 array containing the x,y coordinates 
        (M is the number of points)
        
        k: number of neighbor points used to calculate the tangent plane 
        associated with each point (KNN)
        
    OUTPUT:
        normals: M x 4 array containing the coordinates of the normal of the 
        plane associated to each point
    
    """
    
    # Void's centroid
    v_xmn = np.mean(void_data[:,0])
    v_ymn = np.mean(void_data[:,1])
    
    #We center the data at the (0,0)
    void_data[:,0] = void_data[:,0] - v_xmn
    void_data[:,1] = void_data[:,1] - v_ymn
    
    #Dictionary storing the KNN points to each point
    knn_array = {}      # Array of cooordinates
    knn_list = {}       # List of coordinates
    knn_index = {}      # List of index, where each index corresponds to a row (point) in the void data input matrix
    
    
    # Calculate the K-nearest points to each point (KNN) #
    dist = distance.squareform(distance.pdist(void_data))       # Alternative (and more direct) form to calculate the distances for each point
    closest = np.argsort(dist, axis = 1)           #Axis = 1 because we are sorting the columns
    
    # Extraction of normals and centroids for each point #
    total_pts=np.size(closest,0)
    normals=np.zeros((total_pts,2)) # Coordinates of the normals

    for i in range(total_pts):
    
        normal_vect, knn_pt_id, knn_pt_coord, knn_pt_list, knn_pt_id = tang_2d_v1(closest[i,:],void_data,k)     # Obtention of the normal for the point at the ith row
        normals[i, :] = normal_vect      # Storing of the normal to the tangent plane at that point
        knn_array[(knn_pt_coord[0,0], knn_pt_coord[0,1])] = knn_pt_coord   # Filling the dictionary of the KNN (the KNN are stored in an array format)
        knn_list[(knn_pt_coord[0,0], knn_pt_coord[0,1])] = knn_pt_list          #  Filling the dictionary of the KNN (the KNN are stored in a list format)
        knn_index[knn_pt_id[0]] = list(knn_pt_id)       # Fillin the knn index to each point (an index itself)
     
    ## Normal consistency ##
    norm_consist = normal_tree(knn_index, normals, void_data)
    
    # Graphical representation --> Check for the correct normal consistency (pointing outwards) #
    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(1,1,1)
    #ax2.scatter(v_xmn,v_ymn, facecolors='none',edgecolors = 'c')
    
    #for i in range(total_pts):
        #normal_vect = normals[i,:]
        #xmn = void_data[i,0]
        #ymn = void_data[i,1]
        
        #ax2.scatter(xmn,ymn, facecolors='none',edgecolors = 'c')
        #ax2.plot([xmn,xmn+normal_vect[0]],[ymn,ymn+normal_vect[1]])
        #plt.show()
        #plt.hold(True)
        #ax2.axis('equal')
        
    return  normals, knn_index, norm_consist
    
    
def tang_2d_v1(closest_pt,void_data,k):
    
    """This function gets the centre point c of a cloud of points knni 
    and returns the normal of the tangent plane best fitting the cloud of points.
    The normal is located a the point of interest (centre).
    
        Input: 
            knni: Array of indexes representing the nearest point (index) 
            to the first point of the row knni[i]
            
            void_data (2D_array): y,z coordinates of points defining the surface 
            of the void at a given slice. 
            We will use it to give coordinates to each index in knni, i.e., get
            the coordinates of the nearest neighboor of each point from the knni.
            
            k: Number of closest points we want to calculate for a given point 
            (including the one of interest).
            
    """
    ## Extraction of the coordinates of each point from their indexes and the matrix ellipsoid ##
    knn_pt_id = closest_pt[0:k+1] # Retain only the indexes of the k-closest points
    nb_points = np.size(knn_pt_id)
    knn_pt_coord = np.zeros((nb_points,2))
    knn_pt_list = []
    void_data = np.around(void_data,2)
    
    for i in range(nb_points):
        
        point_i = knn_pt_id[i]
        knn_pt_coord[i,:] = void_data[point_i,:]
        knn_pt_list += [(void_data[point_i, 0], void_data[point_i,1])]
    
    ## Principal component analysis (PCA) ## 
    # Centorid calculation #
    xmn = np.mean(knn_pt_coord[:,0])
    ymn = np.mean(knn_pt_coord[:,1])
    
    c=np.zeros((np.size(knn_pt_coord,0),2))
    
    c[:,0] = knn_pt_coord[:,0] - xmn
    c[:,1] = knn_pt_coord[:,1] - ymn
    
    # Covariance matrix #
    cov=np.zeros((2,2))    
    
    cov[0,0] = np.dot(c[:,0],c[:,0])
    cov[0,1] = cov[1,0] =  np.dot(c[:,0],c[:,1])
    cov[1,1] = np.dot(c[:,1],c[:,1])
   
    ## Single value decomposition (SVD) ##
    u,s,vh = np.linalg.svd(cov) # U contains the orthonormal eigenvectors and S contains the eigenvectors

    # Selection of minimum eigenvalue # 
    smaller_eigenvalue = u[:,1] # The smallest eigenvalue corresponds to the plane normal
    
    # Selection of orthogonal vector corresponing to this eigenvalue --> vector normal to the plane defined by the kpoints"
    normal_vect = smaller_eigenvalue

    return normal_vect, knn_pt_id, knn_pt_coord, knn_pt_list, knn_pt_id


def normalconsist_2d_v1(normals, void_centre):
     
    planes_consist = normals
    angles = 1
    
    return planes_consist, normals, angles
    
    
    