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


def normaldefinition_3D_real(void_data, k):
    
    """
    
    This function calculate the normal of a point taking into account the points surrounding it by 
    calculating containing them and performing a PCA (Principal Component Analysis)
    
    INPUT: 
        ellipsoid_data: M x 3 array containing the x,y and z coordinates (M is the number of points)
        k: number of neighbor points used to calculate the tnagent plane associated with each point (KNN)
        
    OUTPUT:
        normals: M x 6 array containing the coordinates of the normal of the plane associated to each point
    
    """
    
    
    " Calculate the K-nearest points to each point (KNN) "

    dist = distance.squareform(distance.pdist(void_data)) # Alternative (and more direct) form to calculate the distances
    closest = np.argsort(dist, axis = 1) #Axis = 1 because we are sorting the columns
    
    "Extraction of normals and centroids for each point"
    
    total_pts=np.size(closest,0)
    planes=np.zeros((total_pts,6)) # The three first columns contian the coordinates of the normals. The 3 last columns contain the coordinates of the centroid of the plane

    #fig1=plt.figure()
    #ax = fig1.add_subplot(1,1,1, projection='3d')
    #ax.scatter(x,y,z, facecolors='none',edgecolors='r')

    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    
    for i in range(total_pts):
    
        normal_vect, xmn,ymn,zmn, knn_pt_coord = tangentplane_3D_real(closest[i,:],void_data,k) #Obtention of the normal and centroid (and other parametres) for each point in the ellipsoid
    
        planes[i,0:3] = normal_vect #Keep the coordinates of the normal vectors
        planes[i,3:6] = np.array([xmn, ymn, zmn]) #Keep the coordinates of the centroid
        
        #ax.scatter(xmn,ymn,zmn, facecolors='none',edgecolors = 'c')
        #ax.plot([xmn,xmn+normalvect[0]],[ymn,ymn+normalvect[1]],[zmn,zmn+normalvect[2]])
        #plt.show()
        #plt.hold(True)
        
    planes_consist = normalconsistency_3D_real(planes)
    
    return  planes, planes_consist
    

    
def tangentplane_3D_real(closest_pt,ellipsoid_data,k):
    
    """
    
    This function calculates the centre point c of a cloud of points knni and returns
    the normal of the tangent plane best fitting the coud of points
    
        Input: 
            knni: Array of indexes representing the nearest point (index) to the first point of the row knni[i]
            ellipsoid: x,y,z coordinates of points defining the ellipsoid. We will use it to give coordinates to each index in knni
            k: Number of closest points we want to calculate for a given point
            
    """
    
    "Extraction of the coordinates of each point from their indexes and the matrix ellipsoid"
    
    knn_pt_id = closest_pt[0:k] # Retain only the indexes of the k-closest points
    nb_points = np.size(knn_pt_id)
    knn_pt_coord = np.zeros((nb_points,3)) 
    
    for i in range(nb_points):
        
        point_i = knn_pt_id[i]
        
        knn_pt_coord[i,:] = ellipsoid_data[point_i,:]
    
    "Principal component analysis (PCA)"
    
    "Centorid calculation"
        
    xmn = np.mean(knn_pt_coord[:,0])
    ymn = np.mean(knn_pt_coord[:,1])
    zmn = np.mean(knn_pt_coord[:,2])
    
    c=np.zeros((np.size(knn_pt_coord,0),3))
    
    c[:,0] = knn_pt_coord[:,0]-xmn
    c[:,1] = knn_pt_coord[:,1]-ymn
    c[:,2] = knn_pt_coord[:,2]-zmn
    
    "Covariance matrix"
    
    cov=np.zeros((3,3))    
    
    cov[0,0] = np.dot(c[:,0],c[:,0])
    cov[0,1] = np.dot(c[:,0],c[:,1])
    cov[0,2] = np.dot(c[:,0],c[:,2])
    
    cov[1,0] = cov[0,1]
    cov[1,1] = np.dot(c[:,1],c[:,1])
    cov[1,2] = np.dot(c[:,1],c[:,2])
    
    cov[2,0] = cov[0,2]
    cov[2,1] = cov[1,2]
    cov[2,2] = np.dot(c[:,2],c[:,2])
   
    "Single value decomposition (SVD)"
    
    u,s,vh = np.linalg.svd(cov) # U contains the orthonormal eigenvectors and S contains the eigenvectors

    "Selection of minimum eigenvalue"
    
    minevindex = np.argmin(s)
    
    "Selection of orthogonal vector corresponing to this eigenvalue --> vector normal to the plane defined by the kpoints"
    
    normal_vect = u[:,minevindex]

    return normal_vect, xmn, ymn,zmn,knn_pt_coord



def normalconsistency_3D_real(planes):
    
    """
    
    This function checks wherer the normals are oriented towards the outside of the surface, i.e., it 
    checks the consistency of the normals.
    The function changes the direction of the normals that do not point towards the outside of the shape
    The function checks whether the normals are oriented towards the centre of the ellipsoid, 
    and if YES, then, it turns their orientation
    
    INPUTS:
        planes: Vector N x 6, where M is the number of points whose normals and 
        centroid have been calculated. the columns are the coordinates of the normals and the centroids
        
    OUTPUTS:
        planesconsist: N x 6 array, where N is the number of points whose planes have been calculated. This array 
        has all the planes normals pointing outside the surface.
        
    """
    
    nbnormals = np.size(planes, 0)
    planes_consist=np.zeros((nbnormals,6))
    planes_consist[:, 3:6] = planes[:, 3:6] # We just copy the columns corresponding to the coordinates of the centroids (from 3th to 5th)
    
    """ Try the atan2 function : https://uk.mathworks.com/help/vision/ref/pcnormals.html#buxdmoj"""
    
    sensorcentre=np.array([0,0,0])
    
    for i in range(nbnormals):
    
        p1 = (sensorcentre - planes[i,3:6]) / np.linalg.norm(sensorcentre - planes[i,3:6]) # Vector from the centroid to the centre of the ellipsoid (here the sensor is placed)
        p2 = planes[i,0:3]
        
        angle = math.atan2(np.linalg.norm(np.cross(p1,p2)), np.dot(p1,p2) ) # Angle between the centroid-sensor and plane normal
       
        
        if (angle >= -pi/2 and angle <= pi/2): # (angle >= -pi/2 and angle <= pi/2):
            
            planes_consist[i,0] = -planes[i,0]
            planes_consist[i,1] = -planes[i,1]
            planes_consist[i,2] = -planes[i,2]  
            
        else:
            
            planes_consist[i,0] = planes[i,0]
            planes_consist[i,1] = planes[i,1]
            planes_consist[i,2] = planes[i,2]
         
   

    return planes_consist
    
    
    