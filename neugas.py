# -*- coding: utf-8 -*-
"""

  _   _                _____             
 | \ | |              / ____|            
 |  \| |  ___  _   _ | |  __   __ _  ___ 
 | . ` | / _ \| | | || | |_ | / _` |/ __|
 | |\  ||  __/| |_| || |__| || (_| |\__ \
 |_| \_| \___| \__,_| \_____| \__,_||___/
                                                                            

A basic NeuralGas implementation.



Marco Vannucci
Istituto TeCIP, Scuola Superiore Sant'Anna, Pisa, Italy

marco.vannucci AT santannapisa.it

TG: t.me/Taracchi

"""


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


#   ==============================================
#   UTILITY FUNCTIONS
#   ==============================================

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return np.sqrt(np.dot(x, x.T))


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 20, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print() 



#   ==============================================
#   OTHER FUNCTIONS USED BY NEUGAS CLASS
#   ==============================================

def _particle_generator_mixture(X,n_points):
    '''
    Generates initial set of NG particles starting from the original
    data distribution.

    Parameters
    ----------
    X : numpy matrix, float
        The base dataset from which starting NG particles are generated.
    n_points : int
        number of points, drawn at random from X, used to generate
        each particle.

    Returns
    -------
    numpy matrix, float
        Initial set of particles.

    '''
    indices=np.random.choice(X.shape[0],n_points,replace=False)
    return np.mean(X[indices,:],axis=0)
    




def _lr_value(lr0,current_iteration,total_iterations):
    '''
    Calculates learning rate value durting NG training

    Parameters
    ----------
    lr0 : float
        Initial learning rate value.
    current_iteration : int
        Actual iteration of the training.
    total_iterations : int
        Total number of iterations of for the training procedure.

    Returns
    -------
    float
        Learning rate value for the actual iteration.

    '''
    return lr0*np.exp(-2*current_iteration/total_iterations)
    #return lr0*(total_iterations-current_iteration)/total_iterations





def _neigh_value(neigh0,current_iteration,total_iterations,decay_type='linear'):
    '''
    Calculates the weight update coefficent

    Parameters
    ----------
    neigh0 : float
        Initial value for the parameter.
    current_iteration : int
        Actual iteration of the training.
    total_iterations : int
        Total number of iterations of for the training procedure.
    decay_type : string [linear/exponential], optional
        Type of decay function adopted. The default is 'linear'.

    Returns
    -------
    float
        weight update coefficient for the current iteration.

    '''
    if decay_type=='exponential':
        return neigh0*np.exp(-4*current_iteration/total_iterations)
    if decay_type=='linear':
        return neigh0*((total_iterations-current_iteration)/total_iterations)






def quantization_error(centers,X):
    '''
    Calculates quantization error for data X
    with respect to NG particles centers

    Parameters
    ----------
    centers : MxN float, numpy matrix
        NG particles centers.
    X : numpy array, float
        Data for which quantization error is calculated.

    Returns
    -------
    float
        Quantization error.

    '''
    #   calculates quantization error
    closest_particle_dist=np.zeros(X.shape[0])
    for r in range(X.shape[0]):
        #   calculate distance of data point from each particle
        p_distances=np.array([fast_norm(p-X[r,:]) for p in centers])
        #   find closest
        closest_particle_dist[r]=np.min(p_distances)
    return np.mean(closest_particle_dist)




#   ==============================================
#   MAIN CLASS
#   ==============================================


class NeuGas:
    
    def __init__(self,X,n_particles,mixture_gen_points=1):
        '''
        Initializes the NeuralGas
        - setting up initial variables
        - initializes particles

        Parameters
        ----------
        X :  MxN float, numpy matrix
            Data to be quantized.
        n_particles : int
            Number of particles in the neural gas.
        mixture_gen_points : int, optional
            Number of sata samples to be used for the generation of each
            initial particle. The default is 5.

        Returns
        -------
        None.

        '''
        self.n_particles=n_particles
        self.domain_dimension=X.shape[1]
        #   initial particles location
        self.particles=np.zeros((n_particles,self.domain_dimension))
        for i in range(n_particles):
            self.particles[i,:]=_particle_generator_mixture(X,mixture_gen_points)
    

    
    
    def fit_old(self,X,iterations,lr0=1,neigh0=None,progress_bar_refresh=10,decay_type='linear',uprate=0.5):
        '''
        Training of the neural gas by using data X.

        Parameters
        ----------
        X :  MxN float, numpy matrix
            Data to be used for the training.
        iterations : int
            Number of iterations, in terms of samples to be used for the
            training of the neural gas. Data samples are drawn at random from X.
        lr0 : float, optional
            Initial value of learning rate. The default is 1.
        neigh0 : float, optional
            Initial value of neighbors update coefficient.
        progress_bar_refresh : int, optional
            Number of iterations between two progress bar refresh.
            The default is 10. None if you do not want any progress bar
        decay_type : string, optional
            Decay type for learning rate [linear/exponential].
            The default is 'linear'
        uprate : float
            rate of particles (in [0;1]) updated for each sample presented
            during the training. If 1 alla particles are updated. Lower uprate
            speeds-up the training. Default is 0.5

        Returns
        -------
        None.

        '''
        num_samples=X.shape[0]
        #   parameters setting
        if neigh0==None: neigh0=self.n_particles
        #   updating features throughout iterations
        for i in range(iterations):
            # if progress_bar_refresh==None no progress bar is shown
            # otherwise it is refreshed
            if not(progress_bar_refresh==None):
                if (i+1)%progress_bar_refresh==0 or i+1==iterations:
                    printProgressBar(i+1, iterations)
            #   pick a point from X
            picked_point=X[np.random.choice(num_samples),:]
            #   sort features according to distances
            #   1. calculate distances
            particles_distances=[0]*self.n_particles
            for p in range(self.n_particles):
                particles_distances[p]=fast_norm(picked_point-self.particles[p])
            #   2. sort features
            sorted_particles_indices=np.argsort(particles_distances)
            #   update features in order
            updated_particles_number=int(np.round(uprate*len(sorted_particles_indices)))
            for rank,particle_index in enumerate(sorted_particles_indices[0:updated_particles_number]):
                #print(rank,particle_index,particles_distances[particle_index])
                #   calculating DELTA
                delta_coeff=_lr_value(lr0,i,iterations)*np.exp(-(2*rank)/_neigh_value(neigh0,i,iterations,decay_type))
                #   updating
                self.particles[particle_index]+=delta_coeff*(picked_point-self.particles[particle_index])
    



    def fit(self,X,iterations,lr0=1,neigh0='auto',progress_bar_refresh=10,uprate=0.5,delta_coeff_min_update=0):
        '''
        

        Parameters
        ----------
        X :  MxN float, numpy matrix
            Data to be used for the training.
        iterations : int
            Number of iterations, in terms of samples to be used for the
            training of the neural gas. Data samples are drawn at random from X.
        lr0 : float, optional
            Initial value of learning rate. The default is 1.
        neigh0 : float, optional
            Initial value of neighbors update coefficient. (Lower than the number
            of particles NP. Default is NP/20). 
        progress_bar_refresh : int, optional
            Number of iterations between two progress bar refresh.
            The default is 10. None if you do not want any progress bar
        uprate : float
            rate of particles (in [0;1]) updated for each sample presented
            during the training. If 1 alla particles are updated. Lower uprate
            speeds-up the training. Default is 0.5
        delta_coeff_min_update : float
            minimum value of the delta parameter (calculated during the learning phase)
            to update associate particle

        Returns
        -------
        None.


        '''

        num_samples=X.shape[0]
        #   parameters setting
        if neigh0=='auto': neigh0=np.max([self.n_particles//20,2])
        #   updating features throughout iterations
        for i in range(iterations):
            # if progress_bar_refresh==None no progress bar is shown
            # otherwise it is refreshed
            if not(progress_bar_refresh==None):
                if (i+1)%progress_bar_refresh==0 or i+1==iterations:
                    printProgressBar(i+1, iterations)
            #   pick a point from X
            picked_point=X[np.random.choice(num_samples),:]
            #   sort features according to distances
            #   1. calculate distances
            particles_distances=[0]*self.n_particles
            for p in range(self.n_particles):
                particles_distances[p]=fast_norm(picked_point-self.particles[p])
            #   2. sort features
            sorted_particles_indices=np.argsort(particles_distances)
            #   update features in order
            updated_particles_number=int(np.round(uprate*len(sorted_particles_indices)))
            for rank,particle_index in enumerate(sorted_particles_indices[0:updated_particles_number]):
                #print(rank,particle_index,particles_distances[particle_index])
                #   calculating DELTA
                delta_coeff=lr0*((iterations-i)/iterations)*\
                    np.exp(-(2*rank)/(neigh0*((iterations-i)/iterations)**2))
                #   updating
                if delta_coeff>delta_coeff_min_update:
                    self.particles[particle_index]+=delta_coeff*(picked_point-self.particles[particle_index])


    
    def quantize(self,X,progress=False,lowmemory=False):
        '''
        Quantizes an arbitrary data vector X

        Parameters
        ----------
        X : N long numpy array, float
            Data to be quantized by the neural gas.
        progress : boolean
            If True shows a progress bar during training
        lowmemory : if True use a memory efficient (but slower) algorithm

        Returns
        -------
        closest_particle : N long numpy array, int
            list of closest particle to each data point in X.
        closest_particle_dist : N long numpy array, float
            list of distances between closest particle and each data point in X.

        '''
        if lowmemory:
        
            #   initialize closest particle identifier and data point
            #   distance from it
            closest_particle=np.zeros(X.shape[0])
            closest_particle_dist=np.zeros(X.shape[0])
            for r in range(X.shape[0]):
                if progress:
                    printProgressBar(r+1,X.shape[0])
                #   calculate distance of data point from each particle
                p_distances=np.array([fast_norm(p-X[r,:]) for p in self.particles])
                #   find closest
                closest_particle[r]=np.argmin(p_distances).astype(int)
                closest_particle_dist[r]=p_distances[int(closest_particle[r])]                    
        else:
            dist_x_m=euclidean_distances(X,self.particles)
            
            closest_particle=np.argmin(dist_x_m,axis=1).astype(int)
            closest_particle_dist=np.min(dist_x_m,axis=1)
        
        return closest_particle,closest_particle_dist
    
    



            
           
           
             

        

    
    
