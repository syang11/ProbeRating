"""
Created on Sat November 30 20:27:33 2019

@author: Shu Yang

Usage:
> python probeRating_recommender_nn_ry 1 1 10 500 0 10

"""


import scipy.io as sio
import scipy.sparse as sparse
import numpy as np
from keras import models
from keras import layers
from keras import regularizers
from keras import backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import sys
import os
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras import Input
from scipy.linalg import orth


sysuffix=os.getppid()   #get parent .sh process's PID
namePrefix='probeRating_recommender_nn_ry_loss'
syspec=namePrefix+sys.argv[1]+'_optimizer'+sys.argv[2]+'_lr'+sys.argv[3]+'_epochs'+sys.argv[4]+'_batchesPerEpoch'+sys.argv[5]+'_folds'+sys.argv[6]
lossIdx=int(sys.argv[1])
optimizerIdx=int(sys.argv[2])
lrate=float(sys.argv[3])
epochsNum=int(sys.argv[4])
batchesPerEpoch=int(sys.argv[5])
numFolds=int(sys.argv[6])


if lossIdx==1:
    myLoss='mean_squared_error'
elif lossIdx==2:
    myLoss='mean_absolute_percentage_error'
elif lossIdx==3:
    myLoss='mean_squared_logarithmic_error'
elif lossIdx==4:
    myLoss='logcosh'
    


####################################
## Load data
####################################
RRM1612_mat=sio.loadmat('../data/example.mat')#

label=RRM1612_mat['Y'] #241357*164  
labelv=RRM1612_mat['Yv']    
RNAf=RRM1612_mat['D'] #241357*150   
protf=RRM1612_mat['P']  #164*50 





############################################
## Initialize some global var
############################################
# set random number seed for re-producibility of the random numbers
np.random.seed(0)   # np.random.seed(0)   

#when doing MSE, since the intenstiy labels' values are too small, which cause the loss to be small, so do scaling first; but when doing MSLE, no need to do scaling (set syScaleFactor to 1) since do log transform on the labels
syScaleFactor=1 #syScaleFactor=1000  

probeNum=len(RNAf)  # probeNum=241357
proteinNum=len(protf)   #proteinNum=164




#########################
## make data (no generator)
#########################
ridx=np.random.permutation(proteinNum)
protfp=protf[ridx]  #permute row of protein feature matrix
labelp=label[:][:,ridx] #permute col (representing proteins) of rating matrix

k=numFolds    #k-fold CV
fold_ix=np.rint(np.linspace(0,proteinNum,k+1))
fold_ix=fold_ix.astype(int)





############################
## define neural net
############################
# customized PCC loss function
def correlation_coefficient_loss(y_true, y_pred):
    '''Use K.epsilon() == 10^-7 to avoid divide by zero error    
    '''
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.maximum(K.sum(K.square(xm)), K.epsilon()), K.maximum(K.sum(K.square(ym)), K.epsilon())))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(1 - r)



class Kron(layers.merge._Merge):
    """Merge Layer that do outer product on a list of 2 inputs

    It takes as input a list of 2 tensors, do outer-product on them, and returns a single tensor
    """
    def __init__(self, axis=-1, **kwargs):
        """
        **kwargs: standard layer keyword arguments.
        """
        super(Kron, self).__init__(**kwargs)
        self.axis=axis
        self._reshape_required = False  # to be compatible with super class layers.merge._Merge's call() method

    def build(self, input_shape):
        """
        Just to overwrite super class layers.merge._Merge's build(), so that do not need to call _Merge's _compute_elemwise_op_output_shape() method which requires the inputs to be the same dimension in the last axis
        Can also make this method a shape validation method, just like what keras's Concatenate class did in merge.py (did not call _compute_elemwise_op_output_shape() as well)
        """
        pass

    def _merge_function(self, inputs):
        """
        Do outer product on the last axis for the 2 input tensors. Note inputs tensors should have equal dimension for axis=0 case (ie. batchsize should be equal). 

        Alternatively, if inputs tensors have equal dimensions, can see the implementation in outer_product() function below. Have tested these two functions step by step, confirm they give the same results.

        """
        #pass
        output=K.repeat_elements(inputs[0], K.int_shape(inputs[1])[1], -1)
        inputs1_tiled=K.tile(inputs[1], [1, K.int_shape(inputs[0])[1]])
        return output*inputs1_tiled  #element-wise multiply
        
    @staticmethod    
    def outer_product(inputs):
        """
        use the implementation in _merge_function() for outer product, since it is more general. This outer_product() function only serves as a double check, and can only deal with inputs of 2 tensors with equal dimensions
        """
        inputs0, inputs1 = inputs
        batchSize = K.shape(inputs0)[0]
        outerProduct = inputs0[:, :, np.newaxis]*inputs1[:, np.newaxis, :]
        outerProduct = K.reshape(outerProduct, (batchSize, -1))
        # returns a flattened batch-wise set of tensors
        return outerProduct    

    def compute_output_shape(self, input_shape):
        #pass
        if not isinstance(input_shape, list) or len(input_shape)!=2:
            raise ValueError('A `Kronecker` layer should be called on a list of 2 inputs.')
        output_shape=list(input_shape[0])
        shape=list(input_shape[1]) #this should also be fine: shape=input_shape[1]
        if output_shape[self.axis] is None or shape[self.axis] is None:
            output_shape[self.axis]=None
        output_shape[self.axis] *= shape[self.axis]
        return tuple(output_shape)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "axis": self.axis}


# need below function to call Kron class. Also need __init__.py in the custom layers class directory to do "from .Kron import kron"
def kronecker(inputs, **kwargs):
    """Functional interface to the `Kron` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the kronecker/outer product of the inputs.
    """
    return Kron(**kwargs)(inputs)
#






######################################################################
## Do k-fold CV. Need to comment out, when loading pre-trained model to do prediction only and no training
#####################################################################
CCs=np.zeros((proteinNum,4))    #correlation coefficients, 4columns: pcc, pvalue of pcc, scc, pvalue of scc
CCs=np.concatenate([ridx.reshape((-1,1), order='F'), CCs], axis=1)
CCs1=CCs.copy()
for fold in range(k):
    test_ix=np.arange(fold_ix[fold],fold_ix[fold+1])
    train_ix=np.setdiff1d(np.arange(proteinNum), test_ix)

    #selecting training and testing proteins 
    protTrain=protfp[train_ix]
    protTest=protfp[test_ix]
    intensityTrain=labelp[:,train_ix]
    intensityTest=labelp[:,test_ix]

    #make nonparam transformation. Change the task from label/intensity prediction to similarity prediction
    YTY=np.dot(intensityTrain.T, intensityTrain)    #147*147
    YTD=np.dot(intensityTrain.T, RNAf)  #147*150, (YTD)W(P)=YTY

    # make input to nw: protein training and testing matrix
    rnaNum=YTD.shape[0] 
    protTrainIN=protTrain.repeat(rnaNum, axis=0)
    protTestIN=protTest.repeat(rnaNum, axis=0)

    #make input to nw: label training and testing vector
    # reshape similarity as vector
    similarityTrainIN=YTY.reshape((-1,1), order='F') 

    # make input to nw: rna training and testing matrix
    protTrainNum=protTrain.shape[0]
    protTestNum=protTest.shape[0]
    rnaTrainIN=np.tile(YTD,(protTrainNum,1))    
    rnaTestIN=np.tile(YTD,(protTestNum,1)) 

    # optional: permute training data (if permute, means shuffle protein-RNA pairs; if not permute, means protein-RNA pairs are ordered by key1=proteins, key2=RNAs, this would make the validation set in models.fit() below to be always the bottom proteins (ie. these proteins never be seen in the training))
    ridx1=np.random.permutation(similarityTrainIN.shape[0])
    similarityTrainIN=similarityTrainIN[ridx1]
    protTrainIN=protTrainIN[ridx1]
    rnaTrainIN=rnaTrainIN[ridx1]



    #############
    ## set up model
    #############
    activationFunc='tanh'#'relu'
    tensorBoardDir="../deepNN/dl2/%s-%d/tensorBoardLog/fold%d"%(syspec, sysuffix,fold)
    if not os.path.exists(tensorBoardDir):
        os.makedirs(tensorBoardDir)
    checkPtFile='../deepNN/dl2/%s-%d/p2v50d-fold%d.hdf5'%(syspec, sysuffix, fold)
    lossFig='../deepNN/dl2/%s-%d/loss-fold%d.pdf'%(syspec, sysuffix, fold)
    modelFig='../deepNN/dl2/%s-%d/p2v50d-fold%d.pdf'%(syspec, sysuffix, fold)

    protTensor=Input(shape=(protTrainIN.shape[1],), name='FastProt')
    x1=layers.Dropout(0.1)(protTensor)  #x1=layers.Dropout(0.5)(protTensor)
    x1=layers.BatchNormalization()(x1)
    x1=layers.Dense(units=30, activation=activationFunc, kernel_regularizer=regularizers.l2(0.01))(x1) #units=30 or 20, l2(0.001)
    # x1=layers.Dropout(0.2)(x1)
    x1=layers.BatchNormalization()(x1)

    rnaTensor=Input(shape=(rnaTrainIN.shape[1],), name='FastRNA')
    x2=layers.Dropout(0.1)(rnaTensor)   # x2=layers.Dropout(0.5)(rnaTensor)
    x2=layers.BatchNormalization()(x2)
    x2=layers.Dense(units=30, activation=activationFunc, kernel_regularizer=regularizers.l2(0.01))(x2) #units=30 or 20, l2(0.001)
    # x2=layers.Dropout(0.2)(x2)
    x2=layers.BatchNormalization()(x2)

    merged=kronecker([x1, x2]) #layers.dot([x1, x2], -1)    #layers.multiply([x1, x2]) #concatenate
    x12=layers.Dropout(0.2)(merged) #x12=layers.Dropout(0.1)(merged)
    x12=layers.BatchNormalization()(x12)
    # x12=layers.Dense(units=30, activation=activationFunc, kernel_regularizer=regularizers.l2(0.01))(x12)#l2(0.001)

    # x12=layers.Dropout(0.5)(x12)
    # x12=layers.BatchNormalization()(x12)
    similarity=layers.Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(x12)#units=1, l2(0.01)
    network1=models.Model([protTensor, rnaTensor], similarity)


    if optimizerIdx==1:
        myOptimizer=optimizers.RMSprop(lr=lrate, rho=0.9, epsilon=None, decay=0.0) #default for RMSprop: lr=0.001
    elif optimizerIdx==2:
        myOptimizer = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    network1.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss]) # mean_absolute_percentage_error
    callbacksList=[ModelCheckpoint(filepath=checkPtFile, verbose=1, monitor="val_loss", save_best_only=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001), EarlyStopping(monitor="val_loss", patience=50), TensorBoard(tensorBoardDir, histogram_freq=0, embeddings_freq=0)] #If printing histograms, validation_data must be provided, and cannot be a generator.

    #############
    ## fit model
    #############
    history=network1.fit([protTrainIN, rnaTrainIN], similarityTrainIN, batch_size=32, epochs=epochsNum, verbose=2, callbacks=callbacksList, validation_split=0.1, shuffle=True)#

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(1,len(loss)+1)
    plt.figure()
    plt.plot(epochs,loss,'bo',label="Training loss")
    plt.plot(epochs,val_loss,'b',label="Validation loss")
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.savefig(lossFig)

    plot_model(network1, show_shapes=True, to_file=modelFig)


    #############
    ## evaluate model (note that Keras has a model.evaluate() method which only output a limited set of evaluation metrics (losses selected in your model loss) as defined in model.metrics_name, not what I want like PCC, SCC)
    ## ref: https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras
    #############

    predictedSimilarity=network1.predict([protTestIN, rnaTestIN])
    predictedSimilarity=predictedSimilarity.reshape((rnaNum,-1),order='F')  #147*16 
    # option1: Weighted sum reconstruction
    intensityPred=np.dot(intensityTrain, predictedSimilarity)
    # option2:  Moore-Penrose pseudo inverse reconstruction:
    intensityPred1=np.dot(np.linalg.pinv(intensityTrain.T), predictedSimilarity)
   
    for i in range(test_ix.shape[0]):
        CCs[test_ix[i], 1], CCs[test_ix[i], 2]=pearsonr(intensityPred[:,i], intensityTest[:,i])
        CCs[test_ix[i], 3], CCs[test_ix[i], 4]=spearmanr(intensityPred[:,i], intensityTest[:,i])

        CCs1[test_ix[i], 1], CCs1[test_ix[i], 2]=pearsonr(intensityPred1[:,i], intensityTest[:,i])
        CCs1[test_ix[i], 3], CCs1[test_ix[i], 4]=spearmanr(intensityPred1[:,i], intensityTest[:,i])


print("\n\n### the number of folds in cross validation =",k) 
print("The average PCC, pvalue of PCC, SCC, pvalue of SCC:")
print(np.mean(CCs[:,1:], axis=0))
np.savetxt("../deepNN/dl2/%s-%d/p2v50d-performance.csv"%(syspec, sysuffix), CCs, delimiter=',')
print("The average PCC1, pvalue of PCC1, SCC1, pvalue of SCC1:")
print(np.mean(CCs1[:,1:], axis=0))
np.savetxt("../deepNN/dl2/%s-%d/p2v50d-performance1.csv"%(syspec, sysuffix), CCs1, delimiter=',')
