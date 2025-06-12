# =============================================================================
# Imports
# =============================================================================

# General
import warnings
import re

# Specific
import numpy
from sklearn import preprocessing
import gpflow
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import matplotlib.ticker as ticker

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font',**{'family':'serif','serif':['Computer Moder Roman']})
# rc('font',**{'family':'sans-serif'})
rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# =============================================================================
# Configuration
# =============================================================================

# Define normalization methods
featureNorm='MinMax' # None,Standardization,MinMax
labelNorm='LogStand' # None,Standardization,LogStand
# GP Configuration
# gpConfig={'kernel':'Matern12',
#           'useWhiteKernel':True,
#           'trainLikelihood':True,}
# # Example 1: Change the kernel to RBF and use MinMax normalization for features
gpConfig = {'kernel': 'Matern', 'useWhiteKernel': True, 'trainLikelihood': True}
featureNorm = 'MinMax'
labelNorm = 'LogStand'

# Example 2: Use Matern52 kernel and Standardization for both features and labels
# gpConfig = {'kernel': 'RQ', 'useWhiteKernel': True, 'trainLikelihood': True}
featureNorm = 'Standardization'
labelNorm = 'Standardization'
# =============================================================================
# Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log - simply convert x to log(x)
        Defalt: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand' or method=='Log': aux=numpy.log(inputArray)
        else: raise ValueError('Could not recognize method in normalize().')
        if method=='MinMax':
            skScaler=preprocessing.MinMaxScaler().fit(aux)
        elif method=='Log':
            skScaler='NA'
        else:
            skScaler=preprocessing.StandardScaler().fit(aux) 
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        if method!='Log':
            inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand' or method=='Log':
            inputArray=numpy.exp(inputArray)
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand' or method=='Log': aux=numpy.log(inputArray)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='Log':
            inputArray=skScaler.transform(aux)
        else:
            inputArray=aux
    # Return
    return inputArray,skScaler

def buildGP(X_Train,Y_Train,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RBF')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    # Select and initialize kernel
    if kernel=='Matern':
        gpKernel= gpflow.kernels.Matern12() * gpflow.kernels.Polynomial() * gpflow.kernels.Linear()
    if kernel=='RBF': 
        gpKernel=gpflow.kernels.SquaredExponential()
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    if kernel=='Matern12':
        gpKernel=gpflow.kernels.Matern12()
    if kernel=='Exponential':
        gpKernel=gpflow.kernels.Exponential()    
    if kernel=='Polynomial':
        gpKernel=gpflow.kernels.Polynomial()
    if kernel=='Linear':
        gpKernel=gpflow.kernels.Linear()
    if kernel=='White':
        gpKernel=gpflow.kernels.White()
    if kernel=='Constant':
        gpKernel=gpflow.kernels.Constant()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model    
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,noise_variance=10**-5)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           method='SLSQP')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

pp = PdfPages('GPR-gamma_new.pdf')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.0,hspace=0.0)

# =============================================================================
# Main Script
# =============================================================================

# System A
x2_A=[0.9501214919791561,0.805163549869722,0.6585107465359727,0.5238749773685331,0.38769241432842066,
      0.24253661882207567,0.15179807003854653,0.06335061713581094]
G11=[-0.8692598435944665,-1.020440030016593,-1.0740071032653697,-1.1105331644717567,-1.1245482227149048, 
     -1.1435105650331268,-1.1511987296298372,-1.1662456717188494]
G12=[-1.142525582586772,-1.1629479411302674,-1.1721041208769967,-1.1814487522008608,-1.1964906819044379,
     -1.211513176455841,-1.231699516448146,-1.2052458188735449]
G22=[-1.1904559697292174,-1.201122613370382,-1.2164563627270086,-1.228558833366641,-1.2346501496028224,
     -1.2449207490197463,-1.1691150583734105,-1.6350056793220848]
gamma_1=[1.179246667,1.172822783,1.100645814,1.070999324,1.058638945,
         1.037944203,1.027241743,1.013659323]
gamma_2=[0.999419768,1.005453547,0.961756454,0.944352872,0.958408596,
         0.930023817,0.940572928,0.953726921]

# Combine G11, G12, G22 into a feature matrix for System A
X_Train_1 = numpy.concatenate((numpy.array(G11).copy().reshape(-1, 1),
                               numpy.array(G12).copy().reshape(-1, 1),
                               numpy.array(G22).copy().reshape(-1, 1)), axis=1)

# Combine gamma_1 and gamma_2 into a single output matrix for System A
Y_Train_1 = numpy.concatenate((numpy.array(gamma_1).copy().reshape(-1, 1),
                               numpy.array(gamma_2).copy().reshape(-1, 1)), axis=1)

# System B
x2_B = [0.89635227595178,0.7517061447574521,0.6089061293546846,0.527793499146643,0.41703479668860594,
        0.30730520952867346,0.21363922388059375,0.12025300511343805]
G11 = [-3.8201304231959465,-2.8013553651363625,-2.1699767952227327,-1.91699562648137,-1.647279544793204,
       -1.4619132021778365,-1.3370613105760218,-1.2430208367527298]
G12 = [-0.6841932967670188,-0.5292217172200324,-0.46838566288149164,-0.46329216389697664,-0.4862391547522833, 
       -0.5162380984228654,-0.5818682513712815,-0.6998181696825507]
G22 = [-1.2540645142287286,-1.412708055078155,-1.652089105167613,-1.8254372766054243,-2.129506638958376, 
       -2.5960062236669614,-3.1702884425072173,-4.018195693841342]
gamma_1=[0.206225165,0.300658983,0.442790379,0.529586499,0.672808521,
         0.805687443,0.894252792,0.981811383]
gamma_2=[0.940480895,0.830719875,0.650651205,0.546576311,0.399468402,
         0.284732254,0.205525253,0.142044336]

# Combine G11, G12, G22 into a feature matrix for System B
X_Train_2 = numpy.concatenate((numpy.array(G11).copy().reshape(-1, 1),
                               numpy.array(G12).copy().reshape(-1, 1),
                               numpy.array(G22).copy().reshape(-1, 1)), axis=1)

# Combine gamma_1 and gamma_2 into a single output matrix for System B
Y_Train_2 = numpy.concatenate((numpy.array(gamma_1).copy().reshape(-1, 1),
                               numpy.array(gamma_2).copy().reshape(-1, 1)), axis=1)

# System C
x2_C=[0.9580421194338435,0.9160188085595612,0.8772196570418157,0.6790532228075077,0.3649397745506243, 
      0.2596870413687042,0.17132063929869343,0.09137372937672518]
G11=[3.94140373537732,3.6473861527022393,4.053480920773804,3.5092811508166513,0.5729248549073414,
     -0.29315345694687994,-0.8207243903524374,-1.06712759357732]
G12=[-1.5222867872177992,-1.743804991105429,-2.025833657338102,-3.3895333798829212,-3.936346567883935, 
     -3.364655831155444,-2.5676097721800546,-1.9577311257965424]
G22=[ -1.1664704351550994,-1.1262205856881364,-1.0565201682383305,-0.16014856209439204,3.1069467783694904, 
     4.125100755960242,4.0430433564531505,3.9567177905529807]
gamma_1=[4.539850526, 4.139825955, 3.669253522, 2.322434982, 1.423919577, 
         1.243469147, 1.198796011, 1.102960503]
gamma_2=[1.022948324, 1.033631259, 1.066460919, 1.180189507, 1.832369017, 
         2.3023216, 2.887469603, 3.399034206]

# Combine G11, G12, G22 into a feature matrix for System C
X_Train_3 = numpy.concatenate((numpy.array(G11).copy().reshape(-1, 1),
                               numpy.array(G12).copy().reshape(-1, 1),
                               numpy.array(G22).copy().reshape(-1, 1)), axis=1)

# Combine gamma_1 and gamma_2 into a single output matrix for System C
Y_Train_3 = numpy.concatenate((numpy.array(gamma_1).copy().reshape(-1, 1),
                               numpy.array(gamma_2).copy().reshape(-1, 1)), axis=1)

X_Train=numpy.concatenate((X_Train_1,X_Train_2,X_Train_3),axis=0)
Y_Train=numpy.concatenate((Y_Train_1,Y_Train_2,Y_Train_3),axis=0)

# #system 0.9
# x2_1_25 = [0.89377712, 0.78992731, 0.6892642, 0.59391598, 0.49591469,
#           0.40101505, 0.30266877, 0.20333042, 0.103321087]
# G11 = [-4.219792872890755, -3.2363220354432345, -2.635448971817377, -2.220896815155275,
#             -1.905672219271214, -1.664287594417277, -1.4863375738599118,
#             -1.3482135375873023, -1.236731686]
# G12 = [-0.6057639476785153, -0.48939596105762206, -0.42559501041586756, -0.39505891408497723, -0.3828853376070287, -0.40353106841480246,
#             -0.4490793997339826, -0.5108800706782308,-0.656071677]
# G22 = [-1.264583763,-1.381993919, -1.534299068, -1.724507245, -1.977775905,
#             -2.319765521, -2.753424636, -3.486994395234545, -4.630438418]
# gamma_1 = [0.129673295,0.168608172,0.218194912,0.341223734,0.425912262,0.559130326,
#                 0.680578426,0.806740239,0.868036218]
# gamma_2 = [0.948012927, 0.855326409, 0.721552004, 0.569414794, 0.425661575, 
#           0.301721552, 0.213787078, 0.141404269, 0.110595224]

# # Combine G11, G12, G22 into a feature matrix for System C
# X_Train_4 = numpy.concatenate((numpy.array(G11).copy().reshape(-1, 1),
#                                numpy.array(G12).copy().reshape(-1, 1),
#                                numpy.array(G22).copy().reshape(-1, 1)), axis=1)

# # Combine gamma_1 and gamma_2 into a single output matrix for System C
# Y_Train_4 = numpy.concatenate((numpy.array(gamma_1).copy().reshape(-1, 1),
#                                numpy.array(gamma_2).copy().reshape(-1, 1)), axis=1)

# X_Train=numpy.concatenate((X_Train_1,X_Train_2,X_Train_3,X_Train_4),axis=0)
# Y_Train=numpy.concatenate((Y_Train_1,Y_Train_2,Y_Train_3,Y_Train_4),axis=0)


# Normalize
X_Train_N=X_Train.copy()
Y_Train_N=Y_Train.copy()
skScaler_X=None
skScaler_Y_1=None
skScaler_Y_2=None
if featureNorm is not None:
    X_Train_N,skScaler_X=normalize(X_Train,method=featureNorm)
if labelNorm is not None:
    Y_Train_N,skScaler_Y=normalize(Y_Train,method=labelNorm)

# Train GP
model=buildGP(X_Train_N,Y_Train_N,gpConfig=gpConfig)

# Get GP predictions
Y_Train_Pred_N,__=gpPredict(model,X_Train_N)

# Unnormalize
Y_Train_Pred=Y_Train_Pred_N.copy()
if labelNorm is not None:
    Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y,
                              method=labelNorm,reverse=True)

# Compute MRE
MRE=100*numpy.abs(Y_Train_Pred-Y_Train)/Y_Train
MRE=MRE.mean()

# Testing with new data

#system 0.9
x2_new_0_9 = [0.9011388442217876, 0.804410805145677, 0.6976737693586318, 0.5905658218497403, 0.48660837768989024, 0.38224770304579114, 
              0.2834984630719293, 0.1832555007045214, 0.08992908382941511]
G11_new_0_9 = [1.840903066243385, 1.4431782543226708, 1.1194632063728023, 0.6858639751461544, 0.23154617555838125, -0.24298677365716637, 
               -0.6841183674395782, -0.9572020072200575, -1.1089268732668078]
G12_new_0_9 = [-1.5542940908040905, -1.8363956993522885, -2.1579228950735416, -2.401360968763447, -2.546058187312377, -2.530530193491355,
               -2.240945889343394, -1.95801078932589, -1.6407838461263127]
G22_new_0_9 = [-1.1424500797531492, -1.0270770347179023, -0.7774618708046224, -0.39958417871799445, 0.12092383586316457, 0.7356607089479147, 
               1.0501845921183834, 1.4735334698217344, 1.7156420382217195]
gamma_1_0_9 = [2.567618655,2.234567605,1.895190957, 1.611453225, 1.459606765, 1.276981218,
               1.216803436, 1.111069669, 1.077945091]
gamma_2_0_9 = [1.007145257, 1.05312984, 1.110739515, 1.1979926, 1.29029951, 1.456284271, 
               1.691840583, 1.955646658, 2.269363096]


# Combine G11, G12, G22 into a feature matrix
X_New_0_9 = numpy.column_stack((G11_new_0_9, G12_new_0_9, G22_new_0_9))

# Normalize new features using the scaler from training
X_New_N = X_New_0_9.copy()
if featureNorm is not None:
    X_New_N, _ = normalize(X_New_0_9, skScaler=skScaler_X, method=featureNorm)

# Predict gamma_2 using the trained model
Y_New_Pred_N, Y_New_STD_N = gpPredict(model, X_New_N)

# Unnormalize predictions if needed
Y_New_Pred = Y_New_Pred_N.copy()

if labelNorm is not None:
    Y_New_Pred_0_9, _ = normalize(Y_New_Pred_N, skScaler=skScaler_Y, method=labelNorm, reverse=True)

Y_New_Pred_0_9_1 = Y_New_Pred_0_9[:, 0].reshape(-1, 1)
Y_New_Pred_0_9_2 = Y_New_Pred_0_9[:, 1].reshape(-1, 1)

#system 1.1
x2_new_1_1 = [0.8972436077702632, 0.7914671106622541, 0.6923603091635752, 0.5922475538564925, 0.49350003374514123, 0.39666279832328055, 
              0.299917668534429, 0.19987148495001528, 0.10249034320754276]
G11_new_1_1 = [-2.583847531367817, -2.254474967587716, -1.97987765769118, -1.7928252644277378, -1.5840760035996946, -1.4600752966964086, 
               -1.337879648460309, -1.2601378820407125, -1.1999522871962698]
G12_new_1_1 = [-0.8884322521673719, -0.7927077531322532, -0.7427475123859439, -0.695009445410588, -0.7126976894020828, -0.7088501050705824,
               -0.7672594015725076, -0.8184644526450205, -0.9464328433288377]
G22_new_1_1 = [-1.228494667880412, -1.3036452356548454, -1.3988712481184886, -1.5369739784398058, -1.680688133221591, -1.908798823071391,
               -2.1274113493124895, -2.554926753002902, -2.91874599586247]
gamma_1_1_1 = [0.559053564, 0.563149094, 0.62759494, 0.704164786, 0.770191625,
               0.844115256, 0.912062465, 0.973587482, 1.01465767]
gamma_2_1_1 = [0.957725678, 0.919763461, 0.8653463, 0.784496407, 0.71452478, 0.616068337,
               0.536030374, 0.462379567, 0.361966893]


# Combine G11, G12, G22 into a feature matrix
X_New_1_1 = numpy.column_stack((G11_new_1_1, G12_new_1_1, G22_new_1_1))

# Normalize new features using the scaler from training
X_New_N = X_New_1_1.copy()
if featureNorm is not None:
    X_New_N, _ = normalize(X_New_1_1, skScaler=skScaler_X, method=featureNorm)

# Predict gamma_2 using the trained model
Y_New_Pred_N, Y_New_STD_N = gpPredict(model, X_New_N)

# Unnormalize predictions if needed
Y_New_Pred = Y_New_Pred_N.copy()

if labelNorm is not None:
    Y_New_Pred_1_1, _ = normalize(Y_New_Pred_N, skScaler=skScaler_Y, method=labelNorm, reverse=True)

Y_New_Pred_1_1_1 = Y_New_Pred_1_1[:, 0].reshape(-1, 1)
Y_New_Pred_1_1_2 = Y_New_Pred_1_1[:, 1].reshape(-1, 1)

#system 1.25
x2_new_1_25 = [0.89377712, 0.78992731, 0.6892642, 0.59391598, 0.49591469,
          0.40101505, 0.30266877, 0.20333042, 0.103321087]
G11_new_1_25 = [-4.219792872890755, -3.2363220354432345, -2.635448971817377, -2.220896815155275,
            -1.905672219271214, -1.664287594417277, -1.4863375738599118,
            -1.3482135375873023, -1.236731686]
G12_new_1_25 = [-0.6057639476785153, -0.48939596105762206, -0.42559501041586756, -0.39505891408497723, -0.3828853376070287, -0.40353106841480246,
            -0.4490793997339826, -0.5108800706782308,-0.656071677]
G22_new_1_25 = [-1.264583763,-1.381993919, -1.534299068, -1.724507245, -1.977775905,
            -2.319765521, -2.753424636, -3.486994395234545, -4.630438418]
gamma_1_1_25 = [0.129673295,0.168608172,0.218194912,0.341223734,0.425912262,0.559130326,
                0.680578426,0.806740239,0.868036218]
gamma_2_1_25 = [0.948012927, 0.855326409, 0.721552004, 0.569414794, 0.425661575, 
          0.301721552, 0.213787078, 0.141404269, 0.110595224]


# Combine G11, G12, G22 into a feature matrix
X_New_1_25 = numpy.column_stack((G11_new_1_25, G12_new_1_25, G22_new_1_25))

# Normalize new features using the scaler from training
X_New_N = X_New_1_25.copy()
if featureNorm is not None:
    X_New_N, _ = normalize(X_New_1_25, skScaler=skScaler_X, method=featureNorm)

# Predict gamma_2 using the trained model
Y_New_Pred_N, Y_New_STD_N = gpPredict(model, X_New_N)

# Unnormalize predictions if needed
Y_New_Pred = Y_New_Pred_N.copy()

if labelNorm is not None:
    Y_New_Pred_1_1, _ = normalize(Y_New_Pred_N, skScaler=skScaler_Y, method=labelNorm, reverse=True)

Y_New_Pred_1_25_1 = Y_New_Pred_1_1[:, 0].reshape(-1, 1)
Y_New_Pred_1_25_2 = Y_New_Pred_1_1[:, 1].reshape(-1, 1)

#system 1.5
x2_new_1_5 = [0.890320996643666, 0.7876508297373157, 0.6892985078421388, 0.5936338089835054, 0.49756643962368047, 0.4023691117116231, 
              0.30649195243598054, 0.20749420416667058, 0.1066747845478181]
G11_new_1_5 = [1.8716536775104624, 1.402822874817301, 1.219550731433918, 0.7096934482801536, 0.2896179560399686,-0.20653320128333036,
              -0.5441292616332626,-0.9085144299433835,-1.0894768986906986]
G12_new_1_5 = [-1.5955579335237084, -1.892322909957172, -2.2370652578975134, -2.4074915912257016, -2.5521924339627446, -2.4761969305999796,
               -2.42401028642305, -2.0061791014258206, -1.6953132358387557]
G22_new_1_5 = [-1.131692012279248, -0.9956989924649167, -0.7285606543773313, -0.40115904301263283, 0.07755813007732082, 0.516112017519347,
              1.213531864664999, 1.2770659942581624, 1.6520052642251373]
gamma_1_1_5 = [0.010487012, 0.017908912, 0.047325061, 0.097570387, 0.182292986, 
               0.33041185, 0.541406431, 0.764887388, 0.948280403]
gamma_2_1_5 = [0.924622083, 0.755374354, 0.527200146, 0.326082379, 0.189064777, 
               0.089548529, 0.039135126, 0.012578948, 0.007193738]


# Combine G11, G12, G22 into a feature matrix
X_New_1_5 = numpy.column_stack((G11_new_1_5, G12_new_1_5, G22_new_1_5))

# Normalize new features using the scaler from training
X_New_N = X_New_1_5.copy()
if featureNorm is not None:
    X_New_N, _ = normalize(X_New_1_5, skScaler=skScaler_X, method=featureNorm)

# Predict gamma_2 using the trained model
Y_New_Pred_N, Y_New_STD_N = gpPredict(model, X_New_N)

# Unnormalize predictions if needed
Y_New_Pred = Y_New_Pred_N.copy()

if labelNorm is not None:
    Y_New_Pred_1_1, _ = normalize(Y_New_Pred_N, skScaler=skScaler_Y, method=labelNorm, reverse=True)

Y_New_Pred_1_5_1 = Y_New_Pred_1_1[:, 0].reshape(-1, 1)
Y_New_Pred_1_5_2 = Y_New_Pred_1_1[:, 1].reshape(-1, 1)

# =============================================================================
# Plots
# =============================================================================

# axes.plot(x2_C, Y_Train[16:, 0], 'k*', label=r'MD ($\xi = 0.85$)')
# axes.plot(x2_C,Y_Train_Pred[16:,0],'--y',label=r'GP-train ($\xi = 0.85$)')
# axes.plot(x2_A,Y_Train[:8,0],'ko',label=r'MD (A, $\xi = 1.0$)')
# axes.plot(x2_A,Y_Train_Pred[:8,0],'--r',label=r'GP-train ($\xi = 1.0$)')
# axes.plot(x2_B, Y_Train[8:16, 0], 'ks', label=r'MD ($\xi = 1.2$)')
# axes.plot(x2_B,Y_Train_Pred[8:16,0],'--b',label=r'GP-train (B, $\xi = 1.2$)')
# axes.plot(x2_new_0_9, Y_New_Pred_0_9.flatten(), 'ko', label=r'GP-Test ($\xi = 0.9$)')
# axes.plot(x2_new_0_9, gamma_2_0_9, '--', label=r'MD ($\xi = 0.9$)')
# axes.plot(x2_new_1_1, Y_New_Pred_1_1.flatten(), 'ko', label=r'GP-Test ($\xi = 1.1$)')
# axes.plot(x2_new_1_1, gamma_2_1_1, '--', label=r'MD ($\xi = 1.1$)')

axes[0].plot(x2_C, Y_Train[16:24, 0], 'o', color='maroon', markersize=8)
axes[0].plot(x2_C,Y_Train_Pred[16:,0],'--', color='maroon')
axes[1].plot(x2_C, Y_Train[16:24, 1], 'o', color='maroon', markersize=8)
axes[1].plot(x2_C,Y_Train_Pred[16:,1],'--', color='maroon')
axes[0].plot([], [], 'o--',  color='maroon', markersize=8, label=r'$\xi = 0.85$')  

axes[0].plot(x2_new_0_9, gamma_1_0_9, 'sg', markerfacecolor='none', markersize=8)
axes[0].plot(x2_new_0_9, Y_New_Pred_0_9_1.flatten(),'--g')
axes[1].plot(x2_new_0_9, gamma_2_0_9,'sg', markerfacecolor='none', markersize=8)
axes[1].plot(x2_new_0_9,  Y_New_Pred_0_9_2.flatten(),'--g')
axes[0].plot([], [], 'sg--', markersize=8, label=r'$\xi = 0.9$')  

axes[0].plot(x2_A,Y_Train[:8,0],'om', markersize=8)
axes[0].plot(x2_A,Y_Train_Pred[:8,0],'--m')
axes[1].plot(x2_A,Y_Train[:8,1],'om', markersize=8)
axes[1].plot(x2_A,Y_Train_Pred[:8,1],'--m')
axes[0].plot([], [], 'om--', markersize=8, label=r'$\xi = 1$')  

# axes[0].plot(x2_new_1_25, gamma_1_1_25,'sr', markersize=8)
# axes[0].plot(x2_new_1_25, Y_New_Pred_1_25_1.flatten(), '--r')
# axes[1].plot(x2_new_1_25, gamma_2_1_25,'sr', markersize=8)
# axes[1].plot(x2_new_1_25, Y_New_Pred_1_25_2.flatten(), '--r')
# axes[0].plot([], [], 'or--', markersize=8, label=r'$\xi = 1.1$')  

axes[0].plot(x2_new_1_1, gamma_1_1_1,'sr', markerfacecolor='none', markersize=8)
axes[0].plot(x2_new_1_1, Y_New_Pred_1_1_1.flatten(), '--r')
axes[1].plot(x2_new_1_1, gamma_2_1_1,'sr', markerfacecolor='none', markersize=8)
axes[1].plot(x2_new_1_1, Y_New_Pred_1_1_2.flatten(), '--r')
axes[0].plot([], [], 'or--', markersize=8, label=r'$\xi = 1.1$') 

axes[0].plot(x2_B, Y_Train[8:16, 0], 'ob', markersize=8)  # No label here
axes[0].plot(x2_B, Y_Train_Pred[8:16, 0], '--b')  # No label here
axes[1].plot(x2_B, Y_Train[8:16, 1], 'ob', markersize=8)  # No label here
axes[1].plot(x2_B, Y_Train_Pred[8:16, 1], '--b')  # No label here
axes[0].plot([], [], 'ob--', markersize=8, label=r'$\xi = 1.2$')  

# Add legend (unchanged)
axes[0].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, 
               borderaxespad=0.4, handletextpad=0.4, fontsize='24', 
               loc='upper left', handlelength=1)

axes[0].set_ylabel(r'$\mathbf{\gamma}_{\mathbf{1}}$', fontsize=40)
axes[0].set_xlabel(r'$\mathbf{\textit{x}}_{\mathbf{2}}$', fontsize=40)
axes[1].set_ylabel(r'$\mathbf{\gamma}_{\mathbf{2}}$', fontsize=40)
axes[1].set_xlabel(r'$\mathbf{\textit{x}}_{\mathbf{2}}$', fontsize=40)

axes[0].set_ylim(-0.2,5)
axes[1].set_ylim(-0.2,5)

axes[0].set_xticks(numpy.arange(0, 1.1, 0.2))
axes[1].set_xticks(numpy.arange(0, 1.1, 0.2))

plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()

# ================================
# Parity Plot for Activity Coefficients
# ================================

fig_parity, ax_parity = plt.subplots(figsize=(4, 4))

pp = PdfPages('parity-gamma.pdf')

# Ideal parity line
ax_parity.plot([0, 5], [0, 5], 'k--', linewidth=0.5)

# gamma_1 and gamma_2: true vs predicted
# Training systems
# ax_parity.scatter(Y_Train[:8, 0], Y_Train_Pred[:8, 0], c='blue', marker='o', label=r'Train $\gamma_1$', edgecolors='k')
# ax_parity.scatter(Y_Train[:8, 1], Y_Train_Pred[:8, 1], c='red', marker='o', label=r'Train $\gamma_2$', edgecolors='k')
# ax_parity.scatter(Y_Train[8:16, 0], Y_Train_Pred[8:16, 0], c='blue', marker='o', label=r'Train $\gamma_1$', edgecolors='k')
# ax_parity.scatter(Y_Train[8:16, 1], Y_Train_Pred[8:16, 1], c='red', marker='o', label=r'Train $\gamma_2$', edgecolors='k')
# ax_parity.scatter(Y_Train[16:24, 0], Y_Train_Pred[16:24, 0], c='blue', marker='o', label=r'Train $\gamma_1$', edgecolors='k')
# ax_parity.scatter(Y_Train[16:24, 1], Y_Train_Pred[16:24, 1], c='red', marker='o', label=r'Train $\gamma_2$', edgecolors='k')

# # Test systems
# ax_parity.scatter(gamma_1_0_9, Y_New_Pred_0_9_1.flatten(), c='blue', marker='s', label=r'Test $\gamma_1$, $\xi=0.9$', edgecolors='k')
# ax_parity.scatter(gamma_2_0_9, Y_New_Pred_0_9_2.flatten(), c='red', marker='s', label=r'Test $\gamma_2$, $\xi=0.9$', edgecolors='k')

# ax_parity.scatter(gamma_1_1_1, Y_New_Pred_1_1_1.flatten(), c='blue', marker='^', label=r'Test $\gamma_1$, $\xi=1.1$', edgecolors='k')
# ax_parity.scatter(gamma_2_1_1, Y_New_Pred_1_1_2.flatten(), c='red', marker='^', label=r'Test $\gamma_2$, $\xi=1.1$', edgecolors='k')

# === Training sets: solid circles ===
ax_parity.scatter(Y_Train[:, 0], Y_Train_Pred[:, 0], c='blue', marker='o', alpha=0.7, label=r'Train $\gamma_1$')
ax_parity.scatter(Y_Train[:, 1], Y_Train_Pred[:, 1], c='red', marker='o', alpha=0.7, label=r'Train $\gamma_2$')

# === Test sets: hollow squares ===
ax_parity.scatter(gamma_1_0_9, Y_New_Pred_0_9_1.flatten(), facecolors='none', edgecolors='blue', marker='s', label=r'Test $\gamma_1$')
ax_parity.scatter(gamma_2_0_9, Y_New_Pred_0_9_2.flatten(), facecolors='none', edgecolors='red', marker='s', label=r'Test $\gamma_2$')

ax_parity.scatter(gamma_1_1_1, Y_New_Pred_1_1_1.flatten(), facecolors='none', edgecolors='blue', marker='s')
ax_parity.scatter(gamma_2_1_1, Y_New_Pred_1_1_2.flatten(), facecolors='none', edgecolors='red', marker='s')


ax_parity.legend(frameon=False, fontsize=10, loc='upper left')
# ax_parity.grid(True, linestyle='--', alpha=0.3)ax_parity.set_xlabel('MD values', fontsize=16)

ax_parity.set_ylabel(r'$\textbf{Predicted $\boldsymbol{\gamma}_{\boldsymbol{i}}$ (GP)}$', fontsize=16)
ax_parity.set_xlabel(r'$\textbf{Ground Truth $\boldsymbol{\gamma}_{\boldsymbol{i}}$ (MD)}$', fontsize=16)

ax_parity.set_xlim(-0.2, 5.2)
ax_parity.set_ylim(-0.2, 5.2)
ax_parity.set_aspect('equal')
ax_parity.set_xticks(numpy.arange(0, 5.1, 1))
ax_parity.set_yticks(numpy.arange(0, 5.1, 1))
ax_parity.legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, 
               borderaxespad=0.4, handletextpad=0.4, fontsize='18', 
               loc='upper left', handlelength=1)
# ax_parity.grid(True, linestyle='--', alpha=0.3)

def compute_gibbs_duhem(x2, gamma1, gamma2):
    """
    Computes Gibbs-Duhem violation:
    x1 * d(lnγ1)/dx2 + x2 * d(lnγ2)/dx2
    Excludes endpoints to ensure numerical stability.
    """
    x2 = numpy.array(x2)
    gamma1 = numpy.array(gamma1)
    gamma2 = numpy.array(gamma2)

    # Exclude endpoints
    x2 = x2
    x1 = 1 - x2
    ln_gamma1 = numpy.log(gamma1)
    ln_gamma2 = numpy.log(gamma2)

    # Compute derivatives using central differences
    dlnγ1_dx2 = numpy.gradient(ln_gamma1, x2)
    dlnγ2_dx2 = numpy.gradient(ln_gamma2, x2)

    # return x2, x1 * ln_gamma1 + x2 * ln_gamma2  # Return x2 for aligned plotting
    return x2, x1 * dlnγ1_dx2 + x2 * dlnγ2_dx2  # Return x2 for aligned plotting

systems = [
    (r'$\xi=0.85$', x2_C, Y_Train[16:24, 0], Y_Train[16:24, 1],
     Y_Train_Pred[16:24, 0], Y_Train_Pred[16:24, 1], 'maroon', 'o', False),

    (r'$\xi=1.0$', x2_A, Y_Train[0:8, 0], Y_Train[0:8, 1],
     Y_Train_Pred[0:8, 0], Y_Train_Pred[0:8, 1], 'm', 'o', False),

    (r'$\xi=1.2$', x2_B, Y_Train[8:16, 0], Y_Train[8:16, 1],
     Y_Train_Pred[8:16, 0], Y_Train_Pred[8:16, 1], 'b', 'o', False),

    (r'$\xi=0.9$', x2_new_0_9, gamma_1_0_9, gamma_2_0_9,
     Y_New_Pred_0_9_1.flatten(), Y_New_Pred_0_9_2.flatten(), 'g', 'o', True),

    (r'$\xi=1.1$', x2_new_1_1, gamma_1_1_1, gamma_2_1_1,
     Y_New_Pred_1_1_1.flatten(), Y_New_Pred_1_1_2.flatten(), 'r', 'o', True),
]

fig, (ax_gd, ax_cum) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

systems = [
    (r'$\xi=0.85$', x2_C, Y_Train[16:24, 0], Y_Train[16:24, 1],
     Y_Train_Pred[16:24, 0], Y_Train_Pred[16:24, 1], 'maroon', 'o', False),

    (r'$\xi=1.0$', x2_A, Y_Train[0:8, 0], Y_Train[0:8, 1],
     Y_Train_Pred[0:8, 0], Y_Train_Pred[0:8, 1], 'm', 'o', False),

    (r'$\xi=1.2$', x2_B, Y_Train[8:16, 0], Y_Train[8:16, 1],
     Y_Train_Pred[8:16, 0], Y_Train_Pred[8:16, 1], 'b', 'o', False),

    (r'$\xi=0.9$', x2_new_0_9, gamma_1_0_9, gamma_2_0_9,
     Y_New_Pred_0_9_1.flatten(), Y_New_Pred_0_9_2.flatten(), 'g', 'o', True),

    (r'$\xi=1.1$', x2_new_1_1, gamma_1_1_1, gamma_2_1_1,
     Y_New_Pred_1_1_1.flatten(), Y_New_Pred_1_1_2.flatten(), 'r', 'o', True),
]

def compute_gibbs_duhem(x2, gamma1, gamma2):
    """
    Computes Gibbs-Duhem violation:
    x1 * d(lnγ1)/dx2 + x2 * d(lnγ2)/dx2
    """
    x2 = numpy.array(x2)
    gamma1 = numpy.array(gamma1)
    gamma2 = numpy.array(gamma2)
    x1 = 1 - x2

    ln_gamma1 = numpy.log(gamma1)
    ln_gamma2 = numpy.log(gamma2)

    dlnγ1_dx2 = numpy.gradient(ln_gamma1, x2)
    dlnγ2_dx2 = numpy.gradient(ln_gamma2, x2)

    return x2, x1 * dlnγ1_dx2 + x2 * dlnγ2_dx2

pp = PdfPages('Gibbs-duhem.pdf')

fig, (ax_gd, ax_cum) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

for i, (name, x2, gamma1, gamma2, pred1, pred2, color, marker, is_test) in enumerate(systems):
    x2_md, gd_md = compute_gibbs_duhem(x2, gamma1, gamma2)
    x2_gp, gd_gp = compute_gibbs_duhem(x2, pred1, pred2)

    dx_md = numpy.gradient(x2_md)
    dx_gp = numpy.gradient(x2_gp)
    cum_md = numpy.cumsum(gd_md * dx_md)
    cum_gp = numpy.cumsum(gd_gp * dx_gp)

    # Plot pointwise violations on left
    ax_gd.plot(x2_md, gd_md, color=color, linestyle='-', alpha=0.7)
    ax_gd.plot(x2_gp, gd_gp, color=color, linestyle='--', alpha=0.7)
    
    # Plot cumulative violations on right
    ax_cum.plot(x2_md, cum_md, color=color, linestyle='-', label=f'{name}')
    ax_cum.plot(x2_gp, cum_gp, color=color, linestyle='--')

ax_gd.set_xlabel(r'$x_2$', fontsize=40)
ax_gd.set_ylabel('Gibbs-Duhem Violation', fontsize=18)
# ax_gd.set_ylabel(r'$x_1 \, \frac{d \ln \gamma_1}{dx_2} + x_2 \, \frac{d \ln \gamma_2}{dx_2}$', fontsize=24)
ax_gd.set_ylim(-1, 1)

ax_cum.set_xlabel(r'$x_2$', fontsize=40)
ax_cum.set_ylabel('Cumulative Violation', fontsize=18)
# ax_cum.set_ylabel(r'$\int \left( x_1 \frac{d \ln \gamma_1}{dx_2} + x_2 \frac{d \ln \gamma_2}{dx_2} \right) dx_2$', fontsize=24)
ax_cum.axhline(0, color='k', linestyle=':', alpha=0.7)
ax_cum.set_ylim(-1, 1)

# Legend sorting by ξ
lines_gd, labels_gd = ax_gd.get_legend_handles_labels()
lines_cum, labels_cum = ax_cum.get_legend_handles_labels()

def extract_xi(label):
    match = re.search(r'\$\\xi\s*=\s*([0-9.]+)', label)
    return float(match.group(1)) if match else float('inf')

sorted_items = sorted(zip(lines_cum, labels_cum), key=lambda x: extract_xi(x[1]))
sorted_lines, sorted_labels = zip(*sorted_items)

ax_cum.legend(sorted_lines, sorted_labels, frameon=False, loc='upper right', handlelength=0.8, labelspacing=0.2, fontsize=18)

plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()

def compute_g_excess(x2, gamma1, gamma2):
    x1 = 1 - numpy.array(x2)
    ln_gamma1 = numpy.log(gamma1)
    ln_gamma2 = numpy.log(gamma2)
    return x2, x1 * ln_gamma1 + x2 * ln_gamma2

pp = PdfPages('Gibbs-excess.pdf')

fig, ax = plt.subplots(figsize=(6, 5))
for name, x2, gamma1, gamma2, pred1, pred2, color, marker, is_test in systems:
    x2_md, gex_md = compute_g_excess(x2, gamma1, gamma2)
    x2_gp, gex_gp = compute_g_excess(x2, pred1, pred2)

    ax.plot(x2_md, gex_md, color=color, linestyle='-', label=f'{name} MD')
    ax.plot(x2_gp, gex_gp, color=color, linestyle='--', label=f'{name} GP')

ax.set_xlabel(r'$x_2$', fontsize=40)
ax.set_ylabel(r'$G^{\text{ex}} / RT$', fontsize=40)
ax.legend(sorted_lines, sorted_labels, frameon=False, loc='upper right', handlelength=0.8, labelspacing=0.2, fontsize=12)
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()