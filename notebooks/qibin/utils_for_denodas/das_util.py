import torch
import h5py
import numpy as np
from joblib import Parallel, delayed

from ELEP.elep.ensemble_coherence import ensemble_semblance


### Functions to denoise large-N DAS array
def process_3d_array(arr, len1=1500, len2=1500):
    """convert to numpy array"""
    arr = np.array(arr)
    
    """Ensure the array has at least len1 rows and len2 columns"""
    slices, rows, cols = arr.shape
    arr = arr[:, :min(rows, len1), :min(cols, len2)]
    
    """Pad zeros if it has fewer than len1 rows or len2 columns"""
    if rows < len1 or cols < len2:
        padding_rows = max(len1 - rows, 0)
        padding_cols = max(len2 - cols, 0)
        arr = np.pad(arr, ((0, 0), (0, padding_rows), (0, padding_cols)), 'constant')
    
    return arr


def Denoise_largeDAS(data, model_func, devc, repeat=4, norm_batch=False):
    """ This function do the following (it does NOT filter data):
    1) split into multiple 1500-channel segments
    2) call Denoise function for each segments
    3) merge all segments
    
    data: 2D -- [channel, time]
    output: 2D, but padded 0 to have multiple of 1500 channels
    
    This code was primarily designed for the Alaska DAS, but applicable to other networks
    """ 
    data = np.array(data)
    nchan = data.shape[0]
    ntime = data.shape[1]
    
    if (nchan % 1500) == 0:
        n_seg = nchan // 1500
    else:
        n_seg = nchan // 1500 + 1
        
    full_len = int(n_seg * 1500)
    
    pad_data = process_3d_array(data[np.newaxis,:,:], len1=full_len)
    data3d = pad_data.reshape((-1, 1500, 1500))
    
    oneDenoise, mulDenoise = Denoise(data3d, model_func, devc, repeat=repeat, norm_batch=norm_batch)
    
    oneDenoise2d = oneDenoise.reshape((full_len, 1500))[:nchan, :ntime]
    mulDenoise2d = mulDenoise.reshape((full_len, 1500))[:nchan, :ntime]
    
    return oneDenoise2d, mulDenoise2d



def Denoise(data, model_func, devc, repeat=4, norm_batch=False):
    """ This function do the following (it does NOT initialize model):

    1) normalized the data
    2) ensure the data format, precision and size
    3) denoise and scale back the output amplitude
    """ 
    
    """ convert to torch tensors """
    if norm_batch:
        scale = np.std(data[-1]) + 1e-7  ### Avoid potentially bad beginning sub-images
    else:
        scale = np.std(data, axis=(1,2), keepdims=True) + 1e-7
        
    data_norm = data / scale  ## standard scaling
    arr = process_3d_array(data_norm.astype(np.float32), len1=data_norm.shape[1], len2=data_norm.shape[2])
    X = torch.from_numpy(arr).to(devc)
    
    """ denoise - deploy """
    with torch.no_grad():
        oneDenoise = model_func(X)
        mulDenoise = oneDenoise
        
        for i in range(repeat-1):
            mulDenoise = model_func(mulDenoise)

    """ convert back to numpy """
    print(oneDenoise.shape)
    oneDenoise = oneDenoise.to('cpu').numpy() * scale
    mulDenoise = mulDenoise.to('cpu').numpy() * scale
    
    return oneDenoise[:, :len(data[0]), :len(data[0][0])], mulDenoise[:, :len(data[0]), :len(data[0][0])]


### Functions to pick large DAS arrays
def process_p(ista,paras_semblance,batch_pred,istart,sfs):
    
        crap = ensemble_semblance(batch_pred[:, ista, :], paras_semblance)
        imax = np.argmax(crap[istart:])
            
        return float((imax)/sfs)+istart/sfs, crap[istart+imax]
    

def apply_elep(DAS_data, list_models, fs, paras_semblance, device):
    
    """"
    This function takes a array of stream, a list of ML models and 
    apply these models to the data, predict phase picks, and
    return an array of picks .
    DAS_data: NDArray of DAS data: [channel,time stamp - 6000]
    """
    
    twin = 6000  ## needed by EQTransformer
    nsta = DAS_data.shape[0]
    bigS = np.zeros(shape=(DAS_data.shape[0], 3, DAS_data.shape[1]))
    for i in range(nsta):   ### same data copied to three components
        bigS[i,0,:] = DAS_data[i,:]
        bigS[i,1,:] = DAS_data[i,:]
        bigS[i,2,:] = DAS_data[i,:]

    # allocating memory for the ensemble predictions
    batch_pred_P = np.zeros(shape=(len(list_models),nsta,twin)) 
    batch_pred_S = np.zeros(shape=(len(list_models),nsta,twin))
        
    ######### Broadband workflow ################
    crap2 = bigS.copy()
    crap2 -= np.mean(crap2, axis=-1, keepdims= True) # demean data
    # original use std norm
    data_std = crap2 / (np.std(crap2) + 1e-7)
    # could use max data
    mmax = np.max(np.abs(crap2), axis=-1, keepdims=True)
    data_max = np.divide(crap2 , mmax,out=np.zeros_like(crap2), where=mmax!=0)
    del bigS
    
    # data to tensor
    data_tt = torch.from_numpy(data_max).to(device, dtype=torch.float32)
    
    for ii, imodel in enumerate(list_models):
        imodel.eval()
        with torch.no_grad():
            tmp = imodel(data_tt)
            batch_pred_P[ii, :, :] = tmp[1].cpu().numpy()[:, :]
            batch_pred_S[ii, :, :] = tmp[2].cpu().numpy()[:, :]
    
    smb_peak = np.zeros([nsta,2,2], dtype = np.float32)

    smb_peak[:,0,:] =np.array(Parallel(n_jobs=1)(delayed(process_p)(ista,paras_semblance,batch_pred_P,0,fs) 
                                                    for ista in range(nsta)))
    smb_peak[:,1,:] =np.array(Parallel(n_jobs=1)(delayed(process_p)(ista,paras_semblance,batch_pred_S,0,fs) 
                                                    for ista in range(nsta)))
    
    return smb_peak



def extract_metadata(h5file, machine_name='optodas'):
    """Extract metadata from DAS HDF
    Args:
        h5file (str): path to DAS HDF file
        machine_name (str): name of interrogator
    Returns:    
        gl (float): gauge length in meters
        t0 (float): start time in seconds since 1 Jan 1970
        dt (float): sample interval in seconds
        fs (float): sampling rate in Hz
        dx (float): channel interval in meters
        un (str): unit of measurement
        ns (int): number of samples
        nx (int): number of channels
    """
    if machine_name == 'optodas':
        with h5py.File(h5file, 'r') as fp:
            gl = fp['header/gaugeLength'][()]
            t0 = fp['header/time'][()]
            dt = fp['header/dt'][()]
            fs = 1./dt
            dx = fp['header/dx'][()]*10 # not sure why this is incorrect
            un = fp['header/unit'][()]
            ns = fp['/header/dimensionRanges/dimension0/size'][()]
            nx = fp['/header/dimensionRanges/dimension1/size'][()][0]
    elif machine_name == 'onyx':
        with h5py.File(h5file,'r') as fp:      
            gl = fp['Acquisition'].attrs['GaugeLength']
            t0 = fp['Acquisition']['Raw[0]']['RawDataTime'][0]/1e6
            fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
            dt = 1./fs
            dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
            un = fp['Acquisition']['Raw[0]'].attrs['RawDataUnit']
            ns  = len(fp['Acquisition']['Raw[0]']['RawDataTime'][:])
            nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']
    else:
        raise ValueError('Machine name not recognized')
            

    return gl, t0, dt, fs, dx, un, ns, nx