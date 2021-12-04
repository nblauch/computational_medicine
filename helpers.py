import numpy as np
import os
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
import nilearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
import sklearn
import pdb
from IPython import get_ipython
import glob
import cepy
from tqdm import tqdm

# configure as needed
if os.getenv('SERVERNAME') == 'xps':
    BIDS_DIR = '/mnt/d/bids/ds000030'
elif os.getenv('SERVERNAME') == 'mind':
    BIDS_DIR = '/user_data/nblauch/bids/ds000030'
    
# stupid little hack for now, since just focusing on rest data
SUBJECTS = [os.path.basename(os.path.dirname(os.path.dirname(file))) for file in glob.glob(f'{BIDS_DIR}/derivatives/fmriprep/**/func/*_task-rest_bold_space-fsaverage5.L.func.gii')]

def X_is_running():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if not X_is_running() and not isnotebook():
    print('no X server detected. changing default matplotlib backend to Agg for compatibility.')
    import matplotlib
    matplotlib.use('Agg')

GLASSER_ROIS_LONG = {'V1':1, 'medial superior temporal area':2, 'V6':3, 'V2':4, 'V3':5, 'V4':6, 'V8':7, 'primary motor cortex':8,
                    'primary sensory cortex':9, 'FEF':10, 'premotor eye field':11, 'area 55b':12, 'area V3A':13,
                    'retrosplenial complex':14, 'parieto-occipital sulcus area 2':15, 'V7':16, 'intraparietal sulcus area 1':17,
                    'fusiform face complex':18, 'V3B':19, 'LO1':20, 'LO2':21, 'posterior inferotemporal complex':22, 'MT':23,
                    'A1':24, 'perisylvian language area':25, 'superior frontal language area':26, 'precuneus visual area':27,
                    'superior temporal visual area':28, 'medial area 7P':29, '7m':30, 'parieto-occipital sulcus area 1':31,
                    '23d':32, 'area ventral 23 a+b':33, 'area dorsal 23 a+b':34, '31p ventral':35, '5m':36, '5m ventral':37,
                    '23c':38, '5L':39, 'dorsal area 24d':40, 'ventral area 24d':41, 'lateral area 7A':42,
                    'supplemetary and cingulate eye field':43, '6m anterior':44, 'medial area 7a':45, 'lateral area 7p':46,
                    '7pc':47, 'lateral intraparietal ventral':48, 'ventral intraparietal complex':49, 'medial intraparietal area':50,
                    'area 1':51, 'area 2':52,'area 3a':53, 'dorsal area 6':54, '6mp':55, 'ventral area 6':56, 'posterior 24 prime':57,
                    '33 prime':58, 'anterior 24 prime':59, 'p32 prime':60, 'a24':61, 'dorsal 32':62, '8BM':63, 'p32':64, '10r':65,
                    '47m':66, '8Av':67, '8Ad':68, 'area 9 middle':69, 'area 8B lateral':70, 'area 9 posterior':71, '10d':72, '8C':73,
                    'area 44':74, 'area 45':75, 'area 47l':76, 'anterior 47r':77, 'rostral area 6':78, 'area IFJa':79, 'area IFJp':80,
                    'area IFSp':81, 'area IFSa':82, 'area posterior 9-46v':83, 'area 46':84, 'area anterior 9-46v':85, 'area 9-46d':86,
                    'area 9 anterior':87, '10v':88, 'anterior 10p':89, 'polar 10p':90, 'area 11l':91, 'area 13l': 92, 'OFC':93, '47s':94,
                    'lateral intraparietal dorsal':95, 'area 6 anterior':96, 'inferior 6-8 transitional area':97,
                    'superior 6-8 transitional area':98, 'area 43':99, 'area OP4/PV':100, 'area OP1/SII':101, 'area OP2-3/VS':102,
                    'area 52':103, 'retroinsular cortex':104, 'area PFcm':105, 'posterior insula area 2':106, 'area TA2':107,
                    'frontal opercular area 4':108, 'middle insular area':109, 'pirform cortex':110, 'anterior ventral insular area':111,
                    'anterior angranular insula complex':112, 'frontal opercular area 1':113, 'frontal opercular area 3':114,
                    'frontal opercular area 2':115, 'area PFt':116, 'anterior intraparietal area':117, 'entorhinal cortex':118,
                    'preSubiculum':119, 'hippocampus':120, 'proStriate area':121, 'perirhinal ectorhinal cortex':122, 'area STGa':123,
                    'parabelt complex':124, 'A5':125, 'parahippocampal area 1':126, 'parahippocampal area 3':127, 'STSd anterior':128,
                    'STSd posterior':129, 'STSv posterior':130, 'TG dorsal':131, 'TE1 anterior':132, 'TE1 posterior':133, 'TE2 anterior':134,
                    'TF':135, 'TE2 posterior':136, 'PHT':137, 'PH':138, 'temporoparietooccipital junction 1':139,
                    'temporoparietooccipital junction 2':140, 'superior 6-8':141, 'dorsal transitional visual area':142, 'PGp':143,
                    'intraparietal 2':144, 'intraparietal 1':145, 'intraparietal 0':146, 'PF opercular':147, 'PF complex':148, 'PFm complex':149,
                    'PGi':150, 'PGs':151, 'V6A':152, 'ventromedial visual area 1':153, 'ventromedial visual area 3':154, 'parahippocampal area 2':155,
                    'V4t':156, 'FST':157, 'V3CD':158, 'lateral occipital 3':159, 'ventromedial visual area 2':160, '31pd':161, '31a':162,
                    'ventral visual complex':163, 'area 25':164, 's32':165, 'posterior OFC complex':166, 'posterior insular 1':167,
                    'insular granular complex':168, 'frontal opercular 5':169, 'posterior 10p':170, 'posterior 47r':171, 'TG ventral':172,
                    'medial belt complex':173, 'lateral belt complex':174, 'A4':175, 'STSv anterior':176, 'TE1 middle':177, 'parainsular area':178,
                    'anterior 32 prime':179, 'posterior 24':180}
# _,_, names = _read_annot(os.path.join(os.getenv('BIDS'), 'lateralization/derivatives/freesurfer/fsaverage/label/lh.HCPMMP1.annot'))
# GLASSER_ROIS = {names[ii].decode("utf-8").split('_')[1]: ii for ii in range(1,len(names))}
GLASSER_ROIS = {'V1':1, 'MST':2, 'V6':3, 'V2':4, 'V3':5, 'V4':6, 'V8':7, '4':8,
                    '3b':9, 'FEF':10, 'PEF':11, '55b':12, 'V3A':13,
                    'RSC':14, 'POS2':15, 'V7':16, 'IPS1':17,
                    'FFC':18, 'V3B':19, 'LO1':20, 'LO2':21, 'PIT':22, 'MT':23,
                    'A1':24, 'PSL':25, 'SFL':26, 'PCV':27,
                    'STV':28, '7Pm':29, '7m':30, 'POS1':31,
                    '23d':32, 'v23ab':33, 'd23ab':34, '31pv':35, '5m':36, '5mv':37,
                    '23c':38, '5L':39, '24dd':40, '24dv':41, '7AL':42,
                    'SCEF':43, '6ma':44, '7Am':45, '7Pl':46,
                    '7PC':47, 'LIPv':48, 'VIP':49, 'MIP':50,
                    '1':51, '2':52,'3a':53, '6d':54, '6mp':55, '6v':56, 'p24pr':57,
                    '33pr':58, 'a24pr':59, 'p32pr':60, 'a24':61, 'd32':62, '8BM':63, 'p32':64, '10r':65,
                    '47m':66, '8Av':67, '8Ad':68, '9m':69, '8BL':70, '9p':71, '10d':72, '8C':73,
                    '44':74, '45':75, '47l':76, 'a47r':77, '6r':78, 'IFJa':79, 'IFJp':80,
                    'IFSp':81, 'IFSa':82, 'p9-46v':83, '46':84, 'a9-46v':85, '9-46d':86,
                    '9a':87, '10v':88, 'a10p':89, '10pp':90, '11l':91, '13l': 92, 'OFC':93, '47s':94,
                    'LIPd':95, '6a':96, 'i6-8':97,
                    's6-8':98, '43':99, 'OP4':100, 'OP1':101, 'OP2-3':102,
                    '52':103, 'RI':104, 'PFcm':105, 'PoI2':106, 'TA2':107,
                    'FOP4':108, 'MI':109, 'Pir':110, 'AVI':111,
                    'AAIC':112, 'FOP1':113, 'FOP3':114,
                    'FOP2':115, 'PFt':116, 'AIP':117, 'EC':118,
                    'PreS':119, 'H':120, 'ProS':121, 'PeEc':122, 'STGa':123,
                    'PBelt':124, 'A5':125, 'PHA1':126, 'PHA3':127, 'STSda':128,
                    'STSdp':129, 'STSvp':130, 'TGd':131, 'TE1a':132, 'TE1p':133, 'TE2a':134,
                    'TF':135, 'TE2p':136, 'PHT':137, 'PH':138, 'TPOJ1':139,
                    'TPOJ2':140, 'TPOJ3':141, 'DVT':142, 'PGp':143,
                    'IP2':144, 'IP1':145, 'IP0':146, 'PFop':147, 'PF':148, 'PFm':149,
                    'PGi':150, 'PGs':151, 'V6A':152, 'VMV1':153, 'VMV3':154, 'PHA2':155,
                    'V4t':156, 'FST':157, 'V3CD':158, 'LO3':159, 'VMV2':160, '31pd':161, '31a':162,
                    'VVC':163, '25':164, 's32':165, 'pOFC':166, 'PoI1':167,
                    'Ig':168, 'FOP5':169, 'p10p':170, 'p47r':171, 'TGv':172,
                    'MBelt':173, 'LBelt':174, 'A4':175, 'STSva':176, 'TE1m':177, 'PI':178,
                    'a32pr':179, 'p24':180}

def get_mean_parcel_timeseries(timeseries, annot):
    """
    compute mean timeseries for each parcel
    """
    mean_data = []
    for parcel in sorted(np.unique(annot)):
        if parcel == 0:
            continue
        mean_data.append(timeseries[:,annot==parcel].mean(1))
    return np.stack(mean_data)

def map_mean_parcels_to_surf(mean_parcels, annot):
    """
    map the mean value for each parcel to all vertices of that parcel on the full surface
    """
    assert len(mean_parcels.squeeze().shape) == 1, 'mean_parcels must be a vector'
    
    mapped_values = np.zeros_like(annot, dtype=float)
    ii= -1
    for parcel in sorted(np.unique(annot)):
        if parcel == 0:
            continue
        ii += 1
        mapped_values[annot == parcel] = mean_parcels[ii]
    return mapped_values

def get_subject_confounds(subject):
    confounds = pd.read_csv(f'{BIDS_DIR}/derivatives/fmriprep/{subject}/func/{subject}_task-rest_bold_confounds.tsv',sep='\t')
    cons_confounds = confounds[['aCompCor00', 'aCompCor01', 'aCompCor02', 'aCompCor03', 'aCompCor04', 'aCompCor05',
                                'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']]
    confounds_array = cons_confounds.to_numpy()
    return confounds_array

def get_subject_restdat(subject):
    dat = []
    for hemi in ['L', 'R']:
        fn = f'{BIDS_DIR}/derivatives/fmriprep/{subject}/func/{subject}_task-rest_bold_space-fsaverage5.{hemi}.func.gii'
        gii = nib.load(fn)
        all_dat = np.stack([gii.darrays[ii].data for ii in range(len(gii.darrays))])
        dat.append(all_dat)
    return np.concatenate(dat, 1)

def get_subject_connectome(subject, overwrite=False):
    fn = f'{BIDS_DIR}/derivatives/connectomes/{subject}/rest_hcp.npy'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if os.path.exists(fn) and not overwrite:
        conn_z = np.load(fn)
    else:
        restdat = get_subject_restdat(subject)
        annot = nilearn.surface.load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/fsaverage5/label/mh.HCPMMP1.annot') - 1
        mean_data = get_mean_parcel_timeseries(restdat, annot)
        confounds = get_subject_confounds(subject)
        conn = ConnectivityMeasure(kind='correlation').fit_transform([mean_data.T])[0]
    #     conn[conn==1] = .9999
    #     conn[conn==-1] = -.9999
        conn_z = .5*np.log((1+conn)/(1-conn))
        np.save(fn, conn_z)
    return conn_z

def get_group_ce(sparsity=0.3, task='rest', overwrite=False):
    if task != 'rest':
        raise NotImplementedError()
    fn = f'{BIDS_DIR}/derivatives/python/cepy/group_ce_fc-{task}_sparsity-{sparsity}.json'
    if os.path.exists(fn):
        ce_group = cepy.load_model(fn)
    else:
        connectomes = []
        connectomes = []
        for sub in tqdm(SUBJECTS):
            try:
                connectomes.append(get_subject_connectome(sub, overwrite=False))
            except Exception as e:
                print(e)            
        connectomes = np.stack(connectomes)
        connectomes[np.isnan(connectomes)] = 0
        connectomes[np.isinf(connectomes)] = 10
        connectomes[np.isinf(-connectomes)] = -10
        group_connectome = np.mean(connectomes, 0)
        group_connectome = np.abs(group_connectome)
        group_connectome[group_connectome < np.percentile(group_connectome, 1-sparsity)] = 0
        ce_group = cepy.CE(permutations = 1, seed=1)  
        ce_group.fit(group_connectome)
        ce_group.save_model(fn)
    return ce_group
        

def do_full_reg_analysis(X, y, x_name, y_name,
                     do_pca=True,
                     do_pcr=True,
                     save=True,
                     show=False,
                     ):
    """
    replicate a full analysis pipeline for different feature engineering approaches (X) and continuous phenotypes (y)
    """
    if do_pca:
        pca_sol = PCA().fit(X)
        all_PCs = pca_sol.transform(X)
        plt.scatter(all_PCs[:,0], all_PCs[:,1], c=y)
        plt.colorbar()
        if save:
            plt.savefig(f'figures/X-{x_name}_y-{y_name}_PC-0-1.png', dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
            
        corrs = []
        for pc in range(260):
            PC = all_PCs[:,pc]
            corrs.append(spearmanr(y, PC)[0])
        os.makedirs('figures', exist_ok=True)
        plt.plot(corrs)
        plt.title(f'Correlation of {y_name} with PCs of {x_name}')
        plt.xlabel('PC #')
        plt.ylabel('Spearman corelation (r)')
        plt.axhline(0,0,1,color='r',linestyle='--')
        if save:
            plt.savefig(f'figures/X-{x_name}_y-{y_name}_PC-corrs.png', dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    if do_pcr:
        corrs = []
        weights = []
        kfold = KFold(3)
        for train_inds, test_inds in kfold.split(X):
            pca_sol = PCA().fit(X[train_inds])
            X_train_all = pca_sol.transform(X[train_inds])
            X_test_all = pca_sol.transform(X[test_inds])
            these_corrs = []
            these_weights = []
            for n_pcs in range(1,all_PCs.shape[1]):
                X_train = X_train_all[:,:n_pcs]
                X_test = X_test_all[:,:n_pcs]
                model = RidgeCV(alphas=np.logspace(-4,4,num=9)).fit(X_train, y[train_inds])
                preds = model.predict(X_test)
                true = y[test_inds]
                these_corrs.append(spearmanr(true, preds)[0])
                these_weights.append(model.coef_)
            corrs.append(these_corrs)
            weights.append(these_weights)

        corrs = np.mean(corrs, 0)
        weights = np.mean(weights, 0)
        plt.plot(corrs)
        plt.title(f'Regressing {y_name} based on PCs of {x_name}')
        plt.xlabel('# of PCs used')
        plt.ylabel('cross-validated regression prediction (r)')
        plt.axhline(0,0,1,color='r',linestyle='--')
        if save:
            plt.savefig(f'figures/X-{x_name}_y-{y_name}_PCR.png', dpi=200, bbox_inches='tight')
        if show:
            plt.show()    
        plt.close()    
        
        
def do_full_categ_analysis(X, y, x_name, y_name,
                     do_pca=True,
                     do_pcc=True,
                     do_pcc_proper=True,
                     save=True,
                     show=False,
                     ):
    """
    replicate a full analysis pipeline for different feature engineering approaches (X) and categorical phenotypes (y)
    """
    groups = np.unique(y)
    # assert len(groups) == 2, 'only implemented for binary classification'
    
    if do_pca:
        pca_sol = PCA().fit(X)
        all_PCs = pca_sol.transform(X)
        plt.scatter(all_PCs[:,0], all_PCs[:,1], c=y)
        plt.colorbar()
        if save:
            plt.savefig(f'figures/X-{x_name}_y-{y_name}_PC-0-1.png', dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
            
        if len(groups) == 2:
            diffs = cohens_d(all_PCs[y == groups[0]], all_PCs[y == groups[1]])
            plt.plot(diffs, 'o')
            plt.title(f"Cohen's d of PCs of {x_name} across groups of {y_name}")
            plt.xlabel('PC #')
            plt.ylabel("Cohen's d")
            plt.axhline(0,0,1,color='r',linestyle='--')
            if save:
                plt.savefig(f'figures/X-{x_name}_y-{y_name}_PC-diffs.png', dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()

    if do_pcc:
        accs = []
        weights = []
        kfold = StratifiedKFold(5)
        for train_inds, test_inds in kfold.split(X, y):
            pca_sol = PCA(n_components=4*X.shape[0]//5-1).fit(X[train_inds])
            X_train_all = pca_sol.transform(X[train_inds])
            X_test_all = pca_sol.transform(X[test_inds])
            these_accs = []
            these_weights = []
            for n_pcs in range(1,X_train_all.shape[1]):
                X_train = X_train_all[:,:n_pcs]
                X_test = X_test_all[:,:n_pcs]
                model = RidgeClassifierCV(alphas=np.logspace(-4,4,num=9)).fit(X_train, y[train_inds])
                preds = model.predict(X_test)
                true = y[test_inds]
                these_accs.append(sklearn.metrics.balanced_accuracy_score(true, preds, adjusted=False))
                these_weights.append(model.coef_)
            accs.append(these_accs)
            weights.append(these_weights)

        accs = np.mean(accs, 0)
        # weights = np.mean(weights, 0)
        
        plt.plot(accs)
        plt.title(f'Classifying {y_name} based on PCs of {x_name}')
        plt.xlabel('# of PCs used')
        plt.ylabel('cross-validated classification accuracy \n (mean recall)')
        plt.axhline(0.5,0,1,color='r',linestyle='--')
        if save:
            plt.savefig(f'figures/X-{x_name}_y-{y_name}_PCC.png', dpi=200, bbox_inches='tight')
        if show:
            plt.show()    
        plt.close()    
        
    if do_pcc_proper:
        kfold = StratifiedKFold(5)
        n_pcs_sel = []
        accs = []
        for train_inds, test_inds in kfold.split(X, y):
            kfold_inner = StratifiedKFold(5)
            inner_accs = []
            pca_sol = PCA(n_components=4*X.shape[0]//5-1).fit(X[train_inds])
            for inner_train_inds, val_inds in kfold_inner.split(X[train_inds], y[train_inds]):
                X_train_all = pca_sol.transform(X[train_inds[inner_train_inds]])
                X_test_all = pca_sol.transform(X[train_inds[val_inds]])
                these_accs = []
                these_weights = []
                for n_pcs in range(1,X_train_all.shape[1]):
                    X_train = X_train_all[:,:n_pcs]
                    X_test = X_test_all[:,:n_pcs]
                    model = RidgeClassifierCV(alphas=np.logspace(-4,4,num=9)).fit(X_train, y[train_inds[inner_train_inds]])
                    preds = model.predict(X_test)
                    true = y[train_inds[val_inds]]
                    these_accs.append(sklearn.metrics.balanced_accuracy_score(true, preds, adjusted=False))
                    these_weights.append(model.coef_)
                inner_accs.append(these_accs)
            n_pcs = np.argmax(np.nanmean(inner_accs,0))+1
            n_pcs_sel.append(n_pcs)
            
            X_train = pca_sol.transform(X[train_inds])[:,:n_pcs]
            X_test = pca_sol.transform(X[test_inds])[:,:n_pcs]
            model = RidgeClassifierCV(alphas=np.logspace(-4,4,num=9)).fit(X_train, y[train_inds])
            preds = model.predict(X_test)
            true = y[test_inds]
            accs.append(sklearn.metrics.balanced_accuracy_score(true, preds, adjusted=False))
        print(f'Classifying {y_name} based on PCs of {x_name}')
        print(f'mean of {np.mean(n_pcs_sel)} pcs selected \n accuracy: {np.mean(accs)}')
        
        return np.mean(accs), np.mean(n_pcs_sel)

    
def cohens_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x,0) - np.mean(y,0)) / np.sqrt(((nx-1)*np.std(x, axis=0, ddof=1) ** 2 + (ny-1)*np.std(y, axis=0, ddof=1) ** 2) / dof)
