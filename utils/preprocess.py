

import numpy as np
from numpy.random import RandomState
from os import listdir
import nibabel as nib
import math
import csv
import random

class DataLoader:
    """The DataLoader class is intended to be used on our custom-template 
    structural MRI and Jacobian determinant images (placed in folder ../ADNI_volumes_customtemplate_float32)
    The structure of ../ADNI_volumes_customtemplate_float32 is:
        
        stableNL_137_S_4587_MPRAGE_masked_brain.nii.gz 
        stableNL_137_S_4587_MPRAGE_JD_masked_brain.nii.gz  
        ...
        stableMCItoAD_035_S_4784_MPRAGE_masked_brain.nii.gz
        stableMCItoAD_035_S_0869_MPRAGE_JD_masked_brain.nii.gz	
        ...
        stableMCI_029_S_1384_MP-RAGE_masked_brain.nii.gz
        stableMCI_029_S_1384_MP-RAGE_JD_masked_brain.nii.gz
        ...
        stableAD_011_S_0183_MPRAGE_masked_brain.nii.gz
        stableAD_011_S_4949_MPRAGE_JD_masked_brain.nii.gz
        ...
        
        naming convention is: class_subjectID_imageType.nii.gz
        masked_brain denotes structural MRI, JD_masked_brain denotes Jacobian Determinant 
        
        stableNL: healthy controls
        stableMCItoAD: progressive MCI
        stableAD: Alzheimer's subjects
    
    Additionally, we use clinical features from csv file ../LP_ADNIMERGE.csv
    """
    
    
    def __init__(self, target_shape, seed = None):
        self.mri_datapath = '/local/scratch/ses88/ADNI_volumes_customtemplate_float32'
        self.xls_datapath = '/local/scratch/ses88'
        self.target_shape = target_shape
        self.seed = seed
  

    def shuffle_dict_lists (self, dictionary): 
       p = RandomState(self.seed).permutation(len(dictionary.values()[0]))  
       for key in dictionary.iterkeys():
           dictionary[key] = [dictionary[key][i] for i in p]  

    
    def get_filenames (self):
        '''Puts filenames in ../ADNI_volumes_customtemplate_float32 in
        dictionaries according to class (stableMCI, MCItoAD, stableNL and stableAD)
        with keys corresponding to image modality (mri and JD)
        '''
        file_names = sorted(listdir(self.mri_datapath))  
        keys = ['JD', 'mri']
        healthy_dict, ad_dict, smci_dict, pmci_dict = [{key: [] for key in keys} for i in range(4)]
        for _file in file_names:
          if _file[-2:] == 'gz':
            if 'stableNL' in _file:
                if 'JD' in _file:
                    healthy_dict['JD'].append(_file)
                else:
                    healthy_dict['mri'].append(_file)
            elif 'stableAD' in _file:
                if 'JD' in _file:
                    ad_dict['JD'].append(_file)
                else:
                    ad_dict['mri'].append(_file)
            elif 'stableMCItoAD' in _file:
                if 'JD' in _file:
                    pmci_dict['JD'].append(_file)
                else:
                    pmci_dict['mri'].append(_file)
            elif 'stableMCI' in _file and 'AD' not in _file:
                if 'JD' in _file:
                    smci_dict['JD'].append(_file)
                else:
                    smci_dict['mri'].append(_file)
        self.shuffle_dict_lists (healthy_dict) #Randomly shuffle lists healthy_dict ['JD'] and healthy_dict['mri'] in unison
        self.shuffle_dict_lists (ad_dict)
        self.shuffle_dict_lists (smci_dict)
        self.shuffle_dict_lists (pmci_dict)               
        return healthy_dict, ad_dict, smci_dict, pmci_dict
                
                    
    def split_filenames (self, healthy_dict, ad_dict, smci_dict, pmci_dict, val_split =  0.1):
        '''Split filename dictionaries in training/validation and test sets.
        Test set size corresponds to the extra MCI subjects (w.r.t. healthy+AD)
        
        The current version of this method only extracts sMCI and pMCI subjects.
        In order to perform joint training we would need healthy control and 
        Alzheimer's subjects as well.
        
        '''
        keys = ['JD', 'mri']
        train_dict, val_dict, test_dict = [{key: [] for key in keys} for _ in range(3)]

        num_test_samples =  (len(pmci_dict['JD']) + len(smci_dict['JD']) \
                            -len(healthy_dict['JD']) - len (ad_dict['JD']))/2
                            
        num_val_samples =  (int(math.ceil (val_split*(len(smci_dict['JD']) + len(pmci_dict['JD']) - num_test_samples))))/2
            
        for key in healthy_dict.iterkeys():
            test_smci = smci_dict[key][:num_test_samples]
            test_pmci = pmci_dict[key][:num_test_samples]
            test_dict[key] = test_smci + test_pmci
            test_dict['health_state'] = np.concatenate( ( np.zeros( len(test_smci) ),
                                             np.ones (   len(test_pmci) ) ) )
            
            val_smci = smci_dict[key][num_test_samples : num_test_samples + num_val_samples]
            val_pmci = pmci_dict[key][num_test_samples : num_test_samples + num_val_samples]
            val_dict[key] = val_smci + val_pmci
            val_dict['health_state'] = np.concatenate( ( np.zeros( len(val_smci) ),
                                             np.ones (   len(val_pmci) ) ) )

            train_smci = smci_dict[key][num_test_samples + num_val_samples:]
            train_pmci = pmci_dict[key][num_test_samples + num_val_samples:]
            train_dict[key] = train_smci + train_pmci
            train_dict['health_state'] = np.concatenate( ( np.zeros( len(train_smci) ),
                                             np.ones (   len(train_pmci) ) ) )
        
        return train_dict, val_dict, test_dict
                   

    
    #SHOULD FOLLOW SAME ORDER OF SUBJECTS AS mri_file_names
    
    def get_data_xls (self, mri_file_names):
        '''Method extracts clinical variables data for all files in mri_file_names list
        Both mri_file_names and LP_ADNIMERGE.csv are in alphabetical order
        '''
        with open(self.xls_datapath + '/' + 'LP_ADNIMERGE.csv', 'rb') as f:
            reader = csv.reader(f)
            xls = [row for row in reader]  #Extract all data from csv file in a list.
                                                               
        #xls extracts baseline features for patients sorted as in mri_file_names
        test_xls = [row for file_name in mri_file_names for row in xls if row[2] == 'bl' and row[1] in file_name]     
        #convert ethnicity/race/gender features to binary variables
        for row in test_xls:
            row[8] = float(row[8])
            if row[9] == 'Male':
                row[9] = 1.
            else:
                row[9] = 0.
            row[10] = float(row[10])  
                
            if row[11] == 'Hisp/Latino': 
                row[11] = 1.
            else:
                row[11] = 0.
                 
            if row[12] == 'White': #White or non-white only;
                row[12] = 1. #Cluster Am. Indian, unknown, black, asian
                
            else:
                row[12] = 0.
            row[13:] = [float(x) for x in row[13:]]
        clinical_features = np.asarray([row[8:] for row in test_xls])
        return clinical_features
        
    def get_data_mri (self, filename_dict):
         '''Loads subject volumes (MRI and Jacobian images) from filename dictionary
         Returns MRI volume, Jacobian Determinant volume and label     
         '''
         mris = np.zeros( (len(filename_dict.values()[0]),) +  self.target_shape)
         jacs = np.zeros( (len(filename_dict.values()[0]),) +  self.target_shape)
         labels = filename_dict['health_state']
         #keys = ['JD', 'mri']
         keys = ['mri']
         for key in keys:
             for j, filename in enumerate (filename_dict[key]):
                proxy_image = nib.load(self.mri_datapath + '/' + filename)
                image = np.asarray(proxy_image.dataobj)
                if key == 'JD':
                     jacs[j] = np.asarray(np.expand_dims(image, axis = -1))
                else:
                    mris[j] = np.asarray(np.expand_dims(image, axis = -1))
         return mris.astype('float32'), jacs.astype('float32'), labels

            
    def normalize_data (self, train_im, val_im, test_im, mode):
        #We use different normalization procedures depending on data type
        if mode != 'mri' and mode != 'jac' and mode != 'xls':
            print ('Mode has to be mri, jac or xls depending on image data type')
        elif mode == 'mri':
            std = np.std(train_im, axis = 0)
            mean = np.mean (train_im, axis = 0)
            train_im = (train_im - mean)/(std + 1e-20)
            val_im = (val_im - mean)/(std + 1e-20)
            test_im = (test_im - mean)/(std + 1e-20)
        elif mode == 'jac':
            high = np.max(train_im)
            low = np.min(train_im)
            train_im = (train_im - low)/(high - low)
            val_im = (val_im - low)/(high - low)
            test_im = (test_im - low)/(high - low)
        else:
            high = np.max(train_im, axis = 0)
            low = np.min(train_im, axis = 0)
            train_im = (train_im - low)/(high - low)
            val_im = (val_im - low)/(high - low)
            test_im = (test_im - low)/(high - low)
        return train_im, val_im, test_im
            
        
        
    def get_train_val_test (self, val_split = 0.1):
        healthy_dict, ad_dict, smci_dict, pmci_dict = self.get_filenames()
        train_dict, val_dict, test_dict = self.split_filenames (healthy_dict, ad_dict, smci_dict, pmci_dict, val_split = val_split)
        
        train_mri, train_jac, train_labels = self.get_data_mri(train_dict)
        train_xls = self.get_data_xls (train_dict['mri'])
        val_mri, val_jac, val_labels = self.get_data_mri(val_dict)
        val_xls = self.get_data_xls (val_dict['mri'])
        test_mri, test_jac, test_labels = self.get_data_mri(test_dict)
        test_xls = self.get_data_xls (test_dict['mri'])

        train_mri, val_mri, test_mri = self.normalize_data (train_mri, val_mri, test_mri, mode = 'mri')
        train_jac, val_jac, test_jac = self.normalize_data (train_jac, val_jac, test_jac, mode = 'jac')
        train_xls, val_xls, test_xls = self.normalize_data (train_xls, val_xls, test_xls, mode = 'xls')
        
        
        test_data = (test_mri, test_mri, test_xls, test_labels)
        val_data = (val_mri, val_mri, val_xls, val_labels)
        train_data = (train_mri, train_mri, train_xls, train_labels)
    
        
        return train_data, val_data, test_data
