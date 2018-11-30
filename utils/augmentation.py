
import numpy as np
from keras.preprocessing.image import Iterator


class CustomIterator(Iterator):

    def __init__(self, data, batch_size=6, shuffle=False, seed=None,
                 dim_ordering='tf'):
        
        self.mri_data, self.jac_data, self.xls_data, self.labels = data
        self.dim_ordering = dim_ordering
        self.batch_size = batch_size
        super(CustomIterator, self).__init__(self.mri_data.shape[0], batch_size, shuffle, seed)
        #409 mci samples

    def _get_batches_of_transformed_samples(self, index_array):
        batch_mri = np.zeros(tuple([len(index_array)] + list(self.mri_data.shape[1:])))
        batch_jac = np.zeros(tuple([len(index_array)] + list(self.jac_data.shape[1:])))
        batch_xls = np.zeros(tuple([len(index_array)] + list(self.xls_data.shape[1:])))
        batch_labels = np.zeros(tuple([len(index_array)])) 
        
        for i, j in enumerate(index_array):
            mri = self.mri_data[j]
            jac = self.jac_data[j]
            xls = self.xls_data[j]
            batch_mri[i]= mri 
            batch_jac[i]= jac
            batch_xls[i]= xls 
            batch_labels[i] = self.labels[j]

        return [batch_mri, batch_jac, batch_xls], batch_labels

  
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)









