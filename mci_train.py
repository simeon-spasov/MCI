import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix     
from keras import regularizers 
import cPickle as pickle
from utils.preprocess import DataLoader
from utils.models import Parameters, Net


target_rows = 182
target_cols = 218
depth = 182
axis = 1
num_clinical = 13
drop_rate = 0.1
w_regularizer = regularizers.l2(5e-5)
batch_size = 6

model_filepath = '../path'


params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
               'drop_rate': drop_rate, 'epochs': 50,
          'gpu': "/gpu:0", 'model_filepath': model_filepath, 
          'image_shape': (target_rows, target_cols, depth, axis),
          'num_clinical': num_clinical}

params = Parameters(params_dict)

seeds = [np.random.randint(1, 5000) for _ in range(10)]

def evaluate_net (seed):
    data_loader = DataLoader((target_rows, target_cols, depth, axis), seed = seed)
    train_data, val_data, test_data = data_loader.get_train_val_test()
    net = Net(params)
    history = net.train((train_data, val_data))
    #need to nnormalize and arrange my test data (or train data as when training)
    test_loss, test_acc  = net.evaluate (test_data)
    test_preds = net.predict(test_data)  
    fpr_test, tpr_test, thresholds_test = roc_curve(test_data[-1], test_preds) 
    mci_conf_matrix_test = confusion_matrix(y_true = test_data[-1], y_pred = np.round(test_preds))/(float(len(test_data[-1]))/2) 
    
    val_loss, val_acc = net.evaluate (val_data)
    val_preds = net.predict(val_data) 
    fpr_val, tpr_val, thresholds_val = roc_curve(val_data[-1], val_preds)
    mci_conf_matrix_val = confusion_matrix(y_true = val_data[-1], y_pred = np.round(val_preds))/(float(len(val_data[-1]))/2) 

    
    with open('../path' + str(seed)+'.pickle', 'w') as f:
        pickle.dump([[fpr_test, tpr_test, thresholds_test, test_loss, test_acc, mci_conf_matrix_test, test_preds, test_data[-1], val_preds, val_data[-1]],
                     [fpr_val, tpr_val, thresholds_val, val_loss, val_acc, mci_conf_matrix_val]], f)
    
    


for seed in seeds:
    #Load data
    print('Processing seed number ', seed)
    evaluate_net(seed)


    




