'''
    Written by Wafaa Wardah  | USP    | June 2019
    The experiment set up.
'''

from __future__ import print_function
import torch
import sys, os, gc
import datasets, models
import time
import math
import pickle
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, roc_curve, roc_auc_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from sklearn.metrics.ranking import auc

def config():
    print('\n\tInitializing experiment configurations...')
    exp_settings = {}
    exp_settings['batch_size'] = 128
    exp_settings['num_epochs'] = 100
    exp_settings['patience'] = 10
    exp_settings['window_size'] = 7
    exp_settings['space'] = {'lr' : hp.uniform('lr', 0.00001, 0.001),'num_kernels' : 3 + hp.randint('num_kernels', 7)}
    exp_settings['folder'] = 'experiment_output'
    if not os.path.exists(exp_settings['folder']): os.mkdir(exp_settings['folder'])
    exp_settings['datafile'] = exp_settings['folder'] + '/data_summary.txt'
    exp_settings['run_file'] = exp_settings['folder'] + '/eval_summary.txt'
    exp_settings['run_counter'] = 0
    exp_settings['glob_loss'] = 10.0
    return exp_settings
    
def load_all_data(exp_settings):
        
    def load_data(mode):
        dataset = datasets.ProtPep_dataset(ws=exp_settings['window_size'], mode=mode)
        return DataLoader(dataset, shuffle=True, batch_size=exp_settings['batch_size'])
    
    def save_details():
        for mode, loader in sets.items():
            print('Summarizing ', mode, ' set...')
            with open(exp_settings['datafile'], 'a') as f:
                f.write('\n' + mode + ' set contains ' + str(len(loader.dataset)) + ' residues')
   
    print('\n\tLoading datasets...')
    sets = {}
    sets['training'] = load_data('train')
    sets['validation'] = load_data('val')
    sets['test'] = load_data('test')
    save_details()
    return sets


def train(model, lr):
    global exp_settings, data, device
        
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,17])).to(device) # weighted update (1:17)
    optimizer   = optim.Adam(model.parameters(), lr=lr)
    
    max_loss = 10
    patience_counter = exp_settings['patience']
    best_val = {}
    best_val['epoch_counter'] = 0
    best_val['val_loss_list'] = []
    best_val['train_loss_list'] = []
    
    for epoch in range(exp_settings['num_epochs']):
        epoch_train_loss, epoch_val_loss = [], []
        
        model.train()
        for (images, labels) in data['training']:
            inputs          = Variable(images).to(device)
            inputs          = inputs.unsqueeze(1)
            labels          = Variable(labels.long()).to(device)
            outputs         = model(inputs)
            loss            = criterion(outputs, labels)
            epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        best_val['train_loss_list'].append(sum(epoch_train_loss) / len(epoch_train_loss))    
        
        val_outputs, val_labels, val_prob = [], [], []  # for metrics calculations
        model.eval()
        for (images, labels) in data['validation']:
            inputs          = Variable(images).to(device)
            inputs          = inputs.unsqueeze(1)
            labels          = Variable(labels.long()).to(device)
            outputs         = model(inputs)
            val_loss        = criterion(outputs, labels)
            epoch_val_loss.append(val_loss.item())
            out_max = outputs.detach()
            val_prob.append(out_max)
            out_max = torch.argmax(out_max, dim=1)
            val_outputs.append(out_max)
            val_labels.append(labels)      
            
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)    
        best_val['val_loss_list'].append(val_loss)
        
        if val_loss < max_loss:
            max_loss = val_loss
            patience_counter = 0
            best_val['val_outputs'] = torch.cat(val_outputs).cpu().numpy()
            best_val['val_labels'] = torch.cat(val_labels).cpu().numpy()
            best_val['output_prob'] = torch.cat(val_prob).cpu().numpy()[:,1:]
            best_val['model_state'] = model.state_dict()
            best_val['epoch_counter'] = epoch
        else: patience_counter += 1
        if patience_counter == exp_settings['patience']: break
            
    with open(exp_settings['folder'] + '/run' + str(exp_settings['run_counter']) + '.pickle', 'wb') as f:
        pickle.dump(best_val, f)
        
    return best_val

def calc_metrics(iter_dict):
    scores = {}
    temp = confusion_matrix(iter_dict['val_labels'], iter_dict['val_outputs'])
    TN, FP, FN, TP = temp.ravel()
    #scores['misclassification_rate'] = (FP + FN) / (TN + FP + FN + TP ) # or 1-accuracy
    scores['sensitivity'] = TP / (FN + TP) # aka sensitivity or recall
    #scores['false_pos_rate'] = FP / (TN + FP)
    scores['specificity'] = TN / (TN + FP) # aka specificity
    precision = TP / (TP + FP)
    #scores['prevalence'] = (TP + FN) / (TN + FP + FN + TP )
    scores['f_score'] = (2 * scores['sensitivity'] * precision) / (scores['sensitivity'] + precision)
    
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = (TP + FP) * ( TP + FN) * (TN + FP) * (TN + FN)
    if mcc_denominator == 0: mcc_denominator = 1
    scores['mcc'] = mcc_numerator / math.sqrt(mcc_denominator)
    scores['auc'] = roc_auc_score(iter_dict['val_labels'], iter_dict['output_prob'])
    scores['accuracy'] = (TP + TN) / (TN + FP + FN + TP )
    scores['conf_matrix'] = temp
    
    return scores

def pad(x):
    max_len = 11
    x = str(x)
    missing_len = max_len - len(x)
    x = x + (' ' * missing_len)
    return x

def add_header(file, score_dict):
    with open(file, 'a+') as f:
        f.write('\n|' + ('=' * (18 * 11)) + '|')
        f.write('\n|' + pad('Iter') + '|'+ pad('Status')   + '|' + pad('Loss')  + '|'+ pad('So far') + '|'+ pad('Runtime') + 
                '|' + pad('LR') + '|'+ pad('Nodes') + '|'+ pad('Epochs') + '|')
        for key in score_dict.keys():
            if key == 'conf_matrix': f.write(pad('TN FP FN TP') + (' ' * 19) + '|')
            else: f.write(pad(key) + '|')
        f.write('\n|' + ('=' * (18 * 11)) + '|')
        
def save_to_file(file, count, status, loss, sofar, runtime, epochs, lr, num_kernels, score_dict):
    with open(file, 'a+') as f:
        f.write(('\n|' + pad(count)  + '|'+ pad(status)   + '|' + pad(round(loss,6))+ '|'+ pad(round(sofar,6)) + '|'+ 
                 pad(runtime)+ '|'+  pad(round(lr,6)) + '|' + pad(num_kernels) + '|' + pad(epochs+1)   + '|'))
        for key, val in score_dict.items():
            if key == 'conf_matrix': f.write(str(val.ravel()))
            else: f.write(pad(round(val,6)) + '|')
        f.write('\n|' + ('-' * (18 * 11)) + '|')
    
def obj_fn(space):
    global device, exp_settings
    start_time  = time.time()
    lr = space['lr']
    num_kernels = int(math.pow(2, space['num_kernels']))
    model = models.dynamic_model(exp_settings['window_size'], 38, num_kernels).to(device)  
    iter_dict = train(model, lr)
    iter_scores = calc_metrics(iter_dict)
    
    loss = -(iter_scores['auc'])
    
    if iter_scores['mcc'] <= -1.0:
        hp_status = STATUS_FAIL
        status = 'FAIL'
    else:
        hp_status = STATUS_OK        
        if loss < exp_settings['glob_loss']: 
            status = 'BEST'
            exp_settings['glob_loss'] = loss
        else: status = 'ACCEPT'
        
    runtime =  str(datetime.timedelta(seconds=round(time.time() - start_time)))
    if exp_settings['run_counter'] % 20 == 0: add_header(exp_settings['run_file'], iter_scores)
    save_to_file(exp_settings['run_file'], exp_settings['run_counter'], status, loss, exp_settings['glob_loss'], runtime, iter_dict['epoch_counter'], lr, num_kernels, iter_scores)
    
    exp_settings['run_counter'] += 1
    
    return {'loss': loss, 'status': hp_status} 
    

def run_trials():
    global exp_settings, data
    step = 1
    max_trials = 3 # have about 3 or 5 iterations to start with
    file_name = exp_settings['folder'] + '/trials_file.hyperopt'
    
    try:
        trials = pickle.load(open(file_name, 'rb'))
        print("\n\tFound saved Trials! Loading...")
        max_trials = len(trials.trials) + step
        print('Rerunning from {} to {} trials...'.format(len(trials.trials), len(trials.trials) + step))
    except:
        print("\n\tCreating new Trials!...")
        trials = Trials()
       
    best = fmin(fn=obj_fn,
                    space=exp_settings['space'],
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=max_trials)
    
    print('Best : ', best) 
    
    with open(file_name, 'wb') as f:
        pickle.dump(trials, f)

def main():
    global exp_settings, device, data
    print('\nBayesian optimization of CNN hyperparameters, model for predicting protein-peptide binding sites.' + 
          '\n-------------------------------------------------------------------------------------------------')  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)   
    
    exp_settings = config()
    data = load_all_data(exp_settings)
        
    while True:
        run_trials()

if __name__ == '__main__': 
    main()
