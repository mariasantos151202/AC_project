#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
import torch
import torchvision
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import random_split
from dataset import SoundDS, SoundDS_val, SoundDS_valPatch, SoundDS_Patch
import utils
from evaluation import calculate_outcome_scores, calculate_murmur_scores
import validate
from base_model import AudioClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    outcomes = list()
    labels = list()
    recording_files = []
    recording_files_tr = []
    pIDs = []  # all training patient IDS
    labels_tr = []
    pIDs_tr = []  # training patient IDs for all recordings
    age_tr = []
    gender_tr = []
    pregnancy_tr = []
    wide_fea_tr = []
    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i + 1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        # Extract labels and just use numbers
        label = get_murmur(current_patient_data)

        curr_recording_files = get_corr_recordings(current_patient_data, label)
        recording_files_tr.extend(curr_recording_files)
        # recording_files_tmp = get_corr_recordings(current_patient_data, 'Absent')
        # recording_files.append(recording_files_tmp)

        if label in classes:
            current_labels = classes.index(label)
            # current_labels[j] = 1
            labels.append(current_labels)  # without the one-hot encoding

        # get the patient ID
        pID = current_patient_data.split('\n')[0]
        pID = pID.split(' ')[0]
        pIDs.append(pID)
        
        # get the age, sex, and pregnancy features
        # age, gender, pregnancy = get_features_mod(current_patient_data)
        wide_temp = get_features_mod(current_patient_data)

        # repeating the pID and labels
        for j in range(len(curr_recording_files)):
            labels_tr.append(labels[-1])
            pIDs_tr.append(pID)
            wide_fea_tr.append(wide_temp)

        # extract the outcome label and the features
        current_recordings = load_recordings(data_folder, current_patient_data)
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
    features = np.vstack(features)
    outcomes = np.vstack(outcomes)

    # generate training df contains only the filtered training samples
    df = pd.DataFrame()
    df['pID'] = pIDs_tr
    df['relative_path'] = recording_files_tr
    df['label'] = labels_tr
    # df['age'] = age_tr
    # df['gender'] = gender_tr
    # df['pregnancy'] = pregnancy_tr
    # df_wide = pd.concat([pd.get_dummies(df['age']), pd.get_dummies(df['gender']), pd.get_dummies(df['pregnancy'])], axis=1)
    df_wide = pd.DataFrame(wide_fea_tr)
    # 5 folds cross-validation, stratified split according to labels and put
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    for i, (train_idx, val_idx) in enumerate(kf.split(pIDs, labels)):
        print('Fold {} training started'.format(str(i)))
        # df_result_temp = pd.DataFrame()
        # get the training patients and validation patients
        pID_train = np.asarray(pIDs)[train_idx]
        pID_val = np.asarray(pIDs)[val_idx]

        # Extract corresponding df for training and testing
        df_tr = df.loc[df['pID'].isin(pID_train), df.columns]
        df_val = df.loc[df['pID'].isin(pID_val), df.columns]
        
        df_wide_tr = df_wide.loc[df['pID'].isin(pID_train), df_wide.columns]
        df_wide_val = df_wide.loc[df['pID'].isin(pID_val), df_wide.columns]

        # Train and validation data loader
        # train_ds = SoundDS(df_tr, data_path=data_folder, mode='train')
        # val_ds = SoundDS_val(df_val, data_path=data_folder, mode='Val')
        train_ds = SoundDS(df_tr, data_path=data_folder, mode='train', df_wide=df_wide_tr)
        val_ds = SoundDS_Patch(df_val, data_path=data_folder, df_wide=df_wide_val)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        # Train the model.
        if verbose >= 1:
            print('Training model...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = models.densenet.DenseNet(dataset='Physio', pretrained=True).to(device)
        model = AudioClassifier().to(device)
        # nSamples = df_tr['label'].value_counts()
        # nSamples = nSamples.sort_index()
        # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = [5, 3, 1]  # set according to the challenge metrics
        normedWeights = torch.FloatTensor(normedWeights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=normedWeights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = True
        if scheduler:
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 80], gamma=0.1)
        else:
            scheduler = None
        # writer = SummaryWriter(f'C:/Users/huilu/Office/Projects/PCG/runs')
        acc, score, best_acc, best_score, y_test, y_pred, y_prob = train_and_evaluate(model, device,
                                                                                      train_loader, val_loader,
                                                                                      optimizer, loss_fn, i,
                                                                                      model_folder, scheduler)

    # clinical outcome classification
    random_state = 6789  # Random state; set for reproducibility.
    kf_outcome = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    y_all = []
    y_prob = []
    y_pred = []
    outcome_comp = np.zeros((outcomes.shape[0]))
    outcome_comp[outcomes[:, 1] == 1] = 1
    for i, (train_idx, val_idx) in enumerate(kf_outcome.split(features, outcome_comp)):
        print('Fold {} started'.format(i + 1))
        # X_train = features_new[train_idx]
        # X_val = features_new[val_idx]
        X_train = features[train_idx]
        X_val = features[val_idx]
        y_train = outcomes[train_idx]
        y_val = outcomes[val_idx]

        imputer = SimpleImputer().fit(X_train)
        X_train = imputer.transform(X_train)
        X_val = imputer.transform(X_val)
        classifier = RandomForestClassifier(n_estimators=100,  # Number of trees in the forest.
                                            # min_samples_split=2,
                                            # min_samples_leaf=1,
                                            # max_features='auto',
                                            # max_depth=110,
                                            # bootstrap=True,
                                            max_leaf_nodes=36,  # Maximum number of leaf nodes in each tree.
                                            # criterion="log_loss",
                                            class_weight=[{0: 1, 1: 5}, {0: 5, 1: 1}],
                                            # max_leaf_nodes=32,
                                            verbose=1,
                                            random_state=random_state
                                            ).fit(X_train, y_train)
        outcome_probabilities_val = classifier.predict_proba(X_val)
        outcome_probabilities_val = np.asarray(outcome_probabilities_val, dtype=np.float32)[:, :, 1]
        outcome_probabilities_val = outcome_probabilities_val.T
        outcome_labels_val = classifier.predict(X_val)
        # outcome_pred = classifier.predict(X_val)
        # outcome_labels_val = np.zeros([outcome_pred.shape[0], 2])
        # outcome_labels_val[outcome_pred == 1, 1] = 1
        scores = calculate_outcome_scores(outcomes[val_idx], outcome_probabilities_val, outcome_labels_val)
        print('Fold {} outcome score is {}'.format(i + 1, scores[-1]))
        y_all.extend(outcomes[val_idx])
        y_prob.extend(outcome_probabilities_val)
        y_pred.extend(outcome_labels_val)
        save_challenge_model(model_folder, imputer, outcome_classes, classifier, i)
                                                                                      
    if verbose >= 1:
        print('Done.')


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    # load all the trained models into list
    mumrmur_models = []
    outcome_models = []

    for i in range(5):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AudioClassifier().to(device)
        check_path = os.path.join(model_folder, 'model_best_{}.pth.tar'.format(i))
        model = utils.load_checkpoint(check_path, model)
        mumrmur_models.append(model)

        # load the outcome classifiers
        filename = os.path.join(model_folder, 'outcome_model_{}.sav'.format(i))
        outcome_models.append(joblib.load(filename))
    models = {'murmur_models': mumrmur_models, 'outcome_models': outcome_models}
    return models


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    # separate the murmur and outcome classifiers
    murmur_models = model['murmur_models']
    outcome_models = model['outcome_models']

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_record = len(recordings)
    # get the age, sex, and pregnancy features
    # age, gender, pregnancy = get_features_mod(data)
    wide_fea = get_features_mod(data)
    # wide_fea = np.array((age, gender, pregnancy))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels_ens = np.zeros((5, ))
    probabilities_ens = np.zeros((5, 3))
    for i in range(5):
        model_tmp = murmur_models[i]
        model_tmp.eval()

        val_ds = SoundDS_valPatch(recordings, wide_fea)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
        pred_all = []
        prob_all = []
        with torch.no_grad():
            for batch_idx, data_test in enumerate(val_loader):
                inputs = data_test[0]
                inputs1 = data_test[1]
                # print(inputs1.size())
                # outputs = model_tmp(inputs)

                # _, predicted = torch.max(outputs.data, 1)

                # pred_all.extend(predicted.cpu().numpy())
                # prob_temp = torch.softmax(outputs, dim=1)  # get the probability
                # prob_all.extend(prob_temp.data.cpu().numpy())
                prob = []
                for idx, input_idx in enumerate(inputs):
                    outputs = model_tmp(input_idx.to(device), inputs1[idx].to(device))
                    # _, predicted = torch.max(outputs.data, 1)
                    prob_temp = torch.softmax(outputs, dim=1)  # first use the average possibility of patches to classify
                    prob.extend(prob_temp.data.cpu().numpy())
                prob_ave = np.mean(np.asarray(prob), axis=0)
                predicted = np.argmax(prob_ave)
                pred_all.append(predicted)
                prob_all.append(prob_ave)
        # combining results from all the recordings
        prob_all = np.asarray(prob_all)
        pred_all = np.asarray(pred_all)
        if np.all(pred_all == 2):
            labels_ens[i] = 2
            probabilities_ens[i, :] = np.mean(prob_all, axis=0)
        elif np.any(pred_all == 0):
            labels_ens[i] = 0
            probabilities_ens[i, :] = np.mean(prob_all[np.where(pred_all == 0)[0], :], axis=0)
        else:
            labels_ens[i] = 1
            probabilities_ens[i, :] = np.mean(prob_all, axis=0)

    # voting for the final label, the most voted as label and also the corresponding probability to calculate mean
    voting = np.zeros((3, ))
    voting[0] = np.count_nonzero(labels_ens == 0)
    voting[1] = np.count_nonzero(labels_ens == 1)
    voting[2] = np.count_nonzero(labels_ens == 2)
    label = np.argmax(voting, axis=0)  # when the count are the same, take as positive or unknown

    prob_murmur = np.mean(probabilities_ens[np.where(labels_ens == label)[0], :], axis=0)

    # convert label to one-hot vector
    labels_murmur = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = label
    labels_murmur[idx] = 1

    # clinical outcome classification
    outcome_classes = ['Abnormal', 'Normal']

    # Load features.
    features = get_features(data, recordings)
    features = features.reshape(1, -1)

    # 5 folds classifiers
    outcome_prob_all = []
    for i in range(5):
        outcome_model_temp = outcome_models[i]
        # load the model paramters
        imputer = outcome_model_temp['imputer']
        outcome_classifier = outcome_model_temp['outcome_classifier']
        # Impute missing data.
        features_temp = imputer.transform(features)

        outcome_prob = outcome_classifier.predict_proba(features_temp)
        outcome_prob = np.asarray(outcome_prob, dtype=np.float32)[:, 0, 1]
        outcome_prob_all.append(outcome_prob)
    outcome_prob_all = np.asarray(outcome_prob_all)
    prob_outcome_ave = np.mean(outcome_prob_all, axis=0)
    labels_outcome = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(prob_outcome_ave)
    labels_outcome[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((labels_murmur, labels_outcome))
    probabilities = np.concatenate((prob_murmur, prob_outcome_ave))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_classes, outcome_classifier, fold_indx):
    d = {'imputer': imputer, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
    filename = os.path.join(model_folder, 'outcome_model_{}.sav'.format(fold_indx))
    joblib.dump(d, filename, protocol=0)


# Extract features from the data.
def get_features_mod(data):
   # Extract the age group, sex and the pregnancy status features
    age_group = get_age(data)
    age_list = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    is_pregnant = get_pregnancy_status_mod(data)
    if age_group not in ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']:
        if is_pregnant:
            age = 'Young Adult'
        else:
            age = 'Child'
    else:
        age = age_group

    age_fea = np.zeros(5, dtype=int)
    age_fea[age_list.index(age)] = 1
    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)
    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1
    preg_fea = np.zeros(2, dtype=int)
    if is_pregnant:
        preg_fea[0] = 1
    else:
        preg_fea[1] = 1

    wide_fea = np.append(age_fea, [sex_features, preg_fea])
    return wide_fea
    # return age, sex, is_pregnant


def get_pregnancy_status_mod(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                if compare_strings(l.split(': ')[1].strip(), 'True'):
                    is_pregnant = True
                else:
                    is_pregnant = False
                # is_pregnant = bool(l.split(': ')[1].strip())
            except:
                pass
    return is_pregnant
    
    
def get_corr_recordings(data, label):
    # get the recording file names. only for present get the recording in the murmur location
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations + 1]
    recording_files = []
    if label in ['Unknown', 'Absent']:
        for i in range(num_locations):
            entries = recording_information[i].split(' ')
            recording_file = entries[2]
            recording_files.append(recording_file)
    elif label in ['Present']:
        for l in data.split('\n'):
            if l.startswith('#Murmur locations:'):
                try:
                    recordings = l.split(': ')[1].strip()
                    recordings = recordings.split('+')
                    for i in range(num_locations):
                        entries = recording_information[i].split(' ')
                        if entries[0] in recordings:  # only extract recordings in murmur locations
                            recording_file = entries[2]
                            recording_files.append(recording_file)
                except:
                    pass
    else:
        raise Exception('No recording avaiable')
    return recording_files


# Extract features from the data for outcome classifier
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status_mod(data)
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    elif is_pregnant:
        age = 20*12
    else:
        age = float('nan')


    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 0
    elif compare_strings(sex, 'Male'):
        sex_features[0] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    bmi = weight / (height/100)**2

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    # recording_locations = ['AV', 'MV', 'PV', 'TV']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    # recording_features = np.zeros((num_recording_locations, 3), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    # recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 2] = sp.stats.kurtosis(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
                    # recording_features[j, 4] = sp.stats.kurtosis(recordings[i])
    else:
        a = 1

    recording_features = recording_features.flatten()
    # features = np.hstack(([bmi], [age], sex_features, [height], [weight], [is_pregnant], recording_features))
    features = np.hstack(([bmi], [age], sex_features, [height], [weight], [is_pregnant], recording_features))
    return np.asarray(features, dtype=np.float32)


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()
    correct_prediction = 0
    total_prediction = 0
    target_all = []
    prob_all = []
    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs1 = data[0].to(device)
            inputs2 = data[1].to(device)
            # target = data[1].squeeze(1).to(device)
            target = data[2].to(device)

            # outputs = model(inputs)
            outputs = model(inputs1, inputs2)
            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            prob = torch.softmax(outputs, dim=1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == target).sum().item()
            total_prediction += prediction.shape[0]
            target_all.extend(target.cpu().detach().numpy())
            prob_all.extend(prob.cpu().detach().numpy())
        acc = correct_prediction / total_prediction
        auc_score = roc_auc_score(np.asarray(target_all), np.asarray(prob_all), average='macro', multi_class='ovr')
    return loss_avg(), acc, auc_score


def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, split, model_folder,
                       scheduler=None):
    best_acc = 0.0
    best_score = 0
    y_pred_best = []
    y_prob_best = []
    n_stop = 20  # set the early stopping epochs as 10
    n_epoch = 100
    epochs_no_improve = 0
    for epoch in range(n_epoch):
        avg_loss, acc_tr, auc_score_tr = train(model, device, train_loader, optimizer, loss_fn)
        # loss, acc, score, y_test, y_pred, y_prob = validate.evaluate(model, device, val_loader, loss_fn)
        loss, acc, score, y_test, y_pred, y_prob = validate.evaluate_patch(model, device, val_loader, loss_fn)
        # auc = roc_auc_score(np.argmax(y_test, axis=1), y_prob, average='macro', multi_class='ovr')
        auc = score[1]
        print("Epoch {}/{} Loss:{} Train Acc:{} Train AUC : {} ".format(epoch, n_epoch, avg_loss, acc_tr, auc_score_tr))
        print("Epoch {}/{} Valid Loss:{} Valid Acc:{} Valid AUC : {} Valid Score : {} ".format(epoch, n_epoch, loss, acc,
                                                                                               auc, score[-1]))
        is_best = (best_score < score[-1])
        if is_best:
            best_score = score[-1]
            best_acc = acc
            y_pred_best = np.argmax(y_pred, axis=1)
            y_prob_best = y_prob
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_stop:
                print('Early stopping')
                break
        if scheduler:
            scheduler.step()

        utils.save_checkpoint({"epoch": epoch + 1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()}, is_best, split, "{}".format(model_folder))
    return acc, score[-1], best_acc, best_score, np.argmax(y_test, axis=1), y_pred_best, y_prob_best


# if __name__ == '__main__':
#     data_folder = 'data'
#     model_folder = '/model'
#     verbose = 1
#     train_challenge_model(data_folder, model_folder, verbose)
