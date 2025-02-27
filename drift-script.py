import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from glob import glob

from time import process_time

import sys
import re


iterations = 1

rf_estimators = 100

seed_size = 100

monthly_pool_size = 10000
monthly_points = 1000

daily_pool_size = 330
daily_points = 33

query_strategy = uncertainty_sampling

proportion = 0.9

drop_column_list = ['UNIXtime', 'Timestamp', 'SignatureText', 
                    'SignatureID', 'ExtIP', 'IntIP', 'Label']

############################## functions ##############################

def build_rus(file, i, modeldata=None):

    print("Training rus model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    if modeldata != None:

        for past_file in filelist:

            if past_file == file:
                break

            past_data = pd.read_csv(past_file)

            print("Appending", len(past_data), "points from", past_file, 
                  "to training data", file = sys.stderr)

            training_set = pd.concat([training_set, past_data], 
                                     ignore_index=True)

    print("Building rus model on", file, "with", len(training_set), 
          "points", file = sys.stderr)

    X_train = training_set.drop(columns=drop_column_list)

    y_train = training_set['Label']

    # since training_set could be a concatenation of many files and
    # consume a lot of memory, drop it when it is no longer needed

    del training_set
    
    rus = RandomUnderSampler(random_state=i)

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # since X_train and y_train are created from training_set,
    # drop them for memory saving purposes when no longer needed

    del X_train
    del y_train

    clf = RandomForestClassifier(n_estimators=rf_estimators, random_state=i)

    model = clf.fit(X_resampled, y_resampled)

    return {'model' : model}


def build_ros(file, i, modeldata=None):

    print("Training ros model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    if modeldata != None:

        for past_file in filelist:

            if past_file == file:
                break

            past_data = pd.read_csv(past_file)

            print("Appending", len(past_data), "points from", past_file, 
                  "to training data", file = sys.stderr)

            training_set = pd.concat([training_set, past_data],        
                                     ignore_index=True)

    print("Building ros model on", file, "with", len(training_set), 
          "points", file = sys.stderr)

    X_train = training_set.drop(columns=drop_column_list)

    y_train = training_set['Label']

    # since training_set could be a concatenation of many files and
    # consume a lot of memory, drop it when it is no longer needed

    del training_set
    
    ros = RandomOverSampler(random_state=i)

    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # since X_train and y_train are created from training_set,
    # drop them for memory saving purposes when no longer needed

    del X_train
    del y_train

    clf = RandomForestClassifier(n_estimators=rf_estimators, random_state=i)

    model = clf.fit(X_resampled, y_resampled)

    return {'model' : model}


def build_rndoutl(file, i, modeldata=None):

    print("Training rndoutl model on", file, "with rand", i, 
          file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    scas0 = training_set[training_set['SCAS'] == 0]
    scas1 = training_set[training_set['SCAS'] == 1]

    # select the same number of points for fully supervised learning
    # as selected by active learning from the pool

    points = points_from_pool

    num_outliers = int(proportion * points)

    if len(scas1) < num_outliers:
        print("Too few SCAS outliers for the training data, including all", 
              len(scas1), "outliers", file = sys.stderr)
        num_outliers = len(scas1)

    idx = rng.choice(scas0.index, points - num_outliers, replace=False)
    data0 = scas0[scas0.index.isin(idx)]

    idx = rng.choice(scas1.index, num_outliers, replace=False)
    data1 = scas1[scas1.index.isin(idx)]

    data = pd.concat([data0, data1], ignore_index=True)

    temp = data[data['Label'] == 0]

    if len(temp) == 0:
        print("No points with label 0 in data", file = sys.stderr)
        return None

    temp = data[data['Label'] == 1]

    if len(temp) == 0:
        print("No points with label 1 in data", file = sys.stderr)
        return None

    if modeldata != None:

        data = pd.concat([data, modeldata['data']], ignore_index=True)

    print("Building rndoutl model on", file, "with", len(data), 
          "points", file = sys.stderr)

    X_train = data.drop(columns=drop_column_list)

    y_train = data['Label']

    clf = RandomForestClassifier(n_estimators=rf_estimators, random_state=i,
                                 class_weight='balanced')

    model = clf.fit(X_train, y_train)

    return {'model' : model, 'data' : data}


def build_altrad(file, i, modeldata=None):

    print("Training altrad model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    # if the model has not been provided to this function, 
    # build the seed that will be used for creating a new model

    if modeldata == None:

        idx = rng.choice(training_set.index, seed_size, replace=False)
        seed = training_set[training_set.index.isin(idx)]
        training_set = training_set.drop(idx)

        temp = seed[seed['Label'] == 0]

        if len(temp) == 0:
            print("No points with label 0 in seed", file = sys.stderr)
            return None

        temp = seed[seed['Label'] == 1]

        if len(temp) == 0:
            print("No points with label 1 in seed", file = sys.stderr)
            return None

    # build the pool

    idx = rng.choice(training_set.index, pool_size, replace=False)
    pool = training_set[training_set.index.isin(idx)]

    temp = pool[pool['Label'] == 0]

    if len(temp) == 0:
        print("No points with label 0 in pool", file = sys.stderr)
        return None

    temp = pool[pool['Label'] == 1]

    if len(temp) == 0:
        print("No points with label 1 in pool", file = sys.stderr)
        return None

    # if the model has not been provided to this function, 
    # create a new model from the seed

    if modeldata == None:

        print("Building new altrad model on", file, "with seed of", 
              seed_size, "points", file = sys.stderr)

        X_seed = seed.drop(columns=drop_column_list).to_numpy()
        y_seed = seed['Label'].to_numpy()

        clf = RandomForestClassifier(n_estimators=rf_estimators, 
                                     random_state=i,
                                     class_weight='balanced')

        model = ActiveLearner(
            estimator=clf,
            query_strategy=query_strategy,
            X_training=X_seed, y_training=y_seed
        )

    else:
        model = modeldata['model']

    # update the model which was either provided to this function or 
    # created from the seed, using the points from the pool

    print("Updating existing altrad model on", file, "with", 
          points_from_pool, "points from pool of", pool_size, "points", 
          file = sys.stderr)

    X_pool = pool.drop(columns=drop_column_list).to_numpy()
    y_pool = pool['Label'].to_numpy()

    for j in range(points_from_pool):
        
        index, X_instance = model.query(X_pool)
        y_instance = y_pool[index]
    
        model.teach(X_instance, y_instance)

        X_pool = np.delete(X_pool, index, axis=0)
        y_pool = np.delete(y_pool, index, axis=0)

    return {'model' : model}


def build_aloutl(file, i, modeldata=None):

    print("Training aloutl model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    scas0 = training_set[training_set['SCAS'] == 0]
    scas1 = training_set[training_set['SCAS'] == 1]

    # if the model has not been provided to this function, 
    # build the seed that will be used for creating a new model

    if modeldata == None:

        num_outliers = int(proportion * seed_size)

        if len(scas1) < num_outliers:
            print("Too few SCAS outliers for the seed, including all", 
                  len(scas1), "outliers", file = sys.stderr)
            num_outliers = len(scas1)

        idx = rng.choice(scas0.index, seed_size - num_outliers, replace=False)
        seed0 = scas0[scas0.index.isin(idx)]
        scas0 = scas0.drop(idx)

        idx = rng.choice(scas1.index, num_outliers, replace=False)
        seed1 = scas1[scas1.index.isin(idx)]
        scas1 = scas1.drop(idx)

        seed = pd.concat([seed0, seed1], ignore_index=True)

        temp = seed[seed['Label'] == 0]

        if len(temp) == 0:
            print("No points with label 0 in seed", file = sys.stderr)
            return None

        temp = seed[seed['Label'] == 1]

        if len(temp) == 0:
            print("No points with label 1 in seed", file = sys.stderr)
            return None

    # build the pool

    num_outliers = int(proportion * pool_size)

    if len(scas1) < num_outliers:
        print("Too few SCAS outliers for the pool, including all", 
              len(scas1), "outliers", file = sys.stderr)
        num_outliers = len(scas1)

    idx = rng.choice(scas0.index, pool_size - num_outliers, replace=False)
    pool0 = scas0[scas0.index.isin(idx)]

    idx = rng.choice(scas1.index, num_outliers, replace=False)
    pool1 = scas1[scas1.index.isin(idx)]

    pool = pd.concat([pool0, pool1], ignore_index=True)

    temp = pool[pool['Label'] == 0]

    if len(temp) == 0:
        print("No points with label 0 in pool", file = sys.stderr)
        return None

    temp = pool[pool['Label'] == 1]

    if len(temp) == 0:
        print("No points with label 1 in pool", file = sys.stderr)
        return None

    # if the model has not been provided to this function, 
    # create a new model from the seed

    if modeldata == None:

        print("Building new aloutl model on", file, "with seed of", 
              seed_size, "points", file = sys.stderr)

        X_seed = seed.drop(columns=drop_column_list).to_numpy()
        y_seed = seed['Label'].to_numpy()

        clf = RandomForestClassifier(n_estimators=rf_estimators, 
                                     random_state=i,
                                     class_weight='balanced')

        model = ActiveLearner(
            estimator=clf,
            query_strategy=query_strategy,
            X_training=X_seed, y_training=y_seed
        )

    else:
        model = modeldata['model']

    # update the model which was either provided to this function or 
    # created from the seed, using the points from the pool

    print("Updating existing aloutl model on", file, "with", 
          points_from_pool, "points from pool of", pool_size, "points", 
          file = sys.stderr)

    X_pool = pool.drop(columns=drop_column_list).to_numpy()
    y_pool = pool['Label'].to_numpy()

    for j in range(points_from_pool):
        
        index, X_instance = model.query(X_pool)
        y_instance = y_pool[index]
    
        model.teach(X_instance, y_instance)

        X_pool = np.delete(X_pool, index, axis=0)
        y_pool = np.delete(y_pool, index, axis=0)

    return {'model' : model}


def print_results(prefix, timestamp, labels, results):

    precision_list = []
    recall_list = []
    f1_list = []

    for iter in range(iterations):

        precision = precision_score(labels[iter], results[iter])
        recall = recall_score(labels[iter], results[iter])
        f1 = f1_score(labels[iter], results[iter])

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        tn, fp, fn, tp = confusion_matrix(labels[iter], results[iter]).ravel()

        print(prefix, timestamp, "Iteration", iter+1, 
              "Precision", precision, "Recall", recall, "F1-score", f1, 
              "TN", tn, "FP", fp, "FN", fn, "TP", tp,
              "Total", tn+fp+fn+tp)

    print(prefix, timestamp, "Avg_precision", np.mean(precision_list), 
                             "stddev", np.std(precision_list))
    print(prefix, timestamp, "Avg_recall", np.mean(recall_list), 
                             "stddev", np.std(recall_list))
    print(prefix, timestamp, "Avg_f1-score", np.mean(f1_list), 
                             "stddev", np.std(f1_list))


def print_usage():

    print("Usage:", sys.argv[0], "<datadir> <method> <update>", file = sys.stderr)
    print("Files in <datadir> must follow the name format dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv", file = sys.stderr)
    print("<method> = ros|rus|random-outlier|al-trad|al-outlier", file = sys.stderr)
    print("<update> = static|full|cumulative", file = sys.stderr)

############################## main program ##############################

start_time = process_time()

if len(sys.argv) != 4:
    print_usage()
    sys.exit(0)

build_map = { "rus" : build_rus, "ros" : build_ros, 
              "random-outlier" : build_rndoutl,
              "al-trad" : build_altrad, "al-outlier" : build_aloutl }

strategy = sys.argv[2]

if strategy in build_map:
    trainfunc = build_map[strategy]
else:
    print_usage()
    sys.exit(0)
    
update = sys.argv[3]

if update != "static" and update != "full" and update != "cumulative":
    print_usage()
    sys.exit(0)

datadir = sys.argv[1]

filelist = sorted(glob(datadir + '/dataset-*.csv'))

daily_regex = re.compile('dataset-(?P<date>(?P<month>[0-9]{4}-[0-9]{2})-(?P<dayofmonth>[0-9]{2}))\.csv$')
monthly_regex = re.compile('dataset-(?P<month>[0-9]{4}-[0-9]{2})\.csv$')

daily_files = list(filter(daily_regex.search, filelist))
monthly_files = list(filter(monthly_regex.search, filelist))

if len(daily_files) > 0 and len(monthly_files) > 0:

    print("Data file names must either follow the dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv format and not both", file = sys.stderr)

    sys.exit(0)

elif len(daily_files) > 0:

    print("Data file names are following the dataset-YYYY-MM-DD.csv format", file = sys.stderr)

    pool_size = daily_pool_size
    points_from_pool = daily_points

    filelist = daily_files

    daily_datasets = True

elif len(monthly_files) > 0:

    print("Data file names are following the dataset-YYYY-MM.csv format", file = sys.stderr)

    pool_size = monthly_pool_size
    points_from_pool = monthly_points

    filelist = monthly_files

    daily_datasets = False

else:

    print("Data file names are not following the dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv format", file = sys.stderr)

    sys.exit(0)


prediction_time = 0

rfmodel = []

labels = []
results = []

daily_labels = []
daily_results = []

for iter in range(iterations):

    rfmodel.append(None)

    labels.append(None)
    results.append(None)

    daily_labels.append(None)
    daily_results.append(None)


for k in range(len(filelist) - 1):

    training_file = filelist[k]
    test_file = filelist[k+1]

    if daily_datasets:

        timeinfo = daily_regex.search(test_file)
        timestamp = timeinfo.group('month')
        daily_timestamp = timeinfo.group('date')

        if k < len(filelist) - 2:

            next_file = filelist[k+2]

            timeinfo = daily_regex.search(next_file)

            if timeinfo.group('dayofmonth') == "01":
                monthly_report = True
            else:
                monthly_report = False

        else:
            monthly_report = True

    else:

        monthly_report = True

        timeinfo = monthly_regex.search(test_file)
        timestamp = timeinfo.group('month')

    print("Processing test file", test_file, file = sys.stderr)

    test_set = pd.read_csv(test_file)

    if strategy == "al-trad" or strategy == "al-outlier":
        X_test = test_set.drop(columns=drop_column_list).to_numpy()
        y_test = test_set['Label'].to_numpy()
    else:
        X_test = test_set.drop(columns=drop_column_list)
        y_test = test_set['Label']

    iter = 0
    rand = 0

    while iter < iterations:

        if update == "static":
            if k == 0:
                model = trainfunc(training_file, rand)
            else:
                model = rfmodel[iter]

        elif update == "full":
            model = trainfunc(training_file, rand)

        else:
            if k == 0:
                model = trainfunc(training_file, rand)
            else:
                model = trainfunc(training_file, rand, rfmodel[iter])

        if model == None:
            print("Training on", training_file, 
                  "failed for iteration", iter+1, file = sys.stderr)
            rand += 1
            continue

        rfmodel[iter] = model

        time1 = process_time()

        result = rfmodel[iter]['model'].predict(X_test)

        time2 = process_time()

        prediction_time += (time2 - time1)

        if daily_datasets:

            daily_labels[iter] = y_test
            daily_results[iter] = result

            if results[iter] is None:
                labels[iter] = y_test
                results[iter] = result
            else:
                labels[iter] = np.concatenate((labels[iter], y_test))
                results[iter] = np.concatenate((results[iter], result))

        else:

            labels[iter] = y_test
            results[iter] = result

        iter += 1
        rand += 1

    if daily_datasets:

        print_results("Day", daily_timestamp, daily_labels, daily_results)

    if monthly_report:

        print_results("Month", timestamp, labels, results)

        for iter in range(iterations):

            labels[iter] = None
            results[iter] = None


end_time = process_time()

print("Total CPU time: ", end_time - start_time, file = sys.stderr)
print("Prediction CPU time: ", prediction_time, file = sys.stderr)
