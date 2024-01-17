from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tuning import svmTuning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# GLOBAL VARS
prefix = "Mix_2_ALL_Features"
locations = ["middle", "topright", "bottomright", "bottomleft", "topleft"]
# augments = ["NA", "R1", "R2", "M1", "M2"]
augments = ["NA", "R1", "M1"]
augment_keys = {k:n for n,k in enumerate(augments)}
people = ["MP1", "MP2", "MP3"]

# WHOLE DATASET
location_variance = {}

# FN GET WHOLE DATASET
for person in people:
    location_variance[person] = dict()
    d = location_variance[person]
    for location in locations:
        d[location] = {}
        for augment in augments:
            _arr = np.load(f"{person}_{augment}_{location}_features.npy")
            scaler = StandardScaler().set_output(transform="pandas")
            # scaler = MinMaxScaler().set_output(transform="pandas")
            _arr = np.c_[scaler.fit_transform(_arr[:,:7]), _arr[:,7:]]
            # print(_arr)
            # assert _arr.shape == (10,2), f"{prefix}_{person}_{augment}_{location}.csv in wrong shape: {_arr.shape}"
            # print(len(_arr))
            _arr = np.c_[_arr, np.ones(_arr.shape[0])*augment_keys[augment]]

            assert not np.isnan(_arr[:,7:]).any(), f"contains nan in {prefix}_{person}_{augment}_{location}.csv"

            
            # print(_arr)
            d[location][augment] = _arr

# EXAMPLE: M1_ALL_AUGMENTS_ALL_LOCATIONS
def tune(X_train, y_train, X_test, y_test):
    '''tunes both with linear and rbf. For our purposese, only show rsvm_acc'''
    # maxes = np.argsort(dataset[:,102:-1])[-10:]
    # define classifers
    clf_lsvm = SVC(kernel = 'linear')
    clf_rsvm = SVC(kernel = 'rbf')
    
    # tune and print result (tuned_lsvm, tuned_rsvm, lsvm_acc, rsvm_acc)
    results = svmTuning(X_train, y_train, X_test, y_test, clf_lsvm, clf_rsvm, 3)
    # print(results[2:])
    return results

def get_dataset(d):
    '''return copy of the dataset'''
    dataset = []
    for loc in locations:
        for aug in augments:
            dataset.extend(d[loc][aug])
    return np.array(dataset)

def get_dataset_location_picky(d, l):
    '''returns leave one out based on location of the dataset'''
    train = []
    test = []
    for loc in locations:
        if l == loc: 
            for aug in augments:
                test.extend(d[loc][aug])
            continue
        for aug in augments:
            train.extend(d[loc][aug])
    return np.array(train), np.array(test)

# def cell1():
#     MP1 = get_dataset(location_variance["MP1"])
#     X = MP1[:,:-1]
#     y = MP1[:,-1]

#     for i in range(5):
#         for x in X[y == i]:
#             plt.plot(x[2:])
#         plt.title(augments[i])
#         plt.show()
#     # print(X.shape, y.shape)

#     # results = list()
#     # for i in range(10):
#     #     print(i)
#     #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # , random_state=42) # split train and test
#     #     # print(X_train.shape, y_train.shape)
#     #     results.append(tune(X_train, y_train, X_test, y_test))
#     # # results = results.np.array(results)
#     # print([x[3] for x in results])

#     results2 = list()
#     for loc in locations:
#         print(loc)
#         MP1_train, MP1_test = get_dataset_location_picky(location_variance["MP1"], loc)
#         X_train, X_test = MP1_train[:,2:-1], MP1_test[:,2:-1]
#         y_train, y_test = MP1_train[:,-1], MP1_test[:,-1]
#         results2.append(tune(X_train, y_train, X_test, y_test))
#     # results2 = results2.np.array(results)
#     # print([x[3] for x in results2])
#     return results2
from sklearn.metrics import confusion_matrix
def cell2(loc):
    '''raw just fft'''
    # same person same location different configuration
    print(loc)
    MP1_train, MP1_test = get_dataset_location_picky(location_variance["MP1"], loc)
    total_size = len(MP1_train) + len(MP1_test)
    # print(total_size, len(MP1_train), len(MP1_test))
    # print(0.3 * total_size)
    # debug plot features
    # for i in range(5):
    #     for x in X[y == i]:
    #         plt.plot(x)
    #     plt.title(augments[i])
    #     plt.show()
    # print(X.shape, y.shape)

    # results = list()
    # for i in range(max(1, n_features - 10),n_features):
    # for i in range(30):
    _r = list()
    confusions = list()
    # print(MP1_train, MP1_test)
    for a in range(10):
        # print(i,a)
        # X1 = PCA(n_components=i).fit(X).transform(X)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.3) # , random_state=42) # split train and test
        X_train, _, y_train, _ = train_test_split(MP1_train[:, :-1], MP1_train[:, -1],
                                                  train_size=0.5 * total_size / len(MP1_train))
        # print(0.3 * total_size)
        _, X_test, _, y_test = train_test_split(MP1_test[:, :-1], MP1_test[:, -1],
                                                test_size=0.1 * total_size / len(MP1_test))
        _, clf, _, acc = tune(X_train[:,7:], y_train, X_test[:,7:], y_test)
        _r.append(acc)
        confusions.append(confusion_matrix(y_test, clf.predict(y_test)))
    accuracy_results_percentage = [loc, np.average(_r), np.std(_r)]
    return accuracy_results_percentage, confusions
    # results.append(_r)
    # results = results.np.array(results)
    # print(results)
    # return results

def cell2_5(loc):
    '''raw fft + others'''
    # same person same location different configuration
    MP1_train, MP1_test = get_dataset_location_picky(location_variance["MP1"], loc)
    total_size = len(MP1_train) + len(MP1_test)

    # debug plot features
    # for i in range(5):
    #     for x in X[y == i]:
    #         plt.plot(x)
    #     plt.title(augments[i])
    #     plt.show()
    # print(X.shape, y.shape)

    # results = list()
    # for i in range(max(1, n_features - 10),n_features):
    # for i in range(30):
    _r = list()
    confusions = list()
    for a in range(10):
        # print(i,a)
        # X1 = PCA(n_components=i).fit(X).transform(X)
        X_train, _, y_train, _ = train_test_split(MP1_train[:, :-1], MP1_train[:, -1],
                                                  train_size=0.5 * total_size / len(MP1_train))
        # print(0.3 * total_size)
        _, X_test, _, y_test = train_test_split(MP1_test[:, :-1], MP1_test[:, -1],
                                                test_size=0.1 * total_size / len(MP1_test))
        _, clf, _, acc = tune(X_train[:,:], y_train, X_test[:,:], y_test)
        _r.append(acc)
        confusions.append(confusion_matrix(y_test, clf.predict(y_test)))
    accuracy_results_percentage = [loc, np.average(_r), np.std(_r)]
    return accuracy_results_percentage, confusions
    # results.append(_r)
    # results = results.np.array(results)
    # print(results)
    # return results

MAX_PERCENTAGE = 0.9
MAX_TIMES = 10
def cell3(people):
    '''For figure 1, gets accuracy based on % of dataset used for training all locations, only ffts'''
    accuracy_results = list()
    confusions = list()
    for percentage in np.arange(0.1, MAX_PERCENTAGE, 0.2):
        percentage = np.round(percentage, 1)
        _r = list()
        for _i in range(MAX_TIMES):
            X_train = list()
            X_test = list()
            y_train = list()
            y_test = list()
            for person in people:
                for loc in locations:
                    print(percentage, _i, loc)
                    loc_dataset, _ = get_dataset_location_picky(location_variance[person], loc)
                    _X_train, _X_test, _y_train, _y_test = train_test_split(loc_dataset[:,7:-1], loc_dataset[:,-1], test_size=.10, train_size=percentage) # , random_state=42) # split train and test
                    X_train.extend(_X_train)
                    X_test.extend(_X_test)
                    y_train.extend(_y_train)
                    y_test.extend(_y_test)
                # print(len(X_train), len(X_test), len(y_train), len(y_test))
                _, clf, _, acc = tune(X_train, y_train, X_test, y_test)
                _r.append(acc)
                confusions.append(confusion_matrix(y_test, clf.predict(X_test)))
        accuracy_results_percentage = [percentage, np.average(_r), np.std(_r)]
        accuracy_results.append(accuracy_results_percentage)
        # print(accuracy_results_percentage)
    return np.array(accuracy_results), confusions


def cell4(people):
    '''For figure 1, gets accuracy based on % of dataset used for training all locations, ffts + features raw'''
    accuracy_results = list()
    confusions = list()
    for percentage in np.arange(0.1, MAX_PERCENTAGE, 0.2):
        percentage = np.round(percentage, 1)
        _r = list()
        for _i in range(MAX_TIMES):
            X_train = list()
            X_test = list()
            y_train = list()
            y_test = list()
            for person in people:
                for loc in locations:
                    print(percentage, _i, loc)
                    loc_dataset, _ = get_dataset_location_picky(location_variance[person], loc)
                    _X_train, _X_test, _y_train, _y_test = train_test_split(np.c_[loc_dataset[:,:3], loc_dataset[:,7:-1]], loc_dataset[:,-1], test_size=.10, train_size=percentage) # , random_state=42) # split train and test
                    X_train.extend(_X_train)
                    X_test.extend(_X_test)
                    y_train.extend(_y_train)
                    y_test.extend(_y_test)
                # print(len(X_train), len(X_test), len(y_train), len(y_test))
                _, clf, _, acc = tune(X_train, y_train, X_test, y_test)
                _r.append(acc)
                confusions.append(confusion_matrix(y_test, clf.predict(X_test)))
        accuracy_results_percentage = [percentage, np.average(_r), np.std(_r)]
        accuracy_results.append(accuracy_results_percentage)
        # print(accuracy_results_percentage)
    return np.array(accuracy_results), confusions

def cell5(people, n):
    '''For figure 1, gets accuracy based on % of dataset used for training all locations, PCA(ffts,n) + features raw'''
    accuracy_results = list()
    for percentage in np.arange(0.1, MAX_PERCENTAGE, 0.2):
        percentage = np.round(percentage, 1)
        _r = list()
        for _i in range(MAX_TIMES):
            X_train = list()
            X_test = list()
            y_train = list()
            y_test = list()
            for person in people:
                for loc in locations:
                    print(percentage, _i, loc)
                    loc_dataset, _ = get_dataset_location_picky(location_variance[person], loc)
                    _X_train, _X_test, _y_train, _y_test = train_test_split(np.c_[loc_dataset[:,:3], PCA(n_components=n).fit(loc_dataset[:,7:-1]).transform(loc_dataset[:,7:-1])], loc_dataset[:,-1], test_size=.10, train_size=percentage) # , random_state=42) # split train and test
                    X_train.extend(_X_train)
                    X_test.extend(_X_test)
                    y_train.extend(_y_train)
                    y_test.extend(_y_test)
                # print(len(X_train), len(X_test), len(y_train), len(y_test))
                _r.append(tune(X_train, y_train, X_test, y_test)[3])
        accuracy_results_percentage = [percentage, np.average(_r), np.std(_r)]
        accuracy_results.append(accuracy_results_percentage)
        print(accuracy_results_percentage)
    return accuracy_results

def cell6(people):
    '''For figure 1, gets accuracy based on % of dataset used for training all locations, StandardScaler(fft) + features raw'''
    accuracy_results = list()
    confusions = list()
    for percentage in np.arange(0.1, MAX_PERCENTAGE, 0.2):
        percentage = np.round(percentage, 1)
        _r = list()
        for _i in range(MAX_TIMES):
            X_train = list()
            X_test = list()
            y_train = list()
            y_test = list()
            for person in people:
                for loc in locations:
                    print(percentage, _i, loc)
                    loc_dataset, _ = get_dataset_location_picky(location_variance[person], loc)
                    highest_powers = np.argsort(loc_dataset[:, 7:-1], axis=1)
                    # _X_train, _X_test, _y_train, _y_test = train_test_split(np.c_[loc_dataset[:,:3], StandardScaler().fit(loc_dataset[:,7:-1]).transform(loc_dataset[:,7:-1])], loc_dataset[:,-1], test_size=.10, train_size=percentage) # , random_state=42) # split train and test
                    _X_train, _X_test, _y_train, _y_test = train_test_split(
                            np.c_[
                                loc_dataset[:,:7],
                                highest_powers,
                                # StandardScaler().fit(loc_dataset[:,7:-1]).transform(loc_dataset[:,7:-1])],
                            # PCA(n_components=120).fit(loc_dataset[:,:]).transform(loc_dataset[:,:]),
                            ],
                            loc_dataset[:,-1],
                            test_size=.10,
                            train_size=percentage
                    ) # , random_state=42) # split train and test
                    X_train.extend(_X_train)
                    X_test.extend(_X_test)
                    y_train.extend(_y_train)
                    y_test.extend(_y_test)
                # print(len(X_train), len(X_test), len(y_train), len(y_test))
                _, clf, _, acc = tune(X_train, y_train, X_test, y_test)
                _r.append(acc)
                confusions.append(confusion_matrix(y_test, clf.predict(X_test)))
        accuracy_results_percentage = [percentage, np.average(_r), np.std(_r)]
        accuracy_results.append(accuracy_results_percentage)
        # print(accuracy_results_percentage)
    return np.array(accuracy_results), confusions

def cell7(people):
    '''For figure 1, gets accuracy based on % of dataset used for training all locations, StandardScaler(fft) + features raw'''
    accuracy_results = list()
    confusions = list()
    for percentage in np.arange(0.1, MAX_PERCENTAGE, 0.2):
        percentage = np.round(percentage, 1)
        _r = list()
        for _i in range(MAX_TIMES):
            X_train = list()
            X_test = list()
            y_train = list()
            y_test = list()
            for person in people:
                for loc in locations:
                    print(percentage, _i, loc)
                    loc_dataset, _ = get_dataset_location_picky(location_variance[person], loc)
                    super_sum = np.sum(loc_dataset[:,7:-1], axis=1)
                    super_std = np.sum(loc_dataset[:, 7:-1], axis=1)
                    highest_powers = np.argsort(loc_dataset[:, 7:-1], axis=1)
                    # print(np.sum(loc_dataset[:,7:-1], axis=1))
                    _X_train, _X_test, _y_train, _y_test = train_test_split(np.c_[loc_dataset[:,:3], super_sum, super_std, highest_powers], loc_dataset[:,-1], test_size=.10, train_size=percentage) # , random_state=42) # split train and test
                    X_train.extend(_X_train)
                    X_test.extend(_X_test)
                    y_train.extend(_y_train)
                    y_test.extend(_y_test)
                # print(len(X_train), len(X_test), len(y_train), len(y_test))
                _, clf, _, acc = tune(X_train, y_train, X_test, y_test)
                _r.append(acc)
                confusions.append(confusion_matrix(y_test, clf.predict(X_test)))
        accuracy_results_percentage = [percentage, np.average(_r), np.std(_r)]
        accuracy_results.append(accuracy_results_percentage)
        # print(accuracy_results_percentage)
    return np.array(accuracy_results), confusions

if __name__ == "__main__":
    MP1 = get_dataset(location_variance["MP1"])
    X = MP1[:,:-1]
    y = MP1[:,-1]

    # for i in range(5):
    #     for x in X[y == i]:
    #         plt.plot(x[7:])
    #     plt.title(augments[i])
    #     plt.show()
    # results = cell1()
    # plt.scatter(range(5),[x[3] for x in results])
    # plt.title("rbf with increasing PCA components in (middle, tr, br, bl, tl)")
    # plt.savefig("rbf_accuracy_fft_only")
    # plt.show()
    # for loc in locations:
    #     results = cell2(loc, 30)
    #     plt.plot([sum([a[3] for a in x]) / len(x) for x in results])
    #     plt.title(f"rbf with increasing PCA components in direction {loc}")
    #     plt.savefig(f"rbf_accuracy_loc_{loc}")
    #     plt.show()
    # ppl = ["MP1", "MP2", "MP3"]

    ppl = ["MP1"] # ppl being tested

    # for different PCA parameters
    # results_all_pca_fft = [np.array(cell5(ppl, n)) for n in [50,100,150,200]]
    # results_all_fft_s, confusions_fft_s = cell7(ppl)
    results_all_fft_scaled, confusions_fft_scaled = cell6(ppl)
    # results_all_raw, confusions_all_raw = cell4(ppl)
    # results_fft, confusions_fft = cell3(ppl)
    # x_pos = np.arange(len(results_fft))
    x_pos = np.arange(len(results_all_fft_scaled))
    fig, ax = plt.subplots()

    # confusions_fft_s = np.average(confusions_fft_s, axis=0)
    confusions_fft_scaled = np.average(confusions_fft_scaled, axis=0)
    # confusions_all_raw = np.average(confusions_all_raw, axis=0)
    # confusions_fft = np.average(confusions_fft, axis=0)

    # print(confusions_fft_s)
    print(confusions_fft_scaled)
    # print(confusions_all_raw)
    # print(confusions_fft)

    # ax.bar(x_pos-0.1, results_fft[:,1], yerr=results_fft[:,2], alpha=0.5, width=0.1, align='center', label="FFT")
    # ax.bar(x_pos, results_all_raw[:,1], yerr=results_all_raw[:,2], alpha=0.5, width=0.1, align='center', label="FFT with exp decay")
    ax.bar(x_pos+0.1, results_all_fft_scaled[:,1], yerr=results_all_fft_scaled[:,2], alpha=0.5, width=0.1, align='center', label="StandardScaler(FFT) with exp decay")
    # ax.bar(x_pos+0.2, results_all_fft_s[:, 1], yerr=results_all_fft_s[:, 2], alpha=0.5, width=0.1, align='center', label="S")
    # for i,v in enumerate(results_fft):
    #     ax.text(i - 0.1, v + 0.03, str(v), color='black', fontweight='bold')
    # for i,v in enumerate(results_all_raw):
    #     ax.text(i, v + 0.03, str(v), color='black', fontweight='bold')
    # for i,v in enumerate(results_all_fft_scaled):
    #     ax.text(i + 0.1, v + 0.03, str(v), color='black', fontweight='bold')
    # for different PCA parameters
    # for n in range(4):
    #     ax.bar(x_pos+((n + 1) * 0.1), results_all_pca_fft[n][:,1], yerr=results_all_pca_fft[n][:,2], alpha=0.5, width=0.1, align='center', label=f"PCA(FFT, n={(n + 1) * 50}) with others")

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('% of dataset used for training')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in results_all_fft_scaled[:,0]])
    ax.set_title('Accuracy Comparison (Traditional FFT vs FFT + oscilation features)\nwith 10% used for testing')
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True)
    plt.legend()
    plt.show()
    
    # results_per_loc_fft = list()
    # for loc in locations:
    #     results_per_loc_fft.append(cell2(loc))
    #
    # results_per_loc_all = list()
    # for loc in locations:
    #     results_per_loc_all.append(cell2_5(loc))
    # print(results_per_loc_fft, results_per_loc_all)
    # fig, ax = plt.subplots()
    # ax.bar(np.arange(len(results_per_loc_fft)), [x[1] for x in results_per_loc_fft], yerr=[x[2] for x in results_per_loc_fft], alpha=0.5, width=0.1, align="center", label="fft only")
    # ax.bar(np.arange(len(results_per_loc_all)) + 0.1, [x[1] for x in results_per_loc_all], yerr=[x[2] for x in results_per_loc_all], alpha=0.5, width=0.1, align="center", label="fft with exponential decay parameters")
    # ax.set_ylabel('Accuracy')
    # ax.set_xlabel('Direction')
    # ax.set_xticks(np.arange(len(results_per_loc_fft)))
    # ax.set_xticklabels([str(x[0]) for x in results_per_loc_fft])
    # ax.set_title('Accuracy Comparison (Traditional FFT vs FFT + oscilation features)\nwith 50% training/ 10% testing for Each Direction')
    # ax.set_ylim(0.0, 1.0)
    # plt.legend()
    # plt.show()
