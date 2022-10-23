import json
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)
def scquery():
    scquery_train = {'num_samples': [], 'users': [], 'user_data': {}}
    scquery_test = {'num_samples': [], 'users': [], 'user_data': {}}
    dataset_type_list = ['train', 'test', 'valid']
    true_id = 0
    for dataset_type in dataset_type_list:
        with np.load('data/raw_data/{}_scquery.npz'.format(dataset_type), allow_pickle=True) as f:
            features = f['features']
            labels = f['labels']
            accessions = f['accessions']
            acc_name_list = list(set(accessions))
            index = [np.where(f['accessions'] == acc)[0] for acc in list(acc_name_list)]

            for acc_id, acc_name in enumerate(acc_name_list):
                user_X = features[index[acc_id]]
                user_y = labels[index[acc_id]]
                if user_y.shape[0] > 1000 or user_y.shape[0] < 10:
                    continue

                X_train, X_test, y_train, y_test = train_test_split(user_X, user_y, test_size=0.2,
                                                                    random_state=42)

                X_train = X_train.tolist()
                X_test = X_test.tolist()
                y_train = y_train.tolist()
                y_test = y_test.tolist()
                user_name = 'f_{}'.format(true_id)
                true_id += 1
                scquery_train['users'].append(user_name)
                scquery_train['num_samples'].append(len(y_train))
                scquery_train['user_data'][user_name] = {'X': X_train, 'y': y_train}

                scquery_test['users'].append(user_name)
                scquery_test['num_samples'].append(len(y_test))
                scquery_test['user_data'][user_name] = {'X': X_test, 'y': y_test}

                print('{} finished'.format(true_id))

    with open('data/scquery/train/scquery_train.json', 'w') as outfile:
        json.dump(scquery_train, outfile)

    with open('data/scquery/test/scquery_test.json', 'w') as outfile:
        json.dump(scquery_test, outfile)

def pancreas():
    for dataset_type in range(6):
        pancreas_train = {'num_samples': [], 'users': [], 'user_data': {}}
        pancreas_test = {'num_samples': [], 'users': [], 'user_data': {}}
        with np.load('data/raw_data/pancreas{}.npz'.format(dataset_type), allow_pickle=True) as f:
            features = f['features']
            labels = f['labels']
            accessions = f['accessions']
            acc_name_list = list(set(accessions))
            index = [np.where(f['accessions'] == acc)[0] for acc in list(acc_name_list)]
            for acc_id, acc_name in enumerate(acc_name_list):
                user_X = features[index[acc_id]]
                user_y = labels[index[acc_id]]
                if user_y.shape[0] > 1000:
                    random_idx = np.random.choice(range(user_y.shape[0]), size=int(user_y.shape[0] / 10), replace=False)
                    user_X = user_X[random_idx]
                    user_y = user_y[random_idx]

                X_train, X_test, y_train, y_test = train_test_split(user_X, user_y, test_size=0.2,
                                                                        random_state=42)
                X_train = X_train.tolist()
                X_test = X_test.tolist()
                y_train = y_train.tolist()
                y_test = y_test.tolist()
                user_name = 'f_{}'.format(acc_id)
                pancreas_train['users'].append(user_name)
                pancreas_train['num_samples'].append(len(y_train))
                pancreas_train['user_data'][user_name] = {'X': X_train, 'y': y_train}

                pancreas_test['users'].append(user_name)
                pancreas_test['num_samples'].append(len(y_test))
                pancreas_test['user_data'][user_name] = {'X': X_test, 'y': y_test}

        with open('data/pancreas_{}/train/pancreas{}_train.json'.format(dataset_type, dataset_type), 'w') as outfile:
            json.dump(pancreas_train, outfile)

        with open('data/pancreas_{}/test/pancreas{}_test.json'.format(dataset_type, dataset_type), 'w') as outfile:
            json.dump(pancreas_test, outfile)

        print('pancreas_{} finished'.format(dataset_type))

def pbmc():
    pbmc_train = {'num_samples': [], 'users': [], 'user_data': {}}
    pbmc_test = {'num_samples': [], 'users': [], 'user_data': {}}
    with np.load('data/raw_data/pbmc.npz', allow_pickle=True) as f:
        features = f['features']
        labels = f['labels']
        accessions = f['accessions']
        acc_name_list = list(set(accessions))
        index = [np.where(f['accessions'] == acc)[0] for acc in list(acc_name_list)]
        for acc_id, acc_name in enumerate(acc_name_list):
            user_X = features[index[acc_id]]
            user_y = labels[index[acc_id]]
            if user_y.shape[0] > 1000:
                random_idx = np.random.choice(range(user_y.shape[0]), size=int(user_y.shape[0]/10), replace=False)
                user_X = user_X[random_idx]
                user_y = user_y[random_idx]
            X_train, X_test, y_train, y_test = train_test_split(user_X, user_y, test_size=0.2,
                                                                    random_state=42)

            X_train = X_train.tolist()
            X_test = X_test.tolist()
            y_train = y_train.tolist()
            y_test = y_test.tolist()
            user_name = 'f_{}'.format(acc_id)
            pbmc_train['users'].append(user_name)
            pbmc_train['num_samples'].append(len(y_train))
            pbmc_train['user_data'][user_name] = {'X': X_train, 'y': y_train}

            pbmc_test['users'].append(user_name)
            pbmc_test['num_samples'].append(len(y_test))
            pbmc_test['user_data'][user_name] = {'X': X_test, 'y': y_test}

            print('{}/{} finished'.format(acc_id+1, len(acc_name_list)))

    with open('data/pbmc/train/pbmc_train.json', 'w') as outfile:
        json.dump(pbmc_train, outfile)

    with open('data/pbmc/test/pbmc_test.json', 'w') as outfile:
        json.dump(pbmc_test, outfile)


if __name__ == '__main__':
    pancreas()