import json

def scquery():
    scquery_test = json.load(open('../data/scquery/test/scquery_test.json'))
    scq_test = {'num_samples': [], 'users': [], 'user_data': {}}
    scq_test['num_samples'] = scquery_test['num_samples'][-10:]
    scq_test['users'] = scquery_test['users'][-10:]
    X_test = list(scquery_test['user_data'].keys())[-10:]
    scq_test['user_data'] = {key: value for key, value in scquery_test['user_data'].items() if key in X_test}

    with open('../dg_data/scquery/test/scquery.json', 'w') as outfile:
        json.dump(scq_test, outfile)

def pbmc():
    pbmc_test = json.load(open('../data/pbmc/test/pbmc_test.json'))
    pbc_test = {'num_samples': [], 'users': [], 'user_data': {}}
    pbc_test['num_samples'] = pbmc_test['num_samples'][-3:]
    pbc_test['users'] = pbmc_test['users'][-3:]
    X_test = list(pbmc_test['user_data'].keys())[-3:]
    pbc_test['user_data'] = {key: value for key, value in pbmc_test['user_data'].items() if key in X_test}

    with open('../dg_data/pbmc/test/pbmc.json', 'w') as outfile:
        json.dump(pbc_test, outfile)


def pancreas():
    for dataset_type in range(6):
        with open('../data/pancreas_{}/test/pancreas{}_test.json'.format(dataset_type,dataset_type), 'r') as f:
            pancreas_test = json.load(open(f.name))
            pan_test = {'num_samples': [], 'users': [], 'user_data': {}}
            pan_test['num_samples'] = pancreas_test['num_samples'][-2:]
            pan_test['users'] = pancreas_test['users'][-2:]
            X_test = list(pancreas_test['user_data'].keys())[-2:]
            pan_test['user_data'] = {key: value for key, value in pancreas_test['user_data'].items() if key in X_test}

        with open('../dg_data/pancreas_{}/test/pancreas{}_test.json'.format(dataset_type, dataset_type), 'w') as outfile:
            json.dump(pan_test, outfile)


if __name__ == '__main__':
    pancreas()
