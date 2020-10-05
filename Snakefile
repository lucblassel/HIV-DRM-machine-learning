import bz2
import pickle
import pandas as pd
import json
import os.path
from datetime import datetime
from utils_hiv import data_utils, param_utils, learning_utils

repeat_training = {
    'RF': True,
    'Complement': True
    }

no_params = {'Bayes', 'Complement'}

FINAL_REPEATS = config['num_final_repeats']
METRIC = config['metric']
REMOVE_NAIVE_DRMS = config['remove_naive_DRMS']
BALANCE = config['balance']
EXTERNALS = config['external_set']
TEST_SETS = list(config['external_set'].keys()) + ['test']
CROSS_DIR = config.get('cross_dir', 'cross_predictions')

PARAMS = config.get('parameters', {})
DIR = config.get('results_dir', 
    f"results_remove_{config['remove_drms']}_seqs_{config['remove_sequences']}"
)

def add_wild(path, target='{target}', subtype='{subtype}'):
    return os.path.join(DIR, subtype, target, path)

def add_static(path, target='{{target}}', subtype='{{subtype}}'):
    return os.path.join(DIR, subtype, target, path)

def parse_distrib(name, distrib):
    toReturn = "{}: ".format(name)
    if distrib[0]:
        toReturn += "{} ({}, {})".format(distrib[-1], *distrib[1:3])
    else:
        if len(distrib[1:]) > 1:
            toReturn += "choice from {}".format(distrib[1:])
        else:
            toReturn += "{}".format(distrib[1])
    return toReturn

def parse_params(params_path, models):
    params_tot = json.load(open(params_path, 'r'))
    tot = []
    for model, params in params_tot.items():
        if model in models:
            strings = []
            for name, distrib in params.items():
                strings.append(parse_distrib(name, distrib))
            s = "\t\t- {}: \n\t\t\t- {}".format(model, "\n\t\t\t- ".join(strings))
            tot.append(s)
    return '\n'.join(tot)

def print_summary(config_in=config):
    config_copy = {k:v for (k, v) in config_in.items()}
    keys = ['results_dir']

    string = 100 * "=" + "\n"
    string += "Execution starting at: {}\n".format(
        datetime.now())

    datasets = ['train_set', 'test_set', 'external_set', 'metadata']
    keys.extend(datasets)

    string += "\n Input files:\n\n"
    for data_type in datasets:
        string += "\t- {}: {}\n".format(
            data_type, config_copy[data_type])

    string += "\n\n Model(s) trained:\n\n"
    for model in config_copy['model']:
        string += "\t- {}\n".format(model)

    string += "\n\n Subtype(s) used:\n\n"
    for subtype in config_copy['subtype']:
        string += "\t- {}\n".format(subtype)

    keys.extend(['model', 'subtype'])

    string += "\n\n Final model trained {} times.\n".format(
        config_copy['num_final_repeats'])
    keys.extend(['num_final_repeats'])

    for key in keys:
        config_copy.pop(key)

    if config_copy != {}:
        string += "\n\n configuration configuration:\n\n"
        for k, v in config_copy.items():
            string += "\t- {}: {}\n".format(k, v)

    return string

def check_homogeneous(*files):
    cols = set()
    basefile = ''
    for file in files:
        df = pd.read_csv(file, sep='\t', header=0, index_col=0, nrows=3)
        cols = set(df.columns) if cols == set() else cols
        basefile = file if basefile == '' else basefile
        if set(df.columns) != cols:
            raise ValueError(f"{basefile} and {file} not homogeneous.")

summary = os.path.join(DIR, 'summary.log')
env = os.path.join(DIR, 'env_versions.log')

# HANDLERS
onstart:
    print('executing start handler')
    os.makedirs(DIR, exist_ok=True)
    os.makedirs(os.path.join(DIR, '.logs'), exist_ok=True)
    shell('rm -f no_cross_training.txt')
    shell('conda list > {}'.format(env))

    with open(summary, 'a') as out:
        out.write(print_summary())
    check_homogeneous(config['train_set'], config['test_set'], *config['external_set'].values())

onsuccess:
    with open(summary, 'a') as out:
        out.write("\n\nFinished at: {}\n".format(
            datetime.now()))
        out.write(100 * "=" + "\n")

onerror:
    with open(summary, 'a') as out:
        out.write('\n\n### FINISHED WITH ERROR at: {} ###\n'.format(
            datetime.now()))
        out.write(100 * "=" + "\n")

#############################
####### MAIN PIPELINE #######
#############################
        
rule all:
    input:
        repeat_results = expand(
            add_wild('final-{model}-{data}-predictions-num{repeat}.tsv'),
            data=TEST_SETS,
            target=config['target'],
            subtype=config['subtype'],
            model=[m for m in config['model'] if repeat_training.get(m, False)],
            repeat=range(FINAL_REPEATS)
            ),
        single_results = expand(
            add_wild('final-{model}-{data}-predictions-num{repeat}.tsv'),
            data=TEST_SETS,
            target=config['target'],
            subtype=config['subtype'],
            model=[m for m in config['model'] if not repeat_training.get(m, False)],
            repeat=range(1)
        )
    params:
        mem = 100,
        log_dir = os.path.join(DIR, '.logs')

rule all_cross_training:
    input:
        repeated = expand(
            os.path.join(DIR, CROSS_DIR,
            '{data}-{model}-from-{from_target}-{from_subtype}-to-{to_target}-num{repeat}.tsv'),
            data=TEST_SETS,
            model=[m for m in config['model'] if repeat_training.get(m, False)], 
            from_target=config['target'], 
            to_target=config['target'],
            from_subtype=config['subtype'],
            repeat=range(FINAL_REPEATS)
        ),
        not_repeated = expand(
            os.path.join(DIR, CROSS_DIR,
            '{data}-{model}-from-{from_target}-{from_subtype}-to-{to_target}-num{repeat}.tsv'),
            data=TEST_SETS, 
            model=[m for m in config['model'] if not repeat_training.get(m, False)],
            from_target=config['target'],
            to_target=config['target'],
            from_subtype=config['subtype'],
            repeat=range(1)
        )
    params:
        mem = 100,
        log_dir = os.path.join(DIR, '.logs')
    output:
        output = 'cross_training.txt'
    shell:
        '''
        touch {output.output}
        '''

rule subset_data:
    input:
        config_train = config['train_set'],
        config_test = config['test_set'],
        config_external = config['external_set'].values(),
        metadata = config['metadata']
    params:
        mem = 5000,
        log_dir = os.path.join(DIR, '.logs'),
        subtype = lambda wildcards: wildcards.subtype,
    threads: 1
    output:
        train = add_wild('train-set.tsv', target=''),
        test = add_wild('test-set.tsv', target=''),
        external = expand(add_static('{external}-set.tsv', target=''), external=EXTERNALS.keys())
    run:
        train = data_utils.choose_subtype(
            input.config_train, input.metadata, params.subtype)
        test = data_utils.exclude_subtype(
            input.config_test, input.metadata, params.subtype)
        # test = pd.read_csv(input.config_test, sep='\t', header=0, index_col=0)
        externals = {k:pd.read_csv(v, sep='\t', header=0, index_col=0) for k,v in EXTERNALS.items()}
        if config['remove_consensus']:
            train = data_utils.remove_consensus(train, config['subtype'])
            test = data_utils.remove_consensus(test, config['subtype'])
            externals = {k:data_utils.remove_consensus(v, config['subtype']) for k,v in externals.items()}
        if config['deep_clean']:
            train = data_utils.deep_clean_features(train)
            test = data_utils.deep_clean_features(test)
            externals = {k:data_utils.deep_clean_features(v) for k,v in externals.items()}

        train = data_utils.sequence_removal_wrapper(train, config['remove_sequences'])

        train = data_utils.removal_wrapper(train, config['remove_drms'])
        test = data_utils.removal_wrapper(test, config['remove_drms'])
        externals = {k:data_utils.removal_wrapper(v, config['remove_drms']) for k,v in externals.items()}

        train.to_csv(output.train, sep='\t')
        test.to_csv(output.test, sep='\t')
        for data, filename in zip(externals.values(), output.external):
            data.to_csv(filename, sep='\t')

rule train_whole_model:
    input:
        train_set = add_wild('train-set.tsv', target='')
    params:
        mem = 15000,
        log_dir = os.path.join(DIR, '.logs'),
        model = lambda wildcards: wildcards.model,
        target = lambda wildcards: wildcards.target,
        subtype = lambda wildcards: wildcards.subtype,
        parameters = PARAMS
    threads: 4
    wildcard_constraints:
        model = "(?!glmnet).*"
    output:
        model = add_wild('final-{model}-num{repeat}.pkl.bz2'),
        coefficients =add_wild('final-{model}-coefficients-num{repeat}.tsv')
    run:
        parameters = params.parameters.get(params.model, {})
        model, coefs = learning_utils.train_model(
            params.model, input.train_set, parameters, params.target, params.subtype, 
            config['remove_drms'], config['remove_sequences'], BALANCE)
        coefs.to_csv(output.coefficients, sep='\t')

        with bz2.BZ2File(output.model, 'w') as outfile:
            pickle.dump(model, outfile)

rule get_predictions:
    input:
        model = add_wild('final-{model}-num{repeat}.pkl.bz2'),
        test_set = add_wild('test-set.tsv', target=''),
        external_sets = expand(add_static('{external}-set.tsv', target=''), external=EXTERNALS.keys())
    params:
        mem = 15000,
        log_dir = os.path.join(DIR, '.logs'),
        target = lambda wildcards: wildcards.target
    threads: 4
    output:
        predictions_test = add_wild(
            'final-{model}-test-predictions-num{repeat}.tsv'),
        predictions_externals = expand(add_static(
            'final-{{model}}-{external}-predictions-num{{repeat}}.tsv'), external=EXTERNALS.keys())
    run:
        with bz2.BZ2File(input.model, 'r') as infile:
            model = pickle.load(infile)

        test_df = learning_utils.get_predictions(model, input.test_set, params.target, BALANCE)
        test_df.to_csv(output.predictions_test, sep='\t')

        for ext, preds in zip(input.external_sets, output.predictions_externals):
            external_df = learning_utils.get_predictions(model, ext, params.target, BALANCE)
            external_df.to_csv(preds, sep='\t')

rule train_glmnet:
    input:
        train_set = add_wild('train-set.tsv', target=''),
        test_set = add_wild('test-set.tsv', target=''),
        external_set = expand(add_static('{external}-set.tsv', target=''), external=EXTERNALS.keys()),
        script = 'scripts/caret_regression.r'
    params:
        mem = 3500,
        log_dir = os.path.join(DIR, '.logs'),
        metric = METRIC,
        target = lambda wildcards: wildcards.target,
        balance = "--balance" if BALANCE else ""
    threads: 4
    output:
        model = add_wild('final-glmnet-num{repeat}.rds'),
        coefficients =add_wild('final-glmnet-coefficients-num{repeat}.tsv'),
        performance_test = add_wild(
            'final-glmnet-test-results-num{repeat}.json'),
        predictions_test = add_wild(
            'final-glmnet-test-predictions-num{repeat}.tsv'),
        performance_external = expand(add_static(
            'final-glmnet-{external}-results-num{{repeat}}.json'), external=EXTERNALS.keys()),
        predictions_external = expand(add_static(
            'final-glmnet-{external}-predictions-num{{repeat}}.tsv'), external=EXTERNALS.keys())
    shell:
        '''
        Rscript --vanilla {input.script}\
            --train {input.train_set}\
            --test {input.test_set}\
            --external {input.external_set}\
            --output_test {output.performance_test}\
            --output_external {output.performance_external}\
            --pred_test {output.predictions_test}\
            --pred_external {output.predictions_external}\
            --model {output.model}\
            --coefficients {output.coefficients}\
            --metric {params.metric}\
            --target {params.target}\
            --classify\
            {params.balance}
        '''

rule cross_train:
    input:
        model = add_wild('final-{model}-num{repeat}.pkl.bz2', subtype='{from_subtype}', target='{from_target}'),
        data = add_wild('{data}-set.tsv', subtype='{from_subtype}', target='{to_target}')
    params:
        mem = 15000,
        log_dir = os.path.join(DIR, '.logs'),
        target = lambda wildcards: wildcards.to_target
    threads: 4
    wildcard_constraints:
        model = "(?!glmnet).*"
    output:
        performance = os.path.join(DIR, CROSS_DIR, '{data}-{model}-from-{from_target}-{from_subtype}-to-{to_target}-num{repeat}.tsv')
    run:
        with bz2.BZ2File(input.model, 'r') as infile:
            model = pickle.load(infile)
        test_df = learning_utils.get_predictions(model, input.data, params.target, config['balance'])
        test_df.to_csv(output.performance, sep='\t', index=True, header=True)


rule cross_train_glmnet:
    input:
        model = add_wild('final-glmnet-num{repeat}.rds', subtype='{from_subtype}', target='{from_target}'),
        data = add_wild('{data}-set.tsv', subtype='{from_subtype}', target='{to_target}'),
        script = 'scripts/caret_cross_prediction.r'
    params:
        mem = 3500,
        log_dir = os.path.join(DIR, '.logs'),
        target = lambda wildcards: wildcards.to_target
    threads: 4
    output:
        performance = os.path.join(DIR, CROSS_DIR, '{data}-glmnet-from-{from_target}-{from_subtype}-to-{to_target}-num{repeat}.tsv')
    shell:
        '''
        Rscript --vanilla {input.script}\
            --model {input.model}\
            --test {input.data}\
            --target {params.target}\
            --output {output.performance}
        '''

