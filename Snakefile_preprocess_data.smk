
import re
import os
import pandas as pd

if config == dict():
    config = {
        'data_dir': os.path.join('data_pre_split', 'HIVDB_data'),
        'metadata_dir': 'data_pre_split',
        'encoder': os.path.join('utils_hiv', 'utils_hiv', 'data_encoder.py'),
        'output_dir': os.path.join('data_pre_split', 'new_target'),
        'countries': ['Africa', 'UK']
    }

DIR = config.get('data_dir', os.path.join('data_pre_split', 'HIVDB_data'))
METADIR = config.get('metadata_dir', 'data_pre_split')
OUT = config.get('output_dir')

print(config)

rule all:
    input: 
        expand(
        os.path.join(OUT, "{country}-data.tsv"),
        country=config['countries']
    )

rule process_data:
    input:
        naive_seqs = os.path.join(DIR,"{country}", "PrettyRTAA_naive.tsv"),
        treated_seqs = os.path.join(DIR,"{country}", "PrettyRTAA_treated.tsv"),
        naive_res = os.path.join(DIR,"{country}", "ResistanceSummary_naive.tsv"),
        treated_res = os.path.join(DIR,"{country}", "ResistanceSummary_treated.tsv"),
        metadata = os.path.join(METADIR, "{country}-metadata.tsv"),
        script = config['encoder']
    output: temp(os.path.join(DIR, "{country}", "encoded_data.tsv"))
    shell: 
        """
        python3 {input.script} \
            --naive {input.naive_seqs} \
            --treated {input.treated_seqs} \
            --metadata {input.metadata} \
            --resistance {input.naive_res} {input.treated_res} \
            --outfile {output}
        """

rule homogenize_data:
    input: 
        expand(
            os.path.join(DIR, "{country}", "encoded_data.tsv"),
            country=config['countries']
        )
    output: 
        expand(
            os.path.join(OUT, "{country}-data.tsv"),
            country=config['countries']
        )
    run:
        regex_in = r".*/(?P<country>[^/]+)/encoded_data\.tsv$"
        regex_out = r"^.*/(?P<country>[^/]+)-data\.tsv$"
        data = {}
        for input_file in input:
            country = re.match(regex_in, input_file).group('country')
            data[country] = pd.read_csv(input_file, sep='\t').set_index('inputSequence')

        cols = pd.concat([dataset.iloc[:2] for dataset in data.values()], axis=0).columns

        for output_path in output:
            country = re.match(regex_out, output_path).group('country')
            (data[country]
                .reindex(cols)
                .fillna(0)
                .to_csv(output_path, header=True, index=True))