import re
import os
import time
import numpy as np
import csv
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
# from IPython.display import Audio
from datetime import datetime

VAMBN_PATH = "/path/to/vambn-extensions-evaluations"
# VAMBN_PATH = "/path/to/Documents/vambn-extensions-evaluations"
import sys
sys.path.append(VAMBN_PATH)
sys.path.append(VAMBN_PATH + "/HI-VAE/")
import helpers   # Your modified HIVAE helper functions


###########################################################
# Build settings string for each module
###########################################################
def set_settings(opts, nepochs=500, modload=False, save=True):
    """
    Build CLI settings for OU-VAMBN-MT.
    - long modules → include --n_vis
    - stalone     → no --n_vis
    """

    inputf = opts['vargroups'].iloc[0] + '_VIS00'
    missf  = inputf + '_missing.csv'
    typef  = inputf + '_types.csv'

    template = (
        "--epochs NEPOCHS --model_name model_HIVAE_inputDropout --restore MODLOAD "
        "--data_file data_python/INPUT_FILE.csv --types_file data_python/TYPES_FILE "
        "--batch_size NBATCH --save NEPFILL --save_file SAVE_FILE "
        "--dim_latent_s SDIM --dim_latent_z 1 --dim_latent_y YDIM "
        "--miss_percentage_train 0 --miss_percentage_test 0 "
        "--true_miss_file data_python/MISS_FILE --learning_rate LRATE "
        "--n_vis N_VIS"
    )

    # replace placeholders in template
    settings = re.sub('INPUT_FILE', inputf, template)
    settings = re.sub('NBATCH', str(opts['nbatch'].iloc[0]), settings)
    settings = re.sub('NEPOCHS', str(nepochs), settings)
    settings = re.sub('NEPFILL', str(nepochs - 1), settings) if save else re.sub('NEPFILL', str(nepochs * 2), settings)
    settings = re.sub('YDIM', str(opts['ydims'].iloc[0]), settings)
    settings = re.sub('SDIM', str(opts['sdims'].iloc[0]), settings)
    settings = re.sub('MISS_FILE', missf, settings) if not 'stalone' in inputf else re.sub(
        '--true_miss_file data_python/MISS_FILE', '', settings)
    # settings = re.sub('MISS_FILE', missf, settings) if not 'medhist' in inputf else re.sub(
    #     '--true_miss_file data_python/MISS_FILE', '', settings)
    settings = re.sub('TYPES_FILE', typef, settings)
    settings = re.sub('SAVE_FILE', inputf, settings)
    settings = re.sub('LRATE', str(opts['lrates'].iloc[0]), settings)
    settings = re.sub('MODLOAD', '1', settings) if modload else re.sub('MODLOAD', '0', settings)
    settings = re.sub('N_VIS', str(opts['nvis'].iloc[0]), settings)

    return settings



###########################################################
# MAIN SCRIPT
###########################################################
t0 = time.process_time()
print("Begin processing inputs")

sample_size = 1000      # Your OU simulation size

###########################################################
# Discover modules (files)
###########################################################
files = [
    f for f in os.listdir('data_python/')
    if f.endswith('.csv') and '_types' not in f and '_missing' not in f and 'DELETE_PLACEHOLDER' not in f
]

vargroups = sorted(set(f.split('_VIS')[0] for f in files))
print("Detected vargroups:", vargroups)

###########################################################
# Read best hyperparameters
###########################################################
best_hyper = pd.read_csv('ou-full-results.csv')


###########################################################
# Generate type files for all modules
###########################################################
types_dict = {}
with open('ou-full-data_types.csv') as datafile:
    for entry in csv.DictReader(datafile, skipinitialspace=True):
        types_dict[entry['col']] = {k: v for k, v in entry.items() if k != 'col'}

for cf in [i for i in os.listdir('python_names/') if '_cols' in i]:
    with open('python_names/' + cf) as colfile, open('data_python/' + cf.replace('_cols','_types'), 'w') as typefile:
        writer = csv.DictWriter(typefile, fieldnames=['type','dim','nclass'])
        writer.writeheader()
        for row in csv.DictReader(colfile, skipinitialspace=True):
            writer.writerow(types_dict[row['x']])


###########################################################
# Train all modules
###########################################################
print("Begin training")
print('t =', '{:10.4f}'.format(time.process_time()-t0), 'Begin training all vargroups')

for x, vg in enumerate(vargroups):
    opts = dict(best_hyper[best_hyper['vargroups'] == vg])
    settings = set_settings(opts, modload=False, save=True)
    last_loss = helpers.train_network(settings)

    print(f"{x+1}/{len(vargroups)} : trained {vg} with loss {last_loss}")
    attempts = 0
    while np.isnan(last_loss) and attempts < 10:
        last_loss = helpers.train_network(settings)
        attempts += 1
        print(f"Retry {attempts}: loss={last_loss}")


###########################################################
# Embedding extraction
###########################################################
print("Extracting embeddings …")
print('t =', '{:10.4f}'.format(time.process_time()-t0), 'Begin getting embeddings')

dat = []
dfs = []

for vg in vargroups:
    opts = dict(best_hyper[best_hyper['vargroups'] == vg])
    opts['nbatch'].iloc[0] = sample_size
    settings = set_settings(opts, nepochs=1, modload=True, save=False)

    encs, encz, d = helpers.enc_network(settings)

    subj = pd.read_csv(f'python_names/{vg}_VIS00_subj.csv')['x']
    sc = pd.DataFrame({'scode_' + vg: encs, 'SUBJID': subj})
    zc = pd.DataFrame({'zcode_' + vg: [i[0] for i in encz], 'SUBJID': subj})
    enc = pd.merge(sc, zc, on='SUBJID')

    enc.to_csv(f'Saved_Networks/{vg}_meta.csv', index=False)
    dat.append(d)
    dfs.append(enc)

meta = helpers.merge_dat(dfs)
meta.to_csv('metaenc.csv', index=False)


dat_dic = dict(zip(files, dat))
print('t =', '{:10.4f}'.format(time.process_time()-t0), 'Begin plotting SPLOM')
# Plotting embedding distributions
fig = scatter_matrix(
    meta[meta.columns.drop(list(meta.filter(regex='SUBJID|scode_')))],
    figsize=[50, 50],
    marker=".",
    s=10,
    diagonal="kde"
)
for ax in fig.ravel():
    ax.set_xlabel(re.sub('_VIS|zcode_', '', ax.get_xlabel()), fontsize=20, rotation=90)
    ax.set_ylabel(re.sub('_VIS|zcode_', '', ax.get_ylabel()), fontsize=20, rotation=00)

plt.suptitle('HI-VAE embeddings (deterministic)', fontsize=20)

plt.savefig('SPLOM_'+datetime.now().strftime('%b_%d_%H-%M-%S'))



###########################################################
# Reconstruction
###########################################################
print("Reconstructing …")
print('t =', '{:10.4f}'.format(time.process_time()-t0), 'Begin reconstructing data')
meta = pd.read_csv('metaenc.csv')
recdfs = []

for vg in vargroups:
    opts = dict(best_hyper[best_hyper['vargroups'] == vg])
    opts['nbatch'].iloc[0] = sample_size
    settings = set_settings(opts, nepochs=1, modload=True, save=False)

    zcodes = meta['zcode_' + vg]
    scodes = meta['scode_' + vg]
    rec = helpers.dec_network(settings, zcodes, scodes)

    for vis in range(rec.shape[1]):
        rec_vis = rec[:, vis, :]
        subj = pd.read_csv(f'python_names/{vg}_VIS{vis:02d}_subj.csv')['x']
        names = pd.read_csv(f'python_names/{vg}_VIS{vis:02d}_cols.csv')['x']

        df = pd.DataFrame(rec_vis)
        df.columns = names
        df['SUBJID'] = subj
        recdfs.append(df)

data_recon = helpers.merge_dat(recdfs)
data_recon.to_csv('reconRP.csv', index=False)


###########################################################
# Loglikelihoods
###########################################################
print("Computing loglikelihoods …")
print('t =', '{:10.4f}'.format(time.process_time()-t0), 'Begin getting loglikelihoods')
dfs = []
for vg in vargroups:
    opts = dict(best_hyper[best_hyper['vargroups'] == vg])
    opts['nbatch'].iloc[0] = sample_size
    settings = set_settings(opts, nepochs=1, modload=True, save=False)

    zcodes = meta['zcode_' + vg]
    scodes = meta['scode_' + vg]
    loglik_list = helpers.dec_network_loglik(settings, zcodes, scodes)

    for vis, loglik_vis in enumerate(loglik_list):
        loglik_vis = np.nanmean(np.array(loglik_vis).T, axis=1)
        subj = pd.read_csv(f'python_names/{vg}_VIS{vis:02d}_subj.csv')['x']
        df = pd.DataFrame({f"{vg}_VIS{vis:02d}": loglik_vis, "SUBJID": subj})
        dfs.append(df)

decoded = helpers.merge_dat(dfs)
decoded.to_csv("training_logliks.csv", index=False)

print("Finished.")
print('t =', '{:10.4f}'.format(time.process_time()-t0), 'Finished')

plt.show()