import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
from sys import maxsize
import pickle
import numpy as np
import torch
import torch.nn as nn
import copy
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from src.utils import load_yaml_config
from parser_rtsgan import base_parser

# absolute path to RTSGAN repo
# RTSGAN_PATH = "/path/to/Documents/RTSGAN"
RTSGAN_PATH = "/path/to/RTSGAN"
import sys
sys.path.append(RTSGAN_PATH)
sys.path.append(RTSGAN_PATH + "/general")
from general.missingprocessor import *
sys.path.append(RTSGAN_PATH + "/utils")
from utils.general import init_logger, make_sure_path_exists
from physionet2012 import Physio2012


def run(MC_id, base_config):
    
    options = copy.deepcopy(base_config) 

    # options.log_dir = "/benchmark/rtsgan/monte_carlo"
    task_name = options.task_name + "/MC_" + str(MC_id)
    root_dir = "{}/{}/{}".format(options.init_path, options.log_dir, task_name)
    print('Root dir:', root_dir)
    os.makedirs(root_dir, exist_ok=True)
    # make_sure_path_exists(root_dir)

    print("____________ options _____________", options)

    # ===-----------------------------------------------------------------------===
    # Set up logging
    # ===-----------------------------------------------------------------------===
    logger = init_logger(root_dir)

    # ===-----------------------------------------------------------------------===
    # Log some stuff about this run
    # ===-----------------------------------------------------------------------===
    logger.info(' '.join(sys.argv))
    logger.info('')
    logger.info(options)

    seed = MC_id
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    logger.info('Python random seed: {}'.format(seed))

    # ===-----------------------------------------------------------------------===
    # Read in dataset
    # ===-----------------------------------------------------------------------===

    options.dataset = options.init_path + "/benchmark/rtsgan/data_preprocessing/{}/simulated_dataset_MC_{}.pkl".format(options.dataset, MC_id)
    dataset = pickle.load(open(options.dataset, "rb"))
    train_set=dataset["train_set"]
    dynamic_processor=dataset["dynamic_processor"]
    static_processor=dataset["static_processor"]
    train_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len","priv", "nex", "label")
                        
    # ===-----------------------------------------------------------------------===
    # Build model and trainer
    # ===-----------------------------------------------------------------------===

    params=vars(options)
    params["static_processor"]=static_processor
    params["dynamic_processor"]=dynamic_processor
    params["root_dir"]=root_dir
    params["logger"]=logger
    params["device"]=device
    print(params.keys())

    syn = Physio2012((static_processor, dynamic_processor), params)

    if base_config.fix_ae is not None:
        syn.load_ae()
    else:
        syn.train_ae(train_set, options.epochs)

    # h = syn.eval_ae(train_set)
    # sta, dyn = syn.generate_ae(train_set)

    logger.info("Autoencoder training complete.")
        
    # if dataset.get('val_set') is not None:
    #     val_set = dataset['val_set']
    #     val_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len","priv", "nex")
    #     h = syn.eval_ae(val_set)
    #     # with open("{}/hidden".format(root_dir), "wb") as f:
    #     #     pickle.dump(h, f)
    #     vs, vd = syn.generate_ae(val_set)
    #     make_sure_path_exists("{}/val".format(root_dir))
    #     for i, res in enumerate(vd):
    #         res.to_csv("{}/val/p{}.psv".format(root_dir,i), sep='\t', index=False)
    #     # for i, res in enumerate(vs):
    #     #     res.to_csv(f"{root_dir}/val/static_p{i}.psv", sep="\t", index=False)
    #     # del vs, vd, h
    #     # torch.cuda.empty_cache()
    #     logger.info("Finish val")
        
    if options.fix_gan is not None:
        syn.load_generator()
    else:
        syn.train_gan(train_set, options.iterations, options.d_update)

    n_test = len(dataset["val_set"])
    N = base_config.N_generated_datasets
    n_to_generate = N * n_test
    logger.info("\n")
    logger.info(f"Generating {n_to_generate} synthetic samples "
                f"= {N} x len(test_set) ({n_test}).")
    # h = syn.gen_hidden(n_to_generate)
    result = syn.synthesize(n_to_generate)
    dataset["synthetic"] = result
    with open(f"{root_dir}/synth_{N}x.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(f"--- MC {MC_id} completed.")
    print("\n" + "="*50)


    # h = syn.eval_ae(train_set)
    # # with open("{}/hidden".format(root_dir), "wb") as f:
    # #     pickle.dump(h, f)

    # sta, dyn = syn.generate_ae(train_set)
    # # make_sure_path_exists("{}/reconstruction".format(root_dir))
    # # for i, res in enumerate(dyn[:10]):
    # #     res.to_csv("{}/reconstruction/p{}.psv".format(root_dir,i), sep='\t', index=False)
    # logger.info("Finish eval ae")

    # h = syn.gen_hidden(n_test)
    # with open("{}/gen_hidden".format(root_dir), "wb") as f:
    #     pickle.dump(h, f)
        
    # logger.info("\n")
    # logger.info("Generating data!")
    # result = syn.synthesize(len(train_set))
    # dataset["train_set"]=result
    # with open("{}/synth.pkl".format(root_dir),"wb") as f:
    #     pickle.dump(dataset, f)


if __name__ == "__main__":

    base_config, unknown = base_parser(return_unknown=True)
    print("Base config 1:", base_config)

    external_config = load_yaml_config(base_config.config_file)
    for key, value in external_config.items():
        setattr(base_config, key, value)

    print("Base config 2:", base_config)

    if torch.cuda.is_available():
        devices = [int(x) for x in base_config.devi]
        device = torch.device(f"cuda:{devices[0]}")
    else:
        device = torch.device("cpu")

    MC_id = int(unknown[0])
    run(MC_id, base_config)