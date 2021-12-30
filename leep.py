import argparse
import logging
import random
import os
import json
import sys
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_access import DatasetAccess
from models import EncoderNet, ClassifierNet
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_leep(encoder, classifier, target, filename):
    target_dataset = DatasetAccess(target, train=False)
    target_df = target_dataset.get_access()

    target_loader = DataLoader(target_df, batch_size=len(target_df), shuffle=True,
                            num_workers=1, pin_memory=True)
    y = {}
    z_prob = {}
    y_x = []

    for i in range(0,10):
        z_prob[i] = []    
    counter = 0

    for target_x, target_y in target_loader:
        target_x, target_y = target_x.to(device), target_y.to(device)
        y_pred_features = encoder(target_x).view(target_x.shape[0], -1)
        y_pred = classifier(y_pred_features)

        y_pred = F.softmax(y_pred, dim=1)
        y_pred = y_pred.detach().numpy()
        target_y = target_y.detach().numpy()

        for k in range(len(target_y)):
            for i in range(0, 10):
                z_prob[i].insert(k, y_pred[k][i])

            if target_y[k] not in y:
                y[target_y[k]] = []
            y[target_y[k]].append(counter)
            y_x.insert(counter, target_y[k])
            counter = counter + 1

    Pyz = []
    Pz = []
    for i in range(0, 10): # by z
        Pz.insert(i, 0)
        for j in range(0, 10): # by y_target
            sum = 0
            for k in y[j]:
                sum = sum + z_prob[i][k]
            Pyz.insert(i*10+j, sum/counter)
            Pz[i] = Pz[i] + sum/counter

    Py_z = []
    for i in range(0, 10):
        for j in range(0, 10):
            Py_z.insert(i*10+j, Pyz[i*10+j]/Pz[i])

    LEEP = 0
    for i in range(len(target_df)): # by i (y_target)
        local_sum = 0
        for j in range(0, 10): # by z (y_pred)
            local_sum = local_sum + Py_z[y_x[i]+10*j] * z_prob[j][i]
        LEEP = LEEP + math.log(local_sum)
    LEEP = LEEP/counter
    print(LEEP)
    file = open(filename, "w")
    file.write("LEEP = " + str(LEEP) + "\n")
    file.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='LEEP analysis')

    format_str = "%(asctime)s %(name)s [%(levelname)s] - %(message)s"
    logging.basicConfig(format=format_str)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    arg_parser.add_argument('--source_enc_model', type=str, required=False)
    arg_parser.add_argument('--source_clf_model', type=str, required=False)
    arg_parser.add_argument('--file', type=str, required=False, default='leep')
    arg_parser.add_argument('--dataconf', type=str, default = 'MNIST_USPS')

    args = arg_parser.parse_args()
    logging.info(f"Running with {args}")

    # init seed
    seed = random.randint(1, 10000)
    logging.info("Use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # check config path
    data_json_path = str(config.DATA_DIR) + f"/data_args/{args.dataconf}.json"
    if not os.path.exists(data_json_path):
        raise ValueError(f"{data_json_path} doesn't exist")
    
    # open configuration json file
    dataconf = open(data_json_path, "r")
    dataconf = json.load(dataconf)

    output_dir = str(config.DATA_DIR) + f"/trained_models/{dataconf['dataset_name']}"
    logging.info(f"Create output dir {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load models
    source_encoder_model = EncoderNet().to(device)
    source_classifier_model = ClassifierNet().to(device)
    loaded_source = False
    source_encoder = source_encoder_model
    source_classifier = source_classifier_model

    if args.source_enc_model != None:
        logging.info(f"Load model from {args.source_enc_model}")
        source_encoder_model.load_state_dict(torch.load(args.source_enc_model))
        source_encoder = source_encoder_model
        logging.info(f"Load model from {args.source_clf_model}")
        source_classifier_model.load_state_dict(torch.load(args.source_clf_model))
        source_classifier = source_classifier_model
        loaded_source = True

    if not loaded_source:
        logging.info("No source model")
        sys.exit(0)

    ind = args.source_enc_model.rfind("/")
    ind2 = args.source_enc_model.rfind(".")
    filename = str(config.DATA_DIR) + "/" + args.source_enc_model[ind:ind2] + "_" + args.file + ".txt"

    logging.info("Calculate LEEP score")
    calculate_leep(
        source_encoder,
        source_classifier,
        dataconf["target"],
        filename
    )