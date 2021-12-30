import argparse
import logging
import random
import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from dataset_access import DatasetAccess
from models import EncoderNet, ClassifierNet
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_silhoutee(encoder, target, filepath):
    target_dataset = DatasetAccess(target, train=False)
    target_df = target_dataset.get_access()

    target_loader = DataLoader(target_df, batch_size=len(target_df), shuffle=True,
                            num_workers=1, pin_memory=True)
    range_n_clusters = [8, 9, 10]
    
    for n_clusters in range_n_clusters:
        x, _ = next(iter(target_loader))
        x_encoded = encoder(x)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(18, 10)

        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(target_df) + (10 + 1) * 10])

        clusterer = KMeans(n_clusters=10, random_state=10)
        cluster_labels = clusterer.fit_predict(x_encoded.detach())

        silhouette_avg = silhouette_score(x_encoded.detach(), cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        sample_silhouette_values = silhouette_samples(x_encoded.detach(), cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis    labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        print(str(config.DATA_DIR)+"/"+filepath+"_"+str(n_clusters)+".png")
        fig.savefig(str(config.DATA_DIR)+"/"+filepath+"_"+str(n_clusters)+".png")
        plt.close(fig)

    #plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Silhouette analysis')

    format_str = "%(asctime)s %(name)s [%(levelname)s] - %(message)s"
    logging.basicConfig(format=format_str)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    arg_parser.add_argument('--source_enc_model', type=str, required=False)
    arg_parser.add_argument('--source_clf_model', type=str, required=False)
    arg_parser.add_argument('--img', type=str, required=False, default='test')
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

    logging.info("Calculate silhoutte score")
    calculate_silhoutee(
        source_encoder,
        dataconf["target"],
        args.img
    )