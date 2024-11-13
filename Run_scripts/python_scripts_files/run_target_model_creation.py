import csv
import gc
import os
import sys

import numpy as np
import torch
import configparser
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split

sys.path.append("/sise/home/royek/Toshiba_roye/")
from FR_System.Data.data_utils import CelebA_split_ids_train_test, CelebA_target_training_apply_distribution, \
    CelebA_create_yes_records, CelebA_create_no_records, Two_Class_test_Predict_func, neptune_recoder, \
    MAAD_Face_split_ids_train_test, MAAD_Face_target_training_apply_distribution, MAAD_Face_create_yes_records, \
    MAAD_Face_create_no_records, preprocess_celeba_dataset_and_property_distribution, \
    preprocess_maad_face_dataset_and_property_distribution, celeba_create_yes_and_no_record, \
    maad_face_create_yes_and_no_record, choose_predictor_for_eval, load_checkpoint_for_eval, \
    load_embedder_and_predictor_for_eval
from FR_System.Embedder.embedder import Embedder, convert_data_to_net_input
from FR_System.Predictor.predictor import Predictor
from FR_System.fr_system import FR_Api, test_prediction, evaluation
import pytorch_lightning as pl


def datasets_and_target_model_creation(dir, saving_path, seed, backbone, property, distribution, train_emb=True,
                                   is_faceX_zoo=True, model_name='', predictor_architecture_type=1, dataset_name='CelebA', property_annotations_included=None):
    """
    Use the dataset (celebA or MAAD-Face) to create the wanted datasets (target model training set, attack training set and attack test
    set) and induce a target model from it.
    :param dir: Required. Type: str. The path where the dataset images are.
    :param saving_path: Required. Type: str. The path save all the results in.
    :param seed: Required. Type: int. The seed to use for splitting the data and induce the model.
    :param property: Required. Type: str. the property to use when splitting the data.
    :param distribution: Required. Type: dict {property value: int}. The wanted distribution of the property in the target
    model's training set. Must complete to 100.
    :param dataset_name: Required. Type: str. The name of the dataset to use.
    """
    exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path,
                                                        seed,
                                                        backbone,
                                                        property,
                                                        list(distribution.keys())[0],
                                                        distribution.get(list(distribution.keys())[0]))
    # Load / create the datasets
    if model_name.startswith("Attention"):
        batch_size = 32
    else:
        batch_size = 64
    print("exp_path: ", exp_path)
    if not os.path.isdir(exp_path):
        if dataset_name == 'CelebA':
            attack_test_df, attack_train_df, model_train_df_dist = preprocess_celeba_dataset_and_property_distribution(dir,
                                                                                                                   distribution,
                                                                                                                   property,
                                                                                                                   seed)
        elif dataset_name == 'MAAD_Face':
            attack_test_df, attack_train_df, model_train_df_dist = preprocess_maad_face_dataset_and_property_distribution(dir,
                                                                                                                          distribution,
                                                                                                                          property,
                                                                                                                          seed)
        else:
            raise Exception('dataset_name must be CelebA or MAAD_Face')
        os.mkdir(exp_path)
        model_train_df_dist.to_csv("{}model_train_df_dist.csv".format(exp_path))
        attack_train_df.to_csv("{}attack_train_df.csv".format(exp_path))
        attack_test_df.to_csv("{}attack_test_df.csv".format(exp_path))
        # Change target model training to pairs
        if dataset_name == 'CelebA':
            model_train_df_dist_no_pairs, model_train_df_dist_yes_pairs = celeba_create_yes_and_no_record(exp_path,
                                                                                                      model_train_df_dist, property_annotations_included=property_annotations_included, property=property, distribution=distribution, seed=seed) #seed=seed, property_annotations_included=property_annotations_included, property=property)
        elif dataset_name == 'MAAD_Face':
            model_train_df_dist_no_pairs, model_train_df_dist_yes_pairs = maad_face_create_yes_and_no_record(exp_path,
                                                                                                            model_train_df_dist, property_annotations_included=property_annotations_included, property=property, distribution=distribution, seed=seed) #seed=seed, property_annotations_included=property_annotations_included, property=property)
        else:
            raise Exception('dataset_name must be CelebA or MAAD_Face')
        model_train_df_pairs = pd.concat([model_train_df_dist_yes_pairs, model_train_df_dist_no_pairs],
                                         ignore_index=True, axis=0)
        model_train_df_pairs.to_csv("{}model_train_df_pairs.csv".format(exp_path))
    else:
        model_train_df_pairs = pd.read_csv("{}model_train_df_pairs.csv".format(exp_path))
        model_train_df_pairs = model_train_df_pairs.set_index("Unnamed: 0")

    # Create the target model
    # Create an embedder
    print(f' with {property} and {distribution} seed {seed}')
    print("Create embedder")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device, '/ torch version:', torch.version.cuda)

    embeder = Embedder(device=device, model_name=model_name, train=train_emb, faceX_zoo=is_faceX_zoo)
    #if backbone start with
    if model_name.startswith("resnet50") or model_name.startswith("senet50"):
        if not is_faceX_zoo:
            n_in = 2048
            increase_shape = True
        else:
            #throw exception
            raise Exception("model_name must be resnet50 or senet50 and it is not from facexzoo")
    else:
        n_in = 512
        increase_shape = False


    # Split the data to train and test
    # Temp testing data
    test_size = 0.05
    x_train, x_test, y_train, y_test = train_test_split(model_train_df_pairs.drop(["label"], axis=1),
                                                        pd.DataFrame(model_train_df_pairs["label"], columns=["label"]),
                                                        test_size=test_size, random_state=seed)
    y_train_vals = y_train.values

    contrast_y_train_vals = [1 - val[0] for val in y_train_vals]
    y_train['label2'] = contrast_y_train_vals

    columns_titles = ["label2", "label"]
    y_train = y_train.reindex(columns=columns_titles)  # put the value of not same person to be the first class

    y_test_vals = y_test.values

    contrast_y_test_vals = [1 - val[0] for val in y_test_vals]
    y_test['label2'] = contrast_y_test_vals

    columns_titles = ["label2", "label"]
    y_test = y_test.reindex(columns=columns_titles)

    # if property_annotations_included we need to save the csvs and then drop the propties columns, for the use of the csvs as it was before
    #save x_train x_val with proerty annotation with different names
    # f'{property}_path1', f'{property}_path2'
    if property_annotations_included:
        #first save the original x_train and x_test with the witht propety in the names
        if not os.path.isfile("{}model_x_train_with_property.csv".format(exp_path)):
            x_train.to_csv("{}model_x_train_with_property.csv".format(exp_path))
        if not os.path.isfile("{}model_x_val_with_property.csv".format(exp_path)):
            x_test.to_csv("{}model_x_val_with_property.csv".format(exp_path))
        x_train = x_train.drop([f'{property}_path1', f'{property}_path2'], axis=1) #drop so we will not use them in the model
        x_test = x_test.drop([f'{property}_path1', f'{property}_path2'], axis=1) #drop so we will not use them in the model

    if not os.path.isfile("{}model_x_train.csv".format(exp_path)):
        x_train.to_csv("{}model_x_train.csv".format(exp_path))
    if not os.path.isfile("{}model_x_val.csv".format(exp_path)):
        x_test.to_csv("{}model_x_val.csv".format(exp_path))
    if not os.path.isfile("{}model_y_train.csv".format(exp_path)):
        y_train.to_csv("{}model_y_train.csv".format(exp_path))
    if not os.path.isfile("{}model_y_val.csv".format(exp_path)):
        y_test.to_csv("{}model_y_val.csv".format(exp_path))


    print("converting to NN input")

    # bellow for neptune, add hyperparametrs and intiallize the neptune recoder
    hyper_params = {'batch_size': batch_size, 'property': property, 'distribution': distribution, 'seed': seed, 'backbone': model_name, 'train_emb': train_emb, 'predictor_architecture_type': predictor_architecture_type}
    run = neptune_recoder(exp_name='target_model', description='target_model', tags=['target_model'],
                          hyperparameters=hyper_params)

    if not train_emb:
        if not os.path.isfile("{}model_x_train_vectors.npy".format(exp_path)):
            x_train_nn = convert_data_to_net_input(x_train, embeder,
                                                   saving_path_and_name="{}model_x_train_vectors.npy".format(exp_path),
                                                   increase_shape=increase_shape)
        else:
            x_train_nn = np.load("{}model_x_train_vectors.npy".format(exp_path))
        predictor = Predictor(predictor="NN", threshold=0.5, x_train=x_train_nn, y_train=y_train.values,
                              nn_save_path=exp_path, predictor_architecture_type=predictor_architecture_type, n_in=n_in,
                              dataset_name=dataset_name, batch_size=batch_size, increase_shape=increase_shape)
    else:
        if not os.path.isfile("{}model_x_train_vectors.npy".format(exp_path)):
            #create embedding but dont load them
            #we will load them in the first phase trainig and use the actual traning set (x_train) in the second phase
            embeder.embedder.eval()
            _ = convert_data_to_net_input(x_train, embeder,
                                                   saving_path_and_name="{}model_x_train_vectors.npy".format(exp_path),
                                                   increase_shape=increase_shape)
            embeder.embedder.train()
        #need to ceate x_train_nn vectirs for first phase
        predictor = Predictor(predictor="NN", threshold=0.5, x_train=x_train, y_train=y_train.values,
                              nn_save_path=exp_path, embeder=embeder.embedder, device=device, n_in=n_in,
                              predictor_architecture_type=predictor_architecture_type,
                              dataset_name=dataset_name, batch_size=batch_size, increase_shape=increase_shape, emb_name=model_name)
        embeder_inst = predictor.embedder
        #After train now we want to eval
        embeder = Embedder(device=device, train=False, model_name=model_name, faceX_zoo=is_faceX_zoo)
        embeder.embedder = embeder_inst
    fr = FR_Api(embedder=embeder, predictor=predictor)

    # Test and evaluation

    print("Test and evaluation")
    predictions = Two_Class_test_Predict_func(x_test, fr, increase_shape=increase_shape)
    y_test = y_test.values

    evaluations = evaluation(predictions.argmax(axis=1), y_test.argmax(axis=1))
    #bellow for neptune add acuuracy
    run['accuracy'] = evaluations['acc']
    print(evaluations)
    properties = {'Backbone': model_name, 'Seed': seed, 'Property': property, 'Dist': distribution}
    properties.update(evaluations)

    # bellow for neptune to close it after finish use it
    run.stop()
    if train_emb:
        train_emb_folder = "fined_tuning_embedder_Results/"
    else:
        train_emb_folder = "" #not extra folder to add

    if dataset_name == "MAAD_Face":
        dataset_Results_folder = "MAAD_Face_Results/"
    else: #celeba
        dataset_Results_folder = ""
    if predictor_architecture_type == 1:  # not black box different predictor
        if not train_emb:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_target_model_results.csv"
        else:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}predictor_one_different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_one_target_model_results.csv"
    elif predictor_architecture_type == 2:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}predictor_two_different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_target_model_results.csv"
    elif predictor_architecture_type == 3:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_three_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_three_target_model_results.csv"
    elif predictor_architecture_type == 4:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_four_target_model_results.csv"
    elif predictor_architecture_type == 5:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_five_target_model_results.csv"
    elif predictor_architecture_type == 6:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_six_target_model_results.csv"
    elif predictor_architecture_type == 7:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_seven_target_model_results.csv"
    elif predictor_architecture_type == 8: #same as the substitute
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_eight_target_model_results.csv"
    elif predictor_architecture_type == 9:
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_nine_target_model_results.csv"
    else:
        raise Exception("predictor_architecture_type must be 1 - 6")
    # check if path exisit iff not create it
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    if os.path.exists(result_file):
        with open(result_file, 'a') as f:
            w = csv.DictWriter(f, properties.keys())
            w.writerow(properties)
    else:
        with open(result_file, 'w') as f:
            w = csv.DictWriter(f, properties.keys())
            w.writeheader()
            w.writerow(properties)




if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    config = config['DEFAULT']

    dir = config['dir']  # The folder that contains the images of the CelebA dataset.
    # For example: celebA/img_align_celeba/img_align_celeba/
    prefix_saving_path = config['prefix_saving_path']  # The folder to save the outputs of this script.
    seed = int(config['seed'])  # The seed to use (for reproducibility)
    backbone = config['backbone']  # the backbone name
    property = config['property']  # The property to use (should be written as in the CelebA annotation).
    distribution_1_value = int(config["distribution_1_value"])  # The distribution of the value "1" for the property.
    train_emb = config.getboolean('train_emb')  # Whether to train the embedding model or not.
    dataset_name = config['dataset_name']  # The name of the dataset to use.

    # read boolean value of faceX_zoo
    is_faceX_zoo = config.getboolean('is_faceX_zoo')
    train_all_dist = config.getboolean('train_all_dist')
    # read predictor architecture type number as int
    predictor_architecture = int(config['predictor_architecture'])
    print("predictor_architecture type is: ", predictor_architecture)
    property_annotations_included = config.getboolean('property_annotations_included')

    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)

    if train_all_dist:
        print("train all dist")
        for dist in [100, 75, 50, 25, 0]:
            distribution = {1: dist,
                            -1: 100 - dist}
            saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
            print("Starting creating target model with dist: ", dist)
            datasets_and_target_model_creation(dir, saving_path, seed, backbone, property, distribution,
                                       model_name=backbone,
                                       is_faceX_zoo=is_faceX_zoo, train_emb=train_emb,
                                       predictor_architecture_type=predictor_architecture, dataset_name=dataset_name,
                                               property_annotations_included=property_annotations_included)
            print("Finished creating target model with dist: ", dist)
            # clean the current expirement data delete what necesseary use garabage coolector and clean chache of Cuda
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("train only dist: ", distribution_1_value)
        distribution = {1: distribution_1_value,
                        -1: 100 - distribution_1_value}
        saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
        datasets_and_target_model_creation(dir, saving_path, seed, backbone, property, distribution, model_name=backbone,
                                   is_faceX_zoo=is_faceX_zoo, train_emb=train_emb,
                                   predictor_architecture_type=predictor_architecture, dataset_name=dataset_name,
                                           property_annotations_included=property_annotations_included)
