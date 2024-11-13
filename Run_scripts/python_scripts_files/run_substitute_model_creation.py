import configparser
import csv
import gc
import sys
import glob
import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
sys.path.append("/sise/home/royek/Toshiba_roye/")
from FR_System.Embedder.embedder import Embedder, convert_data_to_net_input
from FR_System.Predictor.predictor import Predictor
from FR_System.fr_system import FR_Api, evaluation
from FR_System.Data.data_utils import CelebA_target_training_apply_distribution, CelebA_create_yes_records, \
    CelebA_create_no_records, Two_Class_test_Predict_func, create_yes_records_before_adv, create_no_records_before_adv, \
    load_predictor, neptune_recoder, MAAD_Face_target_training_apply_distribution, celeba_create_yes_and_no_record, \
    maad_face_create_yes_and_no_record
import pytorch_lightning as pl


def datasets_and_substitute_model_creation(traget_model_path, saving_path, seed, backbone, property, distribution, train_emb=True, is_faceX_zoo=True, predictor_architecture_type = 1, dataset_name='CelebA'):
    """
    Use the dataset (celebA or MAAD-Face) to create the wanted dataset (for substitute model) and train the substitute model.
    :param traget_model_path: Required. str. The path the target model is saved at.
    :param saving_path: Required. str. The path save all the results in.
    :param seed: Required. int. The seed to use for splitting the data and induce the model.
    :param property: Required. str. The property to use when splitting the data.
    :param distribution: Required. dict {property value: int}. The wanted distribution of the property in the target
    model's training set. Must complete to 100.
    :param train_emb: Optional. bool. True if the embedder should be trained, False otherwise. Default is True.
    :param is_faceX_zoo: Optional. bool. True if the embedder is FaceX-Zoo, False otherwise. Default is True.
    :param predictor_architecture_type: Optional. int. The type of the predictor architecture. Default is 1.
    :param dataset_name: Optional. str. The name of the dataset. Default is CelebA.
    """
    print(f"predictor_architecture_type: {predictor_architecture_type}")
    exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path,
                                                        seed,
                                                        backbone,
                                                        property,
                                                        list(distribution.keys())[0],
                                                        distribution.get(list(distribution.keys())[0]))


    print("Load data to attack the target model")
    attack_train_df = pd.read_csv("{}attack_train_df.csv".format(traget_model_path))
    attack_train_df = attack_train_df.set_index("Unnamed: 0")
    attack_test_df = pd.read_csv("{}attack_test_df.csv".format(traget_model_path))
    attack_test_df = attack_test_df.set_index("Unnamed: 0")
    # Create embedder
    print("Create embedder")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(device=device, model_name=backbone, train=train_emb, faceX_zoo=is_faceX_zoo)

    if backbone.startswith("resnet50") or backbone.startswith("senet50"):
        if not is_faceX_zoo:
            n_in = 2048
            increase_shape = True
        else:
            #throw exception
            raise Exception("model_name must be resnet50 or senet50 and it is not from facexzoo")
    else:
        n_in = 512
        increase_shape = False

    print("Creat predictor")
    print('Creating predictor for sub model with dist', distribution)
    # preprocess data for substitute model
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
        # preprocess train data for substitute model
        if dataset_name == 'CelebA':
            attack_train_df_dist = CelebA_target_training_apply_distribution(model_train_df=attack_train_df,
                                                                         distribution=distribution,
                                                                         property=property,
                                                                         seed=seed)
            attack_train_df_dist_no_pairs, attack_train_df_dist_yes_pairs = celeba_create_yes_and_no_record(exp_path,
                                                                                                            attack_train_df_dist)

        elif dataset_name == 'MAAD_Face':
            attack_train_df_dist = MAAD_Face_target_training_apply_distribution(model_train_df=attack_train_df,
                                                                         distribution=distribution,
                                                                         property=property,
                                                                         seed=seed)
            attack_train_df_dist_no_pairs, attack_train_df_dist_yes_pairs = maad_face_create_yes_and_no_record(exp_path,
                                                                                                               attack_train_df_dist)
        else:
            raise Exception('dataset_name must be CelebA or MAAD_Face')

        attack_train_df_pairs = pd.concat([attack_train_df_dist_yes_pairs, attack_train_df_dist_no_pairs],
                                          ignore_index=True, axis=0)
        attack_train_df_pairs.to_csv("{}model_train_df_pairs.csv".format(exp_path))
        # preprocess test data for substitute model for adversarial attack(before optimized)
        attack_test_df_yes_pairs = create_yes_records_before_adv(attack_test_df, property, save_to=exp_path)
        attack_test_df_no_pairs = create_no_records_before_adv(attack_test_df, property, yes_pairs_path=exp_path,
                                                           save_to=exp_path)
        attack_test_df_pairs = pd.concat([attack_test_df_yes_pairs, attack_test_df_no_pairs],
                                         ignore_index=True, axis=0)
        attack_test_df_pairs.to_csv("{}attack_test_before_adv_df_pairs.csv".format(exp_path))
    else:
        # preprocess train data for substitute model
        attack_train_df_pairs = pd.read_csv("{}model_train_df_pairs.csv".format(exp_path))
        # preprocess test data for substitute model
        attack_test_df_pairs = pd.read_csv("{}attack_test_before_adv_df_pairs.csv".format(exp_path))


    #####convert the attack data to train test and to classes labels
    #train and test evaluation data to substitute model
    test_size = 0.05
    x_train, x_test, y_train, y_test = train_test_split(attack_train_df_pairs.drop(["label"], axis=1),
                                                        pd.DataFrame(attack_train_df_pairs["label"], columns=["label"]),
                                                        test_size=test_size, random_state=seed)
    #create data that will use to create adv samples for the substitute model (using the
    x_test_adv = attack_test_df_pairs.drop(["label"], axis=1)
    #drop property paths
    x_test_adv = x_test_adv.drop([f"{property}_path1"], axis=1)
    x_test_adv = x_test_adv.drop([f"{property}_path2"], axis=1)
    y_test_adv = pd.DataFrame(attack_test_df_pairs["label"], columns=["label"])

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

    #for before_adv attack data
    y_test_adv_vals = y_test_adv.values

    contrast_y_test_adv_vals = [1 - val[0] for val in y_test_adv_vals]
    y_test_adv['label2'] = contrast_y_test_adv_vals

    columns_titles = ["label2", "label"]
    y_test_adv = y_test_adv.reindex(columns=columns_titles)  # put the value of not same person to be the first class


    if not os.path.isfile("{}model_x_train.csv".format(exp_path)):
        x_train.to_csv("{}model_x_train.csv".format(exp_path))
    if not os.path.isfile("{}model_x_val.csv".format(exp_path)):
        x_test.to_csv("{}model_x_val.csv".format(exp_path))
    if not os.path.isfile("{}model_y_train.csv".format(exp_path)):
        y_train.to_csv("{}model_y_train.csv".format(exp_path))
    if not os.path.isfile("{}model_y_val.csv".format(exp_path)):
        y_test.to_csv("{}model_y_val.csv".format(exp_path))
    if not os.path.isfile("{}model_x_test_adv.csv".format(exp_path)):
        x_test_adv.to_csv("{}model_x_test_adv.csv".format(exp_path))
    if not os.path.isfile("{}model_y_test_adv.csv".format(exp_path)):
        y_test_adv.to_csv("{}model_y_test_adv.csv".format(exp_path))

    epoch_num = 10

    # bellow for neptune to add hyperparametrs and intiallize the neptune recoder
    hyper_params = {'batch_size': 64, 'property': property, 'distribution': distribution, 'seed': seed,
                    'backbone': backbone, 'train_emb': train_emb, 'predictor_architecture_type': predictor_architecture_type}
    run = neptune_recoder(exp_name='substitute_model', description='substitute_model', tags=['substitute_model'],
                          hyperparameters=hyper_params)
    print("converting to NN input")
    if not train_emb:
        if not os.path.isfile("{}model_x_train_vectors.npy".format(exp_path)):
            x_train_nn = convert_data_to_net_input(x_train, embedder,
                                                   saving_path_and_name="{}model_x_train_vectors.npy".format(exp_path),
                                                   increase_shape=increase_shape)
        else:
            x_train_nn = np.load("{}model_x_train_vectors.npy".format(exp_path))
        predictor = Predictor(predictor="NN", threshold=0.5, x_train=x_train_nn, y_train=y_train.values,
                              nn_save_path=exp_path, predictor_architecture_type=predictor_architecture_type, n_in=n_in,
                              dataset_name=dataset_name, increase_shape=increase_shape)
    else:
        if not os.path.isfile("{}model_x_train_vectors.npy".format(exp_path)):
            #create embedding but dont load them
            #we will load them in the first phase trainig and use the actual traning set (x_train) in the second phase
            embedder.embedder.eval()
            _ = convert_data_to_net_input(x_train, embedder,
                                                   saving_path_and_name="{}model_x_train_vectors.npy".format(exp_path),
                                                   increase_shape=increase_shape)
            embedder.embedder.train()
        predictor = Predictor(predictor="NN", threshold=0.5, x_train=x_train, y_train=y_train.values,
                              nn_save_path=exp_path, embeder=embedder.embedder, device=device, n_in=n_in,
                              predictor_architecture_type = predictor_architecture_type,
                              dataset_name=dataset_name, increase_shape=increase_shape, emb_name=backbone)

        embeder_inst = predictor.embedder
        embeder = Embedder(device=device, train=False, model_name=backbone, faceX_zoo=is_faceX_zoo)
        embeder.embedder = embeder_inst
    fr = FR_Api(embedder=embedder, predictor=predictor)

    # Test and evaluation
    print("Test and evaluation")
    predictions = Two_Class_test_Predict_func(x_test, fr, increase_shape=increase_shape)
    y_test = y_test.values

    evaluations = evaluation(predictions.argmax(axis=1), y_test.argmax(axis=1))
    #bellow for neptune add acuuracy
    run['accuracy'] = evaluations['acc']
    print(evaluations)
    properties = {'Backbone': backbone,'Seed':seed ,'Property': property, 'Dist': distribution}
    properties.update(evaluations)

    # bellow for neptune to close it after finish use it
    run.stop()

    if dataset_name == "MAAD_Face":
        dataset_Results_folder = "MAAD_Face_Results/"

    else:
        dataset_Results_folder = ""

    if train_emb:
        train_emb_folder = "fined_tuning_embedder_Results/"
        if predictor_architecture_type == 1:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_one_substitute_model_results.csv"
        elif predictor_architecture_type == 2:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_substitute_model_results.csv"
        elif predictor_architecture_type == 5: #same as predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_five_substitute_model_results.csv"
        elif predictor_architecture_type == 6: #new predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_six_substitute_model_results.csv"
        elif predictor_architecture_type == 7: #new predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_seven_substitute_model_results.csv"
        elif predictor_architecture_type == 8: #susbtitute model should be in a path which is not different predictor, to the case the target is the same with him and htey are in thwy willbe then in the same folder
            #8 us similar to architecture 1 and if the target is the same as rthe susbstittue than it is architecture 8
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_eight_substitute_model_results.csv"
        else:
            raise Exception("predictor_architecture_type must be 1-9")
    else:
        if predictor_architecture_type == 1:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_substitute_model_results.csv"
        elif predictor_architecture_type == 2:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_substitute_model_results.csv"
        else:
            raise Exception("predictor_architecture_type must be 1 or 2")
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

    prefix_traget_model_path = config['prefix_traget_model_path']
    prefix_saving_path = config['prefix_saving_path']
    seed = int(config['seed'])
    property = config['property']
    distribution_1_value = int(config["distribution_1_value"])  # The distribution of the value "1" for the property.
    distribution = {1: distribution_1_value,
                    -1: 100 - distribution_1_value}
    backbone= config['backbone'] #knowledege in white_box attack!!!!
    target_dist_to_load_from = config['target_dist_to_load_from']
    train_emb = config.getboolean('train_emb')
    dataset_name = config['dataset_name']
    #all the target model dist hold in their directory
    # the same attack dataset for the substitute model which we want to train on the attack dataset
    #so it does not realy matter from whome we will load the attack dataset it is the same
    # (and is non overlapping with any of the target model datasets)
    is_faceX_zoo = config.getboolean('is_faceX_zoo')

    train_all_dist = config.getboolean('train_all_dist')

    # read predictor architecture type number as int
    predictor_architecture = int(config['predictor_architecture'])

    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)

    if train_all_dist:
        print("train all dist")
        for dist in [100, 0]:
            distribution = {1: dist,
                            -1: 100 - dist}
            saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
            traget_model_path = f"{prefix_traget_model_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{target_dist_to_load_from}/"
            print("Starting creating target model with dist: ", dist)
            datasets_and_substitute_model_creation(traget_model_path=traget_model_path, saving_path=saving_path, seed=seed,
                              backbone=backbone, property=property,
                              distribution=distribution, train_emb=train_emb, is_faceX_zoo=is_faceX_zoo, predictor_architecture_type = predictor_architecture, dataset_name=dataset_name)

            print("Finished creating target model with dist: ", dist)
            #clean the current expirement data delete what necesseary use garabage coolector and clean chache of Cuda
            torch.cuda.empty_cache()
            gc.collect()



    else:
        print("train only dist: ", distribution_1_value)
        distribution = {1: distribution_1_value,
                        -1: 100 - distribution_1_value}
        saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
        traget_model_path = f"{prefix_traget_model_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{target_dist_to_load_from}/"
        datasets_and_substitute_model_creation(traget_model_path=traget_model_path, saving_path=saving_path, seed=seed, backbone=backbone,
                          property=property,
                          distribution=distribution, train_emb=train_emb, is_faceX_zoo=is_faceX_zoo, predictor_architecture_type = predictor_architecture, dataset_name=dataset_name)
