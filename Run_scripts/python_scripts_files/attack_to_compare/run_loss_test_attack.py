import configparser
import csv
import gc
import sys
import glob
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from torch import nn
sys.path.append("/sise/home/royek/Toshiba_roye/")
from tqdm import tqdm


from FR_System.Data.data_utils import CelebA_target_training_apply_distribution, CelebA_create_yes_records, \
    CelebA_create_no_records, load_predictor, Two_Class_test_Predict_func, random_pairs, save_df_and_df_round_to_4, \
    neptune_recoder, MAAD_Face_target_training_apply_distribution, MAAD_Face_create_yes_records, \
    MAAD_Face_create_no_records, load_embedder_and_predictor_for_eval
from FR_System.fr_system import FR_Api

from FR_System.Embedder.embedder import Embedder, convert_data_to_net_input
from FR_System.Predictor.predictor import Predictor
import pytorch_lightning as pl


def experiment_loss_test(traget_model_path, saving_path, seed, backbone, property, distribution, is_faceX_zoo, target_predictor_architecture = 1,
                      attack_setting="", run=None, dataset="CelebA", is_finetune_emb=False):
    """
    The function runs the experiment on CelebA dataset.
    :param target_predictor_architecture: Required. str. The architecture of the predictor.
    :param is_faceX_zoo: Required. bool. True if the model is from faceX_zoo.
    :param distribution: Required. dict. The distribution of the target training set.
    :param property: Required. str. The property of the target training set.
    :param backbone: Required. str. The backbone of the FR system.
    :param seed: Required. int. The seed of the experiment.
    :param saving_path: Required. str. The path to save the results.
    :param target_model_path: Required. str. The path to the target model.
    :param target_model_name: Required. str. The name of the target model.
    :param target_model_epoch: Required. int. The epoch of the target model.
    :param target_model_architecture: Required. str. The architecture of the target model.
    :param attack_setting: Required. str. The setting of the attack of adversariaLeak,
            in order to save the loss test results in the relevant folder..
    """
    exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path, seed,backbone, property, list(distribution.keys())[0],
                                                     distribution.get(list(distribution.keys())[0]))
    print("Data creation")
    attack_test_df = pd.read_csv("{}attack_test_df.csv".format(traget_model_path))
    attack_test_df = attack_test_df.set_index("Unnamed: 0")

    attack_dist_list = [{1: 100, -1: 0}, {1: 0, -1: 100}]
    dir_loss_test = exp_path + 'loss_test/'

    if not os.path.isdir(dir_loss_test):
        os.mkdir(dir_loss_test)
        for i, attack_dist in enumerate(attack_dist_list):
            if dataset == "CelebA":
                attack_test_df_dist = CelebA_target_training_apply_distribution(model_train_df=attack_test_df,
                                                                                distribution=attack_dist,
                                                                                property=property,
                                                                                seed=seed)
                attack_test_df_dist_yes_pairs = CelebA_create_yes_records(attack_test_df_dist, save_to=dir_loss_test, property=property, property_annotations_included=False)
                attack_test_df_dist_no_pairs = CelebA_create_no_records(attack_test_df_dist, yes_pairs_path=dir_loss_test,
                                                                        save_to=dir_loss_test, property=property, property_annotations_included=False)
            elif dataset == "MAAD_Face":
                attack_test_df_dist = MAAD_Face_target_training_apply_distribution(model_train_df=attack_test_df,
                                                                                distribution=attack_dist,
                                                                                property=property,
                                                                                seed=seed)
                attack_test_df_dist_yes_pairs = MAAD_Face_create_yes_records(attack_test_df_dist, save_to=dir_loss_test, property=property, property_annotations_included=False)
                attack_test_df_dist_no_pairs = MAAD_Face_create_no_records(attack_test_df_dist, yes_pairs_path=dir_loss_test,
                                                                        save_to=dir_loss_test, property=property, property_annotations_included=False)
            else:
                raise Exception("dataset not supported")

            attack_test_df_pairs = pd.concat([attack_test_df_dist_yes_pairs, attack_test_df_dist_no_pairs],
                                              ignore_index=True, axis=0)
            if attack_dist[1] == 100:
                attack_test_df_pairs.to_csv("{}positive_samples.csv".format(dir_loss_test))
                positive_samples = attack_test_df_pairs
            else:
                attack_test_df_pairs.to_csv("{}negative_samples.csv".format(dir_loss_test))
                negative_samples = attack_test_df_pairs
    else:
        positive_samples = pd.read_csv("{}positive_samples.csv".format(dir_loss_test))
        negative_samples = pd.read_csv("{}negative_samples.csv".format(dir_loss_test))


    # Create embedder
    print("Create embedder")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #embedder = Embedder(device=device, model_name=backbone, faceX_zoo=is_faceX_zoo)

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

    print(f"device: {device}")
    print(f"device.type: {device.type}")
    # Create predictor
    print("Creat FR")
    embedder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=backbone, device=device,
                                                                  is_faceX_zoo=is_faceX_zoo,
                                                                  predictor_architecture=target_predictor_architecture,
                                                                  path=traget_model_path, is_finetune_emb=is_finetune_emb,
                                                                  n_in=n_in, dataset_name=dataset)


    # Create complete API
    fr = FR_Api(embedder=embedder, predictor=predictor)

    # Loss Test
    pred_pos = Two_Class_test_Predict_func(positive_samples, fr, increase_shape=increase_shape)
    pred_pos=pred_pos.argmax(axis=1)
    positive_samples['prediction'] = pred_pos
    positive_samples.to_csv("{}positive_samples.csv".format(dir_loss_test))
    pred_neg = Two_Class_test_Predict_func(negative_samples, fr, increase_shape=increase_shape)
    pred_neg=pred_neg.argmax(axis=1)
    negative_samples['prediction'] = pred_neg
    negative_samples.to_csv("{}negative_samples.csv".format(dir_loss_test))

    acc_pos = accuracy_score(positive_samples['label'], positive_samples['prediction'])
    acc_neg = accuracy_score(negative_samples['label'], negative_samples['prediction'])
    print('Accuracy (positive samples):', acc_pos)
    print('Accuracy (negative samples):', acc_neg)
    print('Loss Tess result:', int(acc_pos > acc_neg))
    accuracy_differences = acc_neg - acc_pos ## keep the difference as in my code: dist 0 - dist 100
    print('Accuracy difference:', accuracy_differences)
    Loss_Test_result = int(acc_pos > acc_neg)

    if dataset == "CelebA":
        prefix_for_path = ""
    elif dataset == "MAAD_Face":
        prefix_for_path = "MAAD_Face_Results/"
    else:
        raise Exception("dataset not supported")

    #check if fine tune emb
    if is_finetune_emb:
        fine_tune_path = "fined_tuning_embedder_Results/"
    else:
        fine_tune_path = ""


    #write to loss test csv file, with the seed the backone the property and the distribution of the property in the target model
    result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}property_{property}/seed_{seed}/attack_to_compare/loss_test/property_{property}_seed_{seed}_loss_test_results.csv"
    result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}property_{property}/seed_{seed}/attack_to_compare/loss_test/property_{property}_seed_{seed}_loss_test_results_4_round_digits.csv"
    if attack_setting == 'different_predictor_BlackBox':
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/attack_to_compare/loss_test/property_{property}_seed_{seed}_loss_test_results.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/attack_to_compare/loss_test/property_{property}_seed_{seed}_loss_test_results_4_round_digits.csv"


    # add(append) to the results file if exists else create one and write into it
    dict_for_csv = {'seed': seed, 'backbone': backbone, 'property': property, 'distribution': distribution[1], 'loss_test_result': Loss_Test_result, 'accuracy_difference': accuracy_differences}
    if attack_setting == 'different_predictor_BlackBox':
        dict_for_csv['target_predictor_architecture'] = target_predictor_architecture
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    if os.path.exists(result_file):
        with open(result_file, 'a') as f:
            w = csv.DictWriter(f, dict_for_csv.keys())
            w.writerow(dict_for_csv)
    else:
        with open(result_file, 'w') as f:
            w = csv.DictWriter(f, dict_for_csv.keys())
            w.writeheader()
            w.writerow(dict_for_csv)
    # close the file
    # convert dict for csv if int values to round till 4 digits after the dot
    round_four_dict_for_csv = {k: round(v, 4) if isinstance(v, float) else v for k, v in
                               dict_for_csv.items()}
    # change the value of the gap to the sub 1 attack rounf - sub 2 attack round
    # this is IMPORTANT since the round of gap != to round of sub1 success - round 2 sub 1 success

    # add(append) to the results file if exists else create one and write into it
    print("round_four_dict_for_csv", round_four_dict_for_csv)
    print("result_file_4_round_digits: ", result_file_4_round_digits)
    print("does the file 4 digits exists: ", os.path.exists(result_file_4_round_digits))

    if not os.path.exists(os.path.dirname(result_file_4_round_digits)):
        os.makedirs(os.path.dirname(result_file_4_round_digits))
    if os.path.exists(result_file_4_round_digits):
        with open(result_file_4_round_digits, 'a') as f:
            w = csv.DictWriter(f, round_four_dict_for_csv.keys())
            w.writerow(round_four_dict_for_csv)
    else:
        with open(result_file_4_round_digits, 'w') as f:
            w = csv.DictWriter(f, round_four_dict_for_csv.keys())
            w.writeheader()
            w.writerow(round_four_dict_for_csv)


    #add to run the loss test results
    # 'loss_test_result': Loss_Test_result,
    # 'accuracy_difference': accuracy_differences,
    run['loss_test_result'] = Loss_Test_result
    run['accuracy_difference'] = accuracy_differences

def query_budget_loss_test(traget_model_path, saving_path, seed,
                              backbone, property,
                              distribution, is_faceX_zoo, target_predictor_architecture = 1, attack_setting = "", dataset="CelebA", is_finetune_emb=False):
    """
    load the postive and negative samples
    Then, calculate the accuracy of the positive samples and the negative samples using different random number of query budget
    Finally, calculate the loss test result and the accuracy difference
    """
    # load the positive and negative samples
    exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path, seed, backbone, property,
                                                        list(distribution.keys())[0],
                                                        distribution.get(list(distribution.keys())[0]))
    dir_loss_test = exp_path + 'loss_test/'
    positive_samples = pd.read_csv("{}positive_samples.csv".format(dir_loss_test))
    negative_samples = pd.read_csv("{}negative_samples.csv".format(dir_loss_test))

    # set the max indices for the positive and negative samples
    max_positive_index = positive_samples.shape[0]
    max_negative_index = negative_samples.shape[0]
    #choose the min between the max positive and negative indices
    min_index_samp_list = min(max_positive_index, max_negative_index)
    # set the query budget randomly using the max index and the seed of the experiment
    # take the max indices but ser the order in the random way
    random_indices_of_samples = random_pairs(num_of_given_samples=min_index_samp_list,
                                                       num_of_wanted_samples=min_index_samp_list,
                                                       seed=seed)
    # create a list of the number of unique samples to take
    one_to_hundred_list = list(range(1, 100))
    samples_size_list = list(range(100, min_index_samp_list, 10))
    if (min_index_samp_list > 99):
        # add to the left the one_to_ten_list
        samples_size_list = one_to_hundred_list + samples_size_list
    # create a list of the results
    results_list = []
    for samples_size in samples_size_list:
        #choose the sample size indices from the random indices
        random_indices_of_samples_chosen = random_indices_of_samples[:samples_size]
        # iloc the random indices for the positive and negative samples
        positive_samples_chosen = positive_samples.iloc[random_indices_of_samples_chosen]
        negative_samples_chosen = negative_samples.iloc[random_indices_of_samples_chosen]
        #calc the accuracy of the p chosen ositive and negative samples
        acc_pos = accuracy_score(positive_samples_chosen['label'], positive_samples_chosen['prediction'])
        acc_neg = accuracy_score(negative_samples_chosen['label'], negative_samples_chosen['prediction'])
        print('Accuracy (positive samples):', acc_pos)
        print('Accuracy (negative samples):', acc_neg)
        print('Loss Tess result:', int(acc_pos > acc_neg))
        accuracy_differences = acc_neg - acc_pos  ## keep the difference as in my code: dist 0 - dist 100
        print('Accuracy difference:', accuracy_differences)
        Loss_Test_result = int(acc_pos > acc_neg)
        # save the result in a list
        if attack_setting == 'different_predictor_BlackBox':
            results_list.append([seed, property, backbone,
                                 distribution[1], samples_size, acc_pos, acc_neg,
                                 accuracy_differences, Loss_Test_result, target_predictor_architecture])
        else:
            results_list.append([seed, property, backbone,
                                 distribution[1], samples_size, acc_pos, acc_neg,
                                 accuracy_differences, Loss_Test_result])


        # results_list.append([unique_samples_size, relative_success_according_to_conf_scores_sub_0, relative_success_according_to_conf_scores_sub_1])

    if attack_setting == 'different_predictor_BlackBox':
        results_df = pd.DataFrame(results_list, columns=['seed', 'property', 'backbone',
                                                         'target_distribution', 'samples_size', 'acc_pos', 'acc_neg',
                                                         'accuracy_differences', 'Loss_Test_result', 'target_predictor_architecture'])
    else:
        # write to csv file
        results_df = pd.DataFrame(results_list, columns=['seed', 'property', 'backbone',
                                                         'target_distribution', 'samples_size', 'acc_pos', 'acc_neg',
                                                         'accuracy_differences', 'Loss_Test_result'])

    if dataset == "CelebA":
        prefix_for_path = ""
    elif dataset == "MAAD_Face":
        prefix_for_path = "MAAD_Face_Results/"
    else:
        raise Exception("dataset not supported")

    #check if fine tune emb
    if is_finetune_emb:
        fine_tune_path = "fined_tuning_embedder_Results/"
    else:
        fine_tune_path = ""

    result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/loss_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_loss_test.csv"
    result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/loss_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_loss_test_4_round_digits.csv"
    if attack_setting == 'different_predictor_BlackBox':
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_loss_test.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_loss_test_4_round_digits.csv"

    save_df_and_df_round_to_4(result_file, result_file_4_round_digits, results_df)







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
    backbone = config['backbone']  # knowledege in white_box attack!!!!
    # all the target model dist hold in their directory
    # the same attack dataset for the substitute model which we want to train on the attack dataset
    # so it does not realy matter from whome we will load the attack dataset it is the same
    # (and is non overlapping with any of the target model datasets)
    # load the predictor architecture which is int
    target_predictor_architecture = int(config['target_predictor_architecture'])
    attack_setting_of_AdversariaLeak = config['attack_setting_of_AdversariaLeak']
    is_faceX_zoo = config.getboolean('is_faceX_zoo')
    dataset = config['dataset']
    is_finetune_emb = config.getboolean('is_finetune_emb')

    train_all_dist = config.getboolean('train_all_dist')

    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)

    if train_all_dist:
        print("train all dist")
        for dist in tqdm ([0, 25, 50, 75, 100]):
            distribution = {1: dist,
                            -1: 100 - dist}
            saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
            traget_model_path = f"{prefix_traget_model_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{dist}/"
            print("Starting loss test attack for target model with dist: ", dist)
            # bellow for neptune to add hyperparametrs and intiallize the neptune recoder
            hyper_params = {
                'seed': seed,
                'backbone': backbone,
                'property': property,
                'distribution': distribution[1],
                'target_predictor_architecture': target_predictor_architecture
            }
            run = neptune_recoder(exp_name='loss_test', description='loss_test',
                                  tags=[f'loss_test_predictor_{target_predictor_architecture}'],
                                  hyperparameters=hyper_params)
            experiment_loss_test(traget_model_path=traget_model_path, saving_path=saving_path, seed=seed,
                              backbone=backbone, property=property,
                              distribution=distribution, is_faceX_zoo=is_faceX_zoo,
                              target_predictor_architecture =target_predictor_architecture,
                              attack_setting=attack_setting_of_AdversariaLeak, run=run, dataset=dataset, is_finetune_emb=is_finetune_emb)
            query_budget_loss_test(traget_model_path, saving_path, seed,
                                   backbone, property,
                                   distribution, is_faceX_zoo,
                                   target_predictor_architecture =target_predictor_architecture,
                                   attack_setting=attack_setting_of_AdversariaLeak, dataset=dataset, is_finetune_emb=is_finetune_emb)
            # bellow for neptune to close it after finish use it
            run.stop()
            print("Finished creating target model with dist: ", dist)
            # clean the current expirement data delete what necesseary use garabage coolector and clean chache of Cuda
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("train only dist: ", distribution_1_value)
        distribution = {1: distribution_1_value,
                        -1: 100 - distribution_1_value}
        saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
        traget_model_path = f"{prefix_traget_model_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{distribution_1_value}/"
        hyper_params = {
            'seed': seed,
            'backbone': backbone,
            'property': property,
            'distribution': distribution[1],
            'target_predictor_architecture': target_predictor_architecture
        }
        run = neptune_recoder(exp_name='loss_test', description='loss_test',
                              tags=[f'loss_test_predictor_{target_predictor_architecture}'],
                              hyperparameters=hyper_params)
        experiment_loss_test(traget_model_path=traget_model_path, saving_path=saving_path, seed=seed,
                              backbone=backbone, property=property,
                              distribution=distribution, is_faceX_zoo=is_faceX_zoo,
                          target_predictor_architecture =target_predictor_architecture,
                          attack_setting=attack_setting_of_AdversariaLeak, run=run, dataset=dataset, is_finetune_emb=is_finetune_emb)
        query_budget_loss_test(traget_model_path, saving_path, seed,
                               backbone, property,
                               distribution, is_faceX_zoo,
                               target_predictor_architecture =target_predictor_architecture,
                               attack_setting=attack_setting_of_AdversariaLeak,dataset=dataset, is_finetune_emb=is_finetune_emb)
