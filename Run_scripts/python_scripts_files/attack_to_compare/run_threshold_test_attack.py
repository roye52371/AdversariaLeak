import configparser
import csv
import gc
import sys
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import pytorch_lightning as pl

sys.path.append("/sise/home/royek/Toshiba_roye/")
from FR_System.Embedder.embedder import Embedder
from FR_System.fr_system import FR_Api, evaluation
from FR_System.Data.data_utils import CelebA_target_training_apply_distribution, CelebA_create_yes_records, \
    CelebA_create_no_records, load_predictor, neptune_recoder, \
    load_embedder_and_predictor_for_eval, Two_Class_test_Predict_func, random_pairs, save_df_and_df_round_to_4, \
    MAAD_Face_target_training_apply_distribution, MAAD_Face_create_yes_records, MAAD_Face_create_no_records



def load_two_sub_models(prefix_substitute_models_path, sub_models_backbone, sub_models_predictor_architecture, property, seed, device, is_faceX_zoo_subs, is_finetune_emb=None, n_in=None, dataset=None):
    """
    The function loads the two substitute models.
    :param prefix_substitute_models_path: Required. str. The path to the substitute models.
    :param sub_models_backbone: Required. str. The backbone of the substitute models.
    :param sub_models_predictor_architecture: Required. str. The architecture of the predictor of the substitute models.
    :param property: Required. str. The property of the substitute models.
    :param seed: Required. int. The seed of the experiment.
    :param device: Required. str. The device to run the experiment on.
    :param is_faceX_zoo_subs: Required. bool. True if the substitute models are from faceX_zoo.
    :param is_finetune_emb: Required. bool. True if the substitute models are finetuned.
    :return: list. The list of the substitute models.
    """

    #create empty list
    sub_models = [[], []]
    #load the two substitute models]
    substitute_model_1_path = f"{prefix_substitute_models_path}{property}/seed_{seed}/{seed}_{sub_models_backbone}_property_{property}_dist_1_0/"
    substitute_model_2_path = f"{prefix_substitute_models_path}{property}/seed_{seed}/{seed}_{sub_models_backbone}_property_{property}_dist_1_100/"
    # print("Create predictor as NN classifier")
    embedder_1, predictor_1, _ = load_embedder_and_predictor_for_eval(backbone=sub_models_backbone, device=device,
                                                               is_faceX_zoo=is_faceX_zoo_subs,
                                                               predictor_architecture=sub_models_predictor_architecture,
                                                               path=substitute_model_1_path, is_finetune_emb=is_finetune_emb, n_in=n_in, dataset_name=dataset)
    # Create complete API
    print("Create complete API for 1")
    fr_model_1 = FR_Api(embedder=embedder_1, predictor=predictor_1)
    #do the same for the second substitute model
    embedder_2, predictor_2, _ = load_embedder_and_predictor_for_eval(backbone=sub_models_backbone, device=device,
                                                                   is_faceX_zoo=is_faceX_zoo_subs,
                                                                   predictor_architecture=sub_models_predictor_architecture,
                                                                   path=substitute_model_2_path,
                                                                   is_finetune_emb=is_finetune_emb, n_in=n_in, dataset_name=dataset)
    # Create complete API
    print("Create complete API for 2")
    fr_model_2 = FR_Api(embedder=embedder_2, predictor=predictor_2)
    sub_models[0].append(fr_model_1)
    sub_models[1].append(fr_model_2)
    return sub_models



def experiment_threshold_test(traget_model_path, saving_path, seed, target_backbone, property, distribution, is_faceX_zoo_target, target_predictor_architecture = 1, attack_setting="", run=None,
                      prefix_substitute_models_path="", sub_models_backbone="", is_faceX_zoo_sub=False, sub_models_predictor_architecture=1, dataset=None,
                       is_finetune_target_emb=None, is_finetune_subs_emb=None):
    """
    This function execute the threshold test attack on the FR models.
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
    :param is_finetune_target_emb: Required. bool. True if the target model is finetuned.
    :param is_finetune_subs_emb: Required. bool. True if the substitute models are finetuned.
    :param dataset: Required. str. The dataset name.
    """
    exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path, seed,target_backbone, property, list(distribution.keys())[0],
                                                     distribution.get(list(distribution.keys())[0]))
    print("Data creation")
    attack_test_df = pd.read_csv("{}attack_test_df.csv".format(traget_model_path))
    attack_test_df = attack_test_df.set_index("Unnamed: 0")

    attack_dist_list = [{1: 100, -1: 0}, {1: 0, -1: 100}]
    dir_threshold_test = exp_path + 'threshold_test/'

    if not os.path.isdir(dir_threshold_test):
        os.mkdir(dir_threshold_test)
        #check if the distribution is higher than 0
        if distribution.get(list(distribution.keys())[0]) == 0:
            for i, attack_dist in enumerate(attack_dist_list):
                if dataset == "CelebA":
                    attack_test_df_dist = CelebA_target_training_apply_distribution(model_train_df=attack_test_df,
                                                                                    distribution=attack_dist,
                                                                                    property=property,
                                                                                    seed=seed)
                    attack_test_df_dist_yes_pairs = CelebA_create_yes_records(attack_test_df_dist, save_to=dir_threshold_test, property=property, property_annotations_included=False)
                    attack_test_df_dist_no_pairs = CelebA_create_no_records(attack_test_df_dist, yes_pairs_path=dir_threshold_test,
                                                                            save_to=dir_threshold_test, property=property, property_annotations_included=False)
                elif dataset == "MAAD_Face":
                    attack_test_df_dist = MAAD_Face_target_training_apply_distribution(model_train_df=attack_test_df,
                                                                                    distribution=attack_dist,
                                                                                    property=property,
                                                                                    seed=seed)
                    attack_test_df_dist_yes_pairs = MAAD_Face_create_yes_records(attack_test_df_dist, save_to=dir_threshold_test, property=property, property_annotations_included=False)
                    attack_test_df_dist_no_pairs = MAAD_Face_create_no_records(attack_test_df_dist, yes_pairs_path=dir_threshold_test,
                                                                            save_to=dir_threshold_test, property=property, property_annotations_included=False)
                else:
                    raise Exception("TODO dataset not supported")

                attack_test_df_pairs = pd.concat([attack_test_df_dist_yes_pairs, attack_test_df_dist_no_pairs],
                                                  ignore_index=True, axis=0)
                if attack_dist[1] == 100:
                    attack_test_df_pairs.to_csv("{}positive_samples.csv".format(dir_threshold_test))
                    positive_samples = attack_test_df_pairs
                else:
                    attack_test_df_pairs.to_csv("{}negative_samples.csv".format(dir_threshold_test))
                    negative_samples = attack_test_df_pairs
        else:
            #we can load the positive and negative samples from zero exp path, since all dist in this seed and backbone has the same attack test dist and same substiute models
            zero_dist_exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path, seed, target_backbone, property,
                                                                list(distribution.keys())[0], 0)
            zero_dir_threshold_test = zero_dist_exp_path + 'threshold_test/'
            positive_samples = pd.read_csv("{}positive_samples.csv".format(zero_dir_threshold_test))
            negative_samples = pd.read_csv("{}negative_samples.csv".format(zero_dir_threshold_test))
            positive_samples.to_csv("{}positive_samples.csv".format(dir_threshold_test))
            negative_samples.to_csv("{}negative_samples.csv".format(dir_threshold_test))
            #load celeba_negative_paired_data.csv
            if dataset== "CelebA":
                positive_paired_samples = pd.read_csv("{}celeba_positive_paired_data.csv".format(zero_dir_threshold_test))
                negative_paired_samples = pd.read_csv("{}celeba_negative_paired_data.csv".format(zero_dir_threshold_test))
                positive_paired_samples.to_csv("{}celeba_positive_paired_data.csv".format(dir_threshold_test))
                negative_paired_samples.to_csv("{}celeba_negative_paired_data.csv".format(dir_threshold_test))
            elif dataset == "MAAD_Face":
                positive_paired_samples = pd.read_csv("{}maad_face_positive_paired_data.csv".format(zero_dir_threshold_test))
                negative_paired_samples = pd.read_csv("{}maad_face_negative_paired_data.csv".format(zero_dir_threshold_test))
                positive_paired_samples.to_csv("{}maad_face_positive_paired_data.csv".format(dir_threshold_test))
                negative_paired_samples.to_csv("{}maad_face_negative_paired_data.csv".format(dir_threshold_test))
            else:
                raise Exception("TODO dataset not supported")

    else:
        positive_samples = pd.read_csv("{}positive_samples.csv".format(dir_threshold_test))
        negative_samples = pd.read_csv("{}negative_samples.csv".format(dir_threshold_test))


    # Create embedder
    print("Create embedder")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if target_backbone.startswith("resnet50") or target_backbone.startswith("senet50"):
        if not is_faceX_zoo_target:
            n_in_target = 2048
            increase_shape_target = True
        else:
            #throw exception
            raise Exception("model_name must be resnet50 or senet50 and it is not from facexzoo")
    else:
        n_in_target = 512
        increase_shape_target = False

    #same for substitute models
    if sub_models_backbone.startswith("resnet50") or sub_models_backbone.startswith("senet50"):
        if not is_faceX_zoo_sub:
            n_in_subs = 2048
            increase_shape_subs = True
        else:
            #throw exception
            raise Exception("model_name must be resnet50 or senet50 and it is not from facexzoo")
    else:
        n_in_subs = 512
        increase_shape_subs = False


    print(f"device: {device}")
    print(f"device.type: {device.type}")
    # Create predictor
    print("Creat predictor")

    embedder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=target_backbone, device=device,
                                                                  is_faceX_zoo=is_faceX_zoo_target,
                                                                  predictor_architecture=target_predictor_architecture,
                                                                  path=traget_model_path,
                                                                  is_finetune_emb=is_finetune_target_emb,
                                                                  n_in=n_in_target, dataset_name=dataset)

    # Create complete API for target models
    fr = FR_Api(embedder=embedder, predictor=predictor)

    #get the two substitute models from the same seed property
    sub_models = load_two_sub_models(prefix_substitute_models_path=prefix_substitute_models_path,
                                     sub_models_backbone=sub_models_backbone,
                                     sub_models_predictor_architecture=sub_models_predictor_architecture,
                                     property=property, seed=seed, device=device,
                                     is_faceX_zoo_subs=is_faceX_zoo_sub, is_finetune_emb=is_finetune_subs_emb, n_in=n_in_subs, dataset=dataset)#, is_finetune_emb=is_finetune_subs_emb)

    # 1. Gaps and k
    gap = [0, 0]
    acc_all = np.zeros((2, len(sub_models[0]), 2))  # (model type, number of sub-model, test data type)

    if distribution.get(list(distribution.keys())[0]) == 0:
        # #0 and 1 indices should heve the same length anyway in submodles list
        #check if csv acc_all exists
        if os.path.exists("{}acc_all.npy".format(dir_threshold_test)):
            acc_all = np.load("{}acc_all.npy".format(dir_threshold_test))
        else:
            for i in (0, 1):
                for j, sub_model in enumerate(sub_models[i]):
                    pred_pos = Two_Class_test_Predict_func(positive_samples, sub_model, increase_shape=increase_shape_subs)
                    pred_pos = pred_pos.argmax(axis=1)
                    positive_samples['prediction'] = pred_pos

                    pred_neg = Two_Class_test_Predict_func(negative_samples, sub_model, increase_shape=increase_shape_subs)
                    pred_neg = pred_neg.argmax(axis=1)

                    negative_samples['prediction'] = pred_neg
                    acc_all[i, j, 1] = accuracy_score(positive_samples['label'], positive_samples['prediction'])
                    acc_all[i, j, 0] = accuracy_score(negative_samples['label'], negative_samples['prediction'])
            #save acc_all id dist is 0
            np.save("{}acc_all.npy".format(dir_threshold_test), acc_all)
    else: #load acc_all if dist is not 0, load from zero dist exp path
        zero_dist_exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path, seed, target_backbone, property,
                                                                list(distribution.keys())[0], 0)
        zero_dir_threshold_test = zero_dist_exp_path + 'threshold_test/'
        acc_all = np.load("{}acc_all.npy".format(zero_dir_threshold_test))

    for c in (0, 1):
        gap[c] = np.sum(acc_all[0, :, c]) - np.sum(acc_all[1, :, c])
        print(f'gap[{c}]: {gap[c]}')
    k = int(abs(gap[0]) < abs(gap[1]))
    print(f'k: {k}')

    # 2. Threshold
    Sk = negative_samples if k == 0 else positive_samples
    lambda_candidate = sorted(acc_all[:, :, k].flatten())
    if len(lambda_candidate) > 2:
        lambda_ = 0
        delta_max = 0
        for lambda_tmp in lambda_candidate:
            if gap[k] >= 0:
                delta_tmp = np.sum(acc_all[0, :, k].flatten() >= lambda_tmp) + np.sum(
                    acc_all[1, :, k].flatten() < lambda_tmp)
            else:
                delta_tmp = np.sum(acc_all[0, :, k].flatten() < lambda_tmp) + np.sum(
                    acc_all[1, :, k].flatten() >= lambda_tmp)
            if delta_tmp > delta_max:
                lambda_ = lambda_tmp
                delta_max = delta_tmp
        print(f'lambda_candidate {lambda_candidate}')
        print(f'acc_0 {acc_all[0, :, k].flatten()}')
        print(f'acc_1 {acc_all[1, :, k].flatten()}')
        print(f'lambda_ {lambda_}')
        print(f'delta_max {delta_max}')
    elif len(lambda_candidate) == 2:
        #if we had only two substitute model (one for each) we will receive two lambdas candiadtes,
        #with two candidates the the part of finding the maximal accuracy threshold will bring the same delta for both lambdas
        #therefore the first one will be chosen all the time, and it has no meaning, therefore we will take the median instead
        # lambda will be the median of the two
        lambda_ = np.median(lambda_candidate)
        print(f"lambda median chosen is: {lambda_}, while the candidates are: {lambda_candidate}")
    else:
        raise Exception("lambda_candidate is empty or either have on value which is not possible"
                        " (one candidate should be at list for each distribution and we have 0 and 100 distributions)")

    Sk, b, gap_from_threshold = Threshold_test_target_prediction_phase(Sk, fr, gap, k, lambda_, dir_threshold_test, increase_shape=increase_shape_target)

    print(f'Predicted distribution: {b}')

    if dataset == "CelebA":
        prefix_for_path = ""
    elif dataset == "MAAD_Face":
        prefix_for_path = "MAAD_Face_Results/"
    else:
        raise Exception("dataset not supported")

    #check if fine tune emb
    if is_finetune_target_emb:
        fine_tune_path = "fined_tuning_embedder_Results/"
    else:
        fine_tune_path = ""

    result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}property_{property}/seed_{seed}/attack_to_compare/threshold_test/property_{property}_seed_{seed}_threshold_test_results.csv"
    result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}property_{property}/seed_{seed}/attack_to_compare/threshold_test/property_{property}_seed_{seed}_threshold_test_results_4_round_digits.csv"
    if attack_setting == 'different_predictor_BlackBox':
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/attack_to_compare/threshold_test/property_{property}_seed_{seed}_threshold_test_results.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/property_{property}/seed_{seed}/attack_to_compare/threshold_test/property_{property}_seed_{seed}_threshold_test_results_4_round_digits.csv"


    # add(append) to the results file if exists else create one and write into it
    dict_for_csv = {'seed': seed, 'backbone': target_backbone, 'property': property, 'distribution': distribution[1],
                    'threshold_test_result': b, 'gap_from_threshold': gap_from_threshold}
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

    # add to run the loss test results
    # 'loss_test_result': Loss_Test_result,
    # 'accuracy_difference': accuracy_differences,
    run['threshold_test_result'] = b
    run['gap_from_threshold'] = gap_from_threshold

    #now lets execute the threshold test
    print("enter the query budget function\n")
    query_budget_threshold_test(Sk, fr, gap, k, lambda_, saving_path, seed, target_backbone, property, distribution, target_predictor_architecture, attack_setting, is_finetune_emb=is_finetune_target_emb, dataset=dataset)


def Threshold_test_target_prediction_phase(Sk, fr, gap, k, lambda_, dir_threshold_test, increase_shape=False):
    """
    The function runs the threshold test prediction phase on the target model.
    :param Sk: Required. pd.DataFrame. The samples to test.
    :param fr: Required. FR_Api. The FR system.
    :param gap: Required. list. The gap between the two substitute models.
    :param k: Required. int. The index of the substitute model with the higher accuracy.
    :param lambda_: Required. float. The threshold.
    :param dir_threshold_test: Required. str. The path to save the results.
    :return: int. The predicted distribution.
    """
    # 3. Prediction

    #check if the csv is already exists
    if os.path.exists("{}Sk_samples.csv".format(dir_threshold_test)):
        Sk = pd.read_csv("{}Sk_samples.csv".format(dir_threshold_test))
    else:
        pred = Two_Class_test_Predict_func(Sk, fr, increase_shape=increase_shape)
        pred = pred.argmax(axis=1)
        Sk['prediction'] = pred
        Sk.to_csv("{}Sk_samples.csv".format(dir_threshold_test))
    b, acc_diff_from_threshold = get_b_and_threshold_gap(Sk, gap, k, lambda_)

    return Sk, b, acc_diff_from_threshold


def get_b_and_threshold_gap(Sk, gap, k, lambda_):
    acc_target = accuracy_score(Sk['label'], Sk['prediction'])
    if gap[k] >= 0:
        b = int(acc_target >= lambda_)
        acc_diff_from_threshold = acc_target - lambda_ #lambda_ - acc_target  # want it like fms gap 0 - 100 (so with property will be negative, and without will be positive)
        #did represents the gap between the threshold and the accuracy of 0-100 since we flip fterwards the b in order to adapt it to the right k
        #so we needed to cacluate the gap this way
    else:
        b = int(acc_target < lambda_)
        acc_diff_from_threshold = lambda_ - acc_target #acc_target - lambda_  # want it like fms gap 0 - 100 (so with property will be negative, and without will be positive)
        # did represents the gap between the threshold and the accuracy of 0-100 since we flip fterwards the b in order to adapt it to the right k
        # so we needed to cacluate the gap this way
    b = 1 - b #flip the b in order to adapt it to the right k (since when ga[k] >=0 it is k zero and vice versa, so we need to flip the b)
    # if gap[k] >= 0: then k=0
    # if int(acc_target >= lambda_)==1, then k is the right one, but b will be 1, so we need to flip it to 0
    return b, acc_diff_from_threshold


def query_budget_threshold_test(Sk, fr, gap, k, lambda_, saving_path, seed,
                              backbone, property,
                              distribution, target_predictor_architecture = 1, attack_setting = "", is_finetune_emb=None, dataset=None):
    """
    load the postive and negative samples
    Then, calculate the accuracy of the positive samples and the negative samples using different random number of query budget
    Finally, calculate the loss test result and the accuracy difference
    """
    # load the positive and negative samples
    exp_path = '{}{}_{}_property_{}_dist_{}_{}/'.format(saving_path, seed, backbone, property,
                                                        list(distribution.keys())[0],
                                                        distribution.get(list(distribution.keys())[0]))
    dir_threshold_test = exp_path + 'threshold_test/'

    #get the size of sk
    sk_size = Sk.shape[0]


    # set the query budget randomly using the max index and the seed of the experiment
    # take the max indices but ser the order in the random way
    random_indices_of_samples = random_pairs(num_of_given_samples=sk_size,
                                                       num_of_wanted_samples=sk_size,
                                                       seed=seed)
    # create a list of the number of unique samples to take
    one_to_hundred_list = list(range(1, 100))
    samples_size_list = list(range(100, sk_size, 10))
    if (sk_size > 99):
        # add to the left the one_to_ten_list
        samples_size_list = one_to_hundred_list + samples_size_list
    # create a list of the results
    results_list = []
    print("Start the loop on the samples size list in query budget\n")

    for samples_size in tqdm(samples_size_list):
        #choose the sample size indices from the random indices
        random_indices_of_samples_chosen = random_indices_of_samples[:samples_size]

        #iloc on sk
        Sk_chosen = Sk.iloc[random_indices_of_samples_chosen]
        #calc the accuracy of the p chosen ositive and negative samples
        b, gap_from_threshold = get_b_and_threshold_gap(Sk_chosen, gap, k, lambda_)

        Threshold_Test_result = b
        # save the result in a list
        if attack_setting == 'different_predictor_BlackBox':
            results_list.append([seed, property, backbone,
                                 distribution[1], samples_size,
                                 gap_from_threshold, Threshold_Test_result, target_predictor_architecture])
        else:
            results_list.append([seed, property, backbone,
                                 distribution[1], samples_size,
                                 gap_from_threshold, Threshold_Test_result])



    if attack_setting == 'different_predictor_BlackBox':
        results_df = pd.DataFrame(results_list, columns=['seed', 'property', 'backbone',
                                                         'target_distribution', 'samples_size',
                                                         'gap_from_threshold', 'Threshold_Test_result', 'target_predictor_architecture'])
    else:
        # write to csv file
        results_df = pd.DataFrame(results_list, columns=['seed', 'property', 'backbone',
                                                         'target_distribution', 'samples_size',
                                                         'gap_from_threshold', 'Threshold_Test_result'])


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

    result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/threshold_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_threshold_test.csv"
    result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/threshold_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_threshold_test_4_round_digits.csv"
    if attack_setting == 'different_predictor_BlackBox':
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_threshold_test.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/{backbone}/property_{property}/seed_{seed}/target_dist_{distribution[1]}/query_budget_results_threshold_test_4_round_digits.csv"

    print("save query budget in csv for current distribution in the path:\n")
    print(result_file_4_round_digits)
    print("\n")
    #print the df
    print("the results df\n")
    print(results_df)
    print("\n")
    save_df_and_df_round_to_4(result_file, result_file_4_round_digits, results_df)





if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    config = config['DEFAULT']

    prefix_traget_model_path = config['prefix_traget_model_path']
    prefix_substitute_models_path = config['prefix_substitute_models_path']
    prefix_saving_path = config['prefix_saving_path']
    seed = int(config['seed'])
    property = config['property']
    distribution_1_value = int(config["distribution_1_value"])  # The distribution of the value "1" for the property.
    distribution = {1: distribution_1_value,
                    -1: 100 - distribution_1_value}
    target_backbone = config['target_backbone']  #
    sub_models_backbone = config['sub_models_backbone']  #
    # all the target model dist hold in their directory
    # the same attack dataset for the substitute model which we want to train on the attack dataset
    # so it does not realy matter from whome we will load the attack dataset it is the same
    # (and is non overlapping with any of the target model datasets)
    # load the predictor architecture which is int
    target_predictor_architecture = int(config['target_predictor_architecture'])
    attack_setting_of_AdversariaLeak = config['attack_setting_of_AdversariaLeak']

    train_all_dist = config.getboolean('train_all_dist')

    sub_models_predictor_architecture = int(config['sub_models_predictor_architecture'])
    is_finetune_target_emb = config.getboolean('is_finetune_target_emb')
    is_finetune_subs_emb = config.getboolean('is_finetune_subs_emb')
    dataset = config['dataset']
    if target_backbone == "iresnet100":
        is_faceX_zoo_target = False
    elif target_backbone == "RepVGG_B0":
        is_faceX_zoo_target = True
    else:
        raise Exception("TODO target_backbone not supported")
    if sub_models_backbone == "iresnet100":
        is_faceX_zoo_sub = False
    elif sub_models_backbone == "RepVGG_B0":
        is_faceX_zoo_sub = True
    else:
        raise Exception("TODO sub_models_backbone not supported")
    if attack_setting_of_AdversariaLeak == "different_predictor_BlackBox":
        assert target_predictor_architecture != sub_models_predictor_architecture
        # and backbones must be the same
        assert target_backbone == sub_models_backbone
    if attack_setting_of_AdversariaLeak == "Semi_BlackBox":
        # backbones must be different, and the predictor architecture must be the same
        assert target_backbone != sub_models_backbone
        assert target_predictor_architecture == sub_models_predictor_architecture

    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)

    if train_all_dist:
        print("train all dist")
        for dist in tqdm([0, 25, 50, 75, 100]):
            distribution = {1: dist,
                            -1: 100 - dist}
            saving_path = f"{prefix_saving_path}{property}/seed_{seed}/"
            traget_model_path = f"{prefix_traget_model_path}{property}/seed_{seed}/{seed}_{target_backbone}_property_{property}_dist_1_{dist}/"
            print("Starting threshold test attack for target model with dist: ", dist)
            # bellow for neptune to add hyperparametrs and intiallize the neptune recoder
            hyper_params = {
                'seed': seed,
                'backbone': target_backbone,
                'property': property,
                'distribution': distribution[1],
                'target_predictor_architecture': target_predictor_architecture
            }
            run = neptune_recoder(exp_name='threshold_test', description='threshold_test',
                                  tags=[f'threshold_test_predictor_{target_predictor_architecture}',
                                        f'{attack_setting_of_AdversariaLeak}'],
                                  hyperparameters=hyper_params)

            experiment_threshold_test(traget_model_path=traget_model_path, saving_path=saving_path, seed=seed,
                              target_backbone=target_backbone, property=property, distribution=distribution,
                              is_faceX_zoo_target=is_faceX_zoo_target, target_predictor_architecture=target_predictor_architecture,
                              attack_setting=attack_setting_of_AdversariaLeak, run=run,
                              prefix_substitute_models_path=prefix_substitute_models_path, sub_models_backbone=sub_models_backbone,
                              is_faceX_zoo_sub=is_faceX_zoo_sub,
                              sub_models_predictor_architecture=sub_models_predictor_architecture, dataset=dataset,
                               is_finetune_target_emb=is_finetune_target_emb, is_finetune_subs_emb=is_finetune_subs_emb)

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
        traget_model_path = f"{prefix_traget_model_path}{property}/seed_{seed}/{seed}_{target_backbone}_property_{property}_dist_1_{distribution_1_value}/"
        hyper_params = {
            'seed': seed,
            'backbone': target_backbone,
            'property': property,
            'distribution': distribution[1],
            'target_predictor_architecture': target_predictor_architecture
        }
        run = neptune_recoder(exp_name='threshold_test', description='threshold_test',
                              tags=[f'threshold_test_predictor_{target_predictor_architecture}',
                                    f'{attack_setting_of_AdversariaLeak}'],
                              hyperparameters=hyper_params)

        experiment_threshold_test(traget_model_path=traget_model_path, saving_path=saving_path, seed=seed,
                          target_backbone=target_backbone, property=property, distribution=distribution,
                          is_faceX_zoo_target=is_faceX_zoo_target,
                          target_predictor_architecture=target_predictor_architecture,
                          attack_setting=attack_setting_of_AdversariaLeak, run=run,
                          prefix_substitute_models_path=prefix_substitute_models_path,
                          sub_models_backbone=sub_models_backbone,
                          is_faceX_zoo_sub=is_faceX_zoo_sub,
                          sub_models_predictor_architecture=sub_models_predictor_architecture, dataset=dataset,
                           is_finetune_target_emb=is_finetune_target_emb, is_finetune_subs_emb=is_finetune_subs_emb)

