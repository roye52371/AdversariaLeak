import configparser
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch import optim

from FR_System.Data.data_utils import Demo_creat_yes_records, load_predictor, filter_benign_pairs, \
    load_embedder_and_predictor_for_eval
from FR_System.Embedder.embedder import Embedder, process_image
from FR_System.fr_system import FR_Api
from Run_scripts.python_scripts_files.Attack_ART_new import check_if_all_files_exist, Attack

sys.path.append("/sise/home/royek/Toshiba_roye/")

from Attacks.AdversariaLeak.AdversariaLeak import TargetModel, \
    SubstituteModel, substitute_to_target_attack


def Demo_create_adv_samples(backbone, predictor_path, attack_name, params, demo_yes_pairs_path, demo_curr_sub_folder_path, predictor_architecture_type, is_finetune_emb, dataset_name, n_in, is_faceX_zoo):
    """
    create adversarial samples for the specific FR system substitute model given in the params.
    The function creates adversarial samples for the FR system.
    :param backbone: Required. str. The name of the backbone.
    :param predictor_path: Required. str. The path saves weights of the predictor.
    :param attack_name: Required. str. The name of the attack.
    :param params: Required. dict. The parameters of the attack.
    :param demo_yes_pairs_path: Required. str. The path to the demo yes pairs.
    :param demo_curr_sub_folder_path: Required. str. The path to the current demo sub folder.
    :param is_faceX_zoo: Optional. bool. The flag indicates whether the data is from FaceX-Zoo dataset. Default: True.
    """
    #load the model
    #print("Create embedder")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # loading predictor and embedder for the current epoch
    embeder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=backbone, device=device,
                                                                 is_faceX_zoo=is_faceX_zoo,
                                                                 predictor_architecture=predictor_architecture_type,
                                                                 path=predictor_path,
                                                                 is_finetune_emb=is_finetune_emb,
                                                                 dataset_name=dataset_name, n_in=n_in)

    device_type_str = device.type  # "gpu" # cuda:0

    # Create complete API
    #print("Create complete API")
    fr = FR_Api(embedder=embeder, predictor=predictor)

    # Create yes records
    #print("Create yes records")
    #if all the relevant file before attack exists (meant one art attack executed)
    # load them and use them in the new art attack
    #load positive same person pairs
    positive = pd.read_csv("{}positive_paired_data.csv".format(demo_yes_pairs_path))
    #data_x = positive.drop(["label", f"{property}_path1", f"{property}_path2"], axis=1)

    data_y = pd.DataFrame(positive["label"], columns=["label"])
    # create directories for adverserial  data
    adv_data_path_directory = demo_curr_sub_folder_path + "adv_data/"
    before_adv_attack_data_path_directory = adv_data_path_directory + "before_adv_attack/"
    after_adv_attack_data_path_directory = adv_data_path_directory + "after_adv_attack/"
    after_adv_specific_attack_data_path_directory = after_adv_attack_data_path_directory + attack_name + "/"
    if not os.path.exists(adv_data_path_directory):
        os.makedirs(adv_data_path_directory)
    if not os.path.exists(before_adv_attack_data_path_directory):
        os.makedirs(before_adv_attack_data_path_directory)
    if not os.path.exists(after_adv_attack_data_path_directory):
        os.makedirs(after_adv_attack_data_path_directory)
    if not os.path.exists(after_adv_specific_attack_data_path_directory):
        os.makedirs(after_adv_specific_attack_data_path_directory)


    # convert data to numpy pairs for the pairs of images to fr verification task
    data_x = positive
    batch_x_test_val = []
    for i, row in data_x.iterrows():
        path1 = row["path1"]
        path2 = row["path2"]
        np_image1 = process_image(path1)
        np_image2 = process_image(path2)
        batch_x_test_val.append([np_image1, np_image2])
    #create the labels for the data, same person labels
    data_y_size = data_y.size
    data_y = np.full([data_y_size, 2], [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
    data_x = np.array(batch_x_test_val)

    assert data_x.shape[0] == data_y.shape[0]
    assert data_x.shape[0] == positive.shape[0] #must be the same num of samples
    #print("before filter data_y_shape[0]: ", data_y.shape[0])
    #keep only the all the yes pairs which  realy return the same person prediction whith the FR system
    data_x, _ = filter_benign_pairs(adv_data=data_x, labels=data_y, model=fr, use_properties=False)

    # save the filtered data and create labels correspondly
    data_y_size = data_x.shape[0]
    data_y = np.full([data_y_size, 2], [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
    #print("after filter data_y_shape[0]: ", data_y.shape[0])

    #create the labels for the data, not same person labels
    not_data_y = np.full([data_y_size, 2], [1, 0])
    #print("not_data_y_shape[0]: ", not_data_y.shape[0])
    #print("saving the following the before data in the following path: ")
    #print("{}all_samples.npy".format(before_adv_attack_data_path_directory))
    np.save("{}all_samples.npy".format(before_adv_attack_data_path_directory), data_x)




    #se the loss for the art attack
    if not os.path.exists("{}all_samples.npy".format(after_adv_specific_attack_data_path_directory)):
        loss = torch.nn.BCEWithLogitsLoss()
        attack = Attack(attack_name, fr, params,
                        lossf=loss,
                        input_shape=data_x.shape[1:],
                        nb_classes=2,
                        optimizer=optim.Adam(
                            list(fr.embedder.embedder.parameters()) + list(fr.predictor.nn.parameters()), lr=0.002),
                        channels_first=False,
                        device_type=device_type_str)  ############change to cpu back again
        adv_x = attack.generate(data_x=data_x,
                                data_y=not_data_y)
        #print("data_x: ", data_x)
        #print("adv_x", adv_x)
        #print("data_y: ", data_y)
        #print("not_data_y: ", not_data_y)
        assert data_x.shape == adv_x.shape
        #save the adv data samples
        #print("saving the after data in the following path: ")
        #print("{}all_samples.npy".format(after_adv_specific_attack_data_path_directory))
        #check if the all samples file exists
        np.save("{}all_samples.npy".format(after_adv_specific_attack_data_path_directory), adv_x)
    else:
        adv_x = np.load("{}all_samples.npy".format(after_adv_specific_attack_data_path_directory))
    return adv_x

if __name__ == "__main__":
    """
    The following is demo of AdversariaLeak attack using 5 identities of LFW dataset.
    THe demo is illustrated on the following:
    the pretrained backbone is iresnet100 (you can replace to RepVGG_B0), the property is
    male (you can replace to 5_oclock_shadow or Young), the seed is 42 (you can replace your own seed used) and the evasion attack is PGD
    (you can replace it to Carlini-Wagner L2). 
    Use the demo it after:
     (a) Training the target model using run_target_model_creation.py
     (b) phase 1 of AdversariaLeak attack -
     Training the substitute models using run_substitute_model_creation.py
     
    The demo is:
     (1) creating adversarial samples for each of the two substitute models (phase 2 of AdversariaLeak attack),
     (2) filtering the converged adversarial samples (phase 3 of AdversariaLeak attack),
     (3) then filtering the unique adversarial samples of each adversarial samples set (phase 3 of AdversariaLeak attack).
     (4) Finally calculate the proportion of the unique adversarial samples that mislead the target model,
         for each adversarial samples set, and inferred the property
         of the one who has the higher fraction of misleading samples (phase 4 of AdversariaLeak attack).
         
     *set the data_path and the target (target_model_path) and substitute models paths (sub_0_path and sub_100_path)
      according to your data path and your target and substitute model paths.
     notice the embedder pretrained backbones paths (in the embedder directory files) are according to a path you can access
      - if not then change it to your paths.
      
      In addition, "Run_scripts/gpu_files/all_gpu_options/run_adversariaLeak_demo" contatins the script to run the demo
      on gpu servers, if you need to (need to change the absolute paths their according to you paths)
      
      More information can be found in the guidelines.txt file.
    """
    data_path = "/sise/home/royek/AdversariaLeak_demo_code/FR_System/Data/Patrial_LFW/"  # "FR_System/Data/Patrial_LFW/"

    # Data creation
    print("Data creation")
    print("Creating the records of the positive samples - same person records")
    LFW_yes_records = Demo_creat_yes_records(data_path, ".jpg", mid_saving=False)
    print("crafting the adversarial samples for the two substitute models")
    # craft the adversarial samples
    #######
    backbone = "iresnet100"  # "RepVGG_B0"
    prefix_predictor_path = "/dt/shabtaia/dt-toshiba_2022/Roye/Expiriments/Seed_Everything_Substitute_Models/CelebA_with_clean_backbone/"
    attack_settings = "different_predictor_BlackBox" #
    n_in = 512
    #adjust dataset name according to the dataset you use
    dataset_name = "CelebA" #MAAD_Face
    if dataset_name == "CelebA":
        predictor_architecture_type_subs = 1
        is_fine_tune_subs_backbone = False
        is_fine_tune_target_backbone = False
    elif dataset_name == "MAAD_Face":
        predictor_architecture_type_subs = 8
        is_fine_tune_subs_backbone = True
        is_fine_tune_target_backbone = True
    else:
        #throw error
        raise ValueError("The dataset name is not supported")
    property = "Male"
    seed = 42
    attack_name = "ProjectedGradientDescent"
    params = {
        "max_iter": 15,
        "norm": 2,
        "eps": 0.3,
        "eps_step": 0.1,
        "targeted": True,
        "num_random_init": 0,
        "batch_size": 1
    }

    # set the flag to load a facexzoo model or not
    if backbone == "iresnet100":
        is_faceX_zoo = False
    elif backbone == "RepVGG_B0":
        is_faceX_zoo = True
    else:
        is_faceX_zoo = True

    # set the path to the substitute models - you can change it to your own path
    sub_0_path = f"{prefix_predictor_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_0/"
    sub_100_path = f"{prefix_predictor_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_100/"

    #######
    print("crafting the adversarial samples for the two substitute models - phase 2 of AdversariaLeak attack")
    demo_sub_0_folder_path = f"{data_path}{backbone}/sub_0/"
    demo_sub_100_folder_path = f"{data_path}{backbone}/sub_100/"
    adv_x_sub_0 = Demo_create_adv_samples(backbone, sub_0_path, attack_name, params, data_path, demo_sub_0_folder_path,
                                          is_faceX_zoo=is_faceX_zoo, predictor_architecture_type=predictor_architecture_type_subs, is_finetune_emb=is_fine_tune_subs_backbone, dataset_name=dataset_name, n_in=n_in)
    adv_x_sub_100 = Demo_create_adv_samples(backbone, sub_100_path, attack_name, params, data_path,
                                            demo_sub_100_folder_path, is_faceX_zoo=is_faceX_zoo, predictor_architecture_type=predictor_architecture_type_subs, is_finetune_emb=is_fine_tune_subs_backbone, dataset_name=dataset_name, n_in=n_in)

    # parameters for the rest of the attack phases
    print(
        "filter the converged and unique adversarial samples; calculate FMS rate and infer the property - phase 3 amd 4 of AdversariaLeak attack")

    prefix_target_model_path = "/dt/shabtaia/dt-toshiba_2022/Roye/Expiriments/Seed_Everything_Target_Models/CelebA_with_different_predictor_and_clean_backbone/"

    distribution_1_value_target_model = 75  # The distribution of the value "1" for the property.

    # ptroch lightning seed everything
    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)

    # set the path to the target model - you can change it to your own path
    target_model_path = f"{prefix_target_model_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{distribution_1_value_target_model}/"

    if attack_settings=="different_predictor_BlackBox":
        if dataset_name == "CelebA":
            predictor_architecture_type_target = 4
        elif dataset_name == "MAAD_Face":
            predictor_architecture_type_target = 9
        else:
            #throw error
            raise ValueError("The dataset name is not supported")
    else:
        if dataset_name == "CelebA":
            predictor_architecture_type_target = 1
        elif dataset_name == "MAAD_Face":
            predictor_architecture_type_target = 8
        else:
            #throw error
            raise ValueError("The dataset name is not supported")

    # load the target model
    target_model = TargetModel(seed=seed, property=property, distribution=distribution_1_value_target_model,
                               backbone=backbone,
                               predictor_path=target_model_path, is_faceX_zoo=is_faceX_zoo, predictor_architecture=predictor_architecture_type_target, is_finetune_emb=is_fine_tune_target_backbone, dataset=dataset_name)
    # load the substitute models
    substitute_model_1 = SubstituteModel(seed=seed, property=property, distribution=0,
                                         backbone=backbone,
                                         predictor_path=sub_0_path, attack_name=attack_name,
                                         is_faceX_zoo=is_faceX_zoo,
                                         predictor_architecture=predictor_architecture_type_subs, is_finetune_emb=is_fine_tune_subs_backbone, dataset=dataset_name,
                                         demo_sub_folder_path=demo_sub_0_folder_path,
                                         is_demo=True)
    substitute_model_2 = SubstituteModel(seed=seed, property=property, distribution=100,
                                         backbone=backbone,
                                         predictor_path=sub_100_path, attack_name=attack_name,
                                         is_faceX_zoo=is_faceX_zoo,
                                         predictor_architecture=predictor_architecture_type_subs, is_finetune_emb=is_fine_tune_subs_backbone, dataset=dataset_name,
                                         demo_sub_folder_path=demo_sub_100_folder_path,
                                         is_demo=True)

    # create attack object
    if attack_settings=="different_predictor_BlackBox":
        substitute_to_target_attack_obj = substitute_to_target_attack(target_model=target_model,
                                                                      sub_model1=substitute_model_1,
                                                                      sub_model2=substitute_model_2, property=property,
                                                                      dataset_for_all_models=dataset_name,
                                                                      settings=attack_settings,
                                                                      art_attack_name=attack_name, seed=seed)
    else: #default attack settings allready in the substitute_to_target_attack class
        substitute_to_target_attack_obj = substitute_to_target_attack(target_model=target_model,
                                                                      sub_model1=substitute_model_1,
                                                                      sub_model2=substitute_model_2, property=property,
                                                                      dataset_for_all_models=dataset_name,
                                                                      art_attack_name=attack_name, seed=seed)
    # run the demo attack
    substitute_to_target_attack_obj.demo_attack()