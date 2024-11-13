import configparser
import gc
import sys

import torch
import pytorch_lightning as pl

sys.path.append("/sise/home/royek/Toshiba_roye/")

from Attacks.AdversariaLeak.AdversariaLeak import TargetModel, \
    SubstituteModel, substitute_to_target_attack


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    config = config['DEFAULT']
    target_backbone = config['target_backbone']
    sub_models_backbone = config['sub_models_backbone']
    prefix_target_model_path = config['prefix_target_model_path']
    prefix_substitute_models_path = config['prefix_substitute_models_path']
    seed = int(config['seed'])
    property = config['property']
    distribution_1_value_target_model = int(config["distribution_1_value_target_model"])
    distribution_1_value_sub_1 = int(config["distribution_1_value_sub_1"])  # The distribution of the value "1" for the property.
    distribution_1_value_sub_2 = int(config["distribution_1_value_sub_2"])  # The distribution of the value "1" for the property.
    attack_of_art = config['attack_of_art']
    attack_setting = config['attack_setting']
    dataset_for_all_models = config['dataset']
    train_all_dist = config.getboolean('train_all_dist')
    #load the predictor architecture which is int
    target_predictor_architecture = int(config['target_predictor_architecture'])
    sub_models_predictor_architecture = int(config['sub_models_predictor_architecture'])
    is_finetune_target_emb = config.getboolean('is_finetune_target_emb')
    is_finetune_subs_emb = config.getboolean('is_finetune_subs_emb')
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
    if attack_setting == "different_predictor_BlackBox":
        assert target_predictor_architecture != sub_models_predictor_architecture
        #and backbones must be the same
        assert target_backbone == sub_models_backbone
    if attack_setting == "Semi_BlackBox":
        # backbones must be different, and the predictor architecture must be the same
        assert target_backbone != sub_models_backbone
        assert target_predictor_architecture == sub_models_predictor_architecture
    #ptroch lightning seed everything
    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)

    if train_all_dist:
        print("all properties")
        # adjust according to the dataset evaluated - in celeba we evaluated "5_o_Clock_Shadow", "Young" and "Male"; in maad-face we only evaluated "male" property
        for property in ["5_o_Clock_Shadow", "Young", "Male"]:  #adjust according to the dataset evaluated
            substitute_model_1_path = f"{prefix_substitute_models_path}{property}/seed_{seed}/{seed}_{sub_models_backbone}_property_{property}_dist_1_{distribution_1_value_sub_1}/"
            substitute_model_2_path = f"{prefix_substitute_models_path}{property}/seed_{seed}/{seed}_{sub_models_backbone}_property_{property}_dist_1_{distribution_1_value_sub_2}/"
            print("train all dist")
            for dist in [100,75,50,25,0]:
                print(f"run attak for dist: {dist}")
                distribution_1_value_target_model = dist
                target_model_path = f"{prefix_target_model_path}{property}/seed_{seed}/{seed}_{target_backbone}_property_{property}_dist_1_{distribution_1_value_target_model}/"
                target_model = TargetModel(seed=seed, property=property, distribution=distribution_1_value_target_model,
                                           backbone=target_backbone,
                                           predictor_path=target_model_path, is_faceX_zoo=is_faceX_zoo_target,
                                           predictor_architecture=target_predictor_architecture, is_finetune_emb = is_finetune_target_emb, dataset=dataset_for_all_models)
                # Choose Property to attack
                substitute_model_1 = SubstituteModel(seed=seed, property=property, distribution=distribution_1_value_sub_1,
                                                     backbone=sub_models_backbone,
                                                     predictor_path=substitute_model_1_path, attack_name=attack_of_art, is_faceX_zoo=is_faceX_zoo_sub,
                                                     predictor_architecture=sub_models_predictor_architecture, is_finetune_emb = is_finetune_subs_emb, dataset=dataset_for_all_models)
                substitute_model_2 = SubstituteModel(seed=seed, property=property, distribution=distribution_1_value_sub_2,
                                                     backbone=sub_models_backbone,
                                                     predictor_path=substitute_model_2_path, attack_name=attack_of_art,
                                                     is_faceX_zoo=is_faceX_zoo_sub,
                                                     predictor_architecture=sub_models_predictor_architecture, is_finetune_emb = is_finetune_subs_emb, dataset=dataset_for_all_models)

                substitute_to_target_attack_obj = substitute_to_target_attack(target_model=target_model,
                                                                              sub_model1=substitute_model_1,
                                                                              sub_model2=substitute_model_2,
                                                                              property=property, settings=attack_setting,
                                                                              dataset_for_all_models=dataset_for_all_models,
                                                                              art_attack_name=attack_of_art, seed=seed)
                substitute_to_target_attack_obj.attack()
                substitute_to_target_attack_obj.save_unique_samples_confidence_scores()
                substitute_to_target_attack_obj.caculate_the_relative_success_according_to_conf_scores()
                #clean all varaibles grabage collector chache, gpu
                del substitute_to_target_attack_obj
                del substitute_model_1
                del substitute_model_2
                del target_model
                gc.collect()
                torch.cuda.empty_cache()

    else:
        substitute_model_1_path = f"{prefix_substitute_models_path}{property}/seed_{seed}/{seed}_{sub_models_backbone}_property_{property}_dist_1_{distribution_1_value_sub_1}/"
        substitute_model_2_path = f"{prefix_substitute_models_path}{property}/seed_{seed}/{seed}_{sub_models_backbone}_property_{property}_dist_1_{distribution_1_value_sub_2}/"
        target_model_path = f"{prefix_target_model_path}{property}/seed_{seed}/{seed}_{target_backbone}_property_{property}_dist_1_{distribution_1_value_target_model}/"
        target_model = TargetModel(seed=seed, property= property, distribution=distribution_1_value_target_model,
                                   backbone=target_backbone,
                                   predictor_path= target_model_path, is_faceX_zoo=is_faceX_zoo_target,
                                   predictor_architecture=target_predictor_architecture, is_finetune_emb = is_finetune_target_emb, dataset=dataset_for_all_models)
        # Choose Property to attack
        substitute_model_1 = SubstituteModel(seed=seed, property= property, distribution= distribution_1_value_sub_1,
                                             backbone= sub_models_backbone,
                                             predictor_path= substitute_model_1_path, attack_name= attack_of_art,
                                             is_faceX_zoo=is_faceX_zoo_sub,
                                             predictor_architecture=sub_models_predictor_architecture, is_finetune_emb = is_finetune_subs_emb, dataset=dataset_for_all_models)
        substitute_model_2 = SubstituteModel(seed=seed, property= property, distribution= distribution_1_value_sub_2, backbone= sub_models_backbone,
                                                predictor_path= substitute_model_2_path, attack_name= attack_of_art,
                                             is_faceX_zoo=is_faceX_zoo_sub,
                                             predictor_architecture=sub_models_predictor_architecture, is_finetune_emb = is_finetune_subs_emb, dataset=dataset_for_all_models)


        substitute_to_target_attack_obj = substitute_to_target_attack(target_model=target_model, sub_model1=substitute_model_1,
                                                                      sub_model2=substitute_model_2, property=property, settings=attack_setting,
                                                                      dataset_for_all_models = dataset_for_all_models, art_attack_name=attack_of_art, seed= seed)
        substitute_to_target_attack_obj.attack()
        substitute_to_target_attack_obj.save_unique_samples_confidence_scores()
        substitute_to_target_attack_obj.caculate_the_relative_success_according_to_conf_scores()
