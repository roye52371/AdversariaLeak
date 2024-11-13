import csv
import gc
import os

import numpy as np
import pandas as pd
import torch
#from torch.utils.data import DataLoader
from tqdm import tqdm

#from Run_scripts.python_scripts_files.Attack_ART_new import dataset_from_numpy
from torch.utils.data import Dataset, DataLoader
from FR_System.Data.data_utils import load_predictor, random_pairs, load_embedder_and_predictor_for_eval
from FR_System.Embedder.embedder import Embedder
from FR_System.fr_system import FR_Api


class dataset_from_numpy(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets


    def __getitem__(self, index):

        return (self.data[index], self.targets[index])

    def __len__(self):
        return len(self.targets)



class TargetModel:
    def __init__(self, seed, property, distribution, backbone, predictor_path, data_path=None,is_faceX_zoo=True,
                 predictor_architecture = 1, is_finetune_emb= False, dataset=None):
        self.data_path = data_path
        self.seed = seed
        self.property = property
        self.distribution = distribution
        self.backbone = backbone
        self.predictor_path = predictor_path
        self.is_faceX_zoo = is_faceX_zoo
        self.predictor_architecture = predictor_architecture
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_finetune_emb = is_finetune_emb
        self.dataset = dataset

        embedder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=backbone, device=device, is_faceX_zoo=is_faceX_zoo, predictor_architecture=predictor_architecture, path=predictor_path, is_finetune_emb=is_finetune_emb, dataset_name=dataset)


        self.embeder = embedder
        self.predictor = predictor
        # Create complete API
        print("Create complete API")
        self.fr_model = FR_Api(embedder=self.embeder, predictor=self.predictor)

        print("predictor_path: ", predictor_path)

    def save_unique_samples_confidence_scores_against_target_model(self, unique_adv_samples_sub_0, unique_adv_samples_sub_1, attack_name, attack_setting, sub_0_backbone, sub_1_backbone, sub_0_pred_architecture, sub_1_pred_architecture, fine_tune_info_path=None, dataset_for_all_models=None):
        """
        save the unique samples and the confidence scores of each sample against the target model
        """

        #calculate the confidence scores of the unique samples of sub model 0 on the target model
        confidence_scores_not_same_person_sub_0, confidence_scores_same_person_sub_0, predictions_sub_0 = self.get_confidence_scores(
            unique_adv_samples_sub_0)
        data_0 = {'confidence_scores_not_same_person': confidence_scores_not_same_person_sub_0,
                'confidence_scores_same_person': confidence_scores_same_person_sub_0,
                'predictions': predictions_sub_0}
        uniq_conf_0 = pd.DataFrame(data_0)
        uniq_conf_0['attack_name'] = attack_name
        uniq_conf_0['target_backbone'] = self.backbone
        uniq_conf_0['seed'] = self.seed
        uniq_conf_0['property'] = self.property
        uniq_conf_0['target_distribution'] = self.distribution
        uniq_conf_0['substitute_model_dist'] = 0

        if dataset_for_all_models == "CelebA":
            prefix_for_path = ""
        elif dataset_for_all_models == "MAAD_Face":
            prefix_for_path = "MAAD_Face_Results/"
        else:
            raise ValueError("dataset_for_all_models should be CelebA or MAAD_Face")
        # result_file = self.query_budget_folder_path + f'sub_{self.distribution}_unique_adv_confidence_scores.csv'
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        if attack_setting == 'Semi_BlackBox':
            #add sub models backbones
            uniq_conf_0['substitute_model_backbone'] = sub_0_backbone
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        if attack_setting == 'different_predictor_BlackBox':
            # add sub models backbones
            uniq_conf_0['substitute_model_backbone'] = sub_0_backbone
            #add sub models predictor architecture
            uniq_conf_0['substitute_model_predictor_architecture'] = sub_0_pred_architecture
            #add predictor architecture
            uniq_conf_0['target_predictor_architecture'] = self.predictor_architecture
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        self.save_df_and_df_round_to_4(result_file, result_file_4_round_digits, uniq_conf_0)
        #calculate the confidence scores of the unique samples of sub model 0 on the target model
        confidence_scores_not_same_person_sub_1, confidence_scores_same_person_sub_1, predictions_sub_1 = self.get_confidence_scores(
            unique_adv_samples_sub_1)
        data_1 = {'confidence_scores_not_same_person': confidence_scores_not_same_person_sub_1,
               'confidence_scores_same_person': confidence_scores_same_person_sub_1,
               'predictions': predictions_sub_1}
        uniq_conf_1 = pd.DataFrame(data_1)
        uniq_conf_1['confidence_scores_not_same_person'] = confidence_scores_not_same_person_sub_1
        uniq_conf_1['confidence_scores_same_person'] = confidence_scores_same_person_sub_1
        uniq_conf_1['predictions'] = predictions_sub_1
        uniq_conf_1['attack_name'] = attack_name
        uniq_conf_1['target_backbone'] = self.backbone
        uniq_conf_1['seed'] = self.seed
        uniq_conf_1['property'] = self.property
        uniq_conf_1['target_distribution'] = self.distribution
        uniq_conf_1['substitute_model_dist'] = 100
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        if attack_setting == 'Semi_BlackBox':
            #add sub models backbones
            uniq_conf_1['substitute_model_backbone'] = sub_1_backbone
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        if attack_setting == 'different_predictor_BlackBox':
            # add sub models backbones
            uniq_conf_1['substitute_model_backbone'] = sub_1_backbone
            # add sub models predictor architecture
            uniq_conf_1['substitute_model_predictor_architecture'] = sub_1_pred_architecture
            #add predictor architecture
            uniq_conf_1['target_predictor_architecture'] = self.predictor_architecture
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        self.save_df_and_df_round_to_4(result_file, result_file_4_round_digits, uniq_conf_1)
    def save_df_and_df_round_to_4(self, result_file, result_file_4_round_digits, uniq_conf):
        #save in csv not round results
        # Get the directory of the file
        directory = os.path.dirname(result_file) #the same directory for 4 digit csv file anyway

        # Check if the directory exists
        if not os.path.exists(directory):
            # If the directory does not exist, create it
            os.makedirs(directory)

        if os.path.exists(result_file):
            # If the file exists, append the DataFrame to the existing file
            uniq_conf.to_csv(result_file, mode='a', header=False, index=False)
        else:
            # If the file does not exist, create a new file and write the DataFrame to it
            uniq_conf.to_csv(result_file, mode='w', index=False)
        # round to 4 and save in other csv
        float_df = uniq_conf.select_dtypes(include=[np.number])
        # Round the values of the float columns to 4 decimal places
        float_df = float_df.round(4)
        # Update the original DataFrame with the rounded float columns
        uniq_conf[float_df.columns] = float_df
        # Save the DataFrame to a CSV file
        if os.path.exists(result_file_4_round_digits):
            # If the file exists, append the DataFrame to the existing file
            uniq_conf.to_csv(result_file_4_round_digits, mode='a', header=False, index=False)
        else:
            # If the file does not exist, create a new file and write the DataFrame to it
            uniq_conf.to_csv(result_file_4_round_digits, mode='w', index=False)
    def save_result_file_and_4_digit_file(self, result_file, dict_for_csv, result_file_4_round_digits):
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

        round_four_dict_for_csv = {k: round(v, 4) if isinstance(v, float) else v for k, v in
                                   dict_for_csv.items()}
        # change the value of the gap to the sub 1 attack rounf - sub 2 attack round
        # this is IMPORTANT since the round of gap != to round of sub1 success - round 2 sub 1 success
        # add(append) to the results file if exists else create one and write into it


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

    def load_data(self):
        # Load data
        data = pd.read_csv(self.data_path)
        data = data.set_index("Unnamed: 0")
        return data

    def get_success_rate(self, data,labels, batch_size=1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        self.fr_model.eval()
        num_of_samples = len(data)
        print("num_of_samples (len(data)): ", num_of_samples)
        print("num_of_samples using shape[0]: ", data.shape[0])
        dataset = dataset_from_numpy(data, labels)
        adv_data_loader = DataLoader(dataset, batch_size=batch_size)
        success_sum = 0
        for index, (images, labels) in enumerate(adv_data_loader):
            labels = labels.to(torch.float64)
            assert labels.shape[0] == 1 #because we calc mean of the batch so we cant have batch size > 1
            images, labels = images.to(device), labels.to(device)
            output = self.fr_model(images)
            # print("model prediction output on same person images: ", output)
            output = output.cpu().detach().numpy()
            if output.argmax(axis=1) == labels.cpu().detach().numpy().argmax(axis=1):
                # attack succeed on this sample
                success_sum += 1

        print("success_sum: ", success_sum)
        print("num_of_samples: ", num_of_samples)
        success_rate = success_sum / num_of_samples
        return success_rate

    def get_confidence_scores(self, uniq_samp):
        """
        :param uniq_samp: the unique adv samples
        :return: the confidence scores of the unique adv samples on the substitute/target model
        the output of the model is two numbers, the first is the confidence score of the true label(not same person)
        the secnd is the label of the not same person
        I want to keep both plus the prediction itself
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        self.fr_model.eval()
        dataset = dataset_from_numpy(uniq_samp, np.zeros(uniq_samp.shape[0]))
        adv_data_loader = DataLoader(dataset, batch_size=1)
        confidence_scores_not_same_person = []
        confidence_scores_same_person = []
        predictions = []
        for index, (images, labels) in enumerate(adv_data_loader):
            labels = labels.to(torch.float64)
            images, labels = images.to(device), labels.to(device)
            output = self.fr_model(images)
            output = output.cpu().detach().numpy()
            #want to keep the two confidences score and the prediction itself
            confidence_scores_not_same_person.append(output[0][0])
            confidence_scores_same_person.append(output[0][1])
            predictions.append(output.argmax(axis=1).item())
        return confidence_scores_not_same_person, confidence_scores_same_person, predictions

class SubstituteModel(TargetModel):
    def __init__(self, seed, property, distribution, backbone, predictor_path, attack_name, demo_sub_folder_path, is_demo=False, data_path=None, is_faceX_zoo=True,
                 predictor_architecture=1, is_finetune_emb = False, dataset=None):
        super(SubstituteModel, self).__init__(seed, property, distribution, backbone, predictor_path, is_faceX_zoo=is_faceX_zoo,
                                              predictor_architecture=predictor_architecture, is_finetune_emb=is_finetune_emb, dataset=dataset)

        self.demo_sub_folder_path = demo_sub_folder_path
        self.is_demo = is_demo
        if is_demo:
            self.adv_samples_path = self.demo_sub_folder_path + f'adv_data/after_adv_attack/{attack_name}/'
            self.benign_samples_path = self.demo_sub_folder_path + 'adv_data/before_adv_attack/'
        else:
            self.adv_samples_path = self.predictor_path + f'adv_data/after_adv_attack/{attack_name}/'
            self.benign_samples_path = self.predictor_path + 'adv_data/before_adv_attack/'

        self.attack_name = attack_name
        self.adv_samples = None
        self.benign_samples = None
        self.unique_adv_samples = None
        self.converged_adv_samples = None
        self.all_adv_data_size = None
        self.converged_adv_data_size = None
        self.unique_adv_data_size = None
        self.benign_df_with_images_properties = None
        self.converged_benign_df_with_images_properties = None
        self.unique_benign_df_with_images_properties = None
        self.all_positive_attack_set_df_with_images_properties = None
        self.all_non_filter_benign_df_with_images_properties = None
        self.is_faceX_zoo = is_faceX_zoo

    def get_and_save_unique_samples_confidence_scores(self, attack_setting, fine_tune_info_path=None, dataset_for_all_models=None):
        """load the unique adversarial samples and save the confidence scores of the model on them"""
        self.load_unique_adv_samples()
        confidence_scores_not_same_person, confidence_scores_same_person, predictions = self.get_confidence_scores(self.unique_adv_samples)
        #the confidence score is array
        # and I want what not array to be in each row if the cofidences
        #save the confidence scores of the as solumns in the csv file
        data = {'confidence_scores_not_same_person': confidence_scores_not_same_person,
                'confidence_scores_same_person': confidence_scores_same_person,
                'predictions': predictions}
        uniq_conf = pd.DataFrame(data)
        uniq_conf['attack_name'] = self.attack_name
        uniq_conf['backbone'] = self.backbone
        uniq_conf['seed'] = self.seed
        uniq_conf['property'] = self.property
        uniq_conf['distribution'] = self.distribution

        if dataset_for_all_models == "CelebA":
            prefix_for_path = ""
        elif dataset_for_all_models == "MAAD_Face":
            prefix_for_path = "MAAD_Face_Results/"
        else:
            raise ValueError("dataset_for_all_models should be CelebA or MAAD_Face")
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}query_budget_results/{self.attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.distribution}/sub_{self.distribution}_unique_adv_confidence_scores.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}query_budget_results/{self.attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.distribution}/sub_{self.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
        if attack_setting == 'Semi_BlackBox':
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.distribution}/sub_{self.distribution}_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.distribution}/sub_{self.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
        if attack_setting == 'different_predictor_BlackBox':
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.distribution}/sub_{self.distribution}_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.attack_name}/{self.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.distribution}/sub_{self.distribution}_unique_adv_confidence_scores_4_round_digits.csv"

        self.save_df_and_df_round_to_4(result_file, result_file_4_round_digits, uniq_conf)





    def save_adversarial_samples_evaluation(self,attack_setting, fine_tune_info_path=None, dataset_for_all_models=None):
            """
            save the evaluation of the adversarial samples, from the converged samples (those who succeeed to misled the model)
            to the adversarial samples which we tried to craft in the beggining of the experiment
            saving the float result in one file and the rounding 4 results in another file for extra case
            """
            prec_of_attack_success = (self.converged_adv_data_size / self.all_adv_data_size)*100
            #keep in dict the results
            dict_for_csv = {'backbone': self.backbone, 'seed': self.seed, 'property': self.property,
                            'distribution': self.distribution,
                            'attack_name': self.attack_name,
                            'all_adv_data_size': self.all_adv_data_size,
                            'converged_adv_data_size': self.converged_adv_data_size,
                            '%_of_attack_success': prec_of_attack_success}

            if dataset_for_all_models == "CelebA":
                prefix_for_path = ""
            elif dataset_for_all_models == "MAAD_Face":
                prefix_for_path = "MAAD_Face_Results/"
            else:
                raise ValueError("dataset_for_all_models should be CelebA or MAAD_Face")

            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_adversarial_samples_evaluation.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_adversarial_samples_evaluation_4_round_digits.csv"
            if attack_setting == 'Semi_BlackBox':
                result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_adversarial_samples_evaluation.csv"
                result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}Semi_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_adversarial_samples_evaluation_4_round_digits.csv"
            if attack_setting == 'different_predictor_BlackBox':
                result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_adversarial_samples_evaluation.csv"
                result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_info_path}different_predictor_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_adversarial_samples_evaluation_4_round_digits.csv"

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

            round_four_dict_for_csv = {k: round(v, 4) if isinstance(v, float) else v for k, v in
                                       dict_for_csv.items()}
            # change the value of the gap to the sub 1 attack rounf - sub 2 attack round
            # this is IMPORTANT since the round of gap != to round of sub1 success - round 2 sub 1 success
            # add(append) to the results file if exists else create one and write into it


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


    def load_all_non_filter_benign_df_with_images_properties(self):
        self.all_non_filter_benign_df_with_images_properties= pd.read_csv(self.benign_samples_path + "all_the_non_filter_positive_paired_randomly_chosen.csv")

    def load_all_positive_attack_set_df_with_images_properties(self):
        self.all_positive_attack_set_df_with_images_properties = pd.read_csv(self.predictor_path + "before_adv_positive_paired_data.csv")

    def calc_precentage_of_samp_with_without_the_property(self, samples_df):
        """
        calc the number of samples with and without the property
        when with the property is value 1 in the specific cell, and without is -1
        :param samples_df:
        :return:
        num of  samples with the property
        num of  samples without the property
        **notice a pair could have different property value in the two images
        so we calulate each one separately
        """
        num_of_samp_with_the_property__path_1 = len(samples_df[samples_df[f"{self.property}_path1"] == 1])
        num_of_samp_with_the_property__path_2 = len(samples_df[samples_df[f"{self.property}_path2"] == 1])
        num_of_samp_without_the_property__path_1 = len(samples_df[samples_df[f"{self.property}_path1"] == -1])
        num_of_samp_without_the_property__path_2 = len(samples_df[samples_df[f"{self.property}_path2"] == -1])
        num_of_samp_with_the_property = num_of_samp_with_the_property__path_1 + num_of_samp_with_the_property__path_2
        num_of_samp_without_the_property = num_of_samp_without_the_property__path_1 + num_of_samp_without_the_property__path_2
        #calculate_the_precentage of the unique samples with the property
        precentage_of_samp_with_the_property = num_of_samp_with_the_property / (num_of_samp_with_the_property + num_of_samp_without_the_property)
        #calculate_the_precentage of the unique samples without the property
        precentage_of_samp_without_the_property = num_of_samp_without_the_property / (num_of_samp_with_the_property + num_of_samp_without_the_property)
        return precentage_of_samp_with_the_property, precentage_of_samp_without_the_property


    def load_data(self):
        # Load data
        data = pd.read_csv(self.data_path)
        data = data.set_index("Unnamed: 0")
        return data

    def load_benign_df_with_images_properties(self):
        # Load benign samples
        self.benign_df_with_images_properties = pd.read_csv(self.benign_samples_path + "benign_samples_positive_paired.csv")

    def load_converged_benign_df_with_images_properties(self):
        # Load conv benign samples
        self.converged_benign_df_with_images_properties = pd.read_csv(
            self.adv_samples_path + "converged_benign_samples_positive_paired.csv")

        # save converged benign samples


    def load_adv_samples(self):
        # Load adverserial samples
        self.adv_samples = np.load(self.adv_samples_path + 'all_samples.npy')
        print("adv_samples shape: ", self.adv_samples.shape)

    def load_unique_adv_samples(self):
        # Load unique adverserial samples
        self.unique_adv_samples = np.load(self.adv_samples_path + 'unique_adv_samples.npy')
        print("unique_adv_samples shape: ", self.unique_adv_samples.shape)

    def load_converged_adv_samples(self):
        # Load adverserial samples
        self.converged_adv_samples = np.load(self.adv_samples_path + 'converged_adv_samples.npy')
        print("converged_adv_samples shape: ", self.converged_adv_samples.shape)

    def load_benign_samples(self):
        # Load benign samples
        self.benign_samples_path = np.load(self.benign_samples_path + 'all_samples.npy')
    def set_all_adv_data_size(self):
        self.all_adv_data_size = self.adv_samples.shape[0]
    def set_unique_adv_data_size(self):
        self.unique_adv_data_size = self.unique_adv_samples.shape[0]
    def set_converged_adv_data_size(self):
        self.converged_adv_data_size = self.converged_adv_samples.shape[0]

    def filter_converged_adv(self, labels, adv_data=None, batch_size=1):
        """
                :param labels: labels of the adv_data
                :param adv_data: the adv data to filter unique samples from
                :return: the unique adv samples which success on self model and not on other
                """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        self.fr_model.eval()
        if adv_data is None:
            adv_data = self.adv_samples
            # print("adv_data before converged shape: ", adv_data.shape)
        if os.path.exists(self.adv_samples_path + 'converged_adv_samples.npy') \
                and os.path.exists(self.adv_samples_path + 'converged_benign_samples_positive_paired.csv'):
            # print("THE converged_adv_samples.npy already exists, loading it")
            self.converged_adv_samples = np.load(self.adv_samples_path + 'converged_adv_samples.npy')
            self.converged_benign_df_with_images_properties = pd.read_csv(
                self.adv_samples_path + "converged_benign_samples_positive_paired.csv")
        else:
            dataset = dataset_from_numpy(adv_data, labels)
            adv_data_loader = DataLoader(dataset, batch_size=batch_size)
            samples_index_not_converged = []
            for index, (images, labels) in enumerate(adv_data_loader):
                labels = labels.to(torch.float64)
                images, labels = images.to(device), labels.to(device)
                # get the model prediction on the adv samples
                output = self.fr_model(images)
                # print("model prediction output on same person images: ", output)
                output = output.cpu().detach().numpy()
                if output.argmax(axis=1) != labels.cpu().detach().numpy().argmax(axis=1):
                    # cheking if the prediction is the same as the label (in our caase the label is the Not the same person)
                    samples_index_not_converged.append(index)
            # print("samples_index_not_converged: ", samples_index_not_converged)
            converged_adv_samples = np.delete(adv_data, samples_index_not_converged,
                                              axis=0)  # delete rows(samples) that attack succeed on them( not unique)
            # print("converged_adv_samples.shape: ", converged_adv_samples.shape)
            self.converged_adv_samples = converged_adv_samples
            save_path = self.adv_samples_path + 'converged_adv_samples.npy'
            np.save(save_path, converged_adv_samples)

            # drop the rows of the unconverged adv samples from the benign samples
            if not self.is_demo:
                self.converged_benign_df_with_images_properties = self.benign_df_with_images_properties.drop(
                    self.benign_df_with_images_properties.index[samples_index_not_converged])
                assert self.converged_adv_samples.shape[0] == self.converged_benign_df_with_images_properties.shape[0]
                print("self.converged_benign_df_with_images_properties.shape: ",
                      self.converged_benign_df_with_images_properties.shape)
                self.converged_benign_df_with_images_properties.to_csv(
                    self.adv_samples_path + "converged_benign_samples_positive_paired.csv")

    def filter_unique_adv_samples(self, labels, other_sub_model, adv_data=None, batch_size=1):
        """
        :param labels: labels of the adv_data
        :param other_sub_model: the other sub model
        :param adv_data: the adv data to filter unique samples from
        :return: the unique adv samples which success on self model and not on other
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        other_sub_model.fr_model.eval()
        if adv_data is None:
            # take the convereged adv samples
            adv_data = self.converged_adv_samples
        if os.path.exists(self.adv_samples_path + 'unique_adv_samples.npy') \
                and os.path.exists(self.adv_samples_path + 'unique_benign_samples_positive_paired.csv'):
            print("THE unique_adv_samples already exists, loading it")
            self.unique_adv_samples = np.load(self.adv_samples_path + 'unique_adv_samples.npy')
            self.unique_benign_df_with_images_properties = pd.read_csv(
                self.adv_samples_path + "unique_benign_samples_positive_paired.csv")
        else:
            dataset = dataset_from_numpy(adv_data, labels)
            adv_data_loader = DataLoader(dataset, batch_size=batch_size)
            samples_index_not_unique = []
            for index, (images, labels) in enumerate(adv_data_loader):
                labels = labels.to(torch.float64)
                images, labels = images.to(device), labels.to(device)
                # check the data on the other sub model in order to filter it
                # atpatself = self.fr_model(images)
                output = other_sub_model.fr_model(images)
                # print("model prediction output on same person images: ", output)
                output = output.cpu().detach().numpy()
                if output.argmax(axis=1) == labels.cpu().detach().numpy().argmax(axis=1):
                    # need to delete this sample, because it also success on the other sub model
                    # attack succeed on this sample
                    samples_index_not_unique.append(index)
            unique_adv_samples = np.delete(adv_data, samples_index_not_unique,
                                           axis=0)  # delete rows(samples) that attack succeed on them( not unique)
            # print("unique_adv_samples.shape: ", unique_adv_samples.shape)
            self.unique_adv_samples = unique_adv_samples
            save_path = self.adv_samples_path + 'unique_adv_samples.npy'
            np.save(save_path, unique_adv_samples)

            # drop the rows of the unconverged adv samples from the benign samples
            if not self.is_demo:
                self.unique_benign_df_with_images_properties = self.converged_benign_df_with_images_properties.drop(
                    self.converged_benign_df_with_images_properties.index[samples_index_not_unique])
                assert self.unique_adv_samples.shape[0] == self.unique_benign_df_with_images_properties.shape[0]
                print("self.unique_benign_df_with_images_properties.shape: ",
                      self.unique_benign_df_with_images_properties.shape)
                self.unique_benign_df_with_images_properties.to_csv(
                    self.adv_samples_path + "unique_benign_samples_positive_paired.csv")









class substitute_to_target_attack():
    def __init__(self, target_model, sub_model1, sub_model2, property, art_attack_name, seed,
                 settings="whitebox",dataset_for_all_models = None):
        self.target_model = target_model
        self.sub_model1 = sub_model1
        self.sub_model2 = sub_model2
        self.property = property
        self.settings = settings
        self.dataset_for_all_models = dataset_for_all_models
        self.art_attack_name= art_attack_name
        self.seed = seed
        #when no finetuning the fine tune info path is "" string
        #assert if the two sub models are finetune or not they should be in the same state
        assert self.sub_model1.is_finetune_emb == self.sub_model2.is_finetune_emb

        if self.target_model.is_finetune_emb and self.sub_model1.is_finetune_emb and self.sub_model2.is_finetune_emb:
            self.fine_tune_info_path = "fined_tuning_embedder_Results/"
        elif self.target_model.is_finetune_emb and (not self.sub_model1.is_finetune_emb): #it is enought to check the first sub model finetune because it is assert that the two sub models are in the same state
            #if the target model is finetune and the sub models is not finetune, we need give different name to the fine tune info path
            self.fine_tune_info_path = "fined_tuning_target_embedder_not_fine_tune_sub_emb/"
        else:
            self.fine_tune_info_path = ""
        print("self.fine_tune_info_path: ", self.fine_tune_info_path)

    def attack(self):
        self.sub_model1.load_adv_samples()
        self.sub_model2.load_adv_samples()
        self.sub_model1.set_all_adv_data_size()
        self.sub_model2.set_all_adv_data_size()
        self.filter_the_converged_samples()
        #Need to filter sub1 model alone from all samples to converaged samples to unique samples, and save size and delete all the other samples
        #and then calculae the same for sub model 2 , and
        #save sizes to avoid needed tha samples after we will finish to use them
        self.sub_model1.set_converged_adv_data_size()
        self.sub_model2.set_converged_adv_data_size()
        #clean unsed memory and irrelevant variables
        gc.collect()
        torch.cuda.empty_cache()
        self.sub_model1.adv_samples = None
        self.sub_model2.adv_samples = None
        #load data of sub model1 and sub model2
        #keep only the converged adv samples of each sub model
        if self.sub_model1.converged_adv_samples is None:
            self.sub_model1.load_converged_adv_samples()
            self.sub_model1.load_converged_benign_df_with_images_properties()
        if self.sub_model2.converged_adv_samples is None:
            self.sub_model2.load_converged_adv_samples()
            self.sub_model2.load_converged_benign_df_with_images_properties()
        gc.collect()
        torch.cuda.empty_cache()
        #filter the convered_adv_samples to keep only the unique asv samples of each
        self.filter_the_unique_samples()
        #save sizes to avoid needed tha samples after we will finish to use them
        self.sub_model1.set_unique_adv_data_size()
        self.sub_model2.set_unique_adv_data_size()
        #clean unsed memory and irrelevant variables
        gc.collect()
        torch.cuda.empty_cache()
        self.sub_model1.converged_adv_samples = None
        self.sub_model2.converged_adv_samples = None
        #calculate attack success for the target model using both unique sets of adv samples
        data_sub_1_success_rate, data_sub_2_success_rate = self.calculate_success_rates()
        #clean unsed memory and irrelevant variables
        gc.collect()
        torch.cuda.empty_cache()
        self.sub_model1.unique_adv_samples = None
        self.sub_model2.unique_adv_samples = None
        #return inffered property
        inferred_property = self.property_inferation(data_sub_1_success_rate, data_sub_2_success_rate)
        list_all_positive_property_values, list_all_non_filter_property_values, list_benign_samples_positive_paired_property_values, list_coverage_property_values, list_unique_property_values= self.Extract_precentages_of_property_values_for_the_dataframes()

        #ADD all those precentages to differnet the csv file (except unique set that should be in both csv files)
        self.save_attack_results(data_sub_1_success_rate, data_sub_2_success_rate, inferred_property,
                                 list_all_positive_property_values, list_all_non_filter_property_values,
                                 list_benign_samples_positive_paired_property_values,
                                 list_coverage_property_values, list_unique_property_values)

    def demo_attack(self):
        """
        AdversariaLeak attack demo
        attack the target model with the substitute models
        """
        self.sub_model1.load_adv_samples()
        self.sub_model2.load_adv_samples()
        self.sub_model1.set_all_adv_data_size()
        self.sub_model2.set_all_adv_data_size()
        self.filter_the_converged_samples()
        # Need to filter sub1 model alone from all samples to converaged samples to unique samples, and save size and delete all the other samples
        # and then calculae the same for sub model 2 , and
        # save sizes to avoid needed tha samples after we will finish to use them
        self.sub_model1.set_converged_adv_data_size()
        self.sub_model2.set_converged_adv_data_size()
        # clean unsed memory and irrelevant variables
        gc.collect()
        torch.cuda.empty_cache()
        self.sub_model1.adv_samples = None
        self.sub_model2.adv_samples = None
        # load data of sub model1 and sub model2
        # keep only the converged adv samples of each sub model
        if self.sub_model1.converged_adv_samples is None:
            self.sub_model1.load_converged_adv_samples()
            if not self.sub_model1.is_demo:
                self.sub_model1.load_converged_benign_df_with_images_properties()
        if self.sub_model2.converged_adv_samples is None:
            self.sub_model2.load_converged_adv_samples()
            if not self.sub_model2.is_demo:
                self.sub_model2.load_converged_benign_df_with_images_properties()
        gc.collect()
        torch.cuda.empty_cache()
        # filter the convered_adv_samples to keep only the unique asv samples of each
        self.filter_the_unique_samples()
        # save sizes to avoid needed tha samples after we will finish to use them
        self.sub_model1.set_unique_adv_data_size()
        self.sub_model2.set_unique_adv_data_size()
        # clean unsed memory and irrelevant variables
        gc.collect()
        torch.cuda.empty_cache()
        self.sub_model1.converged_adv_samples = None
        self.sub_model2.converged_adv_samples = None
        # calculate attack success for the target model using both unique sets of adv samples
        data_sub_1_success_rate, data_sub_2_success_rate = self.calculate_success_rates()
        # clean unsed memory and irrelevant variables
        gc.collect()
        torch.cuda.empty_cache()
        self.sub_model1.unique_adv_samples = None
        self.sub_model2.unique_adv_samples = None
        # return inffered property
        inferred_property = self.property_inferation(data_sub_1_success_rate, data_sub_2_success_rate)
        print(f"target model's training set true property distribution is {self.target_model.distribution}% {self.target_model.property}")
        print(f"AdversariaLeak inferred that the target model's training set is {inferred_property}")

    def Extract_precentages_of_property_values_for_the_dataframes(self):
        """
        Extract precentages of property values for the dataframes
        :return:
        lists of all the relevant property values
        """
        # calculate the precentage of with/without property in each unique set
        prec_of_unique_with_property_sub_1, prec_of_unique_without_property_sub_1 = self.sub_model1.calc_precentage_of_samp_with_without_the_property(
            self.sub_model1.unique_benign_df_with_images_properties)
        prec_of_unique_with_property_sub_2, prec_of_unique_without_property_sub_2 = self.sub_model2.calc_precentage_of_samp_with_without_the_property(
            self.sub_model2.unique_benign_df_with_images_properties)
        list_unique_property_values = [prec_of_unique_with_property_sub_1, prec_of_unique_without_property_sub_1,
                                       prec_of_unique_with_property_sub_2, prec_of_unique_without_property_sub_2]
        # calculate the precentage of with/without property in each coverage set
        prec_of_coverage_with_property_sub_1, prec_of_coverage_without_property_sub_1 = self.sub_model1.calc_precentage_of_samp_with_without_the_property(
            self.sub_model1.converged_benign_df_with_images_properties)
        prec_of_coverage_with_property_sub_2, prec_of_coverage_without_property_sub_2 = self.sub_model2.calc_precentage_of_samp_with_without_the_property(
            self.sub_model2.converged_benign_df_with_images_properties)
        list_coverage_property_values = [prec_of_coverage_with_property_sub_1, prec_of_coverage_without_property_sub_1,
                                         prec_of_coverage_with_property_sub_2, prec_of_coverage_without_property_sub_2]
        # calculate the precentage of with/without property in each benign_samples_positive_paired set- the bening samples after the correctly labeled filtering
        self.sub_model1.load_benign_df_with_images_properties()
        self.sub_model2.load_benign_df_with_images_properties()
        prec_of_benign_samples_positive_paired_with_property_sub_1, prec_of_benign_samples_positive_paired_without_property_sub_1 = self.sub_model1.calc_precentage_of_samp_with_without_the_property(
            self.sub_model1.benign_df_with_images_properties)
        prec_of_benign_samples_positive_paired_with_property_sub_2, prec_of_benign_samples_positive_paired_without_property_sub_2 = self.sub_model2.calc_precentage_of_samp_with_without_the_property(
            self.sub_model2.benign_df_with_images_properties)
        list_benign_samples_positive_paired_property_values = [
            prec_of_benign_samples_positive_paired_with_property_sub_1,
            prec_of_benign_samples_positive_paired_without_property_sub_1,
            prec_of_benign_samples_positive_paired_with_property_sub_2,
            prec_of_benign_samples_positive_paired_without_property_sub_2]
        # calculate the precentage of with/without property in each non filter random 3000 samples set
        self.sub_model1.load_all_non_filter_benign_df_with_images_properties()
        self.sub_model2.load_all_non_filter_benign_df_with_images_properties()
        prec_of_all_non_filter_with_property_sub_1, prec_of_all_non_filter_without_property_sub_1 = self.sub_model1.calc_precentage_of_samp_with_without_the_property(
            self.sub_model1.all_non_filter_benign_df_with_images_properties)
        prec_of_all_non_filter_with_property_sub_2, prec_of_all_non_filter_without_property_sub_2 = self.sub_model2.calc_precentage_of_samp_with_without_the_property(
            self.sub_model2.all_non_filter_benign_df_with_images_properties)
        list_all_non_filter_property_values = [prec_of_all_non_filter_with_property_sub_1,
                                               prec_of_all_non_filter_without_property_sub_1,
                                               prec_of_all_non_filter_with_property_sub_2,
                                               prec_of_all_non_filter_without_property_sub_2]
        # calculate the precentage of with/without property in all positive test samples set
        self.sub_model1.load_all_positive_attack_set_df_with_images_properties()
        self.sub_model2.load_all_positive_attack_set_df_with_images_properties()
        prec_of_all_positive_with_property_sub_1, prec_of_all_positive_without_property_sub_1 = self.sub_model1.calc_precentage_of_samp_with_without_the_property(
            self.sub_model1.all_positive_attack_set_df_with_images_properties)
        prec_of_all_positive_with_property_sub_2, prec_of_all_positive_without_property_sub_2 = self.sub_model2.calc_precentage_of_samp_with_without_the_property(
            self.sub_model2.all_positive_attack_set_df_with_images_properties)
        list_all_positive_property_values = [prec_of_all_positive_with_property_sub_1,
                                             prec_of_all_positive_without_property_sub_1,
                                             prec_of_all_positive_with_property_sub_2,
                                             prec_of_all_positive_without_property_sub_2]
        #return the lists
        return list_all_positive_property_values, list_all_non_filter_property_values, list_benign_samples_positive_paired_property_values, list_coverage_property_values, list_unique_property_values

    def save_attack_results(self, data_sub_1_success_rate, data_sub_2_success_rate, inferred_property
                            ,list_all_positive_property_values, list_all_non_filter_property_values,
                            list_benign_samples_positive_paired_property_values,
                            list_coverage_property_values, list_unique_property_values
                            ):
        #take the 4 values from the list unique to variables in one code line
        #prec_of_unique_with_property_sub_1, prec_of_unique_without_property_sub_1, prec_of_unique_with_property_sub_2, prec_of_unique_without_property_sub_2 = list_unique_property_values
        #need to save the precentages only one time so save it in the begining of the distributions
        #need tosave th crafted samples eval only one time so save it in the begining of the distributions
        if self.target_model.distribution == 100:
            print("saving the precentages propvety values")
            self.save_precentage_propety_values(list_all_positive_property_values, list_all_non_filter_property_values,
                                list_benign_samples_positive_paired_property_values,
                                list_coverage_property_values, list_unique_property_values)
            print("saving the adv samples evaluation")
            #save_Adversarial_samples_evaluation
            self.sub_model1.save_adversarial_samples_evaluation(self.settings, fine_tune_info_path=self.fine_tune_info_path, dataset_for_all_models=self.dataset_for_all_models)
            self.sub_model2.save_adversarial_samples_evaluation(self.settings, fine_tune_info_path=self.fine_tune_info_path, dataset_for_all_models=self.dataset_for_all_models)

        dict_for_csv = {'Attack_setting': self.settings, 'Seed': self.seed,
                        'property': self.property,
                        'target_model_backbone': self.target_model.backbone,
                        'target_model_dist': self.target_model.distribution,
                        'sub_model_1_dist': self.sub_model1.distribution,
                        'sub_model_2_dist': self.sub_model2.distribution,
                        'data_sub_1_attack_success': data_sub_1_success_rate,
                        'data_sub_2_attack_success': data_sub_2_success_rate,
                        'attack_success_gap': data_sub_1_success_rate - data_sub_2_success_rate,
                        'sub_model_1_all_adv_data_size': self.sub_model1.all_adv_data_size,
                        'sub_model_2_all_adv_data_size': self.sub_model2.all_adv_data_size,
                        'sub_model_1_converged_adv_samples_size': self.sub_model1.converged_adv_data_size,
                        'sub_model_2_converged_adv_samples_size': self.sub_model2.converged_adv_data_size,
                        'sub_model_1_unique_adv_samples_size': self.sub_model1.unique_adv_data_size,
                        'sub_model_2_unique_adv_samples_size': self.sub_model2.unique_adv_data_size,
                        'inferred_property': inferred_property,
                        'art_attack_name': self.art_attack_name,
                        'dataset_for_all_models': self.dataset_for_all_models
                        }

        if self.dataset_for_all_models == "CelebA":
            prefix_for_path = ""
        elif self.dataset_for_all_models == "MAAD_Face":
            prefix_for_path = "MAAD_Face_Results/"
        else:
            raise ValueError("dataset_for_all_models should be CelebA or MAAD_Face")
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_whiteBoxSetting_SubstituteToTarget_AdversarialAttack_results.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_whiteBoxSetting_SubstituteToTarget_AdversarialAttack_results_4_round_digits.csv"

        if self.settings == "Semi_BlackBox":
            #add the sub model 1 backbone to the dict
            # add th sub_1 and sub_2 backbones to the dict
            dict_for_csv['sub_1_backbone'] = self.sub_model1.backbone
            dict_for_csv['sub_2_backbone'] = self.sub_model2.backbone
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_whiteBoxSetting_SubstituteToTarget_AdversarialAttack_results.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_whiteBoxSetting_SubstituteToTarget_AdversarialAttack_results_4_round_digits.csv"
        if self.settings == "different_predictor_BlackBox":
            # add the sub model 1 backbone to the dict
            # add th sub_1 and sub_2 backbones to the dict
            dict_for_csv['sub_1_backbone'] = self.sub_model1.backbone
            dict_for_csv['sub_2_backbone'] = self.sub_model2.backbone
            #add the sub model 1 predictor architecture to the dict
            dict_for_csv['sub_1_predictor_architecture'] = self.sub_model1.predictor_architecture
            dict_for_csv['sub_2_predictor_architecture'] = self.sub_model2.predictor_architecture
            #add the target model predictor architecture to the dict
            dict_for_csv['target_model_predictor_architecture'] = self.target_model.predictor_architecture
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_SubstituteToTarget_AdversarialAttack_results.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_SubstituteToTarget_AdversarialAttack_results_4_round_digits.csv"


        #add(append) to the results file if exists else create one and write into it
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
        #close the file
        # convert dict for csv if int values to round till 4 digits after the dot
        round_four_dict_for_csv = {k: round(v, 4) if isinstance(v, float) else v for k, v in
                                   dict_for_csv.items()}
        #change the value of the gap to the sub 1 attack rounf - sub 2 attack round
        #this is IMPORTANT since the round of gap != to round of sub1 success - round 2 sub 1 success
        round_four_dict_for_csv['attack_success_gap'] = round_four_dict_for_csv['data_sub_1_attack_success'] - round_four_dict_for_csv['data_sub_2_attack_success']
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





    def save_precentage_propety_values(self, list_all_positive_property_values, list_all_non_filter_property_values,
                                       list_benign_samples_positive_paired_property_values,
                                       list_coverage_property_values,
                                       list_unique_property_values):
        """
        this function write and save to csv the different precentage property values of the different lists
        :param list_all_positive_property_values: list of all positive property values
        :param list_all_non_filter_property_values: list of all non filter property values
        :param list_benign_samples_positive_paired_property_values: list of all benign samples positive paired property values
        :param list_coverage_property_values: list of all coverage property values
        :param list_unique_property_values: list of all unique property values
        """
        prec_of_unique_with_property_sub_1, prec_of_unique_without_property_sub_1, prec_of_unique_with_property_sub_2, prec_of_unique_without_property_sub_2 = list_unique_property_values
        prec_of_coverage_with_property_sub_1, prec_of_coverage_without_property_sub_1, prec_of_coverage_with_property_sub_2, prec_of_coverage_without_property_sub_2 = list_coverage_property_values
        prec_of_benign_samples_positive_paired_with_property_sub_1, prec_of_benign_samples_positive_paired_without_property_sub_1, prec_of_benign_samples_positive_paired_with_property_sub_2, prec_of_benign_samples_positive_paired_without_property_sub_2 = list_benign_samples_positive_paired_property_values
        prec_of_all_non_filter_with_property_sub_1, prec_of_all_non_filter_without_property_sub_1, prec_of_all_non_filter_with_property_sub_2, prec_of_all_non_filter_without_property_sub_2 = list_all_non_filter_property_values
        prec_of_all_positive_with_property_sub_1, prec_of_all_positive_without_property_sub_1, prec_of_all_positive_with_property_sub_2, prec_of_all_positive_without_property_sub_2 = list_all_positive_property_values
        # write to csv
        dict_for_csv = {
            'target_model_backbone': self.target_model.backbone,
            'art_attack_name': self.art_attack_name,
            "% of unique with property sub 1": 100*prec_of_unique_with_property_sub_1,
            "% of unique without property sub 1": 100*prec_of_unique_without_property_sub_1,
            "% of unique with property sub 2": 100*prec_of_unique_with_property_sub_2,
            "% of unique without property sub 2": 100*prec_of_unique_without_property_sub_2,
            "% of coverage with property sub 1": 100*prec_of_coverage_with_property_sub_1,
            "% of coverage without property sub 1": 100*prec_of_coverage_without_property_sub_1,
            "% of coverage with property sub 2": 100*prec_of_coverage_with_property_sub_2,
            "% of coverage without property sub 2": 100*prec_of_coverage_without_property_sub_2,
            "% of labeled filtered benign samples positive paired with property sub 1": 100*prec_of_benign_samples_positive_paired_with_property_sub_1,
            "% of labeled filtered benign samples positive paired without property sub 1": 100*prec_of_benign_samples_positive_paired_without_property_sub_1,
            "% of labeled filtered benign samples positive paired with property sub 2": 100*prec_of_benign_samples_positive_paired_with_property_sub_2,
            "% of labeled filtered benign samples positive paired without property sub 2": 100*prec_of_benign_samples_positive_paired_without_property_sub_2,
            "% of all non filter with property sub 1": 100*prec_of_all_non_filter_with_property_sub_1,
            "% of all non filter without property sub 1": 100*prec_of_all_non_filter_without_property_sub_1,
            "% of all non filter with property sub 2": 100*prec_of_all_non_filter_with_property_sub_2,
            "% of all non filter without property sub 2": 100*prec_of_all_non_filter_without_property_sub_2,
            "% of all positive with property sub 1": 100*prec_of_all_positive_with_property_sub_1,
            "% of all positive without property sub 1": 100*prec_of_all_positive_without_property_sub_1,
            "% of all positive with property sub 2": 100*prec_of_all_positive_with_property_sub_2,
            "% of all positive without property sub 2": 100*prec_of_all_positive_without_property_sub_2
        }
        if self.dataset_for_all_models == "CelebA":
            prefix_for_path = ""
        elif self.dataset_for_all_models == "MAAD_Face":
            prefix_for_path = "MAAD_Face_Results/"
        else:
            raise ValueError("dataset_for_all_models should be CelebA or MAAD_Face")
        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_Property_Percentages_SubstituteToTarget_AdversarialAttack_results.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_Property_Percentages_SubstituteToTarget_AdversarialAttack_results_4_round_digits.csv"

        if self.settings == 'Semi_BlackBox':
            #add th sub_1 and sub_2 backbones to the dict
            dict_for_csv['sub_1_backbone'] = self.sub_model1.backbone
            dict_for_csv['sub_2_backbone'] = self.sub_model2.backbone
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_Property_Percentages_SubstituteToTarget_AdversarialAttack_results.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_Property_Percentages_SubstituteToTarget_AdversarialAttack_results_4_round_digits.csv"
        if self.settings == 'different_predictor_BlackBox':
            #add th sub_1 and sub_2 backbones to the dict
            dict_for_csv['sub_1_backbone'] = self.sub_model1.backbone
            dict_for_csv['sub_2_backbone'] = self.sub_model2.backbone
            #add th sub_1 and sub_2 predctor architetures to the dict
            dict_for_csv['sub_1_predictor_architecture'] = self.sub_model1.predictor_architecture
            dict_for_csv['sub_2_predictor_architecture'] = self.sub_model2.predictor_architecture
            #add the target predictor architecture to the dict
            dict_for_csv['target_predictor_architecture'] = self.target_model.predictor_architecture
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_Property_Percentages_SubstituteToTarget_AdversarialAttack_results.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/property_{self.property}/seed_{self.seed}/property_{self.property}_seed_{self.seed}_Property_Percentages_SubstituteToTarget_AdversarialAttack_results_4_round_digits.csv"

        # add(append) to the results file if exists else create one and write into it
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

        # convert dict for csv if int values to round till 4 digits after the dot
        round_four_dict_for_csv = {k: round(v, 4) if isinstance(v, float) else v for k, v in
                                   dict_for_csv.items()}
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

    def property_inferation(self, data_sub_1_success_rate, data_sub_2_success_rate):
        if data_sub_1_success_rate > data_sub_2_success_rate:
            print("substitute 1 data cause higher success rate when attacking the target model")
            winner_dist = self.sub_model1.distribution
        elif data_sub_1_success_rate < data_sub_2_success_rate:
            print("substitute 2 data cause higher success rate when attacking the target model")
            winner_dist = self.sub_model2.distribution
        else:
            print("substitute 1 and 2 data cause the same success rate when attacking the target model")
            winner_dist = None

        if winner_dist is None:
            is_property = None
            decision = "undecided"
        else:
            if (winner_dist > 50):
                is_property = True
                decision = "with"
            elif (winner_dist < 50):
                is_property = False
                decision = "without"
            else:
                is_property = None
                decision = "undecided"
        print(f"the property that characterize most of the target model training set is: {decision} {self.property}")
        inferred_property = f"mostly {decision} {self.property}"
        return inferred_property

    def calculate_success_rates(self):
        # create not same person labels to sub model1 unique_adv_data
        data_y_size_1 = self.sub_model1.unique_adv_samples.shape[0]
        not_data_y_1 = np.full([data_y_size_1, 2], [1, 0])
        print("sub model 1 unique samples shape[0]: ", self.sub_model1.unique_adv_samples.shape[0])
        print(" not_data_y_1_shape[0]: ", not_data_y_1.shape[0])
        # create not same person labels to sub model2 unique_adv_data
        data_y_size_2 = self.sub_model2.unique_adv_samples.shape[0]
        not_data_y_2 = np.full([data_y_size_2, 2], [1, 0])
        print("sub model 2 unique samples shape[0]: ", self.sub_model2.unique_adv_samples.shape[0])
        print("not_data_y_2_shape[0]: ", not_data_y_2.shape[0])
        # calc success rate of sub model1 on target model using submodel1 unique samples
        data_sub_1_success_rate = self.target_model.get_success_rate(self.sub_model1.unique_adv_samples, not_data_y_1)

        # calc success rate of sub model2 on target model using submodel2 unique samples
        data_sub_2_success_rate = self.target_model.get_success_rate(self.sub_model2.unique_adv_samples, not_data_y_2)
        print("data_sub_1_success_rate: ", data_sub_1_success_rate)
        print("data_sub_2_success_rate: ", data_sub_2_success_rate)
        return data_sub_1_success_rate, data_sub_2_success_rate

    def filter_the_unique_samples(self):
        # create not same person labels to sub model1 adv_data
        labels_1_size = self.sub_model1.converged_adv_samples.shape[0]
        not_same_labels_1 = np.full([labels_1_size, 2], [1, 0])
        # filter sub_model1 data using submodel2
        self.sub_model1.filter_unique_adv_samples(not_same_labels_1, self.sub_model2)
        # create not same person labels to sub model2 adv_data
        labels_2_size = self.sub_model2.converged_adv_samples.shape[0]
        not_same_labels_2 = np.full([labels_2_size, 2], [1, 0])
        # filter sub_model2 data using submodel1
        self.sub_model2.filter_unique_adv_samples(not_same_labels_2, self.sub_model1)

    def filter_the_converged_samples(self):
        """
        this function filter the converged samples of the sub models, using the unique samples of the other sub model
        """
        # check the set of unique examples of each sub model, on the target model
        # self.get_success_rate()
        # sum the rate success attack of each set and average the result,
        # return the property of the sub model, which his set got higher success rate attack
        # return the property of the sub model, which his set got higher success rate attack
        # load data of sub model1 and sub model2
        # load data of target model
        if os.path.exists(self.sub_model1.adv_samples_path + 'converged_adv_samples.npy') \
                and os.path.exists(self.sub_model1.adv_samples_path + 'converged_benign_samples_positive_paired.csv'):
            print(
                "THE converged_adv_samples already exists, loading it, NO NEED TO LOAD ALL SAMPLES AND RECOMPUTE AGAIN")
            self.sub_model1.converged_adv_samples = np.load(
                self.sub_model1.adv_samples_path + 'converged_adv_samples.npy')
            self.sub_model1.converged_benign_df_with_images_properties = \
                pd.read_csv(self.sub_model1.adv_samples_path + 'converged_benign_samples_positive_paired.csv')
        else:
            # load data of sub model1 and sub model2
            self.sub_model1.load_adv_samples()  # we don't need to load this if the converged data allready exists
            if not self.sub_model1.is_demo:
                self.sub_model1.load_benign_df_with_images_properties()
            # create not same person labels to sub model1 adv_data
            labels_1_size = self.sub_model1.adv_samples.shape[0]
            not_same_labels_1 = np.full([labels_1_size, 2], [1, 0])
            # filter sub_model1 data using submodel2
            # given not same person label (allthough the samples are of the same person pairs)
            # and check if the model is success to be mislead and predict that is not the same person (succesfull adversarial attack)
            self.sub_model1.filter_converged_adv(not_same_labels_1)
            # create not same person labels to sub model2 adv_data
        if os.path.exists(self.sub_model2.adv_samples_path + 'converged_adv_samples.npy') \
                and os.path.exists(self.sub_model2.adv_samples_path + 'converged_benign_samples_positive_paired.csv'):
            print(
                "THE converged_adv_samples already exists, loading it, NO NEED TO LOAD ALL SAMPLES AND RECOMPUTE AGAIN")
            # load data of sub model1 and sub model2
            self.sub_model2.converged_adv_samples = np.load(
                self.sub_model2.adv_samples_path + 'converged_adv_samples.npy')
            self.sub_model2.converged_benign_df_with_images_properties = \
                pd.read_csv(self.sub_model2.adv_samples_path + 'converged_benign_samples_positive_paired.csv')
        else:
            # load data of sub model1 and sub model2
            self.sub_model2.load_adv_samples()
            if not self.sub_model2.is_demo:
                self.sub_model2.load_benign_df_with_images_properties()
            labels_2_size = self.sub_model2.adv_samples.shape[0]
            not_same_labels_2 = np.full([labels_2_size, 2], [1, 0])
            # filter sub_model2 data using submodel1
            # given not same person label (allthough the samples are of the same person pairs)
            # and check if the model is success to be mislead and predict that is not the same person (succesfull adversarial attack)
            self.sub_model2.filter_converged_adv(not_same_labels_2)



    def get_property(self):
        return self.property

    def save_unique_samples_confidence_scores(self):
        """save the confidence score of each sample of the unique samples of each sub model,
        In addition save the confidence score of each sample in each unique set against the target model"""
        # load the unique samples of each sub model
        self.sub_model1.load_unique_adv_samples()
        self.sub_model2.load_unique_adv_samples()
        #if dist is 100 it is the first and last time we want to calc the confidence score for the sub model on themeselves
        if (self.target_model.distribution==100):
            self.sub_model1.get_and_save_unique_samples_confidence_scores(self.settings, fine_tune_info_path=self.fine_tune_info_path, dataset_for_all_models=self.dataset_for_all_models)
            self.sub_model2.get_and_save_unique_samples_confidence_scores(self.settings, fine_tune_info_path=self.fine_tune_info_path, dataset_for_all_models=self.dataset_for_all_models)

        # save the confidence score of each sample in each unique set against the target model
        self.target_model.save_unique_samples_confidence_scores_against_target_model(self.sub_model1.unique_adv_samples, self.sub_model2.unique_adv_samples, self.art_attack_name, self.settings,
                                                                                     self.sub_model1.backbone,
                                                                                     self.sub_model2.backbone,
                                                                                     self.sub_model1.predictor_architecture,
                                                                                     self.sub_model2.predictor_architecture, fine_tune_info_path=self.fine_tune_info_path,
                                                                                     dataset_for_all_models=self.dataset_for_all_models)
    def caculate_the_relative_success_according_to_conf_scores(self):
        """
        given the csv file of the target model predictions using the sub model unique samples
        and the csv file of the sub model unique samples confidence scores
        take from 10 to unique samples size in jumping of 10: the unique sample which has the highest confidence score on their sub model
        and choose them in the target model csv file and calculate the  average of their prediction
        then save all the results in a csv file
        """
        if self.dataset_for_all_models == "CelebA":
            prefix_for_path = ""
        elif self.dataset_for_all_models == "MAAD_Face":
            prefix_for_path = "MAAD_Face_Results/"
        else:
            raise ValueError("dataset_for_all_models should be CelebA or MAAD_Face")

        #load csv files
        sub_0_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}query_budget_results/{self.art_attack_name}/{self.sub_model1.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.sub_model1.distribution}/sub_{self.sub_model1.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
        sub_1_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}query_budget_results/{self.art_attack_name}/{self.sub_model2.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.sub_model2.distribution}/sub_{self.sub_model2.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
        if self.settings == "Semi_BlackBox":
            sub_0_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.sub_model1.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.sub_model1.distribution}/sub_{self.sub_model1.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
            sub_1_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.sub_model2.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.sub_model2.distribution}/sub_{self.sub_model2.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
        if self.settings == "different_predictor_BlackBox":
            sub_0_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.sub_model1.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.sub_model1.distribution}/sub_{self.sub_model1.distribution}_unique_adv_confidence_scores_4_round_digits.csv"
            sub_1_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.sub_model2.backbone}/property_{self.property}/seed_{self.seed}/sub_{self.sub_model2.distribution}/sub_{self.sub_model2.distribution}_unique_adv_confidence_scores_4_round_digits.csv"

        sub_0_target_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        sub_100_target_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        if self.settings == "Semi_BlackBox":
            sub_0_target_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
            sub_100_target_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
        if self.settings == "different_predictor_BlackBox":
            sub_0_target_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/sub_0_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"
            sub_100_target_result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/sub_100_vs_target_model_unique_adv_confidence_scores_4_round_digits.csv"

        sub_0_confidence_df = pd.read_csv(sub_0_result_file_4_round_digits)
        sub_1_confidence_df = pd.read_csv(sub_1_result_file_4_round_digits)
        #load the target model predictions using the sub model unique samples
        target_model_predictions_vs_sub_0_df = pd.read_csv(sub_0_target_result_file_4_round_digits)
        target_model_predictions_vs_sub_100_df = pd.read_csv(sub_100_target_result_file_4_round_digits)
        #take the minimum of the two sub model unique samples size
        min_unique_samples_size = min(sub_0_confidence_df.shape[0], sub_1_confidence_df.shape[0])
        #create a list of the number of unique samples to take
        one_to_hundred_list = list(range(1, 100))
        unique_samples_size_list = list(range(100, min_unique_samples_size, 10))
        if (min_unique_samples_size > 99):
            #add to the left the one_to_ten_list
            unique_samples_size_list = one_to_hundred_list + unique_samples_size_list
        #create a list of the results
        results_list = []
        #return a list of the indices sorted from highet confidnce score to lowest
        sub_0_confidence_df_sotrted = sub_0_confidence_df.sort_values(by='confidence_scores_not_same_person', ascending=False)
        sub_1_confidence_df_sotrted = sub_1_confidence_df.sort_values(by='confidence_scores_not_same_person', ascending=False)
        #for each number of unique samples to take
        for unique_samples_size in unique_samples_size_list:
            #take the unique samples which has the highest confidence score on their sub model
            #sub_0_confidence_df_sotrted = sub_0_confidence_df.sort_values(by='confidence_scores_not_same_person', ascending=False)
            sub_0_confidence_df_sotrted_chosen = sub_0_confidence_df_sotrted[:unique_samples_size]
            #choose the samples of the target model using the sub_0_confidence_df_sotrted_chosen
            target_chosen_confidence_df_using_sub_0 = target_model_predictions_vs_sub_0_df.iloc[sub_0_confidence_df_sotrted_chosen.index]
            #target_chosen_confidence_df_using_sub_0 = sub_0_confidence_df_sotrted_chosen.merge(target_model_predictions_vs_sub_0_df)
            #calculate the  average of their prediction when misleading the target model
            mislead_target_chosen_confidence_df_using_sub_0 = target_chosen_confidence_df_using_sub_0[target_chosen_confidence_df_using_sub_0['predictions'] == 0]
            relative_success_according_to_conf_scores_sub_0 = mislead_target_chosen_confidence_df_using_sub_0.shape[0] / unique_samples_size
            #take the unique samples which has the highest confidence score on their sub model
            #sub_1_confidence_df_sotrted = sub_1_confidence_df.sort_values(by='confidence_scores_not_same_person', ascending=False)
            sub_1_confidence_df_sotrted_chosen = sub_1_confidence_df_sotrted[:unique_samples_size]
            #choose them indices (of which choose in the sub model) now in the target model csv file
            target_chosen_confidence_df_using_sub_100 = target_model_predictions_vs_sub_100_df.iloc[sub_1_confidence_df_sotrted_chosen.index]
            #target_chosen_confidence_df_using_sub_1 = sub_1_confidence_df_sotrted_chosen.merge(target_model_predictions_vs_sub_1_df)
            #calculate the  average of their prediction when misleading the target model
            mislead_target_chosen_confidence_df_using_sub_100 = target_chosen_confidence_df_using_sub_100[target_chosen_confidence_df_using_sub_100['predictions'] == 0]
            relative_success_according_to_conf_scores_sub_100 = mislead_target_chosen_confidence_df_using_sub_100.shape[0] / unique_samples_size
            attack_succes_gap = relative_success_according_to_conf_scores_sub_0 - relative_success_according_to_conf_scores_sub_100

            #save the result in a list
            if self.settings == "Semi_BlackBox":
                results_list.append([self.seed, self.art_attack_name, self.property, self.target_model.backbone,
                                     self.target_model.distribution, unique_samples_size,
                                     relative_success_according_to_conf_scores_sub_0,
                                     relative_success_according_to_conf_scores_sub_100,
                                     attack_succes_gap, self.sub_model1.backbone, self.sub_model2.backbone])
            elif self.settings == "different_predictor_BlackBox":
                results_list.append([self.seed, self.art_attack_name, self.property, self.target_model.backbone,
                                     self.target_model.predictor_architecture,
                                     self.target_model.distribution, unique_samples_size,
                                     relative_success_according_to_conf_scores_sub_0,
                                     relative_success_according_to_conf_scores_sub_100,
                                     attack_succes_gap, self.sub_model1.backbone, self.sub_model2.backbone,
                                     self.sub_model1.predictor_architecture, self.sub_model2.predictor_architecture])
            else:
                results_list.append([self.seed, self.art_attack_name, self.property, self.target_model.backbone,
                                     self.target_model.distribution, unique_samples_size,
                                     relative_success_according_to_conf_scores_sub_0,
                                     relative_success_according_to_conf_scores_sub_100,
                                     attack_succes_gap])


        #write to csv file
        if self.settings == "Semi_BlackBox":
            results_df = pd.DataFrame(results_list, columns=['seed', 'attack_name', 'property',
                                                             'target_backbone', 'target_distribution',
                                                             'unique_samples_size',
                                                             'relative_success_using_sub_0',
                                                             'relative_success_using_sub_100',
                                                             'attack_succes_gap',
                                                             'substitute_model_0_Backbone',
                                                             'substitute_model_100_Backbone'])
        elif self.settings == "different_predictor_BlackBox":
            results_df = pd.DataFrame(results_list, columns=['seed', 'attack_name', 'property',
                                                             'target_backbone', 'target_predictor_architecture',
                                                             'target_distribution',
                                                             'unique_samples_size',
                                                             'relative_success_using_sub_0',
                                                             'relative_success_using_sub_100',
                                                             'attack_succes_gap',
                                                             'substitute_model_0_Backbone',
                                                             'substitute_model_100_Backbone',
                                                             'substitute_model_0_predictor_architecture',
                                                             'substitute_model_100_predictor_architecture'])
        else:
            results_df = pd.DataFrame(results_list, columns=['seed', 'attack_name', 'property',
                                                             'target_backbone', 'target_distribution',
                                                             'unique_samples_size',
                                                             'relative_success_using_sub_0',
                                                             'relative_success_using_sub_100',
                                                             'attack_succes_gap'])

        result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores.csv"
        result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores_4_round_digits.csv"
        if self.settings == "Semi_BlackBox":
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}Semi_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores_4_round_digits.csv"
        if self.settings == "different_predictor_BlackBox":
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores.csv"
            result_file_4_round_digits = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{self.fine_tune_info_path}different_predictor_BlackBox_Results/query_budget_results/{self.art_attack_name}/{self.target_model.backbone}/property_{self.property}/seed_{self.seed}/target_dist_{self.target_model.distribution}/FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores_4_round_digits.csv"

        self.target_model.save_df_and_df_round_to_4(result_file, result_file_4_round_digits, results_df)



