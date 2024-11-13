import configparser
import csv
import gc
import json
import os
import resource
import sys

import pandas as pd
import torch
import numpy as np
from PIL import Image
from art import attacks
from art.estimators import classification
import tensorflow as tf
from natsort import natsorted
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
# from Evasion.data_utils import mnist_data_loader_class
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from pathlib import Path
from art.utils import load_mnist
import pytorch_lightning as pl
#add path to run on gpu

sys.path.append("/sise/home/royek/Toshiba_roye/")
from FR_System.Data.data_utils import Demo_creat_yes_records, create_yes_records_before_adv, \
    create_no_records_before_adv, filter_benign_pairs, random_pairs, load_predictor, \
    load_embedder_and_predictor_for_eval
from FR_System.Embedder.embedder import Embedder, process_image
from FR_System.Predictor.predictor import Predictor
from FR_System.fr_system import FR_Api, evaluation, test_prediction



class dataset_from_numpy(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets


    def __getitem__(self, index):

        return (self.data[index], self.targets[index])

    def __len__(self):
        return len(self.targets)


class Attack():
    """
    The general attack class.
    """
    def __init__(self, name, classifier, params, lossf, input_shape, nb_classes, optimizer=None, channels_first=False, device_type="cpu"):
        """
        The attack class constructor.
        :param name: required. type: str. The attack name. Should be suited to the attack instance in the ART library.
        :param classifier: required. type: classifier object (for example, torch.nn.Module). The classifier to attack
                            (the target model).
        :param params: required. type: dict of {str:value}. The attack's parameters. The keys should be suited to the
                        attack instance in the ART library. The values are the parameters' values to use. If None, the
                        attack parameters will be set to default.
        :param lossf: required. type: loss function object (for example, nn.CrossEntropyLoss()). The loss function used
                        by the classifier provided.
        :param input_shape: required. type: set (). The input shape received by the classifier provided.
        :param nb_classes: required. type: int. The number of classes predicted by the classifier provided.
        :param optimizer: optional. type: optimization function (for example, optim.Adam(model.parameters(), lr=0.002)).
                            The optimization function used by the classifier provided. None by default.
        :param channels_first: optional. type: boolean. Whether the channel is first in the input dimensions. False by
                                defualt.
        :param device_type: optional. type: str. The device to use to apply the attack. "cpu" by defualt.
        """
        self.name = name
        if params is None:
            params = self.get_default_params(name)
        self.params = params
        if name=='BasicIterativeMethod' or name=='ProjectedGradientDescent' \
                or name =='FastGradientMethod' or name =='BoundaryAttack':
            classifier_name = 'estimator'
        else:
            classifier_name = 'classifier'
        self.params.update({classifier_name: self.wrap_model(classifier=classifier,
                                                lossf=lossf,
                                                input_shape=input_shape,
                                                nb_classes=nb_classes,
                                                optimizer=optimizer,
                                                channels_first=channels_first,
                                                device_type=device_type,
                                                clip_values = (0.0,1.0))})

    def get_default_params(self, name):
        """
        Provides the default parameters of the attack.
        :param name: required. type: str. The name of the attack.
        :return: dict with {str: value} containing all the default parameters of the attack.
        """
        if name == "HopSkipJump":
            param = {
                "targeted": False,
                "norm": 2,
                "max_iter": 10,
                "init_size": 100
            }
        elif name == "AdversarialPatch":
            param = {
                "binary_search_steps": 10,
                "use_importance": False,
                "nb_parallel": -1,
                "batch_size": 128,
                "max_iter": 1,
                "learning_rate": 1.0
            }
        elif name == "DeepFool":
            param = {
                "max_iter" : 10,
                "epsilon" : 1e-6,
                "nb_grads" : 10,
                "batch_size" : 1
            }
        elif name == "BasicIterativeMethod":
            param = {
                "eps": 0.3,
                "eps_step": 0.1,
                "max_iter": 10,
                "targeted": False,
                "batch_size": 1
            }
        elif name == "CarliniL2Method":
            param = {
                "confidence": 0.0,
                "targeted": False,
                "learning_rate": 0.01,
                "binary_search_steps": 10,
                "max_iter": 200,
                "initial_const": 0.01,
                "max_halving": 5,
                "max_doubling": 5,
                "batch_size": 1

            }
        elif name == "CarliniLInfMethod":
            param = {
                "confidence":  1.0,
                "targeted": False,
                "learning_rate": 0.01,
                "max_iter": 1000,
                "max_halving": 5,
                "max_doubling": 5,
                "eps": 0.9,
                "batch_size":1,
                "verbose": True
            }
        elif name == "ProjectedGradientDescent":
            param = {
                "max_iter": 30,
                "norm": 2,
                "eps": 2.0,
                "eps_step": 2.0,
                "targeted": False,
                "num_random_init": 0,
                "batch_size": 1
            }
        elif name == "ZooAttack":
            param = {
                    "confidence": 0.0,
                    "targeted": False,
                    "learning_rate": 0.1,
                    "max_iter": 10,
                    "binary_search_steps": 1,
                    "initial_const": 1e-3,
                    "abort_early": True,
                    "use_resize": True,
                    "use_importance": True,
                    "nb_parallel": 1,
                    "batch_size": 1,
                    "variable_h": 1e-4
            }
        elif name == "FastGradientMethod":
            param = {
                "targeted": False,
                "num_random_init": 0,
                "batch_size": 128,
                "minimal": False,
                "eps_step": 0.1
            }
        elif name == "SaliencyMapMethod":
            param = {
                "theta": 0.1,
                "gamma": 1.0,
                "batch_size": 1
            }
        elif name == "BoundaryAttack":
            param = {
                "max_iter" : 10,
                "targeted" : False,
                "delta" : 0.01,
                "epsilon" : 0.01,
                "step_adapt" : 0.667,
                "num_trial" : 25,
                "sample_size" : 20,
                "init_size" : 100
            }
        elif name == 'UniversalPerturbation':
            param = {
                "attacker": "fgsm",
                "delta": 0.2,
                "max_iter": 1,
                "eps": 10.0,
                "norm": 2
            }

        else:
            param = dict()
        return param

    def wrap_model(self, classifier, lossf, input_shape, nb_classes,
                   optimizer=None, channels_first=False, device_type="cpu",
                   clip_values = None):
        """
        Wraps the provided model in the ART library wrapper.
        :param classifier: required. type: classifier object (for example, torch.nn.Module). The classifier to attack
                            (the target model).
        :param lossf: required. type: loss function object (for example, nn.CrossEntropyLoss()). The loss function used
                        by the classifier provided.
        :param input_shape: required. type: set (). The input shape received by the classifier provided.
        :param nb_classes: required. type: int. The number of classes predicted by the classifier provided.
        :param optimizer: optional. type: optimization function (for example, optim.Adam(model.parameters(), lr=0.002)).
                            The optimization function used by the classifier provided. None by default.
        :param channels_first: optional. type: boolean. Whether the channel is first in the input dimensions. False by
                                defualt.
        :param device_type: optional. type: str. The device to use to apply the attack. "cpu" by defualt.
        :return: ART wrapper instance. the wrapped classifier.
        """
        # currently only for pytorch models
        return classification.PyTorchClassifier(model=classifier,
                                                loss=lossf,
                                                input_shape=input_shape,
                                                nb_classes=nb_classes,
                                                optimizer=optimizer,
                                                channels_first=channels_first,
                                                device_type=device_type,
                                                clip_values = clip_values)

    def get_attack_instance(self):
        """
        The method construct an instance of the attack using the class parameters and returns the instance.
        :return: ART attack instance (ABCMeta instance).
        """
        a = getattr(attacks.evasion, self.name)
        return a(**self.params)

    def generate(self, data_x, data_y):
        """
        Generates adversarial samples according to the attack name.
        :param data_x: required. type: ndarray. The records to convert them to adversarial.
        :param data_y: required. type: ndarray. The labels of data_x. Can be set to None.
        :return: ndarray. The adversarial samples.
        """
        attack_instance = self.get_attack_instance()
        adv_data = attack_instance.generate(x=data_x, y=data_y)
        return adv_data




def manager(dataset,model,params,all_dataset):
    attack_params = {
        "targeted": False,
        "norm": 2,
        "max_iter": 10,
        "init_size": 100
    }
    # dataset = custom_subset(all_dataset,dataset.indices)
    datasets,labels = process_dataset(dataset)
    output_path = f"{params['output_path']}/Attack"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    tf.compat.v1.disable_eager_execution()
    # attacks = ['BasicIterativeMethod','HopSkipJump','DeepFool',
    #            'CarliniL2Method','ProjectedGradientDescent','ZooAttack',
    #            'FastGradientMethod','SaliencyMapMethod','BoundaryAttack',
    #            'UniversalPerturbation']
    attacks = ['HopSkipJump','BoundaryAttack']
    datasets = datasets[8:]
    labels = labels[8:]

    for i in range (len(datasets)):

        attack_name = attacks[i]
        print(attack_name)
        attack = Attack(attack_name, model, None,
                        lossf=nn.CrossEntropyLoss(),
                        input_shape=(1, params['img size'], params['img size']),
                        nb_classes=10,
                        optimizer=optim.Adam(model.parameters(), lr=0.002),
                        channels_first=False,
                        device_type="cpu")
        curr_data = datasets[i]
        curr_data_labels = labels[i]
        adv_x = attack.generate(data_x=curr_data,
                                data_y=curr_data_labels )
        test_acc, test_loss = test_adversarial_attack(
            adv_x,curr_data_labels,model)
        print(f'{attack_name} accuracy is: {test_acc}, {attack_name} loss '
              f'is: {test_loss}')
        np.save(f'{output_path}/{attack_name}_samples.npy',
                adv_x, allow_pickle=True)

def process_dataset(dataset):
    """
    helper function for resize the dataset.
    :param dataset: required. The dataset on which the experiment will be
    performed.
    :return: smaller dataset.
    """
    datasets = []
    labels = []
    number_of_groups = 10
    group_size = int(len(dataset)/number_of_groups)
    for i in range(number_of_groups):
        reduction = list(range(i, len(dataset), 10))
        sub_dataset = torch.utils.data.Subset(dataset, reduction)
        data_loader = DataLoader(sub_dataset, batch_size=group_size)
        datasets.append(next(iter(data_loader))[0].numpy())
        labels.append(next(iter(data_loader))[1].numpy())

    return datasets,labels

def save_images(data, output_path):
        """
        plots of random images grid and their corresponding label.
        :return: plots of random images grid.
        """
        fig = plt.figure()
        for idx, img in enumerate(data):
            # plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(np.squeeze(img, axis=0),
                       cmap='gray', interpolation='none')
            plt.xticks([])
            plt.yticks([])
            # plt.show()
            fig.savefig(f'{output_path}/{idx}.png', dpi=fig.dpi)

def test_adversarial_attack(adv_data,labels,model):

    """
    Validate the model in a given epoch,
    the evaluation is divided to batches.
    :param test_loader: test data to evaluate the model.
    :return: valid_loss - the loss rate for the given epoch.
             val_correct - number of correct records in the given epoch.
             total_labels - labels of each sample for the given epoch.
             total_predictions - predictions of each sample in the given
             epoch.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          "cpu")
    criterion = nn.CrossEntropyLoss()
    valid_loss, val_correct = 0.0, 0
    model.eval()
    dataset = dataset_from_numpy(adv_data,labels)
    adv_data_loader = DataLoader(dataset,batch_size=1)
    total_labels = []
    total_predictions = []
    for images, labels in adv_data_loader:
        labels = labels.to(torch.float64)
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        print("model prediction output on same person images: ",output)
        loss = criterion(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        ###to delete
        print("prediction is: ", predictions)
        print("label is: ", labels)
        ###to delete
        val_correct += (predictions == labels).sum().item()

    test_loss = valid_loss / len(adv_data_loader.sampler)
    test_acc = val_correct / len(adv_data_loader.sampler) * 100
    return test_acc, test_loss

def evaluation_metrics(self,test_loss, test_correct, labels, predictions, test_loader):
    """
    Function the implements different evaluation metrics on the model
    test set.
    test loss, test accuracy, recall score, precision score, f1 and auc.
    Also saves the model with the best performance.
    :param test_loss:
    :param test_correct: number of correct predictions.
    :param labels: the actual classification of a given data instance.
    :param predictions: model predictions (an ndarray)
    :param test_loader: test dataset as a dataLoader instance.
    :return:
    """
    test_loss = test_loss / len(test_loader.sampler)
    test_acc = test_correct / len(test_loader.sampler) * 100
    return test_acc,test_loss
######################################################################################################
# EXAMPLE
######################################################################################################
# Create example model
class mnist_model(torch.nn.Module):
    """
    Class for define model architecture for MNIST dataset.
    """
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x



def check_if_all_files_exist(before_adv_attack_data_path_directory):
    A = os.path.exists(
        "{}all_the_non_filter_positive_paired_randomly_chosen.csv".format(before_adv_attack_data_path_directory))
    B = os.path.exists("{}all_samples.npy".format(before_adv_attack_data_path_directory))
    C = os.path.exists("{}benign_samples_positive_paired.csv".format(before_adv_attack_data_path_directory))
    return A and B and C


def create_adv_samples(backbone, predictor_path, attack_name, params, part_of_data, chunk_size, random_seed, settings= "white-box", is_faceX_zoo= True,
                       limited_query_budget= True, predictor_architecture_type=1, n_in=512, increase_shape=False, dataset_name="CelebA", train_emb=False):
    """
    The function creates adversarial samples for the FR system.
    :param backbone: Required. str. The name of the backbone.
    :param predictor_path: Required. str. The path saves weights of the predictor.
    :param attack_name: Required. str. The name of the attack.
    :param params: Required. dict. The parameters of the attack.
    :param part_of_data: Required. float. The part of the data to create adversarial samples.
    :param chunk_size: Required. int. The size of the chunk of the data to create adversarial samples for the specific part
    (when the query budget is limit, the chunk size is using as the query budget).
    :param random_seed: Required. int. The random seed.
    :param settings: Optional. str. The settings of the attack. Default: "white-box".
    :param is_faceX_zoo: Optional. bool. The flag indicates whether the data is from FaceX-Zoo dataset. Default: True.
    :param limited_query_budget: Optional. bool. The flag indicates whether the query budget is limited. Default: True.
    :param predictor_architecture_type: Optional. int. The type of the predictor architecture. Default: 1.
    :param n_in: Optional. int. The number of input features of the predictor. Default: 512.
    :param increase_shape: Optional. bool. The flag indicates whether to increase the shape of the images. Default: False.
    :param dataset_name: Optional. str. The name of the dataset. Default: "CelebA".
    """

    print(f"part of data:{part_of_data}")
    print(f"chunk size:{chunk_size}")
    print("Create embedder")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # loading predictor and embedder for the current epoch
    embeder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=backbone, device=device,
                                                                  is_faceX_zoo=is_faceX_zoo,
                                                                  predictor_architecture=predictor_architecture_type,
                                                                  path=predictor_path,
                                                                  is_finetune_emb=train_emb,
                                                                  dataset_name=dataset_name, n_in=n_in)

    device_type_str = device.type  # "gpu" # cuda:0


    # Create complete API
    print("Create complete API")
    fr = FR_Api(embedder=embeder, predictor=predictor)

    # Create yes records
    print("Create yes records")
    #if all the relevant file before attack exists (meant one art attack executed)
    # load them and use them in the new art attack

    # data_before_adv_optimized= pd.read_csv("{}attack_test_before_adv_df_pairs.csv".format(predictor_path))
    # Demo_creat_yes_records(data_path, ".jpg", mid_saving=False)
    positive = pd.read_csv("{}before_adv_positive_paired_data.csv".format(predictor_path))
    data_x = positive.drop(["label", f"{property}_path1", f"{property}_path2"], axis=1)
    #data_x = data_x.drop([f"{property}_path1"], axis=1)
    #data_x = data_x.drop([f"{property}_path2"], axis=1)
    data_y = pd.DataFrame(positive["label"], columns=["label"])
    # create directories for adverserial chunks of data
    adv_data_path_directory = predictor_path + "adv_data/"
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

    if limited_query_budget and check_if_all_files_exist(before_adv_attack_data_path_directory):
        print("all the relevant file before attack exists")
        all_the_non_filter_positive_paired_randomly_chosen = pd.read_csv(
            "{}all_the_non_filter_positive_paired_randomly_chosen.csv".format(before_adv_attack_data_path_directory))
        data_x = np.load("{}all_samples.npy".format(before_adv_attack_data_path_directory))
        updated_samples_df = pd.read_csv(
            "{}benign_samples_positive_paired.csv".format(before_adv_attack_data_path_directory))
        #create y_data
        data_y_size = data_x.shape[0]
        data_y = np.full([data_y_size, 2],
                         [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
        print("after filter data_y_shape[0]: ", data_y.shape[0])

        not_data_y = np.full([data_y_size, 2], [1, 0])
        print("not_data_y_shape[0]: ", not_data_y.shape[0])


    else:

        # choosing random 3000 idices
        if limited_query_budget:
            # save if exists else load it - "{}all_the_non_filter_positive_paired_randomly_chosen.csv".format(before_adv_attack_data_path_directory)
            if not os.path.exists("{}all_the_non_filter_positive_paired_randomly_chosen.csv".format(before_adv_attack_data_path_directory)):


                data_x_num_of_samples = data_x.shape[0]
                print("data_x shape: ", data_x.shape)
                print(f"data_x.shape[0] = {data_x.shape[0]}")
                #choosing between the num of limited budget (chunk size) and the num of samples in the data (the minimal between them)
                num_of_wanted_samples = min(data_x_num_of_samples, chunk_size)
                print("num_of_wanted_samples: ", num_of_wanted_samples)
                #create the randomal idices to vhoose beetween the data using randomnees seed
                sub_model_random_indices_of_samples = random_pairs(num_of_given_samples=data_x_num_of_samples, num_of_wanted_samples=num_of_wanted_samples,
                                                         seed=random_seed)
                #sort the sub model random indices of samples
                print("sub_model_random_indices_of_samples: ", sub_model_random_indices_of_samples)
                print("sort the sub model random indices of samples")
                print("se sort in order to keep the samples order the same as the original data, allthough the indices chosen randomly before it ")
                sub_model_random_indices_of_samples.sort()
                data_x = data_x.iloc[sub_model_random_indices_of_samples, :]
                data_y = data_y.iloc[sub_model_random_indices_of_samples, :]
                print("data_x shape after random: ", data_x.shape)
                print("data_y shape after random: ", data_y.shape)
                #choose the specific  random indices for the rows of the positive csv file

                positive = positive.iloc[sub_model_random_indices_of_samples, :]
                print("positive randomly chosen shape after random: ", positive.shape)
                #save the random indices for the rows of the positive_radomly_chosen to csv file
                positive.to_csv("{}all_the_non_filter_positive_paired_randomly_chosen.csv".format(before_adv_attack_data_path_directory), index=False)
            else:
                positive = pd.read_csv("{}all_the_non_filter_positive_paired_randomly_chosen.csv".format(before_adv_attack_data_path_directory))
                data_x = positive.drop(["label", f"{property}_path1", f"{property}_path2"], axis=1)
                data_y = pd.DataFrame(positive["label"], columns=["label"])
                print("data_x shape after random: ", data_x.shape)
                print("data_y shape after random: ", data_y.shape)
                print("positive randomly chosen shape after random: ", positive.shape)





        else:
            #taking the whole data in case the query budget is not limited
            #taking the whole data in seprated part so we want create huge amounts of samples togetherr which could take enormouse time
            # finish creating directories for adverserial chunks of data
            print(f"data_x.shape[0] = {data_x.shape[0]}")
            num_of_parts = int(data_x.shape[0] / chunk_size)  # num of pairs
            read_start = chunk_size * part_of_data
            read_end = chunk_size * (part_of_data + 1)
            print(f"num of parts: {num_of_parts}")
            print("read end not relevant for last part of data because we read there what left\n")
            print(f"chunck size: {chunk_size}, part of data: {part_of_data}/{num_of_parts-1}, read start: {read_start}, read end: {read_end}")
            #print(f"num_of_parts: {num_of_parts}")
            #print(f"read_start: {read_start}, read_end: {read_end}")
            if part_of_data == num_of_parts - 1:
                data_x = data_x[read_start:]  # take what left
                data_y = data_y[read_start:]
            else:
                data_x = data_x[read_start:read_end]
                data_y = data_y[read_start:read_end]

        # convert data to numpy pairs
        batch_x_test_val = []
        for i, row in data_x.iterrows():
            path1 = row["path1"]
            path2 = row["path2"]
            np_image1 = process_image(path1, increase_shape=increase_shape)
            np_image2 = process_image(path2, increase_shape=increase_shape)
            batch_x_test_val.append([np_image1, np_image2])
        # y_test = y_test.values
        #####convert data to numpy pairs
        data_y_size = data_y.size
        data_y = np.full([data_y_size, 2], [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
        data_x = np.array(batch_x_test_val)

        assert data_x.shape[0] == data_y.shape[0]
        assert data_x.shape[0] == positive.shape[0] #must be the same num of samples
        print("before filter data_y_shape[0]: ", data_y.shape[0])
        #keep only the all the yes pairs which  realy return the same person prediction whith the FR system
        data_x, updated_samples_df, _ = filter_benign_pairs(adv_data=data_x, samples_with_property_info_df= positive, labels=data_y, model=fr, use_properties=True)

        # save the filtered data and create labels correspondly
        data_y_size = data_x.shape[0]
        data_y = np.full([data_y_size, 2], [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
        print("after filter data_y_shape[0]: ", data_y.shape[0])


        not_data_y = np.full([data_y_size, 2], [1, 0])
        print("not_data_y_shape[0]: ", not_data_y.shape[0])
        #data_x = np.array(batch_x_test_val)
        print("saving the following the bofore data in the following path: ")
        if limited_query_budget:
            print("{}all_samples.npy".format(before_adv_attack_data_path_directory))
            np.save("{}all_samples.npy".format(before_adv_attack_data_path_directory), data_x)
            # save updated_samples_df
            updated_samples_df.to_csv(
                "{}benign_samples_positive_paired.csv".format(before_adv_attack_data_path_directory))
            # save updated_samples_df

        else:
            print("{}before_adv_positive_paired_data_x_part_{}_size_{}.npy".format(before_adv_attack_data_path_directory,
                                                                                     part_of_data, data_y.shape[0]))
            np.save("{}before_adv_positive_paired_data_x_part_{}_size_{}.npy".format(before_adv_attack_data_path_directory,
                                                                                     part_of_data, data_y.shape[0]), data_x)
            # save updated_samples_df
            updated_samples_df.to_csv(
                "{}benign_samples_positive_paired_part_{}_size_{}.csv".format(before_adv_attack_data_path_directory,
                                                                                part_of_data, updated_samples_df.shape[0]))




    if settings =="white-box":
        #this is not realy white box but i can not change the whole code here
        # the only thing sinmilar is the loss of the attack on sub models to the  loss used to train the target models, but this does not make it white box attack
        loss = torch.nn.BCEWithLogitsLoss() #in white box we train the model with the same loss as the target model and it is BCEWithLogitsLoss
        print("white-box attack")
        print("the loss is BCEWithLogitsLoss because this is the default loss used to train all of the model is white box settings")
    else: #black-box
        raise Exception("black-box not implemented yet, and the loss of sub model is independent of the target model loss, so it is changes")
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
    print("data_x: ", data_x)
    print("adv_x", adv_x)
    print("data_y: ", data_y)
    print("not_data_y: ", not_data_y)
    assert data_x.shape == adv_x.shape
    print("saving the after data in the following path: ")
    if limited_query_budget:
        print("{}all_samples.npy".format(after_adv_specific_attack_data_path_directory))
        np.save("{}all_samples.npy".format(after_adv_specific_attack_data_path_directory), adv_x)
    else:
        print("{}{}_adv_positive_paired_data_adv_x_part_{}_size_{}.npy".format(after_adv_specific_attack_data_path_directory,
                                                                                 attack_name, part_of_data,
                                                                                 not_data_y.shape[0]))
        np.save("{}{}_adv_positive_paired_data_adv_x_part_{}_size_{}.npy".format(after_adv_specific_attack_data_path_directory,
                                                                                 attack_name, part_of_data,
                                                                                 not_data_y.shape[0]), adv_x)
    assert adv_x.shape[0] == updated_samples_df.shape[0]  # we want to verify that benign df and the adversarial sampless are on the same number of samples

    # np.fliplr(data_y)
    """test_acc, test_loss = test_adversarial_attack(
        adv_x, not_data_y, fr)
    print(f'{attack_name} accuracy is: {test_acc}, {attack_name} loss '
          f'is: {test_loss}')"""
    # real_attack_acc = 100-test_acc
    # print(f'{attack_name} attack accuracy success is: {real_attack_acc} calc as 100- test_acc')





#///////////////////////////////
def prediction_for_adv_eval(model,adv_data, labels, batch_size=1):
    """predict the adverserial samples"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          "cpu")
    model.eval()
    dataset = dataset_from_numpy(adv_data, labels)
    adv_data_loader = DataLoader(dataset, batch_size=batch_size)
    total_predictions = []
    for images, labels in adv_data_loader:
        labels = labels.to(torch.float64)
        images, labels = images.to(device), labels.to(device)
        #print("image_shape",images.shape)
        #print("labels_shape",labels.shape)
        output = model(images)
        #print("model prediction output on same person images: ", output)
        output = output.cpu().detach().numpy()
        total_predictions.append(output)
    predictions = np.vstack(total_predictions)
    print("predictions shape: ", predictions.shape)
    print("predictions: ", predictions)

    return predictions



def adv_sample_eval(predictor_path ,adv_sample_folder, attack_name,seed,distribution,property,is_faceX_zoo, fr_model= None,
                    backbone=None, need_to_stack=False, predictor_architecture_type=1, n_in=512, increase_shape=False, train_emb=False, dataset_name="CelebA"):
    """evaluate the samples before and after the adverserial attack
    adv_sample_folder: the folder where the adverserial samples are stored
    before_adv_sample_folder: the folder where the samples before the adverserial attack are stored
    attack_name: the adversarial attack used to create this adversarial samples
    need a folder because sometimes the samples created in  chunks and not in one attack"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if backbone is None:
        backbone = "iresnet100"

    if fr_model is None:
        device_type_str = device.type  # "gpu" # cuda:0
        embeder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=backbone, device=device,
                                                                     is_faceX_zoo=is_faceX_zoo,
                                                                     predictor_architecture=predictor_architecture_type,
                                                                     path=predictor_path,
                                                                     is_finetune_emb=train_emb,
                                                                     dataset_name=dataset_name, n_in=n_in)


        # Create complete API
        print("Create complete API")
        fr_model = FR_Api(embedder=embeder, predictor=predictor)
    print("predictor_path: ", predictor_path)
    """list = os.listdir(adv_sample_folder)
    for sample in os.listdir(adv_sample_folder):
        if sample.endswith(".npy"):
            adv_sample = np.load(adv_sample_folder + sample)"""
    print("backbone: ", backbone)
    print("attack_name: ", attack_name)
    print("seed: ", seed)
    print("distribution: ", distribution)
    print("property: ", property)
    print("adv_sample_folder: ", adv_sample_folder)


    print("extract all samples together")
    if os.path.exists(adv_sample_folder + '/all_samples.npy'):
        print("all samples already extracted, load it now")
        all_after_adv_attack_Samples_array = np.load(adv_sample_folder + '/all_samples.npy')
    else:
        sorted_list_adv = natsorted(os.listdir(adv_sample_folder))
        print("sorted_list_adv: ", sorted_list_adv)
        """after_adv_attack_samples_list = [np.load(adv_sample_folder + '/' + sample) for sample in sorted_list_adv]
        print("after_adv_attack_samples_list size: ", len(after_adv_attack_samples_list))
        print("stack all samples together")
        all_after_adv_attack_Samples_array = np.vstack(after_adv_attack_samples_list)"""
        all_after_adv_attack_Samples_array = np.load(adv_sample_folder + '/' + sorted_list_adv[0])
        i = 1
        print("with the i and val remvove")
        for val in sorted_list_adv[1:]:
            all_after_adv_attack_Samples_array = np.concatenate((all_after_adv_attack_Samples_array, np.load(adv_sample_folder + '/' + val)), axis=0)
            print("after iteration: ", i, "all_samples shape: ", all_after_adv_attack_Samples_array.shape)
            i += 1
            # remove the current element from the list
            sorted_list_adv.remove(val)
            print("sored_list_adv size: ", len(sorted_list_adv))
            # clean all irrelevant data to save room for memory
            del val
            # clean the memory
            gc.collect()
            # show current memory usage in the code
            print("memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        print("after loop succeed :), exchange vstack in evaluation script, with this part of code")
        #print("after vstack all_after_adv_attack_Samples_array and before stack before array")
        #all_before_adv_attack_Samples_array = np.vstack(before_adv_attack_samples_list)
        print("all_after_adv_Attack_Array shape: ",all_after_adv_attack_Samples_array.shape)
        #print("all_before_adv_Attack_Array shape: ",all_before_adv_attack_Samples_array.shape)
        #assert all_after_adv_attack_Samples_array.shape == all_before_adv_attack_Samples_array.shape
        np.save("{}all_samples.npy".format(adv_sample_folder), all_after_adv_attack_Samples_array)
        #np.save(attack_name + '_before_adv_attack_all_samples.npy', all_before_adv_attack_Samples_array)
    data_y_size = all_after_adv_attack_Samples_array.shape[0]
    data_y = np.full([data_y_size, 2], [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
    print("stata_y_shape[0]: ", data_y.shape[0])
    not_data_y = np.full([data_y_size, 2], [1, 0])
    print("not_data_y_shape[0]: ", not_data_y.shape[0])
    predictions_of_after_adv = prediction_for_adv_eval(fr_model, all_after_adv_attack_Samples_array, not_data_y)
    assert predictions_of_after_adv.shape == not_data_y.shape



    """predictions_of_before_adv = prediction_for_adv_eval(fr_model, all_before_adv_attack_Samples_array, data_y)
    assert predictions_of_before_adv.shape == data_y.shape"""
    print("predictions_of_after_adv shape: ", predictions_of_after_adv.shape)
    print("predictions_of_after_adv: ", predictions_of_after_adv)
    print("not_data_y shape: ", not_data_y.shape)
    print("not_data_y: ", not_data_y)
    predictions_arg_max = predictions_of_after_adv.argmax(axis=1)
    print("predictions_arg_max shape: ", predictions_arg_max.shape)
    print("predictions_arg_max: ", predictions_arg_max)
    not_data_y_arg_max = not_data_y.argmax(axis=1)
    print("not_data_y_arg_max shape: ", not_data_y_arg_max.shape)
    print("not_data_y_arg_max: ", not_data_y_arg_max)
    label_representation = [True, False] #true labels representation, true is what equatly the not_y has in it ( true in not same 1 in same)
    evaluation_after = evaluation(predictions_arg_max, not_data_y_arg_max, label_representation,
                                  is_y_true_one_class_labels = True, adv_mode_eval=True)
    #is_y_true_one_class_labels = True cause y_true is with only one class labels (not same person/ same person)
    #evaluation_before = evaluation(predictions_of_before_adv.argmax(axis=1), data_y.argmax(axis=1))
    #rename the key with the name accuracy to be named attack_succcess
    print("evaluation_after: ", evaluation_after)
    evaluation_after['attack_success'] = evaluation_after.pop('acc')





    print("evaluation_after: ", evaluation_after)
    properties = {'Attack_name': attack_name, 'Backbone': backbone, 'Seed': seed, 'Property': property, 'Dist': distribution, 'num_of_samples': all_after_adv_attack_Samples_array.shape[0]}
    properties.update(evaluation_after)

    if dataset_name == "MAAD_Face":
        dataset_Results_folder = "MAAD_Face_Results/"

    else:
        dataset_Results_folder = ""

    if train_emb:
        train_emb_folder = "fined_tuning_embedder_Results/"
        if predictor_architecture_type == 1:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_one_Adv_Samples_Eval.csv"
        elif predictor_architecture_type == 2:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_Adv_Samples_Eval.csv"
        elif predictor_architecture_type == 5: #same as predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_five_Adv_Samples_Eval.csv"
        elif predictor_architecture_type == 6: #new predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_six_Adv_Samples_Eval.csv"
        elif predictor_architecture_type == 7: #new predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_seven_Adv_Samples_Eval.csv"
        elif predictor_architecture_type == 8: #susbtitute model should be in a path which is not different predictor, to the case the target is the same with him and htey are in thwy willbe then in the same folder
            #8 us similar to architecture 1 and if the target is the same as rthe susbstittue than it is architecture 8
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_eight_Adv_Samples_Eval.csv"
        else:
            raise Exception("predictor_architecture_type must be 1-9")
    else:
        if predictor_architecture_type ==1:
            result_file =f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/property_{property}/seed_{seed}/property_{property}_seed_{seed}_Adv_Samples_Eval.csv"
        elif predictor_architecture_type ==2:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_Adv_Samples_Eval.csv"
        else:
            raise ValueError("predictor_architecture_type is not 1 or 2")
    #check if path exisit iff not create it
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
    #





    return all_after_adv_attack_Samples_array


def before_adv_sample_eval(before_adv_sample_folder, seed, distribution, property, predictor_path, is_faceX_zoo,
                           fr_model= None, backbone=None, need_to_stack=False, predictor_architecture_type=1, n_in=512,
                           increase_shape=False, train_emb=False, dataset_name="CelebA"):
    """evaluate the samples before and after the adverserial attack
        adv_sample_folder: the folder where the adverserial samples are stored
        before_adv_sample_folder: the folder where the samples before the adverserial attack are stored
        attack_name: the adversarial attack used to create this adversarial samples
        need a folder because sometimes the samples created in  chunks and not in one attack"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if backbone is None:
        backbone = "iresnet100"

    if fr_model is None:
        device_type_str = device.type  # "gpu" # cuda:0
        embeder, predictor, _ = load_embedder_and_predictor_for_eval(backbone=backbone, device=device,
                                                                     is_faceX_zoo=is_faceX_zoo,
                                                                     predictor_architecture=predictor_architecture_type,
                                                                     path=predictor_path,
                                                                     is_finetune_emb=train_emb,
                                                                     dataset_name=dataset_name, n_in=n_in)


        # Create complete API
        print("Create complete API")
        fr_model = FR_Api(embedder=embeder, predictor=predictor)
    print("predictor_path: ", predictor_path)

    print("backbone: ", backbone)
    #print("attack_name: ", attack_name)
    print("seed: ", seed)
    print("distribution: ", distribution)
    print("property: ", property)
    #print("adv_sample_folder: ", adv_sample_folder)

    print("extract all samples together")
    #after_adv_attack_samples_list = [np.load(adv_sample_folder + '/' + sample) for sample in
    #                                 os.listdir(adv_sample_folder)]
    if os.path.exists(before_adv_sample_folder+ '/' + 'all_samples.npy'):
        print("all samples already extracted, load it now")
        all_before_adv_attack_Samples_array = np.load(before_adv_sample_folder+ '/' + 'all_samples.npy')
    else:
        sort_list_bef_adv = natsorted(os.listdir(before_adv_sample_folder))
        print("sort_list_bef_adv: ", sort_list_bef_adv)
        """before_adv_attack_samples_list = [np.load(before_adv_sample_folder + '/' + sample) for sample in sort_list_bef_adv]
        print("before_adv_attack_samples_list size: ", len(before_adv_attack_samples_list))
        print("stack all samples together")
        all_before_adv_attack_Samples_array = np.vstack(before_adv_attack_samples_list)"""
        all_before_adv_attack_Samples_array = np.load(before_adv_sample_folder + '/' + sort_list_bef_adv[0])
        i = 1
        print("with the i and val remvove")
        for val in sort_list_bef_adv[1:]:
            all_before_adv_attack_Samples_array = np.concatenate((all_before_adv_attack_Samples_array, np.load(before_adv_sample_folder + '/' + val)), axis=0)
            print("after iteration: ", i, "all_samples shape: ", all_before_adv_attack_Samples_array.shape)
            i += 1
            # remove the current element from the list
            sort_list_bef_adv.remove(val)
            print("sored_list_adv size: ", len(sort_list_bef_adv))
            # clean all irrelevant data to save room for memory
            del val
            # clean the memory
            gc.collect()
            # show current memory usage in the code
            print("memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        print("after loop succeed :), exchange vstack in evaluation script, with this part of code")
        #print("all_after_adv_Attack_Array shape: ", all_after_adv_attack_Samples_array.shape)
        print("all_before_adv_Attack_Array shape: ",all_before_adv_attack_Samples_array.shape)
        # assert all_after_adv_attack_Samples_array.shape == all_before_adv_attack_Samples_array.shape
        #np.save("{}{}_after_adv_attack_all_samples.npy".format(adv_sample_folder, attack_name),
        #        all_after_adv_attack_Samples_array)
        print("save all before samples together")
        np.save("{}all_samples.npy".format(before_adv_sample_folder), all_before_adv_attack_Samples_array)
    data_y_size = all_before_adv_attack_Samples_array.shape[0]
    data_y = np.full([data_y_size, 2], [0, 1])  # create labels with 0 in no same person and 1 in yes the same person
    print("stata_y_shape[0]: ", data_y.shape[0])
    not_data_y = np.full([data_y_size, 2], [1, 0])
    print("not_data_y_shape[0]: ", not_data_y.shape[0])
    #predictions_of_after_adv = prediction_for_adv_eval(fr_model, all_after_adv_attack_Samples_array, not_data_y)
    #assert predictions_of_after_adv.shape == not_data_y.shape



    predictions_of_before_adv = prediction_for_adv_eval(fr_model, all_before_adv_attack_Samples_array, data_y)
    assert predictions_of_before_adv.shape == data_y.shape
    #print("predictions_of_after_adv shape: ", predictions_of_after_adv.shape)
    #print("predictions_of_after_adv: ", predictions_of_after_adv)
    print("data_y shape: ", data_y.shape)
    print("data_y: ", data_y)
    predictions_arg_max = predictions_of_before_adv.argmax(axis=1)
    print("predictions_arg_max shape: ", predictions_arg_max.shape)
    print("predictions_arg_max: ", predictions_arg_max)
    data_y_arg_max = data_y.argmax(axis=1)
    print("data_y_arg_max shape: ", data_y_arg_max.shape)
    print("data_y_arg_max: ", data_y_arg_max)
    label_representation = [False,
                            True]  # true labels representation, true is what equatly the y has in it ( true in  same person)
    evaluation_before = evaluation(predictions_arg_max, data_y_arg_max, label_representation,
                                  is_y_true_one_class_labels=True, adv_mode_eval=True)
    # is_y_true_one_class_labels = True cause y_true is with only one class labels (not same person/ same person)


    # #print(evaluation_after)
    print("evaluation_before: ", evaluation_before)
    properties = {'Backbone': backbone, 'Seed': seed, 'Property': property, 'Dist': distribution,
                  'num_of_samples': all_before_adv_attack_Samples_array.shape[0]}
    properties.update(evaluation_before)
    if dataset_name == "MAAD_Face":
        dataset_Results_folder = "MAAD_Face_Results/"

    else:
        dataset_Results_folder = ""

    if train_emb:
        train_emb_folder = "fined_tuning_embedder_Results/"
        if predictor_architecture_type == 1:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_one_yes_pairs_results_before_attack.csv"
        elif predictor_architecture_type == 2:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_yes_pairs_results_before_attack.csv"
        elif predictor_architecture_type == 5: #same as predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_five_yes_pairs_results_before_attack.csv"
        elif predictor_architecture_type == 6: #new predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_six_yes_pairs_results_before_attack.csv"
        elif predictor_architecture_type == 7: #new predictor
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_seven_yes_pairs_results_before_attack.csv"
        elif predictor_architecture_type == 8: #susbtitute model should be in a path which is not different predictor, to the case the target is the same with him and htey are in thwy willbe then in the same folder
            #8 us similar to architecture 1 and if the target is the same as rthe susbstittue than it is architecture 8
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{dataset_Results_folder}{train_emb_folder}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_eight_yes_pairs_results_before_attack.csv"
        else:
            raise Exception("predictor_architecture_type must be 1-9")
    else:
        if predictor_architecture_type ==1:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/property_{property}/seed_{seed}/property_{property}_seed_{seed}_yes_pairs_results_before_attack.csv"
        elif predictor_architecture_type ==2:
            result_file = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_two_yes_pairs_results_before_attack.csv"
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
    return all_before_adv_attack_Samples_array


def convert_samples_to_images(folder, num_of_wanted_pair_images=10, is_Adv = True):
    """
    convert samples to images
    :param folder: folder of samples
    :param num_of_wanted_pair_images: number of wanted pair images
    :param is_Adv: if the samples are adv samples
    """
    print("load all samples")
    all_samples = np.load(folder + '/all_samples.npy')

    print("after load all samples")
    images_dir = folder + '/images/'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for i in range(num_of_wanted_pair_images):
        sample = all_samples[i]
        image_1 = sample[0]
        image_2 = sample[1]

        image_1 = image_1[0].transpose(1, 2, 0)#image_1.reshape(image_1.shape[2], image_1.shape[3], image_1.shape[1])#reshape from(1,3,112, 112) to (3,112, 112)
        image_2 = image_2[0].transpose(1, 2, 0)#image_2.reshape(image_2.shape[2], image_2.shape[3], image_2.shape[1]) # reshape from(1,3,112, 112) to (3,112, 112)

        image_1 = np.uint8(image_1*255)
        image_2 = np.uint8(image_2*255)

        img_1_photo = Image.fromarray(image_1)#, "RGB")
        img_2_photo = Image.fromarray(image_2)#, "RGB")

        img_1_photo = img_1_photo.resize((178,218))
        img_2_photo = img_2_photo.resize((178,218))



        curr_pair_dir = images_dir + 'pair_{}/'.format(i)

        if not os.path.exists(curr_pair_dir):
            os.makedirs(curr_pair_dir)
        image_1_filename = curr_pair_dir + "image_0.png"
        image_2_filename = curr_pair_dir + "image_1.png"
        img_1_photo.save(image_1_filename)
        img_2_photo.save(image_2_filename)


def cos_sim(x1, x2):
    dot_product= np.dot(x1, x2)
    norm_x1 =np.linalg.norm(x1)
    norm_x2= np.linalg.norm(x2)
    return dot_product/(norm_x1*norm_x2)


def calc_cosine_similarity(adv_samp, benign_samp, data_path, attack_name):
    #print("calc_cosine_similarity")
    cos_sim_list = [1. -cdist(sample_1.reshape(1, -1), sample_2.reshape(1, -1),'cosine')[0][0] for
                    sample_1, sample_2 in
                    zip(adv_samp, benign_samp)]


    #write cosine to csv file
    samples_index = list(np.arange(1, len(cos_sim_list)+1))
    data_dict = {'samples_index': samples_index, 'cosine_similarity': cos_sim_list}
    cos_sim_df = pd.DataFrame(data_dict)
    cos_sim_df.to_csv(data_path + f'{attack_name}_cosine_similarity.csv', index=False)


def evaluate_samples(predictor_path, property, distribution, seed,
                     attack_name,backbone, need_to_stack,is_faceX_zoo, predictor_architecture_type= 1, n_in=512, increase_shape=False, dataset_name="CelebA", train_emb=False):
    """
    evaluate samples
    :param predictor_path: path to predictor
    :param property: property of the samples
    :param distribution: distribution of the samples
    :param seed: seed of the samples
    :param attack_name: name of the attack
    :param backbone: backbone of the predictor
    :param need_to_stack: if the samples need to be stacked
    :param is_faceX_zoo: if the samples are from faceX zoo
    :param predictor_architecture_type: type of the predictor architecture
    :param n_in: number of input features
    :param increase_shape: if the shape of the images need to be increased
    :param dataset_name: name of the dataset
    """
    adv_path = predictor_path + "adv_data/"
    adv_sample_folder = adv_path + "after_adv_attack/" + attack_name + "/"
    before_adv_sample_folder = adv_path + "before_adv_attack/"

    # convert the lines below to and  function

    adv_samp = adv_sample_eval(predictor_path=predictor_path,
                               adv_sample_folder=adv_sample_folder, attack_name=attack_name, seed=seed,
                               distribution=distribution, property=property,
                               backbone=backbone, need_to_stack=need_to_stack,is_faceX_zoo=is_faceX_zoo,
                               predictor_architecture_type=predictor_architecture_type, n_in=n_in,
                               increase_shape=increase_shape, dataset_name=dataset_name, train_emb=train_emb)


    print("after adv sample eval")
    del adv_samp
    gc.collect()
    convert_samples_to_images(folder=adv_sample_folder, num_of_wanted_pair_images=10)
    print("after convert adv samples to images")
    gc.collect()
    benign_samp = before_adv_sample_eval(before_adv_sample_folder=before_adv_sample_folder, seed=seed,
                                         distribution=distribution, property=property, predictor_path=predictor_path,
                                         backbone=backbone, need_to_stack=need_to_stack,is_faceX_zoo=is_faceX_zoo,
                                         predictor_architecture_type=predictor_architecture_type, n_in=n_in,
                                         increase_shape=increase_shape, dataset_name=dataset_name, train_emb=train_emb)
    print("after benign sample eval")
    del benign_samp
    gc.collect()
    convert_samples_to_images(folder=before_adv_sample_folder, num_of_wanted_pair_images=10)
    # print("after convert benign samples to images")
    # clean all irrelevant variables to save memory usage
    gc.collect()

    adv_samp_cosine = np.load(adv_sample_folder + 'all_samples.npy')
    benign_samp_cosine = np.load(before_adv_sample_folder + 'all_samples.npy')
    # print("calc cosine similarity between benign and adv samples")
    cosine_similarity = calc_cosine_similarity(adv_samp_cosine, benign_samp_cosine, adv_path, attack_name)
#///////////////////////////////


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    config = config['DEFAULT']
    backbone = config['backbone']
    prefix_predictor_path = config['prefix_predictor_path']
    property = config['property']
    sub_model_dist = config['sub_model_dist']
    seed = int(config['seed'])
    attack_name = config['attack_name']
    params = json.loads(config['params'])
    part_of_data = int(config['part_of_data'])
    chunk_size = int(config['chunk_size'])
    settings = config['settings']
    is_faceX_zoo = config.getboolean('is_faceX_zoo')
    limited_query_budget = config.getboolean('limited_query_budget')
    train_emb = config.getboolean('train_emb')
    all_dist = config.getboolean('all_dist')
    # read predictor architecture type number as int
    predictor_architecture = int(config['predictor_architecture'])
    dataset_name = config['dataset_name']

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

    #pytorch lighting seed eveything
    print("pytorch lighting seed_everything")
    pl.seed_everything(seed)
    if all_dist:
        print("train all dist")
        for dist in [0, 100]:
            sub_model_dist = dist
            predictor_path = f"{prefix_predictor_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{sub_model_dist}/"
            print("create the adversarial samples")
            create_adv_samples(backbone, predictor_path, attack_name, params, part_of_data, chunk_size,
                               random_seed=seed, settings=settings,
                               is_faceX_zoo=is_faceX_zoo, limited_query_budget=limited_query_budget, predictor_architecture_type=predictor_architecture, n_in=n_in,
                               increase_shape=increase_shape, dataset_name=dataset_name, train_emb=train_emb)
            need_to_stack = False if limited_query_budget else True  # when limited there is no need to stack
            # since there is only one np file of all the samples
            print("evaluate the adversarial samples")
            evaluate_samples(predictor_path=predictor_path, property=property, distribution=sub_model_dist,
                             seed=seed, attack_name=attack_name, backbone=backbone, need_to_stack=need_to_stack,
                             is_faceX_zoo=is_faceX_zoo, predictor_architecture_type=predictor_architecture, n_in=n_in,
                             increase_shape=increase_shape, dataset_name=dataset_name, train_emb=train_emb)


    else:
        predictor_path = f"{prefix_predictor_path}{property}/seed_{seed}/{seed}_{backbone}_property_{property}_dist_1_{sub_model_dist}/"
        print("create the adversarial samples")
        create_adv_samples(backbone, predictor_path, attack_name, params, part_of_data, chunk_size,random_seed=seed, settings=settings,
                           is_faceX_zoo=is_faceX_zoo, limited_query_budget=limited_query_budget, predictor_architecture_type=predictor_architecture,
                           n_in=n_in, increase_shape=increase_shape, dataset_name=dataset_name, train_emb=train_emb)
        need_to_stack = False if limited_query_budget else True #when limited there is no need to stack
        #since there is only one np file of all the samples
        print("evaluate the adversarial samples")
        evaluate_samples(predictor_path=predictor_path, property=property, distribution=sub_model_dist,
                         seed=seed, attack_name=attack_name, backbone=backbone, need_to_stack=need_to_stack,
                         is_faceX_zoo=is_faceX_zoo, predictor_architecture_type=predictor_architecture, n_in=n_in,
                         increase_shape=increase_shape, dataset_name=dataset_name, train_emb=train_emb)





