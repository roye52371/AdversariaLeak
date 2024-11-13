import ast
import csv
import os
import random
from pathlib import Path

import neptune
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from tqdm import tqdm
from itertools import combinations

from torch.utils.data import Dataset, DataLoader
from FR_System.Embedder.embedder import process_image, Embedder
from FR_System.Predictor.predictor import choose_predictor_and_epoch_size


from FR_System.Predictor.predictor import Predictor
from torch import nn
#from Run_scripts.python_scripts_files.Attack_ART_new import dataset_from_numpy


class dataset_for_utils(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets


    def __getitem__(self, index):

        return (self.data[index], self.targets[index])

    def __len__(self):
        return len(self.targets)



def Demo_creat_yes_records(data_path, suffix, mid_saving=False):
    """
    Create data of pairs of images from the same person.
    :param data_path: Required. Type: str. The data folder which contains the images.
                      The data folder structure should consist of folder for each identity,
                      which contains images of the identity from a single suffix (JPG, PNG, etc.).
    :param suffix: Required. Type: str. The suffix of an image.
    :param mid_saving: Optional. Type: boolean. Whether to save the new data after every iteration.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    data = pd.DataFrame(columns=["path1","path2","label"])
    i = 0

    identities = os.listdir(data_path)

    for identity in identities:
        if os.path.isdir(data_path+identity):
            files = os.listdir(data_path+identity)
            for file in files[:]:
                if not file.endswith(suffix):
                    files.remove(file)

            for file_n1 in files:
                for file_n2 in files:
                    path1 = "{}{}{}{}".format(data_path,identity,"/",file_n1)
                    path2 = "{}{}{}{}".format(data_path,identity,"/",file_n2)
                    # data = data.append({"path1":path1,"path2":path2,"label":1}, ignore_index=True)
                    data = pd.concat([data, pd.DataFrame.from_records([{"path1": path1, "path2": path2, "label": 1}])])

                    i = i+1
                if mid_saving:
                    data.to_csv("{}positive_paired_data.csv".format(data_path), index=False)
    data.to_csv("{}positive_paired_data.csv".format(data_path), index=False)
    return data


def Demo_create_no_records(data_path, suffix, mid_saving=False, positive_run=True):
    """
    Create data of pairs of images from different identities.
    :param data_path: Required. Type: str. The data folder which contains the images.
                      The data folder structure should consist of folder for each identity,
                      which contains images of the identity from a single suffix (JPG, PNG, etc.).
    :param suffix: Required. Type: str. The suffix of an image.
    :param mid_saving: Optional. Type: boolean. Whether to save the new data after every iteration.
    :param positive_run: Optional. Type: boolean. Whether the positive pairs already were extracted.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    if positive_run:
        positive = pd.read_csv("{}positive_paired_data.csv".format(data_path))
    else:
        positive = Demo_creat_yes_records(data_path, suffix, mid_saving)
    data = pd.DataFrame(columns=["path1","path2","label"])
    for i, row in positive.iterrows():
        path1 = row["path1"]
        identity = path1.split("/")[len(path1.split("/"))-2]
        optional_path2 = list(positive[~positive["path2"].str.contains(identity)]["path2"])
        for path2 in optional_path2:
            # data = data.append({"path1":path1,"path2":path2,"label":0}, ignore_index=True)
            data =  pd.concat([data, pd.DataFrame.from_records([{"path1":path1,"path2":path2,"label":0}])])

        if mid_saving:
            data.to_csv("{}negative_paired_data.csv".format(data_path), index=False)
    data.to_csv("{}negative_paired_data.csv".format(data_path), index=False)
    return data


def Demo_split_train_test(outdir, balanced=True, test_size=0.30):
    """
    split the data into train and test.
    :param outdir: Required. Type: str. The data folder which contains the positive and negative csv.
    :param balanced: Optional. Type: boolean. Whether the combined data should be label-balanced. By default True.
    :param test_size: Optional. Type: float between (0,1). The test percentage out of the combined data. By default 0.3.
    :return: DataFrames of x_train, x_test, y_train, y_test.
    """
    positive = pd.read_csv("{}positive_paired_data.csv".format(outdir))
    negative = pd.read_csv("{}negative_paired_data.csv".format(outdir))
    if balanced:
        negative = negative.sample(n=positive.shape[0], replace=False, random_state=1).copy().reset_index(
            drop=True)
    total_data = pd.concat([positive, negative], ignore_index=True)
    data_x = total_data.drop(["label"], axis=1)
    data_y = pd.DataFrame(total_data["label"], columns=["label"])

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=42)
    x_train.to_csv("{}x_train.csv".format(outdir), index=False)
    x_test.to_csv("{}x_test.csv".format(outdir), index=False)
    y_train.to_csv("{}y_train.csv".format(outdir), index=False)
    y_test.to_csv("{}y_test.csv".format(outdir), index=False)
    return x_train, x_test, y_train, y_test




#####celebA
def load_CelebA(dir, property="None", print_progress=True):
    """
    The method makes Dataframe from CelebA dataset.
    :param dir: Required. Type: str. The directory where images exist.
    :param property: Optional. Type: str. The property to use.
    :param print_progress: bool. To print the progress of this function.
    :return: Dataframe of CelebA images information.
    """
    id_df = pd.read_csv(os.path.join(dir, '../../identity_CelebA.csv'), delimiter=' ', names=['path', 'id'])
    id_df['path'] = dir + id_df['path'].astype(str) # Concatenate dir path to every img name
    path_df = pd.DataFrame(columns=['path'])
    property_df = pd.read_csv(os.path.join(dir, '../../list_attr_celeba.csv'), usecols=['image_id', property], header=0)
    property_df = property_df.rename(columns={'image_id': 'path'})
    property_df['path'] = dir + property_df['path'].astype(str) # Concatenate dir path to every img name
    files = glob.glob(dir + '*.jpg') + glob.glob(dir + '*.JPG')
    path_df["path"] = files
    df = path_df.merge(id_df, on='path').merge(property_df, on='path')
    return df


def CelebA_split_ids_train_test(dir, property="None", seed=1):
    """
    Use to split the different CelebA IDs to the three different data sets.
    :param dir: Required. Type: str. The folder that contains the images.
    :param property: Optional. Type: str. The property to use.
    :param seed: Optional. Type: int. The seed to use.
    :return: three lists of IDs (model_train_ids, attack_train_ids, attack_test_ids) and the original data Dataframe
    (celeba_df).
    """
    celeba_df = load_CelebA(dir, property=property)
    ids_property_df = celeba_df.drop_duplicates(subset='id')[['id', property]]
    positive_records = ids_property_df[ids_property_df[property] == 1]
    negative_records = ids_property_df[ids_property_df[property] == -1]
    minority_attr = 1 if len(positive_records) < len(negative_records) else -1
    minority_ids = ids_property_df[ids_property_df[property] == minority_attr]
    majority_ids = ids_property_df[ids_property_df[property] == -minority_attr]
    minority_model_train_ids, minority_attack_ids = train_test_split(minority_ids, test_size=0.5, random_state=seed)
    minority_attack_train_ids, minority_attack_test_ids = train_test_split(minority_attack_ids, test_size=0.5, random_state=seed)
    majority_model_train_ids, majority_attack_ids = train_test_split(majority_ids, test_size=len(minority_attack_ids), random_state=seed) #####
    majority_attack_train_ids, majority_attack_test_ids = train_test_split(majority_attack_ids, test_size=0.5, random_state=seed)
    model_train_ids = np.concatenate((minority_model_train_ids['id'].values, majority_model_train_ids['id'].values))
    attack_train_ids = np.concatenate((minority_attack_train_ids['id'].values, majority_attack_train_ids['id'].values))
    attack_test_ids = np.concatenate((minority_attack_test_ids['id'].values, majority_attack_test_ids['id'].values))
    return model_train_ids, attack_train_ids, attack_test_ids, celeba_df


def CelebA_create_yes_records(data, property_annotations_included, property, min_images_per_identity=15, save_to=None):
    """
    Create data of pairs of images from the same person.
    Currently, this function supposes to be applied on the CelebA dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param save_to: Optional. Type: str. The saving path.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    clip_dfs = []
    ids = data["id"].unique()
    for id in ids:
        data_id = data[data["id"]==id]
        data_id = data_id[:min_images_per_identity]# it is 15 always 15]
        clip_dfs.append(data_id)
    data = pd.concat(clip_dfs, ignore_index=True)
    df = data.groupby('id')['path'].apply(combinations,2)\
                     .apply(list).apply(pd.Series)\
                     .stack().apply(pd.Series)\
                     .set_axis(['path1','path2'],1,inplace=False)\
                     .reset_index(level=0)
    df = df.drop('id', axis=1)
    df['label'] = 1
    if property_annotations_included:
        # Create a map from path to property
        path_to_property = data.set_index('path')[property].to_dict()

        # For celeba_positive_paired_data
        df[f"{property}_path1"] = df['path1'].map(path_to_property)
        df[f"{property}_path2"] = df['path2'].map(path_to_property)

    if save_to is not None:
        df.to_csv(f'{save_to}celeba_positive_paired_data.csv', index=False)
    return df

def create_yes_records_before_adv(data, property, save_to=None):
    """
    Create data of pairs of images from the same person for adversarial attack use (before optimized).
    Currently, this function supposes to be applied on the CelebA dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param save_to: Optional. Type: str. The saving path.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    clip_dfs = []
    ids = data["id"].unique()
    for id in ids:
        data_id = data[data["id"]==id]
        data_id = data_id[:15]
        clip_dfs.append(data_id)
    data = pd.concat(clip_dfs, ignore_index=True)
    df = data.groupby('id')['path'].apply(combinations,2)\
                     .apply(list).apply(pd.Series)\
                     .stack().apply(pd.Series)\
                     .set_axis(['path1','path2'],1,inplace=False)\
                     .reset_index(level=0)
    df = df.drop('id', axis=1)
    df['label'] = 1
    #iterate row in df csv file and to each pair of path add the property of each path
    print("Adding properties to the data")
    #df['path1_property'] = df['path1'].apply(lambda x: data.loc[data['path'] == x, f"{property}"].iloc[0])
    #df['path2_property'] = df['path2'].apply(lambda x: data.loc[data['path'] == x, f"{property}"].iloc[0])
    #print("Adding properties to the data")
    df[f"{property}_path1"] = df.apply(lambda row: data[data['path'] == row['path1']].iloc[0][f"{property}"], axis=1)
    df[f"{property}_path2"] = df.apply(lambda row: data[data['path'] == row['path2']].iloc[0][f"{property}"], axis=1)
    print("check this after the end of the function")


    if save_to is not None:
        df.to_csv(f'{save_to}before_adv_positive_paired_data.csv', index=False)
    return df

def create_no_records_before_adv(data, property, yes_pairs_path, save_to=None, seed=0):
    """
    Create data of pairs of images from the same person for adversarial attack use (before optimized).
    Currently, this function supposes to be applied on the CelebA dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param yes_pairs_path: Required. Type: str. The path of the "celeba_positive_paired_data" file.
    :param save_to: Optional. Type: str. The saving path.
    :param seed: Optional. Type: int. The seed to use.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    random.seed(seed)
    pairs = pd.read_csv("{}before_adv_positive_paired_data.csv".format(yes_pairs_path))
    before_changes_pairs = pairs.copy()
    pairs["path"] = pairs["path1"]
    pairs = pairs.join(data.set_index("path"), on="path")
    #before_changes_pairs = pairs.copy()
    pairs = pairs[["path1","path2","label","id"]]
    path2 = []
    ids = list(pairs["id"].unique())
    ids_options = dict()
    for id in tqdm(ids):
        ids_options.update({id:list(pairs[pairs["id"]!=id]["path2"].unique())})
    for i, row in tqdm(pairs.iterrows()):
        id = row["id"]
        options = ids_options.get(id)
        new_path2 = random.sample(options, 1)[0]
        path2.append(new_path2)
        options.remove(new_path2)
        ids_options.update({id: options})
    pairs["path2"] = path2
    pairs = pairs.drop(["id"], axis=1)
    pairs["label"] = 0

    # iterate over df rows and to each pair of path add the property which they have on data dataframe
    print("Adding properties to the data")
    pairs[f"{property}_path1"] = pairs.apply(lambda row: before_changes_pairs[before_changes_pairs['path1'] == row['path1']].iloc[0][f"{property}_path1"], axis=1)
    pairs[f"{property}_path2"] = pairs.apply(lambda row: before_changes_pairs[before_changes_pairs['path2'] == row['path2']].iloc[0][f"{property}_path2"], axis=1)

    print("check this after the end of the function")

    if save_to is not None:
        pairs.to_csv(f'{save_to}before_adv_negative_paired_data.csv', index=False)
    return pairs

def CelebA_create_no_records(data, property_annotations_included, yes_pairs_path, property, save_to=None, seed=0):
    """
    Create data of pairs of images from the same person.
    Currently, this function supposes to be applied on the CelebA dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param yes_pairs_path: Required. Type: str. The path of the "celeba_positive_paired_data" file.
    :param save_to: Optional. Type: str. The saving path.
    :param seed: Optional. Type: int. The seed to use.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    random.seed(seed)
    pairs = pd.read_csv("{}celeba_positive_paired_data.csv".format(yes_pairs_path))
    pairs["path"] = pairs["path1"]
    pairs = pairs.join(data.set_index("path"), on="path")
    pairs = pairs[["path1","path2","label","id"]]
    path2 = []
    ids = list(pairs["id"].unique())
    ids_options = dict()
    for id in tqdm(ids):
        ids_options.update({id:list(pairs[pairs["id"]!=id]["path2"].unique())})
    for i, row in tqdm(pairs.iterrows()):
        id = row["id"]
        options = ids_options.get(id)
        new_path2 = random.sample(options, 1)[0]
        path2.append(new_path2)
        options.remove(new_path2)
        ids_options.update({id: options})
    pairs["path2"] = path2
    pairs = pairs.drop(["id"], axis=1)
    pairs["label"] = 0

    if property_annotations_included:
        # Create a map from path to property
        path_to_property = data.set_index('path')[property].to_dict()
        # For celeba_positive_paired_data
        pairs[f"{property}_path1"] = pairs['path1'].map(path_to_property)
        pairs[f"{property}_path2"] = pairs['path2'].map(path_to_property)

    if save_to is not None:
        pairs.to_csv(f'{save_to}celeba_negative_paired_data.csv', index=False)
    return pairs


def CelebA_target_training_apply_distribution(model_train_df, distribution, property, seed):
    """
    Apply the wanted distribution on the provided dataset
    :param model_train_df: Required. Type: Dataframe. the training data for the target mode.
    :param distribution:  Required. Type: dict. the wanted distribution according to the property values.
                            For example, {1: 80, -1: 20}.
    :param property: Required. Type: str. The property to use.
    :param seed: Required. Type: int. the random seed to sample from.
    :return: The dataset with the wanted data distribution.
    """
    random.seed(seed)
    assert (sum(distribution.values()) == 100)
    minority = model_train_df[property].value_counts().idxmin()
    minority_value = model_train_df[property].value_counts().min()
    minority_dist = distribution.get(minority)
    if minority_dist == 0:
        all_indexes = list(model_train_df[model_train_df[property] != minority].index)
    elif minority_dist == 100:
        all_indexes = list(model_train_df[model_train_df[property] == minority].index)
    else:
        majority_dist = 100 - minority_dist
        majority_value = len(model_train_df) - minority_value
        num_total = min(100*minority_value//minority_dist, 100*majority_value//majority_dist)
        num_from_majority = num_total*majority_dist//100
        minority_index = list(model_train_df[model_train_df[property] == minority].index)
        minority_index_by_dist = minority_index
        majority_index = list(model_train_df[model_train_df[property] != minority].index)
        random.seed(seed)
        majority_index_by_dist = random.sample(majority_index, num_from_majority)
        all_indexes = majority_index_by_dist + minority_index_by_dist
    return model_train_df.loc[all_indexes]

# def celeba_create_yes_and_no_record(exp_path, model_train_df_dist, seed, property, property_annotations_included):
#     #if property_annotations_included call the function with the property name
#     if property_annotations_included:
#         model_train_df_dist_yes_pairs = create_yes_records_with_property_values_annotations(model_train_df_dist, property=property, save_to=exp_path, dataset_prefix="celeba")
#         model_train_df_dist_no_pairs = create_no_records_with_property_values_annotations(model_train_df_dist, property=property, yes_pairs_path=exp_path, save_to=exp_path, seed=seed, dataset_prefix="celeba")
#     else:
#         model_train_df_dist_yes_pairs = CelebA_create_yes_records(model_train_df_dist, save_to=exp_path)
#         model_train_df_dist_no_pairs = CelebA_create_no_records(model_train_df_dist, yes_pairs_path=exp_path,
#                                                                 save_to=exp_path)
#     return model_train_df_dist_no_pairs, model_train_df_dist_yes_pairs

def celeba_create_yes_and_no_record(exp_path, model_train_df_dist, property_annotations_included, property, distribution, seed, min_images_per_identity=15): #seed, property_annotations_included, property):
    #adjust ids distriubtion to the wanted distribution, and create yes and no pairs
    #if distribution of key 1 is 25
    if distribution[1] == 25: # TODO, change it for all properties
        model_train_df_dist = dataset_target_training_apply_distribution_id_based(model_train_df_dist, distribution=distribution, seed=seed, property=property, min_images_per_identity=min_images_per_identity, exp_path=exp_path)
    model_train_df_dist_yes_pairs = CelebA_create_yes_records(model_train_df_dist, save_to=exp_path, property_annotations_included=property_annotations_included, property=property, min_images_per_identity=min_images_per_identity)
    model_train_df_dist_no_pairs = CelebA_create_no_records(model_train_df_dist, yes_pairs_path=exp_path,
                                                            save_to=exp_path, property_annotations_included=property_annotations_included, property=property) #seed=seed, property=property)
    return model_train_df_dist_no_pairs, model_train_df_dist_yes_pairs

def preprocess_celeba_dataset_and_property_distribution(dir, distribution, property, seed):
    model_train_ids, attack_train_ids, attack_test_ids, celeba_df = CelebA_split_ids_train_test(dir=dir,
                                                                                                property=property,
                                                                                                seed=seed)
    model_train_df = celeba_df[celeba_df['id'].isin(model_train_ids)]
    model_train_df_dist = CelebA_target_training_apply_distribution(model_train_df=model_train_df,
                                                                    distribution=distribution, property=property,
                                                                    seed=seed)
    attack_train_df = celeba_df[celeba_df['id'].isin(attack_train_ids)]
    attack_train_df = CelebA_target_training_apply_distribution(model_train_df=attack_train_df,
                                                                distribution={1: 50, -1: 50},
                                                                property=property,
                                                                seed=seed)
    attack_test_df = celeba_df[celeba_df['id'].isin(attack_test_ids)]
    attack_test_df = CelebA_target_training_apply_distribution(model_train_df=attack_test_df,
                                                               distribution={1: 50, -1: 50},
                                                               property=property,
                                                               seed=seed)
    return attack_test_df, attack_train_df, model_train_df_dist
###end celebA

def load_predictor(exp_path, epoch_num=10, device='cpu', predictor_architecture=1, n_in=512):
    """
    The function loads the predictor of the FR system.
    :param exp_path: Required. str. The path saves weights of the predictor.
    """
    n_in, n_out = n_in, 1
    #load the predictor using architecutre type
    epoch_num, NN = choose_predictor_and_epoch_size(n_in, n_out, predictor_architecture, device)
    # NN = nn.Sequential(nn.Linear(n_in, 64).to(device),
    #                    nn.ReLU().to(device),
    #                    nn.Linear(64, 8).to(device),
    #                    nn.ReLU().to(device),
    #                    nn.Linear(8, n_out).to(device),
    #                    nn.Sigmoid().to(device))
    optimizer = torch.optim.Adam(NN.parameters(), lr=0.0001)
    if predictor_architecture >2:
        checkpoint_name = "predictor_{}_checkpoints".format(predictor_architecture)
    else:
        checkpoint_name = "checkpoints"
    checkpoint = torch.load("{}state_dict_model_epoch_{}.pt".format("{}{}/".format(exp_path, checkpoint_name), epoch_num-1))
    # checkpoint = torch.load("{}state_dict_model_epoch_{}.pt".format("{}checkpoints/".format(exp_path), epoch_num-1))
    NN.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    NN.eval()
    print(f"Predictor loaded successfully from epoch {epoch_num}, when the predictor  architecture"
          f" is {predictor_architecture} and the predictor path is {exp_path}")
    predictor = Predictor(predictor="NN", nn_instance=NN, threshold=0.5, nn_save_path=exp_path, device=device, predictor_architecture_type=predictor_architecture)
    return predictor
def Two_Class_test_Predict_func(x_test,fr, increase_shape=False):
    pred = []
    for i, row in x_test.iterrows():
        path1 = row["path1"]
        path2 = row["path2"]
        np_image1 = process_image(path1, increase_shape=increase_shape)
        np_image2 = process_image(path2, increase_shape=increase_shape)
        prediction = two_class_fr_predict(fr, np_image1, np_image2)
        if torch.is_tensor(prediction):
            pred.append(prediction.detach().numpy()[0])
        else:
            pred.append(prediction)
    pred = np.asarray(pred)#.reshape((len(pred), 1))
    return pred
def two_class_fr_predict(fr,image1, image2):
    if torch.is_tensor(image1):
        image1_emb = fr.embedder.predict(image1)
        image2_emb = fr.embedder.predict(image2)
    else:
        image1_emb = fr.embedder.predict(image1)
        image2_emb = fr.embedder.predict(image2)
    vector1= image1_emb
    vector2= image2_emb
    if torch.is_tensor(vector1):
        #print(f"fr.predictor.device: {fr.predictor.device}")
        #print(f"fr.predictor.nn.device: {fr.predictor.nn.device}")
        diff = np.subtract(vector1.cpu().detach().numpy(), vector2.cpu().detach().numpy())
        proba = fr.predictor.nn(torch.tensor(diff, device=fr.predictor.device).float())
    else:
        diff = np.subtract(vector1, vector2)
        proba = fr.predictor.nn(torch.tensor(diff).float())
    ################ to delete
    proba_comp = 1-proba
    proba_two_class = torch.cat((proba_comp,proba), -1)
    #################
    return list(map(int, (proba_two_class >= fr.predictor.threshold).reshape(-1)))  # take all classes predoction instead only [0]



def filter_benign_pairs(labels, model, adv_data, samples_with_property_info_df=None, batch_size=1, use_properties= True):
    """
    Filter benign pairs which allready not predicted( before adversarial attack)
    as the true labels in the model,
    which mean that this samples want change in the adversarial attack and are not needed
    """
    # filter benign pairs
    #keep only the pair samples wich the model predict their real label correctly
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          "cpu")
    model.eval()
    print("adv_data shape: ", adv_data.shape)
    print("adv_data num of samples: ", len(adv_data))
    #print("samples_with_property_info_df shape: ", samples_with_property_info_df.shape)
    dataset = dataset_for_utils(adv_data, labels)
    adv_data_loader = DataLoader(dataset, batch_size=batch_size)
    samples_index_not_unique = []
    for index, (images, labels) in enumerate(adv_data_loader):
        labels = labels.to(torch.float64)
        images, labels = images.to(device), labels.to(device)
        # print("image_shape", images.shape)
        # print("labels_shape", labels.shape)
        # check the data on the other sub model in order to filter it
        # atpatself = self.fr_model(images)
        output = model(images)
        # print("model prediction output on same person images: ", output)
        output = output.cpu().detach().numpy()
        if output.argmax(axis=1) != labels.cpu().detach().numpy().argmax(axis=1):
            # attack succeed on this sample
            #sample retun different prediction than it should,
            # so it will not be needed in the adversarial attack
            #needed to delete this sample
            samples_index_not_unique.append(index)
    unique_adv_samples = np.delete(adv_data, samples_index_not_unique,
                                   axis=0)  # delete rows(samples) that attack succeed on them( not unique)
    if use_properties:
        #print(samples_index_not_unique)
        #delete the specifix rows using the index of the samples
        updated_samples_with_property_info_df = samples_with_property_info_df.drop(samples_with_property_info_df.index[samples_index_not_unique])
        assert unique_adv_samples.shape[0] == updated_samples_with_property_info_df.shape[0]
        print("unique_adv_samples.shape: ", unique_adv_samples.shape)
        print("unique_adv_samples num of samples: ", len(unique_adv_samples))
        print("updated_samples_with_property_info_df shape: ", updated_samples_with_property_info_df.shape)
        return unique_adv_samples, updated_samples_with_property_info_df, samples_index_not_unique
    else:
        print("unique_adv_samples.shape: ", unique_adv_samples.shape)
        print("unique_adv_samples num of samples: ", len(unique_adv_samples))
        return unique_adv_samples, samples_index_not_unique


def random_pairs(num_of_given_samples, num_of_wanted_samples, seed):
    random.seed(seed)
    return random.sample(range(0, num_of_given_samples), num_of_wanted_samples)


def save_df_and_df_round_to_4(result_file, result_file_4_round_digits, uniq_conf):
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
#def save_result_file_and_4_digit_file(self, result_file, dict_for_csv, result_file_4_round_digits):
def save_result_file_and_4_digit_file(result_file, dict_for_csv, result_file_4_round_digits):
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
    # print("round_four_dict_for_csv", round_four_dict_for_csv)
    # print("result_file_4_round_digits: ", result_file_4_round_digits)
    # print("does the file 4 digits exists: ", os.path.exists(result_file_4_round_digits))

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

def neptune_recoder(exp_name, description, tags, hyperparameters):
    """

    """
    run = neptune.init_run(
    project="AdversariaLeak/Black-Box-different-predictors",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNGJlZDBhNy0xMjA3LTQxNzUtODA1MC0wNTNmMjgwMDA2NjMifQ==",
    tags = tags,
    name = exp_name,
)  # your credentials

    run['hyper-parameters'] = hyperparameters
    run['sys/description'] = description
    run["sys/tags"].add(tags)

    return run


def average_target_model_performance(results_general_path):
    """
    results_general_path: path to the results folder
    """
    #loop acording to seed and property and add to all results csv
    #get all the csv files in the folder
    properties = ["5_o_Clock_Shadow", "Young", "Male"]
    backbones = ["iresnet100", "RepVGG_B0"]
    distributions = [0, 25, 50, 75, 100]
    seeds= [15, 32, 42]
    #stringofseeds
    stringofseeds = "_".join(str(x) for x in seeds)
    #create df to all results
    all_target_performance_csv = pd.DataFrame()

    #get the csv and append to all results csv
    #loop first property then seed
    for property in properties:
        for seed in seeds:
            current_target_fermoance_csv_path = f'{results_general_path}property_{property}/seed_{seed}/property_{property}_seed_{seed}_predictor_four_target_model_results.csv'
            current_target_fermoance_csv = pd.read_csv(current_target_fermoance_csv_path)
            #replace Dist value to Dist[0]

            #add to all results csv
            all_target_performance_csv = all_target_performance_csv.append(current_target_fermoance_csv)
    #save all results csv
    all_target_performance_path= f'{results_general_path}all_target_performance_csv.csv'
    #if file not exitsts create it else append to it
    #check if file exists
    if not os.path.exists(all_target_performance_path):
        all_target_performance_csv.to_csv(all_target_performance_path, index=False)
    else:
        #read the file
        all_target_performance_csv = pd.read_csv(all_target_performance_path)

    # Convert the 'Dist' column from string to dictionary
    all_target_performance_csv['Dist'] = all_target_performance_csv['Dist'].apply(ast.literal_eval)
    #create a column with the first value of the dictionary
    all_target_performance_csv['Dist_val'] = all_target_performance_csv['Dist'].apply(lambda x: list(x.values())[0])

    # # Filter the DataFrame based on the condition
    # filtered_df = df[df['Dist'].apply(lambda x: list(x.values())[0] == 100)]
    #create df to all mean and std accuracy results
    all_target_acc_mean_and_std_csv = pd.DataFrame()
    #average the results according to the property and backbone
    #loop first property then backbone
    for property in properties:
        for backbone in backbones:
            for dist in distributions:
                #get the mean and std of the accuracy for the specific property and backbone

                #get the mean and std of the accuracy
                current_target_acc_mean_and_std_csv = all_target_performance_csv[(all_target_performance_csv['Property'] == property) & (all_target_performance_csv['Backbone'] == backbone) & (all_target_performance_csv['Dist_val'] == dist)].groupby(['Property', 'Backbone', 'Dist_val']).agg({'acc': ['mean', 'std']}).reset_index()
                #round to 4 digits after the dot float result
                current_target_acc_mean_and_std_csv = current_target_acc_mean_and_std_csv.round(4)
                #add to all mean and std accuracy results
                all_target_acc_mean_and_std_csv = all_target_acc_mean_and_std_csv.append(current_target_acc_mean_and_std_csv)
            #

    #save all mean and std accuracy results
    all_target_acc_mean_and_std_path = f'{results_general_path}all_target_accuracy_mean_and_std_seeds_{stringofseeds}_round_to_4.csv'
    if not os.path.exists(all_target_acc_mean_and_std_path):
        all_target_acc_mean_and_std_csv.to_csv(all_target_acc_mean_and_std_path, index=False)


#load checkpoints for eval
def load_checkpoint_for_eval(path, predictor, pred_arch_num, embedder=None, checkpoint_name=None, embedder_check_name=None, device="cuda", checkpoint_number=None):
    """
    The method loads the last checkpoint from the given path.
    :param path: Required. Type: str. The path to the checkpoint.
    :param model: Required. Type: nn.Sequential. The model to load.
    :param optimizer: Required. Type: torch.optim. The optimizer to load.
    :param embedder: Optional. Type: nn.Sequential. The embedder to load.
    :return: The model, optimizer and the epoch number.
    """
    emb_checkpoint_loaded_epoch_num = None
    pred_checkpoint_loaded_epoch_num = None
    if embedder is not None:
        #return error not implemented messeage
        #raise NotImplementedError("the bellow code is problematic and might not be right, TO CHECK and fix before use it")
        embedder_path = "{}{}".format(path, embedder_check_name)
        embedder_checkpoint_list = sorted(Path(embedder_path).iterdir(), key=os.path.getmtime, reverse=True)#[0]
        if checkpoint_number is not None:
            if pred_arch_num == 8 or pred_arch_num == 9: #the embedder start from 20 to be saved
                reveresed_checkpoint_number = (len(embedder_checkpoint_list)-1) - checkpoint_number+20
            else:
                reveresed_checkpoint_number = (len(embedder_checkpoint_list)-1) - checkpoint_number
            embedder_checkpoint = embedder_checkpoint_list[reveresed_checkpoint_number]
            # if len(embedder_checkpoint_list) == len(sorted(Path(path).iterdir(), key=os.path.getmtime, reverse=True)):
            #     reveresed_checkpoint_number = (len(embedder_checkpoint_list)-1) - checkpoint_number
            # else: #embedder of archtecturee 8 or 9 has only 10 checkpoints and not start from 0
            #     reveresed_checkpoint_number = (len(embedder_checkpoint_list)-1) - (checkpoint_number+10)
            # embedder_checkpoint = embedder_checkpoint_list[reveresed_checkpoint_number]
        else:
            embedder_checkpoint = embedder_checkpoint_list[0]  # last checkpoint
        # embedder_checkpoint = sorted(Path(embedder_path).iterdir(), key=os.path.getmtime, reverse=True)[0]
        print("loading embedder from checkpoint: {}".format(embedder_checkpoint))
        checkpoint = torch.load(embedder_checkpoint, map_location=device)
        embedder.load_state_dict(checkpoint['model_state_dict'])
        embedder.eval()
        emb_checkpoint_loaded_epoch_num = checkpoint['epoch'] #for checking and assertion
        #self.embedder = embedder #to fix wrong line

    preditor_path = "{}{}".format(path, checkpoint_name)
    preditor_checkpoint_list = sorted(Path(preditor_path).iterdir(), key=os.path.getmtime, reverse=True)
    if checkpoint_number is not None:
        reveresed_checkpoint_number = (len(preditor_checkpoint_list)-1) - checkpoint_number
        preditor_checkpoint = preditor_checkpoint_list[reveresed_checkpoint_number]
    else:
        preditor_checkpoint = preditor_checkpoint_list[0]  # last checkpoint
    # last_checkpoint = sorted(Path(preditor_path).iterdir(), key=os.path.getmtime, reverse=True)[0]
    checkpoint = torch.load(preditor_checkpoint, map_location=device)
    predictor.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #print("loading embedder from checkpoint: {}".format(embedder_checkpoint))
    print("loading predictor from checkpoint: {}".format(preditor_checkpoint))
    predictor.eval()
    pred_checkpoint_loaded_epoch_num = checkpoint['epoch'] #for checking and assertion
    if embedder is not None: #make sure that the embedder and predictor are from the same epoch, preventing problems
        assert pred_checkpoint_loaded_epoch_num == emb_checkpoint_loaded_epoch_num
    return embedder, predictor #, optimizer, epoch

#choose the predictor for eval
def choose_predictor_for_eval(predictor_architecture, device, n_in=512, n_out=1, dataset_name='CelebA'):
    """
    Choose the predictor architecture and the number of epochs to train it.
    :param n_in: Required. Type: int. The number of input features.
    :param n_out: Required. Type: int. The number of output features.
    :param predictor_architecture: Required. Type: int. The number of the architecture to use.
    :param device: Required. Type: str. The device to use.
    :return: The predictor model and the number of epochs to train it.
    """

    if predictor_architecture == 1 or predictor_architecture == 5 or predictor_architecture==8:  # default
        model = nn.Sequential(nn.Linear(n_in, 64).to(device),
                              nn.ReLU().to(device),
                              nn.Linear(64, 8).to(device),
                              nn.ReLU().to(device),
                              nn.Linear(8, n_out).to(device),
                              nn.Sigmoid().to(device))

        if dataset_name == 'CelebA':
            epoch_num = 10
        elif dataset_name == 'MAAD_Face':
            epoch_num = 20
        else:
            raise Exception("dataset_name must be CelebA or MAAD_Face")
        if predictor_architecture == 8:
            epoch_num = epoch_num + 10  # 10 or 20 for train solo the predictor the 10 rest is fine tune them toghther
        #epoch_num = 10
    elif predictor_architecture == 2:
        model = nn.Sequential(
            nn.Linear(n_in, 256).to(device),
            nn.ReLU().to(device),
            nn.Linear(256, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128, 64).to(device),
            nn.ReLU().to(device),
            nn.Linear(64, 32).to(device),
            nn.ReLU().to(device),
            nn.Linear(32, 16).to(device),
            nn.ReLU().to(device),
            nn.Linear(16, 8).to(device),
            nn.ReLU().to(device),
            nn.Linear(8, n_out).to(device),
            nn.Sigmoid().to(device))
        epoch_num = 30
    elif predictor_architecture == 3: #decreased layer 3 of architecture 2 - the layer with the 64 nuerons (try to get less complex model than it but not dramatically)
        model = nn.Sequential(
            nn.Linear(n_in, 256).to(device),
            nn.ReLU().to(device),
            nn.Linear(256, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128, 32).to(device),
            nn.ReLU().to(device),
            nn.Linear(32, 16).to(device),
            nn.ReLU().to(device),
            nn.Linear(16, 8).to(device),
            nn.ReLU().to(device),
            nn.Linear(8, n_out).to(device),
            nn.Sigmoid().to(device))
        epoch_num = 30
    elif predictor_architecture == 4 or predictor_architecture == 6 or predictor_architecture==9:
        model = nn.Sequential(
            nn.Linear(n_in, 512).to(device),
            nn.ReLU().to(device),
            nn.Linear(512, 256).to(device),
            nn.ReLU().to(device),
            nn.Linear(256, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128, 64).to(device),
            nn.ReLU().to(device),
            nn.Linear(64, 32).to(device),
            nn.ReLU().to(device),
            nn.Linear(32, 16).to(device),
            nn.ReLU().to(device),
            nn.Linear(16, 8).to(device),
            nn.ReLU().to(device),
            nn.Linear(8, n_out).to(device),
            nn.Sigmoid().to(device))
        if predictor_architecture == 9:
            epoch_num = 40 # 30 for train solo the predictor the 10 rest is fine tune them toghther
        else:
            epoch_num = 30 #default

    elif predictor_architecture == 7:  # increase architecture 4 to bottleneck of like autoencoder
        model = nn.Sequential(
            nn.Linear(n_in, 512).to(device),
            nn.ReLU().to(device),
            nn.Linear(512, 256).to(device),
            nn.ReLU().to(device),
            nn.Linear(256, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128, 64).to(device),
            nn.ReLU().to(device),
            nn.Linear(64, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128, 512).to(device),
            nn.ReLU().to(device),
            nn.Linear(512, 256).to(device),
            nn.ReLU().to(device),
            nn.Linear(256, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128, 64).to(device),
            nn.ReLU().to(device),
            nn.Linear(64, 32).to(device),
            nn.ReLU().to(device),
            nn.Linear(32, 16).to(device),
            nn.ReLU().to(device),
            nn.Linear(16, 8).to(device),
            nn.ReLU().to(device),
            nn.Linear(8, n_out).to(device),
            nn.Sigmoid().to(device))
        epoch_num = 30
    else:
        raise ValueError("predictor_architecture must be 1-9")
    print("predictor_architecture: ", predictor_architecture)
    print("epoch_num: ", epoch_num)
    return model, epoch_num

def load_embedder_and_predictor_for_eval(backbone, device, is_faceX_zoo, predictor_architecture, path, is_finetune_emb=False, n_in=512, n_out=1, checkpoint_number=None, dataset_name=None):
    if predictor_architecture > 2:  # predictor 3 and 4 and 5 are in the same folder (CelebA_with_different_predictor_and_clean_backbone) unlike predictor 2 (CelebA_with_predictor_two_and_clean_backbone))
        # checkpoint_name is predictor_{predictor_architecture}_checkpoints
        checkpoint_name = "predictor_{}_checkpoints".format(predictor_architecture)
        embedder_check_name = "checkpoints_emb_using_predictor_{}".format(predictor_architecture)

        # if not os.path.exists("{}predictor_{}_checkpoints".format(saving_path, predictor_architecture)):
        #     os.mkdir("{}checkpoints".format(saving_path))
    else:  # predictor 2 and 1 checkpoints are in other folders and their checkpoint folder call checkpoints
        checkpoint_name = "checkpoints"
        embedder_check_name = "checkpoints_emb"

    embeder = Embedder(device=device, model_name=backbone, faceX_zoo=is_faceX_zoo)
    pretrained_backbone = embeder.embedder #need the backbone sturcture in order to load the finedtuned dict in the currect structure
    predictor_arch, epochs_size = choose_predictor_for_eval(predictor_architecture, device, n_in=n_in, n_out=n_out, dataset_name=dataset_name)
    if is_finetune_emb:
        loaded_fined_tune_backbone, predictor_nn = load_checkpoint_for_eval(path=path, predictor=predictor_arch, pred_arch_num=predictor_architecture, embedder=pretrained_backbone,
                                                       checkpoint_name=checkpoint_name,
                                                       embedder_check_name=embedder_check_name, device=device, checkpoint_number=checkpoint_number)
        embeder.embedder = loaded_fined_tune_backbone #update to backbone of the embedder to the loaded finetuned weights of it
    else: #give none to embedder so it will not be loaded and we will use its pretrained weights as is (not finetune)
        _, predictor_nn = load_checkpoint_for_eval(path=path, predictor=predictor_arch, pred_arch_num=predictor_architecture, embedder=None,
                                                checkpoint_name=checkpoint_name,
                                                embedder_check_name=embedder_check_name, device=device, checkpoint_number=checkpoint_number)

    #create the predictor with its relevant predictor astructure predictor_nn
    predictor = Predictor(predictor="NN", nn_instance=predictor_nn, device=device,
                          predictor_architecture_type=predictor_architecture)
    return embeder, predictor, epochs_size

##################Start MAAD-Face functions##################
def load_MAAD_Face(dir, images_dir='/dt/shabtaia/dt-toshiba/VGG-Face2/', property="None", print_progress=True): #
    """
    The method makes Dataframe from MAAD_Face dataset.
    :param dir: Required. Type: str. The directory where images exist (needs to be the vggface2 images - maad_face based on them).
    :param property: Optional. Type: str. The property to use.
    :param print_progress: bool. To print the progress of this function.
    :return: Dataframe of VGG-Face2 images information.
    """
    #maad_face_property_df = pd.read_csv(os.path.join(dir, 'MAAD_Face.csv'), delimiter=',', names=['Filename', 'Identity', f'{property}']).head(100)
    # maad_face_property_df = pd.read_csv(os.path.join(dir, 'MAAD_Face.csv'), usecols=['Filename', 'Identity', f"{property}"])#.head(100)

    # Convert 'Identity' column to string type
    clean_file_name = 'MAAD_Face_after_identity_and_male_clean.csv'
    #check if clean MAAD_Face.csv file does not exist
    if not os.path.exists(os.path.join(dir, clean_file_name)): #need cause we have '713\\' in the Identity column
        # maad_face_property_df = pd.read_csv(os.path.join(dir, 'MAAD_Face.csv'),
        #                                     usecols=['Filename', 'Identity', f"{property}"])
        maad_face_property_df = pd.read_csv(os.path.join(dir, 'MAAD_Face.csv'))
        maad_face_property_df['Identity'] = maad_face_property_df['Identity'].astype(str)
        # Find indices where the string cannot represent a numeric value
        non_numeric_indices = maad_face_property_df[~maad_face_property_df['Identity'].str.isnumeric()].index
        index_values = non_numeric_indices.values
        # Clean non-numeric values by extracting only the numeric part
        maad_face_property_df.loc[index_values, 'Identity'] = maad_face_property_df.loc[
            non_numeric_indices, 'Identity'].str.strip('\\') #extract(r'(\d+\\)')#r'(\d+)')

        #GET THE NEW VALUE OF THE NEW [non_numeric_indices, 'Identity']
        #non_numeric_indiceVAL =  maad_face_property_df.loc[250634, 'Identity']
        if property == "Male":
            maad_face_property_df['Identity'] = maad_face_property_df['Identity'].astype(str)

            ###to dleete
            multiple_property_values = maad_face_property_df.groupby('Identity')[property].nunique()
            identities_with_multiple_values = multiple_property_values[multiple_property_values > 1].index
            # male_value_counts = maad_face_property_df.loc[
            #     maad_face_property_df['Identity'].isin(identities_with_multiple_values), 'Male'].value_counts()

            # male_value_counts = maad_face_property_df[maad_face_property_df['Identity'].isin(identities_with_multiple_values)][
            #     'Male'].value_counts()
            male_value_counts = maad_face_property_df[
                maad_face_property_df['Identity'].isin(identities_with_multiple_values)].groupby(
                ['Identity', property]).size().unstack()
            # in male_value_counts take the max value for each identity
            # create empty dataframe
            property_values_maad_df = pd.DataFrame()
            property_values = male_value_counts.idxmax(axis=1)
            property_values_maad_df['Identity'] = property_values.index.astype(
                str)  # .astype(int).sort_values().astype(str)
            property_values_maad_df[property] = property_values.values
            # update the maad_face_property_df with the new male values according to the compatible Identitiy valeus
            # maad_face_property_df = maad_face_property_df.merge(property_values_maad_df,  on=['Male', 'Identity'], how='left')# on='Identity', how='left')
            # Merge the fixed_male_values_df with maad_face_property_df to update the male values
            maad_face_property_df = pd.merge(maad_face_property_df, property_values_maad_df, on='Identity', how='left')

            # Update the male column in maad_face_property_df with the fixed male values
            maad_face_property_df[property] = maad_face_property_df[f'{property}_y'].fillna(maad_face_property_df[f'{property}_x'])

            # Drop the unnecessary columns (Male_x, Male_y)
            maad_face_property_df = maad_face_property_df.drop([f'{property}_x', f'{property}_y'], axis=1)
            # convert datate frame column male to be int
            maad_face_property_df[property] = maad_face_property_df[property].astype(int)
            # get from maad_face_property_df identitty '1'
            identity_1_rows = maad_face_property_df[maad_face_property_df['Identity'] == '1']
            # do them same for '3183'
            identity_3183_rows = maad_face_property_df[maad_face_property_df['Identity'] == '3183']
            # check again if the problems of more then one annotation value of male per identitiy has solved
            multiple_property_values = maad_face_property_df.groupby('Identity')[property].nunique()
            identities_with_multiple_values = multiple_property_values[multiple_property_values > 1].index
            updated_male_value_counts = maad_face_property_df[
                maad_face_property_df['Identity'].isin(identities_with_multiple_values)].groupby(
                ['Identity', property]).size().unstack()


        #save the new csv file
        maad_face_property_df.to_csv(os.path.join(dir, clean_file_name), index=False)
    else:
        maad_face_property_df = pd.read_csv(os.path.join(dir, clean_file_name), usecols=['Filename', 'Identity', f"{property}"])
        #maad_face_property_df = pd.read_csv(os.path.join(dir, 'MAAD_Face.csv'), usecols=['Filename', 'Identity', f"{property}"])

    maad_face_property_df['Identity'] = maad_face_property_df['Identity'].astype(str)
    maad_face_property_df = maad_face_property_df.rename(columns={'Filename': 'path', 'Identity': 'id'})
    maad_face_property_df['path'] = images_dir + 'data/train+test/' + maad_face_property_df['path'].astype(str) # Concatenate dir path to every img name
    #drop the rows in the property column which has unkown property value (0 value)
    maad_face_property_df_no_unknown = maad_face_property_df[maad_face_property_df[f"{property}"] != 0]
    #print the number of rows in the df
    #print(f"number of rows in the maad_face_property with all {property} values: {len(maad_face_property_df)}")
    #print(f"number of rows in the maad_face_property_df_no_uknown {property} values: {len(maad_face_property_df_no_unknown)}")
    #print info about property value 1 and for property value -1
    #print(f"number of rows in the maad_face_property_df_no_unknown {property} values with value 1: {len(maad_face_property_df_no_unknown[maad_face_property_df_no_unknown[f'{property}'] == 1])}")
    #print(f"number of rows in the maad_face_property_df_no_unknown {property} values with value -1: {len(maad_face_property_df_no_unknown[maad_face_property_df_no_unknown[f'{property}'] == -1])}")
    return maad_face_property_df_no_unknown

def MAAD_Face_split_ids_train_test(dir, property="None", seed=1): #
    """
    Use to split the different MAAD-Face IDs to the three different data sets.
    :param dir: Required. Type: str. The folder that contains the images.
    :param property: Optional. Type: str. The property to use.
    :param seed: Optional. Type: int. The seed to use.
    :return: three lists of IDs (model_train_ids, attack_train_ids, attack_test_ids) and the original data Dataframe
    (vggface_df).
    """
    maad_face_df = load_MAAD_Face(dir, property=property)
    ids = maad_face_df['id'].unique()
    ids_property_df = maad_face_df.drop_duplicates(subset='id')[['id', property]]
    positive_records = ids_property_df[ids_property_df[property] == 1]
    negative_records = ids_property_df[ids_property_df[property] == -1]
    minority_attr = 1 if len(positive_records) < len(negative_records) else -1
    minority_ids = ids_property_df[ids_property_df[property] == minority_attr]
    majority_ids = ids_property_df[ids_property_df[property] == -minority_attr]
    minority_model_train_ids, minority_attack_ids = train_test_split(minority_ids, test_size=0.5, random_state=seed)
    minority_attack_train_ids, minority_attack_test_ids = train_test_split(minority_attack_ids, test_size=0.5, random_state=seed)
    majority_model_train_ids, majority_attack_ids = train_test_split(majority_ids, test_size=len(minority_attack_ids), random_state=seed) #####
    majority_attack_train_ids, majority_attack_test_ids = train_test_split(majority_attack_ids, test_size=0.5, random_state=seed)
    model_train_ids = np.concatenate((minority_model_train_ids['id'].values, majority_model_train_ids['id'].values))
    attack_train_ids = np.concatenate((minority_attack_train_ids['id'].values, majority_attack_train_ids['id'].values))
    attack_test_ids = np.concatenate((minority_attack_test_ids['id'].values, majority_attack_test_ids['id'].values))
    return model_train_ids, attack_train_ids, attack_test_ids, maad_face_df

def MAAD_Face_create_yes_records(data, property_annotations_included, property,min_images_per_identity=15, save_to=None): #
    """
    Create data of pairs of images from the same person.
    Currently, this function supposes to be applied on the MAAD-Face dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param save_to: Optional. Type: str. The saving path.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    clip_dfs = []
    ids = data["id"].unique()
    for id in ids:
        data_id = data[data["id"]==id]
        data_id = data_id[:min_images_per_identity]# min_images_per_identity is always 15; 15]
        clip_dfs.append(data_id)
    data = pd.concat(clip_dfs, ignore_index=True)
    df = data.groupby('id')['path'].apply(combinations,2)\
                     .apply(list).apply(pd.Series)\
                     .stack().apply(pd.Series)\
                     .set_axis(['path1','path2'],1,inplace=False)\
                     .reset_index(level=0)
    df = df.drop('id', axis=1)
    df['label'] = 1

    if property_annotations_included:
        # Create a map from path to property
        path_to_property = data.set_index('path')[property].to_dict()

        # For celeba_positive_paired_data
        df[f"{property}_path1"] = df['path1'].map(path_to_property)
        df[f"{property}_path2"] = df['path2'].map(path_to_property)

    if save_to is not None:
        df.to_csv(f'{save_to}maad_face_positive_paired_data.csv', index=False)
    return df

def MAAD_Face_create_no_records(data, property_annotations_included, yes_pairs_path, property, save_to=None, seed=0): #
    """
    Create data of pairs of images from the same person.
    Currently, this function supposes to be applied on the MAAD-Face dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param yes_pairs_path: Required. Type: str. The path of the "vggface_positive_paired_data" file.
    :param save_to: Optional. Type: str. The saving path.
    :param seed: Optional. Type: int. The seed to use.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    random.seed(seed)
    pairs = pd.read_csv("{}maad_face_positive_paired_data.csv".format(yes_pairs_path))
    pairs["path"] = pairs["path1"]
    pairs = pairs.join(data.set_index("path"), on="path")
    pairs = pairs[["path1","path2","label","id"]]
    path2 = []
    ids = list(pairs["id"].unique())
    ids_options = dict()
    for id in tqdm(ids):
        ids_options.update({id:list(pairs[pairs["id"]!=id]["path2"].unique())})
    for i, row in tqdm(pairs.iterrows()):
        id = row["id"]
        options = ids_options.get(id)
        new_path2 = random.sample(options, 1)[0]
        path2.append(new_path2)
        options.remove(new_path2)
        ids_options.update({id: options})
    pairs["path2"] = path2
    pairs = pairs.drop(["id"], axis=1)
    pairs["label"] = 0

    if property_annotations_included:
        # Create a map from path to property
        path_to_property = data.set_index('path')[property].to_dict()
        # For celeba_positive_paired_data
        pairs[f"{property}_path1"] = pairs['path1'].map(path_to_property)
        pairs[f"{property}_path2"] = pairs['path2'].map(path_to_property)

    if save_to is not None:
        pairs.to_csv(f'{save_to}maad_face_negative_paired_data.csv', index=False)
    return pairs

def MAAD_Face_target_training_apply_distribution(model_train_df, distribution, property, seed):#), is_reduce_data=False, reduce_factor=None): #
    """
    Apply the wanted distribution on the provided dataset
    :param model_train_df: Required. Type: Dataframe. the training data for the target mode.
    :param distribution:  Required. Type: dict. the wanted distribution according to the property values.
                            For example, {1: 80, -1: 20}.
    :param property: Required. Type: str. The property to use.
    :param seed: Required. Type: int. the random seed to sample from.
    :return: The dataset with the wanted data distribution.
    """
    random.seed(seed)
    assert (sum(distribution.values()) == 100)
    minority = model_train_df[property].value_counts().idxmin()
    minority_value = model_train_df[property].value_counts().min()
    minority_dist = distribution.get(minority)
    if minority_dist == 0:
        all_indexes = list(model_train_df[model_train_df[property] != minority].index)
    elif minority_dist == 100:
        all_indexes = list(model_train_df[model_train_df[property] == minority].index)
    else:
        majority_dist = 100 - minority_dist
        majority_value = len(model_train_df) - minority_value
        num_total = min(100*minority_value//minority_dist, 100*majority_value//majority_dist)
        # if is_reduce_data: #cut the data by the factor, e,g, 0.5 is the factor, so take half of the data
        #     num_total = int(num_total*reduce_factor)
        num_from_minority = num_total*minority_dist//100
        num_from_majority = num_total*majority_dist//100
        minority_index = list(model_train_df[model_train_df[property] == minority].index)
        minority_index_by_dist = random.sample(minority_index, num_from_minority)
        majority_index = list(model_train_df[model_train_df[property] != minority].index)
        random.seed(seed)
        majority_index_by_dist = random.sample(majority_index, num_from_majority)
        all_indexes = majority_index_by_dist + minority_index_by_dist
    return model_train_df.loc[all_indexes]

def maad_face_create_yes_and_no_record(exp_path, model_train_df_dist, property_annotations_included, property, distribution, seed, min_images_per_identity=15): #seed, property):
    if distribution[1] == 25: # TODO, change it for all properties
        print("enter adjust dist by unique ids")
        model_train_df_dist = dataset_target_training_apply_distribution_id_based(model_train_df_dist, distribution=distribution, seed=seed, property=property, min_images_per_identity=min_images_per_identity, exp_path=exp_path)
    model_train_df_dist_yes_pairs = MAAD_Face_create_yes_records(model_train_df_dist, save_to=exp_path, property_annotations_included=property_annotations_included, property=property, min_images_per_identity=min_images_per_identity)
    model_train_df_dist_no_pairs = MAAD_Face_create_no_records(model_train_df_dist, yes_pairs_path=exp_path,
                                                            save_to=exp_path, property_annotations_included=property_annotations_included, property=property) #seed=seed, property_annotations_included=property_annotations_included, property=property)
    return model_train_df_dist_no_pairs, model_train_df_dist_yes_pairs

def preprocess_maad_face_dataset_and_property_distribution(dir, distribution, property, seed):
    model_train_ids, attack_train_ids, attack_test_ids, maad_face = MAAD_Face_split_ids_train_test(dir=dir,
                                                                                                property=property,
                                                                                                seed=seed)
    model_train_df = maad_face[maad_face['id'].isin(model_train_ids)]

    model_train_df_dist = MAAD_Face_target_training_apply_distribution(model_train_df=model_train_df,
                                                                    distribution=distribution, property=property,
                                                                    seed=seed)
    if distribution[1] == 25 or distribution[1] == 50 or distribution[1] == 75: #reduce the train distribution data by half cause those dists have to many data to train with - which effect the target model performance to be insufficient
        #cut ids to half
        model_train_df_dist = reduce_identities_and_maintain_distribution(model_train_df_dist, 'id', property, distribution, seed,
                                                       reduce_factor=0.5)
    attack_train_df = maad_face[maad_face['id'].isin(attack_train_ids)]
    attack_train_df = MAAD_Face_target_training_apply_distribution(model_train_df=attack_train_df,
                                                                distribution={1: 50, -1: 50},
                                                                property=property,
                                                                seed=seed)
    attack_test_df = maad_face[maad_face['id'].isin(attack_test_ids)]
    attack_test_df = MAAD_Face_target_training_apply_distribution(model_train_df=attack_test_df,
                                                               distribution={1: 50, -1: 50},
                                                               property=property,
                                                               seed=seed)
    return attack_test_df, attack_train_df, model_train_df_dist



def reduce_identities_and_maintain_distribution(model_train_df_dist, id_column, property, distribution, seed,
                                                reduce_factor=0.5):
    """
    Reduce the identities by a factor and then sample images to maintain the property distribution
    based on the total images available after the identity reduction.

    :param model_train_df_dist: DataFrame with initial property distribution.
    :param id_column: Column name containing unique identifiers.
    :param property: Property column name for distribution.
    :param distribution: Desired property distribution, e.g., {1: 25, -1: 75}.
    :param seed: Random seed for reproducibility.
    :param reduce_factor: Factor to reduce identities by, e.g., 0.5 for half.
    :return: DataFrame maintaining the property distribution with identities reduced.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Reduce identities for each property group
    reduced_indices = {}
    for prop_value in distribution.keys():
        group_df = model_train_df_dist[model_train_df_dist[property] == prop_value]
        unique_ids = group_df[id_column].unique()

        # Randomly select a reduced set of unique IDs
        reduced_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * reduce_factor), replace=False)

        # Get the indices of the rows corresponding to the reduced set of IDs
        reduced_indices[prop_value] = group_df[group_df[id_column].isin(reduced_ids)].index.tolist()

    # Calculate the total number of available images after reduction
    total_images_after_reduction = sum(len(reduced_indices[prop_value]) for prop_value in distribution)

    # Now, sample images to maintain the property distribution
    sampled_indices = []
    for prop_value, prop_dist in distribution.items():
        # Calculate the number of samples to maintain the distribution from the reduced IDs
        num_samples = min(len(reduced_indices[prop_value]), int(total_images_after_reduction * (prop_dist / 100)))

        # Randomly sample indices from the reduced set
        if num_samples > 0:
            sampled_group_indices = random.sample(reduced_indices[prop_value], num_samples)
            sampled_indices.extend(sampled_group_indices)

    # Use the loc function to return the final sampled dataset based on the sampled indices
    final_df = model_train_df_dist.loc[sampled_indices].reset_index(drop=True)

    return final_df


##################function to create pairs with the proerties values##################
def create_yes_records_with_property_values_annotations(data, property, save_to=None, dataset_prefix=None):
    """
    Create data of pairs of images from the same person for adversarial attack use (before optimized).
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id", property].
    :param property: Required. Type: str. The property name to include in the pair data.
    :param save_to: Optional. Type: str. The saving path.
    :return: DataFrame that contains the columns ["path1", "path2", "label", property+"_path1", property+"_path2"].
    """
    clip_dfs = []
    ids = data["id"].unique()
    for id in ids:
        data_id = data[data["id"] == id]
        data_id = data_id[:15]  # Limit to first 15 entries if applicable
        clip_dfs.append(data_id)
    data_clipped = pd.concat(clip_dfs, ignore_index=True)

    #Generate pairs including the property values directly
    def generate_pair_properties(group):
        pairs = list(combinations(group[['path', property]].to_dict('records'), 2))
        return [{'path1': pair[0]['path'], 'path2': pair[1]['path'],
                 f'{property}_path1': pair[0][property], f'{property}_path2': pair[1][property]} for pair in pairs]

    pairs_data = data_clipped.groupby('id').apply(generate_pair_properties).explode().reset_index(drop=True).apply(pd.Series)
    pairs_data['label'] = 1

    if save_to is not None:
        pairs_data.to_csv(f'{save_to}{dataset_prefix}_positive_paired_data.csv', index=False)
    return pairs_data

def create_no_records_with_property_values_annotations(data, property, yes_pairs_path, save_to=None, seed=None,
                                                           dataset_prefix=None):
    random.seed(seed)
    pairs = pd.read_csv(f"{yes_pairs_path}{dataset_prefix}_positive_paired_data.csv")
    before_changes_pairs = pairs.copy()
    pairs["path"] = pairs["path1"]
    pairs = pairs.join(data.set_index("path"), on="path")
    # before_changes_pairs = pairs.copy()
    pairs = pairs[["path1", "path2", "label", "id", f"{property}_path1", f"{property}_path2"]]
    path2 = []
    path2_property_val = []
    ids = list(pairs["id"].unique())
    ids_options = dict()
    for id in tqdm(ids):
        ids_options.update({id: list(pairs[pairs["id"] != id]["path2", f"{property}_path2"].unique())})
    for i, row in tqdm(pairs.iterrows()):
        id = row["id"]
        options = ids_options.get(id)
        new_path2 = random.sample(options, 1)[0]
        # append new path
        path2.append(new_path2[0])
        # append new path property value
        path2_property_val.append(new_path2[1])
        # path2.append(new_path2)
        options.remove(new_path2)
        ids_options.update({id: options})
    pairs["path2"] = path2
    pairs[f"{property}_path2"] = path2_property_val
    pairs = pairs.drop(["id"], axis=1)
    pairs["label"] = 0

    if save_to is not None:
        pairs.to_csv(f'{save_to}{dataset_prefix}_negative_paired_data.csv', index=False)
    return pairs


##################################new apply adjsut distribution function acording to unique ids, with thorwing ids with less than min value
def dataset_target_training_apply_distribution_id_based(model_train_df, distribution, property, seed, min_images_per_identity, exp_path):
    """
    Adjust the dataset to achieve the desired distribution of unique IDs based on a specified property,
    taking into account the actual distribution of the property among unique IDs.

    :param model_train_df: DataFrame containing the training data.
    :param distribution: Desired distribution of the property among unique IDs (e.g., {1: 25, -1: 75}).
    :param property: Property to adjust by (e.g., 'gender').
    :param seed: Random seed for reproducibility.
    :param min_images_per_identity: Minimum number of images per identity to consider.
    :return: Adjusted DataFrame with the target distribution of unique IDs.
    """
    random.seed(seed)
    assert sum(distribution.values()) == 100, "Distribution must sum to 100."

    #min_images_per_identity must be equal to the number we set to the pairs creation
    #take only the ids with equal or more than 15 images, need to prevent to piars cretion to ruin completely the distribution we adjusted
    model_train_df = model_train_df[model_train_df['id'].map(model_train_df['id'].value_counts()) >= min_images_per_identity]

    # Group by ID and get the first occurrence of the property for each ID
    ids_property = model_train_df.groupby('id')[property].first().reset_index()

    # Calculate the actual distribution of the property among unique IDs
    actual_counts = ids_property[property].value_counts()
    total_ids = actual_counts.sum()
    actual_distribution = {prop: (count / total_ids) * 100 for prop, count in actual_counts.items()}

    # Determine the minority and majority based on the actual distribution
    actual_minority_property, _ = min(actual_distribution.items(), key=lambda x: x[1])
    actual_majority_property, _ = max(actual_distribution.items(), key=lambda x: x[1])

    # Calculate counts for minority and majority properties among unique IDs
    minority_count = actual_counts.get(actual_minority_property, 0)
    majority_count = actual_counts.get(actual_majority_property, 0)

    # Calculate how many IDs we can actually sample based on the desired distribution and available counts
    minority_dist = distribution[actual_minority_property]
    majority_dist = distribution[actual_majority_property]

    num_total = min(minority_count * 100 // minority_dist, majority_count * 100 // majority_dist)
    num_from_minority = num_total * minority_dist // 100
    num_from_majority = num_total * majority_dist // 100

    # Sample IDs to achieve the adjusted target distribution
    minority_ids = ids_property[ids_property[property] == actual_minority_property]['id'].tolist()
    majority_ids = ids_property[ids_property[property] == actual_majority_property]['id'].tolist()

    sampled_minority_ids = random.sample(minority_ids, min(len(minority_ids), num_from_minority))
    sampled_majority_ids = random.sample(majority_ids, min(len(majority_ids), num_from_majority))

    # Combine sampled IDs and filter the original DataFrame to include these IDs
    final_sampled_ids = sampled_minority_ids + sampled_majority_ids
    balanced_df = model_train_df[model_train_df['id'].isin(final_sampled_ids)]

    #save to csv exp_path
    balanced_df.to_csv("{}model_train_df_dist_based_unique_ids.csv".format(exp_path))
    # balanced_df.to_csv(f'model_train_df_based_unique_ids', index=False)

    return balanced_df
########################################################################################
