import configparser
import gc
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from matplotlib import ticker

sys.path.append("/sise/home/royek/Toshiba_roye/")




#threshold test functions
def create_all_Results_threshold_test_df(results_path,aggregate_csv_name, backbones, properties, seeds, distributions):
    """create a dataframe with all the results of the query budget folder
    (from art attack, backbones, properties, seeds, distributions)"""

    all_results_df = pd.DataFrame()
    for backbone in backbones:
        for property in properties:
            for seed in seeds:
                for distribution in distributions:
                    results_file = results_path + "/" + backbone + "/property_" + property + "/seed_" + str(seed) + "/target_dist_" + str(distribution) + "/" + aggregate_csv_name
                    results_df = pd.read_csv(results_file)
                    all_results_df = all_results_df.append(results_df)
    all_results_df.drop_duplicates()
    return all_results_df
def threshold_test_query_budget_graphs(attack_setting="different_predictor_BlackBox", dataset_used="CelebA", is_finetuned_modles=False):
    print("in Threshold_Test_query_budget_results_analyze")
    if dataset_used == "CelebA":
        prefix_for_path = ""
    elif dataset_used == "MAAD_Face":
        prefix_for_path = "MAAD_Face_Results/"
    else:
        raise Exception("dataset not supported")

    #check if fine tune emb
    if is_finetuned_modles:
        fine_tune_path = "fined_tuning_embedder_Results/"
    else:
        fine_tune_path = ""
    #attack_setting = "different_predictor_BlackBox" #"whitebox"  # "Semi_BlackBox"
    if attack_setting == "Semi_BlackBox":
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/plots/FINAL_all_results.csv"
    elif attack_setting == "different_predictor_BlackBox":
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/plots/FINAL_all_results.csv"
    else:
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/threshold_test/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/threshold_test/plots/FINAL_all_results.csv"
    # example_path = "/dt/shabtaia/dt-toshiba_2022/Roye/Reports/query_budget_results/attacks_to_compare/loss_test/"
    # save_path = "/dt/shabtaia/dt-toshiba_2022/Roye/Reports/query_budget_results/attacks_to_compare/loss_test/FINAL_all_results.csv"
    properties = ["Male"] #["5_o_Clock_Shadow", "Young", "Male"]
    distributions = [0, 25, 50, 75, 100]
    seeds = [15, 32, 42]
    backbones = ["iresnet100", "RepVGG_B0"]
    #art_attack = ["ProjectedGradientDescent", "CarliniL2Method"]  # , "SaliencyMapMethod"]
    aggregate_csv_name = "query_budget_results_threshold_test_4_round_digits.csv"
    # if file not exist, create it
    if not os.path.exists(save_path):
        all_results_df = create_all_Results_threshold_test_df(example_path, aggregate_csv_name, backbones, properties,
                                               seeds, distributions)
        all_results_df.to_csv(save_path)
    else:
        all_results_df = pd.read_csv(save_path)
    font_size = 16
    for samples_size in ["24000"]: #["10", "20", "30", "100", "24000", "all"]: #["24000", "all"]: #["10", "20", "30", "100", "all"]: #["1000", "3000", "25000"]
        print("in samples_size: ", samples_size)
        for backbone in backbones:
            print("in backbone: ", backbone)
            for distribution in distributions:
                print("in distribution: ", distribution)

                curr_res = all_results_df[
                    ( all_results_df["backbone"] == backbone) & (
                                                              all_results_df["target_distribution"] == distribution)]
                curr_res = curr_res.reset_index(drop=True)
                # Slice the data to only include the first 10 values of the unique samples size
                if samples_size != "all":
                    # curr_res = curr_res[:int(unique_samples_size)]
                    curr_res = curr_res[curr_res["samples_size"] <= int(samples_size)]

                sns.set(font_scale=1.2)  # for font size = 12
                sns.set_style("white")
                plt.figure(figsize=(6, 3))

                if len(properties) == 1:  # pnly male is there
                    palette = sns.color_palette(["green"])
                    sns.relplot(
                        data=curr_res, kind="line",
                        x="samples_size", y="gap_from_threshold", hue="property", legend=False,
                        # color="green"  # Set your desired color directly
                        palette=palette
                    )
                else:
                    sns.relplot(
                        data=curr_res, kind="line",
                        x="samples_size", y="gap_from_threshold", hue="property", legend=False
                    )
                # rename axis names
                plt.xlabel("Samples Size", fontsize=font_size)
                # plt.ylabel("Unique samples success gap")
                plt.ylabel("gap_from_threshold", fontsize=font_size, labelpad=10)


                attack_title = "threshold test"
                plt.title(f"Results on {backbone} with {attack_title}\nTarget Distribution: {distribution}",
                          fontsize=font_size)  # , fontsize=15)
                plt.yticks(np.arange(-1, 1.2, 0.5), fontsize=font_size)
                if samples_size == "20":  # becasie for some reason it is giving not naturql numbers in this case
                    plt.xticks(np.arange(0, 22, 2), fontsize=font_size)
                if samples_size == "24000":
                    plt.xticks(np.arange(0, 24002, 8000), fontsize=font_size)#[0,8000, 16000, 24000]
                # set plt x ticks font size
                plt.xticks(fontsize=font_size)
                plt.axhline(y=0, color=(0.5, 0.5, 0.5), linestyle='--')

                # get the y-axis
                ax = plt.gca()
                if samples_size == "24000":
                    # Set the x-axis tick positions and labels
                    x_ticks = [0, 8000, 16000, 24000]
                    x_tick_labels = ['0', '8k', '16k', '24k']

                    # Set the tick formatter
                    tick_formatter = ticker.FuncFormatter(
                        lambda x, pos: x_tick_labels[x_ticks.index(x)] if x in x_ticks else '')

                    # Apply the formatter to the x-axis ticks
                    ax.xaxis.set_major_formatter(tick_formatter)

                if distribution == 0:
                    ax.annotate('No Property', xy=(-0.17, 0.5), xycoords='axes fraction', xytext=(-0.17, 1.04),
                                arrowprops=dict(arrowstyle='<-', color='lightblue', lw=2),
                                ha='center', va='center')

                    # Add the downward arrow
                    ax.annotate('Property', xy=(-0.17, 0.5), xycoords='axes fraction', xytext=(-0.17, -0.06),
                                # 0.04
                                arrowprops=dict(arrowstyle='<-', color='lightseagreen', lw=2),
                                ha='center', va='center')


                if attack_setting == "different_predictor_BlackBox":
                    attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/plots/"
                    samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/threshold_test/plots/samples_limit:_{samples_size}/"

                else:
                    attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/threshold_test/plots/"
                    samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/threshold_test/plots/samples_limit:_{samples_size}/"


                if not os.path.exists(attack_path):
                    os.makedirs(attack_path)
                if not os.path.exists(samples_limit_path):
                    os.makedirs(samples_limit_path)
                plt.savefig(
                    samples_limit_path + f"Query_budget_results_for_threshold_test_attack_{backbone}_target_dist={distribution}.png",
                    bbox_inches="tight")
                plt.close()

#loss test functions
def create_all_Results_loss_test_df(results_path,aggregate_csv_name, backbones, properties, seeds, distributions):
    """create a dataframe with all the results of the query budget folder
    (from art attack, backbones, properties, seeds, distributions)"""

    all_results_df = pd.DataFrame()
    for backbone in backbones:
        for property in properties:
            for seed in seeds:
                for distribution in distributions:
                    results_file = results_path + "/" + backbone + "/property_" + property + "/seed_" + str(seed) + "/target_dist_" + str(distribution) + "/" + aggregate_csv_name
                    results_df = pd.read_csv(results_file)
                    all_results_df = all_results_df.append(results_df)
    all_results_df.drop_duplicates()
    return all_results_df
def loss_test_query_budget_graphs(attack_setting="different_predictor_BlackBox", dataset_used="CelebA", is_finetuned_modles=False):
    print("in Loss_Test_query_budget_results_analyze")

    if dataset_used == "CelebA":
        prefix_for_path = ""
    elif dataset_used == "MAAD_Face":
        prefix_for_path = "MAAD_Face_Results/"
    else:
        raise Exception("dataset not supported")

    #check if fine tune emb
    if is_finetuned_modles:
        fine_tune_path = "fined_tuning_embedder_Results/"
    else:
        fine_tune_path = ""
    #attack_setting = "different_predictor_BlackBox" #"whitebox"  # "Semi_BlackBox"
    if attack_setting == "Semi_BlackBox":
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/plots/FINAL_all_results.csv"
    elif attack_setting == "different_predictor_BlackBox":
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/plots/FINAL_all_results.csv"
    else:
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/loss_test/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/loss_test/plots/FINAL_all_results.csv"
    # example_path = "/dt/shabtaia/dt-toshiba_2022/Roye/Reports/query_budget_results/attacks_to_compare/loss_test/"
    # save_path = "/dt/shabtaia/dt-toshiba_2022/Roye/Reports/query_budget_results/attacks_to_compare/loss_test/FINAL_all_results.csv"
    properties = ["Male"] #["5_o_Clock_Shadow", "Young", "Male"]
    distributions = [0, 25, 50, 75, 100]
    seeds = [15, 32, 42]
    backbones = ["iresnet100", "RepVGG_B0"]
    #art_attack = ["ProjectedGradientDescent", "CarliniL2Method"]  # , "SaliencyMapMethod"]
    aggregate_csv_name = "query_budget_results_loss_test_4_round_digits.csv"
    # if file not exist, create it
    if not os.path.exists(save_path):
        all_results_df = create_all_Results_loss_test_df(example_path, aggregate_csv_name, backbones, properties,
                                               seeds, distributions)
        all_results_df.to_csv(save_path)
    else:
        all_results_df = pd.read_csv(save_path)
    font_size = 16
    for samples_size in ["24000"]:  #["10", "20", "30", "100", "all"]: #["1000", "3000", "25000"]
            for backbone in backbones:
                print("in backbone: ", backbone)
                for distribution in distributions:
                    print("in distribution: ", distribution)

                    curr_res = all_results_df[
                        ( all_results_df["backbone"] == backbone) & (
                                                                  all_results_df["target_distribution"] == distribution)]
                    curr_res = curr_res.reset_index(drop=True)
                    # Slice the data to only include the first 10 values of the unique samples size
                    if samples_size != "all":
                        # curr_res = curr_res[:int(unique_samples_size)]
                        curr_res = curr_res[curr_res["samples_size"] <= int(samples_size)]

                    sns.set(font_scale=1.2)  # for font size = 12
                    sns.set_style("white")
                    plt.figure(figsize=(6, 3))

                    if len(properties) == 1:  # pnly male is there
                        palette = sns.color_palette(["green"])
                        sns.relplot(
                            data=curr_res, kind="line",
                            x="samples_size", y="accuracy_differences", hue="property", legend=False,
                            # color="green"  # Set your desired color directly
                            palette=palette
                        )
                    else:
                        sns.relplot(
                            data=curr_res, kind="line",
                            x="samples_size", y="accuracy_differences", hue="property", legend=False
                        )
                    # rename axis names
                    plt.xlabel("Samples Size", fontsize=font_size)
                    # plt.ylabel("Unique samples success gap")
                    plt.ylabel("Accuracy Differences", fontsize=font_size, labelpad=10)


                    attack_title = "loss test"
                    plt.title(f"Results on {backbone} with {attack_title}\nTarget Distribution: {distribution}",
                              fontsize=font_size)  # , fontsize=15)
                    plt.yticks(np.arange(-1, 1.2, 0.5), fontsize=font_size)
                    if samples_size == "20":  # becasie for some reason it is giving not naturql numbers in this case
                        plt.xticks(np.arange(0, 22, 2), fontsize=font_size)
                    if samples_size == "24000":
                        plt.xticks(np.arange(0, 24002, 8000), fontsize=font_size)#[0,8000, 16000, 24000]
                    # set plt x ticks font size
                    plt.xticks(fontsize=font_size)
                    plt.axhline(y=0, color=(0.5, 0.5, 0.5), linestyle='--')

                    # get the y-axis
                    ax = plt.gca()
                    if samples_size == "24000":
                        # Set the x-axis tick positions and labels
                        x_ticks = [0, 8000, 16000, 24000]
                        x_tick_labels = ['0', '8k', '16k', '24k']

                        # Set the tick formatter
                        tick_formatter = ticker.FuncFormatter(
                            lambda x, pos: x_tick_labels[x_ticks.index(x)] if x in x_ticks else '')

                        # Apply the formatter to the x-axis ticks
                        ax.xaxis.set_major_formatter(tick_formatter)

                    if distribution == 0:
                        ax.annotate('No Property', xy=(-0.17, 0.5), xycoords='axes fraction', xytext=(-0.17, 1.04),
                                    arrowprops=dict(arrowstyle='<-', color='lightblue', lw=2),
                                    ha='center', va='center')

                        # Add the downward arrow
                        ax.annotate('Property', xy=(-0.17, 0.5), xycoords='axes fraction', xytext=(-0.17, -0.06),
                                    # 0.04
                                    arrowprops=dict(arrowstyle='<-', color='lightseagreen', lw=2),
                                    ha='center', va='center')



                    if attack_setting == "different_predictor_BlackBox":
                        attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/plots/"
                        samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/attacks_to_compare/loss_test/plots/samples_limit:_{samples_size}/"

                    else:
                        attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/loss_test/plots/"
                        samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/attacks_to_compare/loss_test/plots/samples_limit:_{samples_size}/"


                    if not os.path.exists(attack_path):
                        os.makedirs(attack_path)
                    if not os.path.exists(samples_limit_path):
                        os.makedirs(samples_limit_path)
                    plt.savefig(
                        samples_limit_path + f"Query_budget_results_for_loss_test_attack_{backbone}_target_dist={distribution}.png",
                        bbox_inches="tight")
                    plt.close()
def create_all_Results_df(results_path,aggregate_csv_name, art_attack, backbones, properties, seeds, distributions):
    """create a dataframe with all the results of the query budget folder
    (from art attack, backbones, properties, seeds, distributions)"""

    all_results_df = pd.DataFrame()
    for attack in art_attack:
        for backbone in backbones:
            for property in properties:
                #if art attack is "SaliencyMapMethod", then the property "Male" is not exis, so we skip it
                if attack == "SaliencyMapMethod" and property == "Male":
                    continue
                for seed in seeds:
                    for distribution in distributions:
                        results_file = results_path + attack + "/" + backbone + "/property_" + property + "/seed_" + str(seed) + "/target_dist_" + str(distribution) + "/" + aggregate_csv_name
                        results_df = pd.read_csv(results_file)
                        all_results_df = all_results_df.append(results_df)
    all_results_df.drop_duplicates()
    return all_results_df


def AdversariaLeak_query_budget_results_analyze(attack_setting="different_predictor_BlackBox", dataset_used="CelebA", is_finetuned_modles=False):
    print("in AdversariaLeak_query_budget_results_analyze")
    #global attack, backbone, distribution
    #predictor_architecture = 4
    if dataset_used == "CelebA":
        prefix_for_path = ""
    elif dataset_used == "MAAD_Face":
        prefix_for_path = "MAAD_Face_Results/"
    else:
        raise Exception("dataset not supported")

    #check if fine tune emb
    if is_finetuned_modles:
        fine_tune_path = "fined_tuning_embedder_Results/"
    else:
        fine_tune_path = ""


    if attack_setting == "Semi_BlackBox":
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/plots/FINAL_all_results.csv"
    elif attack_setting == "different_predictor_BlackBox":
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/plots/FINAL_all_results.csv"
    else:
        example_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/"
        save_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/plots/FINAL_all_results.csv"
    properties = ["Male"] #["5_o_Clock_Shadow", "Young", "Male"]
    distributions = [0, 25, 50, 75, 100]
    seeds = [15, 32, 42]
    backbones = ["iresnet100", "RepVGG_B0"]
    art_attack = ["ProjectedGradientDescent", "CarliniL2Method"]  # , "SaliencyMapMethod"]
    aggregate_csv_name = "FINAL_relative_success_using_top_mislead_unique_adv_confidence_scores_4_round_digits.csv"
    # if file not exist, create it
    if not os.path.exists(save_path):
        all_results_df = create_all_Results_df(example_path, aggregate_csv_name, art_attack, backbones, properties,
                                               seeds, distributions)
        all_results_df.to_csv(save_path)
    else:
        all_results_df = pd.read_csv(save_path)
    font_size = 16
    # set a minimal high value to run and updated according to max value of sample size  each iteration
    min_sample_size = 1000000
    for unique_samples_size in ["10", "20", "30", "100",
                                "all"]:  # we set the query budget to 750 since the max value we can use is 780 due to some unique size in property male
        for attack in art_attack:
            for backbone in backbones:
                for distribution in distributions:
                    curr_res = all_results_df[(all_results_df["attack_name"] == attack) & (
                                        all_results_df["target_backbone"] == backbone) & (
                                                                  all_results_df["target_distribution"] == distribution)]
                    curr_res = curr_res.reset_index(drop=True)
                    # Slice the data to only include the first 10 values of the unique samples size
                    if unique_samples_size != "all":
                        # curr_res = curr_res[:int(unique_samples_size)]
                        curr_res = curr_res[curr_res["unique_samples_size"] <= int(unique_samples_size)]

                    if unique_samples_size == "all":
                        # get the max value of the unique samples size for each property
                        # then take the min value of all the max values
                        max_val_prop = curr_res.groupby("property")["unique_samples_size"].max()
                        print("max_val_prop", max_val_prop)
                        min_max_sample_size = min(max_val_prop)
                        # print("min_max_sample_size acordint to max value for  each property", min_max_sample_size)
                        # update the min_sample_size
                        if min_max_sample_size < min_sample_size:
                            min_sample_size = min_max_sample_size
                    sns.set(font_scale=1.2)  # for font size = 12
                    sns.set_style("white")
                    plt.figure(figsize=(6, 3))

                    # Set a custom color palette
                    #if properties is only male, then we set the color to green
                    if len(properties) == 1: #pnly male is there
                        # Create a palette with one color, in this case, green
                        palette = sns.color_palette(["green"])
                        sns.relplot(
                            data=curr_res, kind="line",
                            x="unique_samples_size", y="attack_succes_gap", hue="property", legend=False,
                            # color="green"  # Set your desired color directly
                            palette=palette  # Use the palette with a single color
                        )
                    else:
                        sns.relplot(
                            data=curr_res, kind="line",
                            x="unique_samples_size", y="attack_succes_gap", hue="property", legend=False
                        )

                        # Show the plot
                    # plt.show()
                    # rename axis names
                    plt.xlabel("Unique Samples Size", fontsize=font_size)
                    # plt.ylabel("Unique samples success gap")
                    plt.ylabel("FMS Gap", fontsize=font_size, labelpad=10)


                    if attack == "ProjectedGradientDescent":
                        # convert name to PGD
                        attack_title = "PGD"
                    elif attack == "CarliniL2Method":
                        # convert name to PGD
                        attack_title = "CW"
                    else:
                        attack_title = attack
                    # plt.title(f"Results on {backbone} with {attack_title}\nTarget Distribution: {distribution}",
                    #          fontsize=font_size)
                    plt.title(f"Results on {backbone} with {attack_title}\nPD = {distribution}",
                              fontsize=font_size)  # , fontsize=15)
                    plt.yticks(np.arange(-1, 1.2, 0.5), fontsize=font_size)
                    if unique_samples_size == "20":  # becasie for some reason it is giving not naturql numbers in this case
                        plt.xticks(np.arange(0, 22, 2), fontsize=font_size)
                    # IF IT THIS 750, THEN WE NEED TO CHANGE THE X AXIS TO 0, 250, 500, 750
                    if unique_samples_size == "750":
                        plt.xticks(np.arange(0, 751, 250), fontsize=font_size)
                    # set plt x ticks font size
                    plt.xticks(fontsize=font_size)
                    plt.axhline(y=0, color=(0.5, 0.5, 0.5), linestyle='--')

                    # get the y-axis
                    ax = plt.gca()

                    ax.annotate('No Property', xy=(-0.17, 0.5), xycoords='axes fraction', xytext=(-0.17, 1.04),
                                arrowprops=dict(arrowstyle='<-', color='lightblue', lw=2),
                                ha='center', va='center')

                    # Add the downward arrow
                    ax.annotate('Property', xy=(-0.17, 0.5), xycoords='axes fraction', xytext=(-0.17, -0.06),  # 0.04
                                arrowprops=dict(arrowstyle='<-', color='lightseagreen', lw=2),
                                ha='center', va='center')


                    # check if dir exisit
                    if attack_setting == "Semi_BlackBox":
                        attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/plots/{attack}/"
                        samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}Semi_BlackBox_Results/query_budget_results/plots/{attack}/samples_limit:_{unique_samples_size}/"
                    elif attack_setting == "different_predictor_BlackBox":
                        attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/plots/{attack}/"
                        samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}different_predictor_BlackBox_Results/query_budget_results/plots/{attack}/samples_limit:_{unique_samples_size}/"
                    else:
                        attack_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/plots/{attack}/"
                        samples_limit_path = f"/dt/shabtaia/dt-toshiba_2022/Roye/Reports/{prefix_for_path}{fine_tune_path}query_budget_results/plots/{attack}/samples_limit:_{unique_samples_size}/"
                    if not os.path.exists(attack_path):
                        os.makedirs(attack_path)
                    if not os.path.exists(samples_limit_path):
                        os.makedirs(samples_limit_path)
                    plt.savefig(
                        samples_limit_path + f"Query_budget_results_for_{attack}_{backbone}_target_dist={distribution}.png",
                        bbox_inches="tight")
                    plt.close()

    # print the min val
    print("min_sample_size is of all maxes: ", min_sample_size)


if __name__ == '__main__':
    attack_setting = "different_predictor_BlackBox" #"whitebox" #"Semi_BlackBox"
    dataset_used = "MAAD_Face" #"CelebA"
    is_finetuned_modles = True #False
    AdversariaLeak_query_budget_results_analyze(attack_setting=attack_setting, dataset_used=dataset_used, is_finetuned_modles=is_finetuned_modles)
    loss_test_query_budget_graphs(attack_setting=attack_setting, dataset_used=dataset_used, is_finetuned_modles=is_finetuned_modles)
    threshold_test_query_budget_graphs(attack_setting=attack_setting, dataset_used=dataset_used, is_finetuned_modles=is_finetuned_modles)


