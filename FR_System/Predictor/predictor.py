import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from FR_System.Embedder.embedder import process_image



class Predictor():
    """
    The predictor class.
    """
    def __init__(self, predictor=None, x_train=None, y_train=None,
                 nn_save_path="", nn_instance=None, threshold=0.5, embeder=None, device="cpu", n_in=None,NN_epochs=10, predictor_architecture_type = 1, batch_size=None,
                 dataset_name='CelebA', increase_shape=False, emb_name= None):
        """
        Constructor.
        :param predictor: Optional. Type: str. The type of predictor to use.
                          Options: "cosine similarity", "euclidean distance", "chi-square distance",
                          "bhattacharyya distance", "NN".
        :param x_train: Optional. Type: ndarray. The training data for the NN. Must be provided in case
                        predictor="NN" and nn_instance=None.
        :param y_train: Optional. Type: ndarray. The training labels for the NN. Must be provided in case
                        predictor="NN" and nn_instance=None.
        :param nn_save_path: Optional. Type: str. The saving path of the trained NN.
        :param nn_instance:  Optional. Type: torch model. Instance of NN to use as the predictor.
        :param threshold: Optional. Type: float between [0,1]. The threshold for the predictions.
                          If None, the probability will be returned.
        :param predictor_architecture_type: Optional. Type: int. The type of the predictor architecture (two types possible currently - 1 or 2).
        """
        if predictor is None:
            predictor = "cosine similarity"
        self.predictor = predictor
        self.device = device
        self.embedder = embeder
        if predictor == "NN":
            if nn_instance is None:
                assert (x_train is not None)
                assert (y_train is not None)
                if predictor_architecture_type == 1 or predictor_architecture_type == 5 or predictor_architecture_type==8:
                    #lr = 0.0001
                    if dataset_name == 'CelebA':
                        self.NN_epochs = 10
                    elif dataset_name == 'MAAD_Face':
                        self.NN_epochs = 20
                    else:
                        raise Exception("dataset_name must be CelebA or MAAD_Face")
                elif predictor_architecture_type == 2 or predictor_architecture_type == 3 or predictor_architecture_type == 4 or predictor_architecture_type == 6 or predictor_architecture_type == 7 or predictor_architecture_type == 9:
                    self.NN_epochs = 30

                else:
                    raise Exception("predictor_architecture_type must be 1-9")
                if predictor_architecture_type==8 or predictor_architecture_type==9:
                    self.NN_epochs = self.NN_epochs + 10 #10 for finetune
                if batch_size is None:
                    self.nn = self.train_NN(x_train, y_train, saving_path=nn_save_path, embeder=self.embedder, n_in=n_in, epoch_num=NN_epochs,
                                            predictor_architecture=predictor_architecture_type, dataset_name=dataset_name, increase_shape=increase_shape, emb_name=emb_name)
                else:
                    self.nn = self.train_NN(x_train, y_train, saving_path=nn_save_path, embeder=self.embedder, batch_size=batch_size, n_in=n_in, epoch_num=NN_epochs,
                                            predictor_architecture=predictor_architecture_type, dataset_name=dataset_name, increase_shape=increase_shape, emb_name=emb_name)

            else:
                self.nn = nn_instance
        self.threshold = threshold



    def predict(self, vector1, vector2, proba=False):
        """
        The method predicts whether the two images are of the same person, according ti the predictor type.
        :param vector1: Required. Type: ndarray/torch tensor. Image vector 1
        :param vector2: Required. Type: ndarray/torch tensor. Image vector 2
        :param proba: Optional. Type: boolean. Whether to return the raw probability.
        :return: The probability of them to be the same person.
        """
        if proba:
            threshold = None
        else:
            threshold = self.threshold
        if self.predictor == "cosine similarity":
            return self.cosine_similarity(vector1, vector2, threshold)
        elif self.predictor == "euclidean distance":
            return self.euclidean_distance(vector1, vector2, threshold)
        elif self.predictor == "NN":
            return self.net(vector1, vector2, return_proba=proba)

    def cosine_similarity(self, vector1, vector2, threshold):
        """
        The method returns the cosine similarity of vector1 and vector2.
        :param vector1: Required. Type: ndarray/torch tensor. Image vector 1
        :param vector2: Required. Type: ndarray/torch tensor. Image vector 2
        :param threshold: Required. Type: float between [0,1]. The threshold for the predictions.
                          If None, the probability will be returned.
        :return: float. The cosine similarity between corresponding 1-D vectors
                    in the two vectors.
        """
        if torch.is_tensor(vector1):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            output = cos(vector1, vector2)
        else:
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            output = cos(torch.Tensor(vector1), torch.Tensor(vector2))
        proba = output.cpu().detach().numpy()
        if threshold is None:
            return proba
        else:
            return int(proba>=threshold)

    def euclidean_distance(self, vector1, vector2, threshold):
        """
        The method returns the euclidean distance of vector1 and vector2.
        :param vector1: Required. Type: ndarray/torch tensor. Image vector 1
        :param vector2: Required. Type: ndarray/torch tensor. Image vector 2
        :param threshold: Required. Type: float between [0,1]. The threshold for the predictions.
                          If None, the probability will be returned.
        :return: float. The euclidean distance between corresponding 1-D vectors
                    in the two vectors.
        """
        if not torch.is_tensor(vector1):
            vector1 = torch.Tensor(vector1)
            vector2 = torch.Tensor(vector2)
        vector1 -= torch.min(vector1)
        vector1 /= torch.max(vector1)
        vector2 -= torch.min(vector2)
        vector2 /= torch.max(vector2)
        dist = torch.linalg.norm(vector1 - vector2, axis=1)
        if threshold is None:
            return dist
        else:
            return int(dist<=threshold)

    def train_NN(self, x_train, y_train, lossf=torch.nn.BCEWithLogitsLoss(), batch_size=64, epoch_num=10,
                 lr=0.0001, saving_path="", embeder=None, n_in=None, predictor_architecture = 1, dataset_name='CelebA', increase_shape=False, emb_name=None):
        """
        Train an NN to use as a predictor.
        :param x_train: Required. Type: ndarray/torch tensor. The training data for the NN.
        :param y_train: Required. Type: ndarray/torch tensor. The training labels for the NN.
        :param lossf: Optional. The loss function instance to use. If not given, it is equal to cross entropy loss.
        :param batch_size: Optional. The batch size to use during training. If not given, it is equal to 64.
        :param epoch_num: Optional. The number of epoch to do during training. If not given, it is equal to 10.
        :param lr: Optional. Type: float. The learning rate used durring training. If not given, it is equal to 0.0001.
        :param saving_path: Optional. Type: str. The location to save the checkpoints and complete net.
        :return: nn.Sequensial a trained NN.
        """
        #convert lossf to lossf with regularization
        print("predictor_architecture type is: ")
        print(predictor_architecture)

        assert x_train.shape[0] == y_train.shape[0]
        #check if check point of the specific architecture exists

        if predictor_architecture > 2: #predictor 3 and 4 and 5 are in the same folder (CelebA_with_different_predictor_and_clean_backbone) unlike predictor 2 (CelebA_with_predictor_two_and_clean_backbone))
            #checkpoint_name is predictor_{predictor_architecture}_checkpoints
            checkpoint_name = "predictor_{}_checkpoints".format(predictor_architecture)
            embedder_check_name = "checkpoints_emb_using_predictor_{}".format(predictor_architecture)

        else: #predictor 2 and 1 checkpoints are in other folders and their checkpoint folder call checkpoints
            checkpoint_name = "checkpoints"
            embedder_check_name = "checkpoints_emb"
        if not os.path.exists("{}{}".format(saving_path, checkpoint_name)):
            os.mkdir("{}{}".format(saving_path, checkpoint_name))
        

        if embeder is not None:
            if not os.path.exists("{}{}".format(saving_path, embedder_check_name)):
                os.mkdir("{}{}".format(saving_path, embedder_check_name))
        torch.manual_seed(0)
        np.random.seed(0)
        if n_in is None:
            n_in = x_train.shape[1]
        n_out = 1

        epoch_num, model = choose_predictor_and_epoch_size(n_in, n_out, predictor_architecture, device=self.device, dataset_name=dataset_name)
        print("epoch_num is: ")
        print(epoch_num)


        if embeder is None: # or pre trained or fine tuning in two steps first use pre trained and then fine tune
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            #if embeder is iresnet100
            if emb_name== "iresnet100":
                # Freeze lower layers
                for layer in [embeder.conv1, embeder.bn1, embeder.prelu, embeder.layer1, embeder.layer2, embeder.layer3]:
                    for param in layer.parameters():
                        param.requires_grad = False

                # Unfreeze top layers
                for layer in [embeder.layer4, embeder.bn2, embeder.fc, embeder.features]:
                    for param in layer.parameters():
                        param.requires_grad = True

                # Set up optimizer with different learning rates
                fine_tune_params = [p for p in embeder.parameters() if p.requires_grad]
                lr_fine_tune = 0.00001  # Learning rate for fine-tuning
                predictor_params = list(model.parameters())
                fine_tune_and_pred_params = fine_tune_params + predictor_params
                if predictor_architecture == 8 or predictor_architecture == 9: #fine tuninig only after trainng solo the predictor
                    optimizer = torch.optim.Adam(predictor_params, lr=lr) #need to adjust the optimizer in the last 10 epochs
                    finetune_optimizer = torch.optim.Adam(fine_tune_and_pred_params, lr=lr_fine_tune)
                else:
                    raise Exception("predictor_architecture must be 8 or 9")
                #optimizer = torch.optim.Adam(fine_tune_params, lr=lr_fine_tune)
            elif emb_name == "RepVGG_B0":
                # Assuming 'repvgg_model' is your RepVGG model instance

                # Freeze earlier stages
                for stage_name in ['stage0', 'stage1', 'stage2', 'stage3']:
                    for param in getattr(embeder.module, stage_name).parameters():
                        param.requires_grad = False

                # Unfreeze stage 4 and output layer
                for layer_name in ['stage4', 'output_layer']:
                    for param in getattr(embeder.module, layer_name).parameters():
                        param.requires_grad = True

                # Set up optimizer with different learning rates
                fine_tune_params = [p for p in embeder.parameters() if p.requires_grad]
                lr_fine_tune = 0.00001  # Learning rate for fine-tuning
                predictor_params = list(model.parameters())
                fine_tune_and_pred_params = fine_tune_params + predictor_params
                if predictor_architecture == 8 or predictor_architecture == 9:  # fine tuninig only after trainng solo the predictor
                    optimizer = torch.optim.Adam(predictor_params,
                                                 lr=lr)  # need to adjust the optimizer in the last 10 epochs
                    finetune_optimizer = torch.optim.Adam(fine_tune_and_pred_params, lr=lr_fine_tune)
                else:
                    raise Exception("predictor_architecture must be 8 or 9")
            else:
                params = list(model.parameters()) + list(embeder.parameters())
                optimizer = torch.optim.Adam(params, lr=lr)

        epoch_start = 0
        #if checkpoints exists
        #load model and optimizer using load_checkpoint
        if os.path.exists("{}{}".format(saving_path, checkpoint_name)):
            if embeder is not None:
                #check if folder of predictor and embedder is not empty, it might be empty if the training was interrupted
                if predictor_architecture == 8 or predictor_architecture == 9:
                    if len(os.listdir("{}{}".format(saving_path, checkpoint_name))) > 0: #if only predictor checkpoints exists
                        if len(os.listdir("{}{}".format(saving_path, embedder_check_name))) > 0:
                            #in this phase we allready have checkpoint from finetune phase so we will use  the embedder and optimizer from the finetune phase
                            model, optimizer, epoch = self.load_checkpoint(path=saving_path, model=model, embedder=embeder ,optimizer=finetune_optimizer,
                                                                           checkpoint_name=checkpoint_name, embedder_check_name=embedder_check_name)
                            epoch_start = epoch + 1
                        else:
                            #in this phase we dont have checkpoint from finetune phase so we will use the predictor checpoint and optimizer from the pretrain phase with pretrained embeddere
                            #ttooddeelrte the checkpoint number it is just for debuging
                            model, optimizer, epoch = self.load_checkpoint(path=saving_path, model=model, optimizer=optimizer,
                                                                           checkpoint_name=checkpoint_name, embedder_check_name=embedder_check_name)#, checkpoint_number=0)
                            epoch_start = epoch + 1
                else:
                    if (len(os.listdir("{}{}".format(saving_path, checkpoint_name))) > 0) and (len(os.listdir("{}{}".format(saving_path, embedder_check_name))) > 0):
                        model, optimizer, epoch = self.load_checkpoint(path=saving_path, model=model, optimizer=optimizer, embedder=embeder, checkpoint_name=checkpoint_name, embedder_check_name=embedder_check_name)
                        epoch_start = epoch + 1 #because we want to start from the next epoch
                    #because the epoch here is the last one trained and saved
            else:
                # check if folder of predictor is not empty (no embedder trained), it might be empty if the training was interrupted
                if len(os.listdir("{}{}".format(saving_path, checkpoint_name))) > 0:
                    # model, optimizer, epoch = self.load_checkpoint(path=saving_path, model=model, optimizer=optimizer, embedder=embeder, checkpoint_name=checkpoint_name, embedder_check_name=embedder_check_name)
                    model, optimizer, epoch = self.load_checkpoint(path=saving_path, model=model, optimizer=optimizer, checkpoint_name=checkpoint_name, embedder_check_name=embedder_check_name)
                    epoch_start = epoch + 1
        ####### for the architecture that will use scheduler
        # Assuming 'optimizer' is already defined
        # Calculate the total number of batches per epoch
        if predictor_architecture == 5 or predictor_architecture == 6:
            batches_per_epoch = x_train.shape[0] // batch_size
            if x_train.shape[0] % batch_size != 0:
                batches_per_epoch += 1  # Account for the last batch that might be smaller

            total_steps = epoch_num * batches_per_epoch

            # Initialize the OneCycleLR scheduler
            max_lr = 0.01  # This is the peak learning rate
            initial_lr = lr
            div_factor = max_lr / initial_lr  # To start at initial_lr
            final_div_factor = div_factor  # To end at initial_lr or higher

            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                anneal_strategy='linear',
                div_factor=div_factor,
                final_div_factor=final_div_factor
            )
            # scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, anneal_strategy='linear')
        else:
            scheduler = None
        #######enddddddd


        model.train()
        if epoch_start == epoch_num-1:
            if embeder is not None:
                self.embedder = embeder.eval()
            model.eval()
            return model

        #new code for finetuning
        #first is the archtiecture that need two phase training
        if predictor_architecture == 8 or predictor_architecture == 9: #solo predictor training should be less 10 epochs that wwill be used for the finetuning
            #train the predictor with pre trained embedder (none to avoid emb train) and left 10 epochs for finetuning
            #first phase need pretrained embedder, than load the embeddings
            train_embed_output = np.load("{}model_x_train_vectors.npy".format(saving_path))
            embeder.to(self.device).eval()
            if epoch_num-epoch_start > 10:
                #embedder here is pretrained and will return none cause we sent none
                model, _ = self.training_loops(batch_size=batch_size, checkpoint_name=checkpoint_name, embeder=None,
                                                     embedder_check_name=embedder_check_name, epoch_num=epoch_num-10,
                                                     epoch_start=epoch_start, increase_shape=increase_shape, lossf=lossf,
                                                     model=model, n_in=n_in, optimizer=optimizer,
                                                     predictor_architecture=predictor_architecture, saving_path=saving_path,
                                                     x_train=train_embed_output, y_train=y_train, scheduler=scheduler)
            #after trainng solo the predictor, we will train the predictor and embedder together with new optimizer and new learning rate for 10 epochs
            embeder.to(self.device).train()
            if emb_name== "iresnet100":
                # Freeze lower layers
                for layer in [embeder.conv1, embeder.bn1, embeder.prelu, embeder.layer1, embeder.layer2, embeder.layer3]:
                    for param in layer.parameters():
                        param.requires_grad = False

                # Unfreeze top layers
                for layer in [embeder.layer4, embeder.bn2, embeder.fc, embeder.features]:
                    for param in layer.parameters():
                        param.requires_grad = True

                # Set up optimizer with different learning rates
                fine_tune_params = [p for p in embeder.parameters() if p.requires_grad]
                lr_fine_tune = 0.00001  # Learning rate for fine-tuning
                predictor_params = list(model.parameters())
                pred_and_emb_fine_tune_params = fine_tune_params + predictor_params
                finetune_optimizer = torch.optim.Adam(pred_and_emb_fine_tune_params, lr=lr_fine_tune)
                #optimizer = torch.optim.Adam(fine_tune_params, lr=lr_fine_tune)
            elif emb_name == "RepVGG_B0":
                # Assuming 'repvgg_model' is your RepVGG model instance

                # Freeze earlier stages
                for stage_name in ['stage0', 'stage1', 'stage2', 'stage3']:
                    for param in getattr(embeder.module, stage_name).parameters():
                        param.requires_grad = False

                # Unfreeze stage 4 and output layer
                for layer_name in ['stage4', 'output_layer']:
                    for param in getattr(embeder.module, layer_name).parameters():
                        param.requires_grad = True

                # Set up optimizer with different learning rates
                fine_tune_params = [p for p in embeder.parameters() if p.requires_grad]
                lr_fine_tune = 0.00001  # Learning rate for fine-tuning
                predictor_params = list(model.parameters())
                pred_and_emb_fine_tune_params = fine_tune_params + predictor_params
                finetune_optimizer = torch.optim.Adam(pred_and_emb_fine_tune_params, lr=lr_fine_tune)

            else:
                #throw error
                raise Exception("embeder name is not iresnet100 or RepVGG_B0, please check the embeder name")

            if os.path.exists("{}{}".format(saving_path, checkpoint_name)):
                if (len(os.listdir("{}{}".format(saving_path, checkpoint_name))) > 0) and (
                        len(os.listdir("{}{}".format(saving_path, embedder_check_name))) > 0): #load only if both predictor and embedder checkpoints are not empty
                    model, finetune_optimizer, epoch = self.load_checkpoint(path=saving_path, model=model, optimizer=finetune_optimizer,
                                                                   embedder=embeder, checkpoint_name=checkpoint_name,
                                                                   embedder_check_name=embedder_check_name)
                    epoch_start = epoch + 1 # because we want to start from the next epoch
                else:
                    print("the embedder folder is probably empty, probably the embedder not strat train it yet, then the loaded model before is sufficient for use")
                    print("the epoch start in this case will be epoch_num-10")
                    epoch_start = epoch_num-10

            #train the predictor and embedder together with new optimizer and new learning rate for 10 epochs
            #if epoch and start are bigger than 10, than there is and error in the code because we supoosed to enter here if we have left 10 epochs to train in total
            if epoch_num - epoch_start > 10:
                #throw error
                raise Exception("epoch_num - epoch_start > 10 when reaching finetuning phase, please check the code")
            #second phase of training, finetuning with updating both predictor and embedder
            #embeder.to(self.device).train()
            model, embeder = self.training_loops(batch_size=batch_size, checkpoint_name=checkpoint_name, embeder=embeder,
                                                 embedder_check_name=embedder_check_name, epoch_num=epoch_num,
                                                 epoch_start=epoch_start, increase_shape=increase_shape, lossf=lossf,
                                                 model=model, n_in=n_in, optimizer=finetune_optimizer,
                                                 predictor_architecture=predictor_architecture, saving_path=saving_path,
                                                 x_train=x_train, y_train=y_train, scheduler=scheduler)
        else: #if not finetuning than just train the predictor solo with pre trained embedder
            model, embeder = self.training_loops(batch_size=batch_size, checkpoint_name=checkpoint_name, embeder=embeder,
                                                 embedder_check_name=embedder_check_name, epoch_num=epoch_num,
                                                 epoch_start=epoch_start, increase_shape=increase_shape, lossf=lossf,
                                                 model=model, n_in=n_in, optimizer=optimizer,
                                                 predictor_architecture=predictor_architecture, saving_path=saving_path,
                                                 x_train=x_train, y_train=y_train, scheduler=scheduler)
        #        if os.path.exists("{}{}".format(saving_path, checkpoint_name)):
        #new code end
        if embeder is not None:
            self.embedder = embeder.eval()
        return model

    def training_loops(self, batch_size, checkpoint_name, embedder_check_name, embeder, epoch_num, epoch_start,
                       increase_shape, lossf, model, n_in, optimizer, predictor_architecture, saving_path,
                       x_train, y_train, scheduler=None):
        for epoch in tqdm(range(epoch_start, epoch_num)):
            for i in range(0, x_train.shape[0], batch_size):
                optimizer.zero_grad()
                batch_x = x_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                if embeder is not None:
                    x_1 = [process_image(x, increase_shape=increase_shape) for x in batch_x["path1"]]
                    x_2 = [process_image(x, increase_shape=increase_shape) for x in batch_x["path2"]]
                    x1 = np.vstack(x_1)
                    x2 = np.vstack(x_2)
                    x1_t = torch.tensor(x1, device=self.device)
                    x2_t = torch.tensor(x2, device=self.device)
                    # check if x1_t and x2_t bach sizes is one
                    # the embedder if use batch norm will results in error for one sample (need at list 2 samples in the batch)
                    if x1_t.shape[0] == 1 or x2_t.shape[
                        0] == 1:  # they are anyn way in the shame batch shape so  we dont have to do and between them, one is enough
                        # duplicate the batch
                        x1_t = torch.cat((x1_t, x1_t), 0)
                        x2_t = torch.cat((x2_t, x2_t), 0)
                        # also dup batch y labels to match the smaples batch we duplicated above (we duplicated the samples because we need at list 2 samples in the batch for batch norm)
                        batch_y = np.repeat(batch_y, 2, axis=0)

                    try:
                        embeding1 = embeder(x1_t)
                    except ValueError as e:
                        print("Error occurred:", e)
                        print("Shape of x1_t:", x1_t.shape)
                    try:
                        embeding2 = embeder(x2_t)
                    except ValueError as e:
                        print("Error occurred:", e)
                        print("Shape of x2_t:", x2_t.shape)
                    input_x = torch.subtract(embeding1, embeding2)
                    y_pred = model(input_x)
                else:
                    input_x = batch_x
                    y_pred = model(torch.tensor(input_x, device=self.device).float())
                flattened_batch_y = torch.tensor(batch_y.astype(float), device=self.device)
                y_pred_comp = 1.0 - y_pred
                T_cat = torch.cat((y_pred_comp, y_pred), -1)  # concatenate the two tensors to two classes tensor
                y_pred_two_class = T_cat
                loss = lossf(y_pred_two_class.float(), torch.tensor(batch_y.astype(float), device=self.device))
                loss.backward()
                optimizer.step()
                # if archtieture that use scheduler
                if predictor_architecture == 5 or predictor_architecture == 6:
                    scheduler.step()  # Update the learning rate
                self.embedder = embeder
                print('epoch: ', epoch, ' part: ', i, ' loss: ', loss.item())

            if predictor_architecture == 5 or predictor_architecture == 6:
                print('epoch:', epoch, 'current lr:', scheduler.get_last_lr()[0], 'loss:', loss.item())
            else:
                print('epoch: ', epoch, ' loss: ', loss.item())

            if predictor_architecture == 5 or predictor_architecture == 6:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'input_size': n_in,
                }, "{}state_dict_model_epoch_{}.pt".format("{}{}/".format(saving_path, checkpoint_name), epoch))
                if embeder is not None:
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.embedder.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss,
                                }, "{}state_dict_embedder_epoch_{}.pt".format(
                        "{}{}/".format(saving_path, embedder_check_name), epoch))

            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'input_size': n_in,
                }, "{}state_dict_model_epoch_{}.pt".format("{}{}/".format(saving_path, checkpoint_name), epoch))
                if embeder is not None:
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.embedder.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, "{}state_dict_embedder_epoch_{}.pt".format(
                        "{}{}/".format(saving_path, embedder_check_name), epoch))
        return model, embeder

    def net(self, vector1, vector2, return_proba=False):
        """
        The method returns the probability for class 1 according to the trained NN.
        :param vector1: Required. Type: ndarray/torch tensor. Image vector 1
        :param vector2: Required. Type: ndarray/torch tensor. Image vector 2
        :param return_proba: Optional. Type: boolean. Whether to return the probability. Default is False.
        :return:
        """
        if torch.is_tensor(vector1):
            diff = np.subtract(vector1.cpu().detach().numpy(), vector2.cpu().detach().numpy())
            proba = self.nn(torch.tensor(diff, device=self.device).float())
        else:
            diff = np.subtract(vector1, vector2)
            proba = self.nn(torch.tensor(diff).float())
        if return_proba:
            return proba.flatten().tolist()[0]
        else:
            return list(map(int, (proba >= self.threshold).reshape(-1)))[0]

    def load_checkpoint(self, path, model, optimizer, embedder=None, checkpoint_name=None, embedder_check_name=None, checkpoint_number=None, predictor_architecture_type=None):
        """
        The method loads the last checkpoint from the given path.
        :param path: Required. Type: str. The path to the checkpoint.
        :param model: Required. Type: nn.Sequential. The model to load.
        :param optimizer: Required. Type: torch.optim. The optimizer to load.
        :param embedder: Optional. Type: nn.Sequential. The embedder to load.
        :return: The model, optimizer and the epoch number.
        """

        if embedder is not None:
            #return error not implemented messeage
            #raise NotImplementedError("the bellow code is problematic and might not be right, TO CHECK and fix before use it")
            embedder_path = "{}{}".format(path, embedder_check_name)
            embedder_checkpoint_list = sorted(Path(embedder_path).iterdir(), key=os.path.getmtime, reverse=True)
            if checkpoint_number is not None:
                reveresed_checkpoint_number = (len(embedder_checkpoint_list)-1) - checkpoint_number
                embedder_checkpoint = embedder_checkpoint_list[reveresed_checkpoint_number]
            else:
                embedder_checkpoint = embedder_checkpoint_list[0] #last checkpoint
            # embedder_checkpoint = sorted(Path(embedder_path).iterdir(), key=os.path.getmtime, reverse=True)[0]
            print("loading embedder from checkpoint: {}".format(embedder_checkpoint))
            checkpoint = torch.load(embedder_checkpoint, map_location=self.device)
            embedder.load_state_dict(checkpoint['model_state_dict'])
            self.embedder = embedder #to fix wrong line

        preditor_path = "{}{}".format(path, checkpoint_name)
        preditor_checkpoint_list = sorted(Path(preditor_path).iterdir(), key=os.path.getmtime, reverse=True)
        if checkpoint_number is not None:
            reveresed_checkpoint_number = (len(preditor_checkpoint_list)-1) - checkpoint_number
            preditor_checkpoint = preditor_checkpoint_list[reveresed_checkpoint_number]
        else:
            preditor_checkpoint = preditor_checkpoint_list[0] #last checkpoint
        # last_checkpoint = sorted(Path(preditor_path).iterdir(), key=os.path.getmtime, reverse=True)[0]
        checkpoint = torch.load(preditor_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        model.train()
        return model, optimizer, epoch
def choose_predictor_and_epoch_size(n_in, n_out, predictor_architecture, device, dataset_name='CelebA'):
    """
    Choose the predictor architecture and the number of epochs to train it.
    :param n_in: Required. Type: int. The number of input features.
    :param n_out: Required. Type: int. The number of output features.
    :param predictor_architecture: Required. Type: int. The number of the architecture to use.
    :param device: Required. Type: str. The device to use.
    :return: The predictor model and the number of epochs to train it.
    """
    #lr_scheduler_indicator = False
    if predictor_architecture == 1 or predictor_architecture == 5 or predictor_architecture==8:  # 1 is default, 5 uses scheduler
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
            epoch_num = epoch_num + 10 # 10 or 20 for train solo the predictor the 10 rest is fine tune them toghther
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
    elif predictor_architecture == 4 or predictor_architecture == 6 or predictor_architecture==9: #6 uses scheduler
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
    elif predictor_architecture == 7: #increase architecture 4 to bottleneck of like autoencoder
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
        raise ValueError("predictor_architecture must be 1-7")
    return epoch_num, model


