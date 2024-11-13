from FR_System.Embedder.embedder import Embedder, process_image
from FR_System.Predictor.predictor import Predictor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import torch



def test_prediction(x_test, fr):
    """
    Get the FR system for the test set given.
    :param x_test: Required. Type: Dataframe. Dataframe of pairs path.
    :param fr: Required. Type: FR_System. The used face recognition system.
    :return: ndarray of the predictions.
    """
    pred = []
    for i, row in x_test.iterrows():
        path1 = row["path1"]
        path2 = row["path2"]
        np_image1 = process_image(path1)
        np_image2 = process_image(path2)
        prediction = fr.predict(np_image1, np_image2)
        if torch.is_tensor(prediction):
            pred.append(prediction.detach().numpy()[0])
        else:
            pred.append(prediction)
    pred = np.asarray(pred).reshape((len(pred), 1))
    return pred


def evaluation(pred, labels, label_representation = None, is_y_true_one_class_labels = False,adv_mode_eval = False):
    """
    The method evaluates results between predictions and labels.
    :param pred: Required. Type: ndarray. An array like object with the same dimensions as labels.
    :param labels: Required. Type: ndarray. An array like object with the same dimensions as pred.
    :return: dict. Evaluation results.
    """
    evaluation = {}
    labels = labels.astype(int)
    pred = pred.astype(int)
    if label_representation is None:
        conf_mat = confusion_matrix(labels, pred)
    else:
        conf_mat = confusion_matrix(labels, pred, labels=label_representation)
    print("Confusion matrix: ", conf_mat)
    evaluation['tn'] = conf_mat[0][0]
    evaluation['fp'] = conf_mat[0][1]
    evaluation['fn'] = conf_mat[1][0]
    evaluation['tp'] = conf_mat[1][1]
    evaluation['acc'] = accuracy_score(labels, pred)
    if adv_mode_eval == True:
        evaluation['precision'] = evaluation['tp'] / (evaluation['tp'] + evaluation['fp'])
        evaluation['recall'] = evaluation['tp'] / (evaluation['tp'] + evaluation['fn'])
        evaluation['f1'] = 2 * evaluation['precision'] * evaluation['recall'] / (evaluation['precision'] + evaluation['recall'])
    else:
        evaluation['precision'] = precision_score(labels, pred)
        evaluation['recall'] = recall_score(labels, pred)
        evaluation['f1'] = f1_score(labels, pred)
    if not is_y_true_one_class_labels:
        evaluation['auc'] = roc_auc_score(labels, pred)
    else:
        evaluation['auc'] = -1.0 # "give -1.0 as default value error: Only one class present in y_true. ROC AUC score is not defined in that case" #-1.0
        print("give -1.0 as default value error: Only one class present in y_true. ROC AUC score is not defined in that case") #-1.0
    return evaluation



class FR_Api(torch.nn.Module):
#class FR_Api():
    """
    The face recognition API class.
    """
    def __init__(self, embedder, predictor):
        super(FR_Api, self).__init__()
        """
        Constructor.
        :param embedder: Required. Type: Embeder object.
        :param predictor: Required. Type: Predictor object.
        """
        assert isinstance(embedder, Embedder)
        assert isinstance(predictor, Predictor)
        self.embedder = embedder
        self.predictor = predictor

    def predict(self, image1, image2, proba=False):
        """
        The method predicts whether the two images are of the same person.
        :param image1: Required. Type: ndarray / torch.tensor. An array like object with 4 dimensions:
                        (batch size, channels, image hight, image width).
                        For example: (24, 3, 112, 112).
        :param image2: Required. Type: ndarray / torch.tensor. An array like object with 4 dimensions:
                        (batch size, channels, image hight, image width).
                        For example: (24, 3, 112, 112).
        :param proba: Optional. Type: boolean. Whether to predict the probabilities. Default is False.
        :return: the probability of them to be the same person.
        """
        if torch.is_tensor(image1):
            image1_emb = self.embedder.predict(image1)
            image2_emb = self.embedder.predict(image2)
        else:
            image1_emb = self.embedder.predict(image1)
            image2_emb = self.embedder.predict(image2)
        prediction = self.predictor.predict(image1_emb, image2_emb, proba=proba)
        return prediction

    """def forward(self, x):
        ''' Forward pass of the model. '''
        prediction_array=[]
        #for val in x:
        image_pair = x[0]
        image_1, image_2 = image_pair[0], image_pair[1]
        # x_1 = torch.from_numpy(np.array(x_1, dtype='int32'))
        # x_2 = torch.from_numpy(np.array(x_2, dtype='int32'))
        prob = True
        posterior_proba = self.predict(image_1, image_2, proba=prob)
        # if prob:
        #     posterior_proba = x.item()
        # else:
        posterior_proba = float(posterior_proba)
        # size_x = x.size()
        posterior_proba_not_same = 1 - posterior_proba
        # x= np.full([size_x,2],(posterior_proba_not_same,posterior_proba))

        prediction_array.append([posterior_proba_not_same,posterior_proba])
        #a.append([posterior_proba_not_same, posterior_proba])
        ndarr_prediction=np.array(prediction_array)
        # print("hello1\n")
        # a= np.full([size_x,2],0)
        # print("hello2\n")
        prediction = torch.from_numpy(ndarr_prediction)
        prediction.requires_grad = True
        return prediction"""


    def forward(self, x):
        ''' Forward pass of the model. '''
        #print("in forward")
        prediction_array = []

        #take the two image pair - according to the shape of them
        image_pair = x[0]
        #take each image
        image_1, image_2 = image_pair[0], image_pair[1]
        # x_1 = torch.from_numpy(np.array(x_1, dtype='int32'))
        # x_2 = torch.from_numpy(np.array(x_2, dtype='int32'))
        prob = True
        #posterior_proba = self.predict(image_1, image_2, proba=prob)
        #print(image_1.device)
        emb_pred= self.embedder.embedder(image_1)
        emb_pred2 = self.embedder.embedder(image_2)
        #print(emb_pred.device)
        res_of_sub =torch.subtract(emb_pred,emb_pred2)
        #print(res_of_sub.device)
        #print(self.predictor.nn.device)
        y= self.predictor.nn(res_of_sub)
        #####
        y_comp= 1.0-y
        y_two_class = torch.cat((y_comp, y), -1)
        #####
        return y_two_class