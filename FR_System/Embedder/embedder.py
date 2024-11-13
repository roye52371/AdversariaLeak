#from matplotlib import pyplot as plt
import pickle

from torchvision.transforms import transforms
import FR_System.Embedder.iresnet as AFBackbone
#import FR_System.Embedder.inception_resnet as IRBackbone
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import onnx
from onnx2pytorch import ConvertModel
import copy
from ModelX_Zoo.test_protocol.utils.model_loader import ModelLoader
from ModelX_Zoo.backbone.backbone_def import BackboneFactory
# from FR_System.Embedder.ResNet_for_VGGFace2 import get_resnet50_model
# from FR_System.Embedder.SENet_for_VGGFace2 import get_senet50_model
# #from facenet_pytorch import InceptionResnetV1
# from FR_System.Embedder.Inception_ResNet_V1_Model import InceptionResnetV1




def process_image(image_path, increase_shape=False):
    """
    Process the image to fit FR_Api.
    :param image_path: Required. Type: str. The path of the image.
    :param increase_shape: Optional. Type: bool. Default: False. If True, the image will be resized to (224, 224).
    :return: ndarray with the shape of (112, 112, 3).
    """
    if '/storage/users/dt-toshiba' in image_path:
        image_path = image_path.replace('/storage/users/dt-toshiba/', '/dt/shabtaia/dt-toshiba/')
    image = Image.open("{}".format(image_path))
    ### to delete
    #plt.imshow(image)
    #plt.show()
    ###to delete
    if increase_shape:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
    np_image = test_transform(image).numpy()
    ### to delete
    #plt.imshow(np_image.transpose(1, 2, 0))
    #plt.show()
    ### to delete
    np_image = np_image.reshape((1, np_image.shape[0], np_image.shape[1], np_image.shape[2]))
    ### to delete
    #np_image_2 = np_image.transpose(0, 2, 3, 1)
    #plt.imshow(np_image[0].transpose(1, 2, 0))
    #plt.show()
    ### to delete

    return np_image


def get_embeddings(embedder, image_path, increase_shape=False):
    """
    Create embedding vectors from an image.
    :param embedder: Required. Type: Embedder object.
    :param image_path: Required. Type: str. The path of the image.
    :return: ndarray with the shape of (1, 512).
    """
    np_image = process_image(image_path, increase_shape=increase_shape)
    return embedder.predict(np_image)


def convert_data_to_net_input(data, embedder, saving_path_and_name, increase_shape=False):
    """
    Convert records of pairs of images into the form of the embedding vectors for training models.
    :param data: Required. Type: DataFrame. data must include column ['path1', 'path2'].
                 Each column should contain the path to the images in str.
    :param embedder: Required. Type: Embedder object.
    :param saving_path_and_name: Required. Type: str. The saving path and file name.
    :return: ndarray with the shape of (X, 1024), where X is the length of data.
    """
    data_vectors = []
    for i, row in tqdm(data.iterrows()):
        path1 = row["path1"]
        path2 = row["path2"]
        embedding1 = get_embeddings(embedder, path1, increase_shape=increase_shape).cpu().detach().numpy().flatten()
        embedding2 = get_embeddings(embedder, path2, increase_shape=increase_shape).cpu().detach().numpy().flatten()
        current = np.subtract(embedding1, embedding2)
        data_vectors.append(current)
    data_vectors = np.vstack(data_vectors)
    np.save(saving_path_and_name, data_vectors)
    return data_vectors


class Embedder():
    """
    The Embedder class.
    """

    def __init__(self, device, model_name='', train=False, faceX_zoo=True):
        """
        Constructor.
        :param device: Required. Type: str. The device the network will use.
                       Options:"cpu" / "cuda:0" / other coda name. The device used by pytorch.
        """
        if model_name == 'iresnet100':
            if train:
                embedder = AFBackbone.iresnet100(pretrained=True).to(device).train()

            else:
                embedder = AFBackbone.iresnet100(pretrained=True).to(device).eval()
        ######################################
        # FaceX-Zoo
        ######################################
        if faceX_zoo:
            print(f"Model Name: {model_name}")
            model_path = f"/sise/home/royek/Toshiba_roye/Pretrained_Backbones/{model_name}.pt"
            conf_path = r"/sise/home/royek/Toshiba_roye/ModelX_Zoo/test_protocol/backbone_conf.yaml"
            ModelFactory = BackboneFactory(model_name, conf_path)
            model_loader = ModelLoader(ModelFactory)
            embedder = model_loader.load_model(model_path=model_path, device=device)


        self.embedder = embedder
        self.device = device

    def predict(self, input):
        """
        The method returns the input's embedding vectors produced by the embedder.
        :param input: Required. Type: ndarray / torch.tensor. An array like object with 4 dimensions:
                        (batch size, channels, image hight, image width).
                        For example: (24, 3, 112, 112).
        :return: torch.tensor. The embedding vectors for the inputs.
                    Shape: (batch size, embedding vector size).
                    For example: (24, 512)
        """
        if torch.is_tensor(input):
            return self.embedder(input.float()).to(self.device)
        else:
            inp = torch.tensor(input).float().to(self.device)
            pred = self.embedder(inp).detach()
            return pred