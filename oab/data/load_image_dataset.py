import os
import shutil
import tarfile
import wget
import zipfile
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image

from oab.data.classification_dataset import ClassificationDataset
from oab.data.unsupervised import UnsupervisedAnomalyDataset
from oab.data.semisupervised import SemisupervisedAnomalyDataset
from oab.data.utils_image import (tensorflow_datasets, mvtec_ad_suffixes,
    mvtec_ad_datasets, url_dict, mvtec_ad_bw_datasets)


crack_credits = """
2018 – Özgenel, Ç.F., Gönenç Sorguç, A. “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”, ISARC 2018, Berlin.

Lei Zhang , Fan Yang , Yimin Daniel Zhang, and Y. J. Z., Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road Crack Detection Using Deep Convolutional Neural Network. In 2016 IEEE International Conference on Image Processing (ICIP). http://doi.org/10.1109/ICIP.2016.7533052

See https://data.mendeley.com/datasets/5y9wdsg2zt/2 for more information. The dataset is published under CC BY 4.0.
"""



def _load_image_dataset(dataset_name: str, anomaly_dataset: bool = True,
        preprocess_classification_dataset: bool = False,
        semisupervised: bool = False,
        dataset_folder: str = "datasets"):
    """
    Loads image datasets.

    :param dataset_name: Name of the dataset to be loaded
    :param anomaly_dataset: If this is set, an anomaly dataset is returned. Else,
        a classification dataset is returned. Defaults to True
    :param preprocess_classification_dataset: This is only used when a
        classification dataset is being returned. If this is set, preprocessing
        will be applied, i.e., all values will be scaled with 1/255. If it is
        set to `False`, preprocessing is not applied. Defaults to False
    :param semisupervised: If this is set to `True`, a semisupervised anomaly
        dataset is returned. If set to `False`, an unsupervised anomaly dataset
        is returned. Defaults to False
    :param dataset_name: Name of the folder in which a dataset is stored locally
        if necessary, defaults to "datasets"

    :return: The specified image dataset as specified, i.e., as classficication
        dataset, unsupervised anomaly dataset or semisupervised anomaly dataset
    """
    if dataset_name.startswith('mvtec_ad_'):
        return _load_mvtec_ad_dataset(dataset_name, anomaly_dataset,
            preprocess_classification_dataset, semisupervised, dataset_folder)

    if dataset_name == "crack":
        x, y = _load_crack_dataset(dataset_folder)

    elif dataset_name in tensorflow_datasets:
        x, y = _load_tensorflow_image_dataset(dataset_name)

    else: # dataset_name not in tensorflow_datasets
        raise NotImplementedError()

    # with x and y, make classification dataset
    cd = ClassificationDataset(x, y, name=dataset_name)
    if (not anomaly_dataset and not preprocess_classification_dataset):
        return cd
    # preprocessing
    cd.scale(scaling_factor=1/255)
    if (not anomaly_dataset and preprocess_classification_dataset):
        return cd
    # turn into anomaly dataset
    if semisupervised: # semisupervised anomaly detection dataset
        ad = SemisupervisedAnomalyDataset(cd, normal_labels=[0])
    else: # unsupervised anomaly detection dataset
        ad = UnsupervisedAnomalyDataset(cd, normal_labels=[0])
    return ad


def _load_mvtec_ad_dataset(dataset_name: str, anomaly_dataset: bool = True,
        preprocess_classification_dataset: bool = False,
        semisupervised: bool = False,
        dataset_folder: str = "datasets"):
    """
    Helper to load MVTec AD datasets.
    Loads the specified dataset into an unsupervised or semisupervised anomaly or classification dataset, with
    preprocessing optionally applied (if classification dataset to be loaded),
    as specified by the parameters.

    :param dataset_name: Name of the MVTec AD dataset to be loaded
    :param anomaly_dataset: If this is set, an anomaly dataset is returned. Else,
        a classification dataset is returned. Defaults to True
    :param preprocess_classification_dataset: This is only used when a
        classification dataset is being returned. If this is set, preprocessing
        will be applied, i.e., all values will be scaled with 1/255. If it is
        set to `False`, preprocessing is not applied. Defaults to False
    :param semisupervised: If this is set to `True`, a semisupervised anomaly
        dataset is returned. If set to `False`, an unsupervised anomaly dataset
        is returned. Defaults to False
    :param dataset_name: Name of the folder in which a dataset is stored locally
        if necessary, defaults to "datasets"

    :return: The specified image dataset as specified, i.e., as classficication
        dataset, unsupervised anomaly dataset or semisupervised anomaly dataset
    """
    # get dataset name (without mvtec_ad_)
    mvtec_ad_dataset_name = dataset_name[9:]
    if not mvtec_ad_dataset_name in mvtec_ad_suffixes: # check if dataset exists
        raise ValueError(f"Cannot load {dataset_name}. All available datasets from MVTec AD are {mvtec_ad_datasets}.")

    # if dataset is not loaded, download dataset, correct file structure and image size
    if not os.path.isfile(Path(dataset_folder) / mvtec_ad_dataset_name / "applied_modification.txt"):
        _download_mvtec_ad(mvtec_ad_dataset_name, dataset_folder)

    # load classification dataset and return if specified
    dataset_path = Path(dataset_folder) / mvtec_ad_dataset_name
    normals_folder_path = dataset_path / "normal"
    anomalies_folder_path = dataset_path / "anomaly"
    n_normals, n_anomalies = len(os.listdir(normals_folder_path)), \
        len(os.listdir(anomalies_folder_path))
    if dataset_name in mvtec_ad_bw_datasets:
        dimensionality = 256 * 256
    else:
        dimensionality = 256 * 256 * 3
    x = np.zeros((n_normals+n_anomalies, dimensionality))
    y = np.zeros(n_normals+n_anomalies)
    x_normal_test_mask = np.zeros(n_normals) # this array holds a value for each normal data point
    # indicating whether or not it belongs to the original data's test set (1) or training (0)
    # load normals
    for idx, file in enumerate(os.listdir(normals_folder_path)):
        image = Image.open(normals_folder_path / file)
        data = np.asarray(image).reshape(-1)
        x[idx], y[idx] = data, 0
        if file.startswith('01'): # original test samples start with '01':
            x_normal_test_mask[idx] = 1
    # load anomalies
    for idx, file in enumerate(os.listdir(anomalies_folder_path)):
        image = Image.open(anomalies_folder_path / file)
        data = np.asarray(image).reshape(-1)
        x[n_normals+idx], y[n_normals+idx] = data, 1
    cd = ClassificationDataset(x, y, name=dataset_name)
    cd.x_normal_test_mask = x_normal_test_mask
    if (not anomaly_dataset and not preprocess_classification_dataset):
        return cd

    # preprocessing and return if specified
    cd.scale(scaling_factor=1/255)
    if (not anomaly_dataset and preprocess_classification_dataset):
        return cd

    # make anomaly dataset
    if not semisupervised:
        ad = UnsupervisedAnomalyDataset(cd, normal_labels=[0])
    else:
        ad = SemisupervisedAnomalyDataset(cd, normal_labels=[0])
    return ad



def _download_mvtec_ad(mvtec_ad_dataset_name: str, dataset_folder: str) -> None:
    """
    Downloads the specified MVTec AD dataset into the specified folder.
    Additionally, in the folder structure dataset_folder/mvtec_ad_dataset_name,
    folders "normal" and "anomaly" are created and the normals and anomalies
    are saved in these respectively. All images are rescaled to 256x256 pixels.

    :param mvtec_ad_dataset_name: Name of the MVTec AD dataset, e.g., 'wood'
    :param dataset_folder: Folder in which dataset is to be stored. Note that
        in this folder, a new folder for the dataset will be created.
    """

    # check if dataset exists
    if not mvtec_ad_dataset_name in mvtec_ad_suffixes:
        raise ValueError(f"Cannot load {dataset_name}. All available datasets from MVTec AD are {mvtec_ad_datasets}.")

    # set dataset path
    dataset_folder_path = Path(dataset_folder)
    dataset_path = dataset_folder_path / mvtec_ad_dataset_name

    # get url
    url = url_dict[mvtec_ad_dataset_name]

    # delete everything in folder and folder itself (if it has content)
    try:
        shutil.rmtree(dataset_path)
    except:
        pass

    # setup folder structure
    dataset_folder_path.mkdir(parents=True, exist_ok=True)

    # download data
    wget.download(url, str(dataset_folder_path))

    # unpack data
    fname = dataset_folder_path / f"{mvtec_ad_dataset_name}.tar.xz"
    tar = tarfile.open(fname, "r:xz")
    tar.extractall(dataset_folder_path)

    # restructure folders
    os.chmod(dataset_path, 0o755)
    normals_folder_path = dataset_path / "normal"
    anomalies_folder_path = dataset_path / "anomaly"
    normals_folder_path.mkdir(parents=True, exist_ok=True)
    anomalies_folder_path.mkdir(parents=True, exist_ok=True)
    os.chmod(normals_folder_path, 0o755)
    os.chmod(anomalies_folder_path, 0o755)

    ## normals from training data
    source_normals_training = dataset_path / "train" / "good"
    os.chmod(source_normals_training, 0o755)
    for file in os.listdir(source_normals_training):
	    shutil.move(source_normals_training / file, normals_folder_path / f"00{file}")

    ## normals from test data
    source_normals_test = dataset_path / "test" / "good"
    os.chmod(source_normals_test, 0o755)
    for file in os.listdir(source_normals_test):
	    shutil.move(source_normals_test / file, normals_folder_path / f"01{file}")

    ## anomalies
    for idx, foldername in enumerate(os.listdir(dataset_path / "test")):
        if foldername != "good":
            full_foldername = dataset_path / "test" / foldername
            os.chmod(full_foldername, 0o755)
            for file in os.listdir(full_foldername):
                shutil.move(full_foldername / file, anomalies_folder_path / f"{idx}{file}")

    # resize images
    for folder_path in [normals_folder_path, anomalies_folder_path]:
        for file in os.listdir(folder_path):
            filepath = folder_path / file
            os.chmod(filepath, 0o755)
            # print(filepath)
            image = Image.open(filepath)
            new_image = image.resize((256, 256))
            new_image.save(filepath)
            del image
            del new_image

    # add description (i.e., that data was resized)
    description_path = dataset_path / "applied_modification.txt"
    with open(description_path, "w+") as f:
        f.write(f"Reordered images into normals and anomalies folder\n")
        f.write(f"Rescaled images to 256x256\n")

    return


def _load_crack_dataset(dataset_folder: str = 'datasets'):
    """
    Returns x, y for crack dataset. If dataset is not downloaded yet,
    the dataset will be downloaded.

    :dataset_folder: Folder in which a subfolder 'crack' is created. Dataset
        is in subfolder 'crack'.
    """
    print(f"Credits: {crack_credits}")
    dataset_folder = Path(dataset_folder)

    dataset_path = dataset_folder / 'crack'
    normals_folder_path = dataset_path / "Negative"
    anomalies_folder_path = dataset_path / "Positive"

    # if dataset is already loaded, do not reload. -> Check it.
    if (not os.path.isfile(dataset_folder / "crack" / "info.txt")):
        # download dataset
        zipname = "crack.zip" # name of the zip file
        link = "https://www.dropbox.com/s/pd0r3wphnp2jc8d/crack.zip?dl=1"
        wget.download(link, str(dataset_folder))

        # delete all previously existing files in the structure this will create
        try:
            shutil.rmtree(dataset_path)
        except:
            pass

        # extract downloaded zip, delete zip file
        with zipfile.ZipFile(dataset_folder / zipname, 'r') as zipref:
            zipref.extractall(dataset_folder)
        os.remove(dataset_folder / zipname)

        # resize images
        for folder_path in [normals_folder_path, anomalies_folder_path]:
            for file in os.listdir(folder_path):
                filepath = folder_path / file
                os.chmod(filepath, 0o755)
                # print(filepath)
                image = Image.open(filepath)
                new_image = image.resize((128, 128))
                new_image.save(filepath)
                del image
                del new_image

    # load images
    n_normals, n_anomalies = len(os.listdir(normals_folder_path)), \
        len(os.listdir(anomalies_folder_path))
    x = np.zeros((n_normals+n_anomalies, 128*128*3))
    y = np.zeros(n_normals+n_anomalies)
    # load normals
    for idx, file in enumerate(os.listdir(normals_folder_path)):
        with Image.open(normals_folder_path / file) as image:
            data = np.asarray(image).reshape(-1)
            x[idx], y[idx] = data, 0
            del data
    # load anomalies
    for idx, file in enumerate(os.listdir(anomalies_folder_path)):
        with Image.open(anomalies_folder_path / file) as image:
            data = np.asarray(image).reshape(-1)
            x[n_normals+idx], y[n_normals+idx] = data, 1
            del data

    return x, y



def _load_tensorflow_image_dataset(dataset_name: str):
    """
    Loads datasets from tensorflow's dataset library.

    :param dataset_name: Name of the dataset to be loaded
    :return: Tuple (x, y) of the specified dataset
    """
    dataset = getattr(tf.keras.datasets, dataset_name)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    x, y = _combine_values_and_labels((x_train, x_test), (y_train, y_test))
    if dataset_name == 'cifar100':
        y = y // 5 # because we want superclasses
    return x, y



def _combine_values_and_labels(xs, ys):
    """
    Helper to combine xs and ys, e.g., combine xs and ys of train and test
    set.
    """
    x = np.vstack(xs)
    y = np.hstack(ys)
    return x, y
