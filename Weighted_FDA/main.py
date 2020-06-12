import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from my_OWFDA import My_OWFDA
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import os
import glob
import math
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from sklearn.model_selection import train_test_split




def main():
    # ---- settings:
    kernel = "rbf"  #kernel over data (X) --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’
    method = "kernel_OWFDA"  #--> OWFDA, kernel_OWFDA, other_methods, other_methods_kernel, CWFDA, kernel_CWFDA
    save_variable_images = True
    report_progress_of_gradient_descent = False
    dataset = "ETH"  #--> ORL_faces, ETH, MNIST
    read_dataset_again = False
    max_iterations_in_gradient_descent = 1
    max_epochs = None
    step_checkpoint = 1
    K_in_KNN_for_OWFDA_and_kernelOWFDA = None   #--> integer \in [1, n_classes-1] --> excluding self in classes, if None --> take all neighbor classes (=n_classes-1)
    K_in_KNN_for_KNN_weighting = None  #--> integer \in [1, n_classes-1] --> excluding self in classes, if None --> take all neighbor classes (=n_classes-1)
    type_gradient = 2
    lambda_regularization = 0.01

    # ---- read dataset:
    if dataset == "ORL_faces":
        scale = 0.4
        # scale = 0.6
        path_dataset = "C:\\Users\\bghojogh\\Desktop\\Datasets\\Face_datasets\\Att_(ORL)_faces\\ORL\\"
        image_height, image_width = 112, 92
        n_classes = 20
        if read_dataset_again:
            X, y = read_ORL_dataset(path_dataset=path_dataset, image_height=image_height, image_width=image_width,
                                    n_classes=n_classes, do_resize=True, scale=scale)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="X", arr=X)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="y", arr=y)
            # ---------- split to train and test sets:
            X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.33, random_state=42)
            X_train, X_test = X_train.T, X_test.T
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="X_train", arr=X_train)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="X_test", arr=X_test)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="y_train", arr=y_train)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="y_test", arr=y_test)
        else:
            X = np.load("./datasets/ORL/X.npy")
            y = np.load("./datasets/ORL/y.npy")
            X_train = np.load("./datasets/ORL/X_train.npy")
            y_train = np.load("./datasets/ORL/y_train.npy")
            X_test = np.load("./datasets/ORL/X_test.npy")
            y_test = np.load("./datasets/ORL/y_test.npy")
        image_height, image_width = int(image_height * scale), int(image_width * scale)
        # print(image_height)
        # print(image_width)
        # input("hi")
    elif dataset == "ETH":
        scale = 0.15
        # scale = 0.3
        # scale = 0.6
        samples_per_class = 25  #--> <= 410
        path_dataset = "C:\\Users\\bghojogh\\Desktop\\Datasets\\ETH_80\\ETH_togetherFolders\\"
        image_height, image_width = 256, 256
        n_classes = 8
        if read_dataset_again:
            X, y = read_ETH_dataset(path_dataset=path_dataset, image_height=image_height, image_width=image_width,
                                    n_classes=n_classes, do_resize=True, scale=scale, samples_per_class=samples_per_class)
            save_numpy_array(path_to_save="./datasets/ETH/", arr_name="X", arr=X)
            save_numpy_array(path_to_save="./datasets/ETH/", arr_name="y", arr=y)
            # ---------- split to train and test sets:
            X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.33, random_state=42)
            X_train, X_test = X_train.T, X_test.T
            save_numpy_array(path_to_save="./datasets/ETH/", arr_name="X_train", arr=X_train)
            save_numpy_array(path_to_save="./datasets/ETH/", arr_name="X_test", arr=X_test)
            save_numpy_array(path_to_save="./datasets/ETH/", arr_name="y_train", arr=y_train)
            save_numpy_array(path_to_save="./datasets/ETH/", arr_name="y_test", arr=y_test)
        else:
            X = np.load("./datasets/ETH/X.npy")
            y = np.load("./datasets/ETH/y.npy")
            X_train = np.load("./datasets/ETH/X_train.npy")
            y_train = np.load("./datasets/ETH/y_train.npy")
            X_test = np.load("./datasets/ETH/X_test.npy")
            y_test = np.load("./datasets/ETH/y_test.npy")
        image_height, image_width = int(image_height * scale), int(image_width * scale)
        # print(image_height)
        # print(image_width)
        # input("hi")
    elif dataset == 'MNIST':
        image_height, image_width = 28, 28
        subset_of_MNIST = True
        pick_subset_of_MNIST_again = True
        MNIST_subset_cardinality_training = 5000
        MNIST_subset_cardinality_testing = 1000
        path_dataset_save = "C:\\Users\\bghojogh\\Desktop\\Datasets\\ETH_80\\ETH_togetherFolders\\"
        path_dataset_save2 = "./datasets/MNIST/"
        if read_dataset_again:
            file = open(path_dataset_save + 'X_train.pckl', 'rb')
            X_train = pickle.load(file);
            file.close()
            file = open(path_dataset_save + 'y_train.pckl', 'rb')
            Y_train = pickle.load(file);
            file.close()
            Y_train = np.asarray(Y_train)
            file = open(path_dataset_save + 'X_test.pckl', 'rb')
            X_test = pickle.load(file);
            file.close()
            file = open(path_dataset_save + 'y_test.pckl', 'rb')
            Y_test = pickle.load(file);
            file.close()
            Y_test = np.asarray(Y_test)
            if subset_of_MNIST:
                    X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                    X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                    y_train_picked = Y_train[0:MNIST_subset_cardinality_training]
                    y_test_picked = Y_test[0:MNIST_subset_cardinality_testing]
                    save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save2)
                    save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save2)
                    save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save2)
                    save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save2)
        else:
            file = open(path_dataset_save2 + 'X_train_picked.pckl', 'rb')
            X_train_picked = pickle.load(file)
            file.close()
            file = open(path_dataset_save2 + 'X_test_picked.pckl', 'rb')
            X_test_picked = pickle.load(file)
            file.close()
            file = open(path_dataset_save2 + 'y_train_picked.pckl', 'rb')
            y_train_picked = pickle.load(file)
            file.close()
            file = open(path_dataset_save2 + 'y_test_picked.pckl', 'rb')
            y_test_picked = pickle.load(file)
            file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            y_train = y_train_picked
            y_test = y_test_picked
        X_train = X_train.T
        X_test = X_test.T
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train.T)
        X_train = (scaler.transform(X_train.T)).T
        X_test = (scaler.transform(X_test.T)).T
        # print(image_height)
        # print(image_width)
        # input("hi")

    my_owFDA = My_OWFDA(X=X_train, y=y_train, X_test=X_test, y_test=y_test, dataset_name=dataset,
                        image_height=image_height, image_width=image_width,
                        max_iterations_in_gradient_descent=max_iterations_in_gradient_descent,
                        K_ell0=K_in_KNN_for_OWFDA_and_kernelOWFDA, lambda_regularization=lambda_regularization, n_components=None,
                        type_gradient=type_gradient,
                        kernel=kernel, kernel_OWFDA=False, save_variable_images=save_variable_images,
                        report_progress_of_gradient_descent=report_progress_of_gradient_descent)
    if method == "OWFDA":
        my_owFDA.train_OWFDA(max_epochs=max_epochs, step_checkpoint=step_checkpoint)
    elif method == "kernel_OWFDA":
        my_owFDA.train_kernel_OWFDA(max_epochs=max_epochs, step_checkpoint=step_checkpoint)
    elif method == "other_methods":
        my_owFDA.train_FDA(kernel_method=False)
        print("FDA done...")
        my_owFDA.train_FDA_weighted_POW(power_=3, kernel_method=False)
        print("FDA-POW done...")
        my_owFDA.train_FDA_weighted_KNN(k=K_in_KNN_for_KNN_weighting, kernel_method=False)
        print("FDA-KNN done...")
        my_owFDA.train_FDA_weighted_CDM(LDA_or_QDA="QDA", kernel_method=False)
        print("FDA-CDM done...")
        my_owFDA.train_FDA_weighted_APAC(kernel_method=False)
        print("FDA-APAC done...")
    elif method == "other_methods_kernel":
        my_owFDA.train_FDA(kernel_method=True)
        print("kernel FDA done...")
        my_owFDA.train_FDA_weighted_POW(power_=3, kernel_method=True)
        print("kernel FDA-POW done...")
        my_owFDA.train_FDA_weighted_KNN(k=K_in_KNN_for_KNN_weighting, kernel_method=True)
        print("kernel FDA-KNN done...")
        my_owFDA.train_FDA_weighted_CDM(LDA_or_QDA="QDA", kernel_method=True)
        print("kernel FDA-CDM done...")
        my_owFDA.train_FDA_weighted_APAC(kernel_method=True)
        print("kernel FDA-APAC done...")
    elif method == "CWFDA":
        my_owFDA.train_FDA_weighted_cosine(kernel_version=None, kernel_method=False)
    elif method == "kernel_CWFDA":
        my_owFDA.train_FDA_weighted_cosine(kernel_version=1, kernel_method=True)
        my_owFDA.train_FDA_weighted_cosine(kernel_version=2, kernel_method=True)

def read_ORL_dataset(path_dataset, image_height, image_width, n_classes=None, do_resize=False, scale=1):
    if n_classes is None:
        n_samples = 400
    else:
        n_samples = n_classes * 10
    if resize:
        data = np.zeros((int(image_height * scale) * int(image_width * scale), n_samples))
    else:
        data = np.zeros((image_height*image_width, n_samples))
    labels = np.zeros((1, n_samples))
    for image_index in range(n_samples):
        img = load_image(address_image=path_dataset+str(image_index+1)+".jpg",
                        image_height=image_height, image_width=image_width, do_resize=do_resize, scale=scale)
        data[:, image_index] = img.ravel()
        labels[:, image_index] = math.floor(image_index / 10) + 1
    # ---- cast dataset from string to float:
    data = data.astype(np.float)
    # ---- change range of images from [0,255] to [0,1]:
    data = data / 255
    data_notNormalized = data
    # ---- normalize (standardation):
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
    data = (scaler.transform(data.T)).T
    # ---- show one of the images:
    print("dimensionality: " + str(data.shape[0]))
    if False:
        if resize:
            an_image = data[:, 0].reshape((int(image_height * scale), int(image_width * scale)))
        else:
            an_image = data[:, 0].reshape((image_height, image_width))
        plt.imshow(an_image, cmap='gray')
        plt.colorbar()
        plt.show()
    return data, labels.ravel()

def read_ETH_dataset(path_dataset, image_height, image_width, n_classes=None, do_resize=False, scale=1, samples_per_class=None):
    if samples_per_class is None:
        samples_per_class = 410
    else:
        samples_per_class = min(samples_per_class, 410)
    if n_classes is None:
        n_samples = 8 * samples_per_class
        n_classes = 8
    else:
        n_samples = n_classes * samples_per_class
    if resize:
        data = np.zeros((int(image_height * scale) * int(image_width * scale), n_samples))
    else:
        data = np.zeros((image_height*image_width, n_samples))
    labels = np.zeros((1, n_samples))
    image_index = -1
    for class_index in range(n_classes):
        image_index_per_class = -1
        path_of_class = path_dataset + str(class_index) + "/*.png"
        for address_image in glob.glob(path_of_class):
            image_index += 1
            image_index_per_class += 1
            if image_index_per_class > samples_per_class:
                break
            img = load_image(address_image=address_image,
                             image_height=image_height, image_width=image_width, do_resize=do_resize, scale=scale)
            data[:, image_index] = img.ravel()
            labels[:, image_index] = class_index
    # ---- cast dataset from string to float:
    data = data.astype(np.float)
    # ---- change range of images from [0,255] to [0,1]:
    data = data / 255
    data_notNormalized = data
    # ---- normalize (standardation):
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
    data = (scaler.transform(data.T)).T
    # ---- show one of the images:
    print("dimensionality: " + str(data.shape[0]))
    if False:
        if resize:
            an_image = data[:, 0].reshape((int(image_height * scale), int(image_width * scale)))
        else:
            an_image = data[:, 0].reshape((image_height, image_width))
        plt.imshow(an_image, cmap='gray')
        plt.colorbar()
        plt.show()
    # print(data.shape)
    # print(labels.shape)
    # input("hi")
    return data, labels.ravel()


def load_image(address_image, image_height, image_width, do_resize=False, scale=1):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    if do_resize:
        size = int(image_height * scale), int(image_width * scale)
        # img.thumbnail(size, Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = resize(img_arr, (int(img_arr.shape[0]*scale), int(img_arr.shape[1]*scale)), order=5, preserve_range=True, mode="constant")
    return img_arr

def save_numpy_array(path_to_save, arr_name, arr):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.save(path_to_save+arr_name+".npy", arr)

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

if __name__ == '__main__':
    main()

