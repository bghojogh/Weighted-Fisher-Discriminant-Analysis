import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from my_generalized_eigen_problem import My_generalized_eigen_problem
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from scipy.special import erf   #--> error function
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn.metrics import accuracy_score


class My_OWFDA:

    def __init__(self, X, y, X_test, y_test, dataset_name, image_height, image_width,
                max_iterations_in_gradient_descent, K_ell0, lambda_regularization=0.2,
                n_components=None, type_gradient=2,
                kernel="linear", kernel_OWFDA=False,
                save_variable_images=True, report_progress_of_gradient_descent=False):
        # X: rows are features and columns are samples
        # pixel intensity range: [0,1]
        self.dataset_name = dataset_name
        self.image_height = image_height
        self.image_width = image_width
        self.X = X
        self.y = y
        self.X_classes, _ = self.separate_samples_of_classes_2(X=self.X, y=self.y)
        print("Samples of classes separated....")
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_classes = len(self.X_classes)
        self.N = np.zeros((self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            self.N[class_index, class_index] = self.X_classes[class_index].shape[1]
        self.save_variable_images = save_variable_images
        self.report_progress_of_gradient_descent = report_progress_of_gradient_descent
        self.type_gradient = type_gradient
        self.max_iterations_in_gradient_descent = max_iterations_in_gradient_descent
        self.S_W = None
        self.kernel_S_W = None
        self.M = None
        self.class_means = None
        self.lambda_regularization = lambda_regularization
        self.X_test = X_test
        self.y_test = y_test
        self.kernel = kernel
        if K_ell0 is None:
            self.K_ell0 = self.n_classes - 1  #--> to exclude self (take all classes)
        else:
            self.K_ell0 = K_ell0
        if n_components != None:
            self.n_components = n_components  # --> p
        else:
            if not kernel_OWFDA:
                self.n_components = min([self.n_dimensions, self.n_samples - 1, self.n_classes - 1])
            else:
                self.n_components = min([self.n_samples, self.n_classes - 1])

    def train_OWFDA(self, max_epochs=None, step_checkpoint=10):
        V = np.random.rand(self.n_dimensions, self.n_components)  # --> rand in [0,1)
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            # temp = np.random.rand(self.n_classes)
            temp = np.ones((self.n_classes,))
            A[:, :, class_index] = np.diagflat(temp)
        self.save_variable(variable=V, name_of_variable="V_epochs_initial", path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/V/')
        self.save_variable(variable=A, name_of_variable="A_epochs_initial", path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/A/')
        if self.save_variable_images:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/"+ self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/V_figs/initial/")
            self.save_image_of_cxc_variable(A, path_to_save="./saved_files/"+ self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/A_figs/initial/")
        epoch_index = -1
        objective_value_epochs = np.zeros((step_checkpoint, 1))
        error_V_epochs = np.zeros((step_checkpoint, 1))
        error_A_epochs = np.zeros((step_checkpoint, 1))
        while True:
            epoch_index = epoch_index + 1
            print("----- epoch #" + str(epoch_index))
            V_previous_epoch = V.copy()
            weights_all_together_previous_epoch = self.put_all_weights_in_a_matrix(A=A)
            V = self.optimization_for_V(A)
            print("optimization of V done....")
            for class_index in range(self.n_classes):
                A[:, :, class_index] = self.gradient_descent_A(class_index=class_index, V=V, A=A,
                                                                A_class=A[:, :, class_index],
                                                                max_iterations=self.max_iterations_in_gradient_descent)
                print("gradient decent of A, class " + str(class_index) + " done....")
            # errors of epoch:
            index_to_save = int(epoch_index % step_checkpoint)
            objective_value = self.calculate_objective_value_overall(V=V, A=A)
            objective_value_epochs[index_to_save] = objective_value
            print("----- objective value of epoch #" + str(epoch_index) + ": " + str(objective_value))
            error_V = LA.norm(V - V_previous_epoch, ord="fro")
            error_V_epochs[index_to_save] = error_V
            weights_all_together = self.put_all_weights_in_a_matrix(A=A)
            error_A = LA.norm(weights_all_together - weights_all_together_previous_epoch, ord="fro")
            error_A_epochs[index_to_save] = error_A
            print("----- Change of V in epoch #" + str(epoch_index) + ": " + str(error_V))
            print("----- Change of A in epoch #" + str(epoch_index) + ": " + str(error_A))
            # save the information at checkpoints:
            if (epoch_index+1) % step_checkpoint == 0:
                print("Saving the checkpoint in epoch #" + str(epoch_index))
                checkpoint_index = int(np.floor(epoch_index / step_checkpoint))
                self.save_variable(variable=objective_value_epochs, name_of_variable="objective_value_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/objective_value/')
                self.save_np_array_to_txt(variable=objective_value_epochs, name_of_variable="objective_value_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/objective_value/')
                self.save_variable(variable=error_V_epochs, name_of_variable="error_V_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/V_error/')
                self.save_np_array_to_txt(variable=error_V_epochs, name_of_variable="error_V_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/V_error/')
                self.save_variable(variable=error_A_epochs, name_of_variable="error_A_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/A_error/')
                self.save_np_array_to_txt(variable=error_A_epochs, name_of_variable="error_A_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/A_error/')
                self.save_variable(variable=V, name_of_variable="V_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/V/')
                self.save_variable(variable=A, name_of_variable="A_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/OWFDA/k=' + str(self.K_ell0) + '/A/')
                if self.save_variable_images:
                    self.save_image_of_dxd_variable(V, path_to_save="./saved_files/"+ self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/V_figs/epoch_"+str(epoch_index)+"/")
                    V2 = self.optimization_for_V_notDirtySolution(A)
                    self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/"+ self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/V_figs_2/epoch_"+str(epoch_index)+"/")
                    self.save_image_of_cxc_variable(A, path_to_save="./saved_files/"+ self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/A_figs/epoch_"+str(epoch_index)+"/")
                    self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/"+ self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/projections/epoch_"+str(epoch_index)+"/")
                self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name +"/OWFDA/k=" + str(self.K_ell0) + "/KNN/epoch_"+str(epoch_index)+"/", kernel_method=False)
            # termination check:
            if max_epochs != None:
                if epoch_index >= max_epochs:
                    break

    def train_kernel_OWFDA(self, max_epochs=None, step_checkpoint=10):
        V = np.random.rand(self.n_samples, self.n_components)  # --> rand in [0,1)
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            # temp = np.random.rand(self.n_classes)
            temp = np.ones((self.n_classes,))
            A[:, :, class_index] = np.diagflat(temp)
        self.save_variable(variable=V, name_of_variable="V_epochs_initial", path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/V/')
        self.save_variable(variable=A, name_of_variable="A_epochs_initial", path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/A/')
        if self.save_variable_images:
            self.save_image_of_cxc_variable(A, path_to_save="./saved_files/"+ self.dataset_name +"/kernel_OWFDA/k=" + str(self.K_ell0) + "/A_figs/initial/")
        epoch_index = -1
        objective_value_epochs = np.zeros((step_checkpoint, 1))
        error_V_epochs = np.zeros((step_checkpoint, 1))
        error_A_epochs = np.zeros((step_checkpoint, 1))
        while True:
            epoch_index = epoch_index + 1
            print("----- epoch #" + str(epoch_index))
            V_previous_epoch = V.copy()
            weights_all_together_previous_epoch = self.put_all_weights_in_a_matrix(A=A)
            V = self.kernel_optimization_for_V(A)
            for class_index in range(self.n_classes):
                A[:, :, class_index] = self.gradient_descent_A(class_index=class_index, V=V, A=A,
                                                                A_class=A[:, :, class_index],
                                                                max_iterations=self.max_iterations_in_gradient_descent,
                                                               kernel_method=True)
                print("gradient decent of A, class " + str(class_index) + " done....")
            # errors of epoch:
            index_to_save = int(epoch_index % step_checkpoint)
            objective_value = self.kernel_calculate_objective_value_overall(V=V, A=A)
            objective_value_epochs[index_to_save] = objective_value
            print("----- objective value of epoch #" + str(epoch_index) + ": " + str(objective_value))
            error_V = LA.norm(V - V_previous_epoch, ord="fro")
            error_V_epochs[index_to_save] = error_V
            weights_all_together = self.put_all_weights_in_a_matrix(A=A)
            error_A = LA.norm(weights_all_together - weights_all_together_previous_epoch, ord="fro")
            error_A_epochs[index_to_save] = error_A
            print("----- Change of V in epoch #" + str(epoch_index) + ": " + str(error_V))
            print("----- Change of A in epoch #" + str(epoch_index) + ": " + str(error_A))
            # save the information at checkpoints:
            if (epoch_index+1) % step_checkpoint == 0:
                print("Saving the checkpoint in epoch #" + str(epoch_index))
                checkpoint_index = int(np.floor(epoch_index / step_checkpoint))
                self.save_variable(variable=objective_value_epochs, name_of_variable="objective_value_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/objective_value/')
                self.save_np_array_to_txt(variable=objective_value_epochs, name_of_variable="objective_value_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/objective_value/')
                self.save_variable(variable=error_V_epochs, name_of_variable="error_V_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/V_error/')
                self.save_np_array_to_txt(variable=error_V_epochs, name_of_variable="error_V_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/V_error/')
                self.save_variable(variable=error_A_epochs, name_of_variable="error_A_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/A_error/')
                self.save_np_array_to_txt(variable=error_A_epochs, name_of_variable="error_A_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/A_error/')
                self.save_variable(variable=V, name_of_variable="V_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/V/')
                self.save_variable(variable=A, name_of_variable="A_epochs_"+str(checkpoint_index), path_to_save='./saved_files/'+ self.dataset_name +'/kernel_OWFDA/k=' + str(self.K_ell0) + '/A/')
                if self.save_variable_images:
                    self.save_image_of_cxc_variable(A, path_to_save="./saved_files/"+ self.dataset_name +"/kernel_OWFDA/k=" + str(self.K_ell0) + "/A_figs/epoch_"+str(epoch_index)+"/")
                    self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/"+ self.dataset_name +"/kernel_OWFDA/k=" + str(self.K_ell0) + "/projections/epoch_"+str(epoch_index)+"/")
                self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name +"/kernel_OWFDA/k=" + str(self.K_ell0) + "/KNN/epoch_"+str(epoch_index)+"/", kernel_method=True)
            # termination check:
            if max_epochs != None:
                if epoch_index >= max_epochs:
                    break

    def train_FDA(self, kernel_method=False):
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            temp = np.ones((self.n_classes,))
            A[:, :, class_index] = np.diagflat(temp)
        if not kernel_method:
            V = self.optimization_for_V(A)
            V2 = self.optimization_for_V_notDirtySolution(A)
            name = "FDA"
        else:
            V = self.kernel_optimization_for_V(A)
            name = "kernel_FDA"
        self.save_variable(variable=V, name_of_variable="V", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/V/')
        self.save_variable(variable=A, name_of_variable="A", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/A/')
        if not kernel_method:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs/")
            self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs_2/")
        self.save_image_of_cxc_variable(A, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/A_figs/")
        if not kernel_method:
            self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=False)
        else:
            self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=True)


    def train_FDA_weighted_POW(self, power_=3, kernel_method=False):
        self.calculate_M()
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            for class_index_2 in range(self.n_classes):
                if class_index != class_index_2:
                    A[class_index_2, class_index_2, class_index] = 1 / (LA.norm(self.M[:, class_index_2, class_index]) ** power_)
                else:  #--> the class with itself
                    A[class_index_2, class_index_2, class_index] = 0
        if not kernel_method:
            V = self.optimization_for_V(A)
            V2 = self.optimization_for_V_notDirtySolution(A)
            name = "FDA_weighted_POW"
        else:
            V = self.kernel_optimization_for_V(A)
            name = "kernel_FDA_weighted_POW"
        self.save_variable(variable=V, name_of_variable="V", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/V/')
        self.save_variable(variable=A, name_of_variable="A", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/A/')
        if not kernel_method:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs/")
            self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs_2/")
        self.save_image_of_cxc_variable(A, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/A_figs/")
        if not kernel_method:
            self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=False)
        else:
            self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=True)

    def train_FDA_weighted_KNN(self, k, kernel_method=False):
        if k is None:
            k = self.n_classes - 1  # excluding self
        self.calculate_M()
        connectivity_matrix = KNN(X=self.class_means.T, n_neighbors=k, mode='connectivity', include_self=False, n_jobs=-1)
        connectivity_matrix = connectivity_matrix.toarray()
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            A[:, :, class_index] = np.diagflat(connectivity_matrix[class_index, :])
        if not kernel_method:
            V = self.optimization_for_V(A)
            V2 = self.optimization_for_V_notDirtySolution(A)
            name = "FDA_weighted_KNN/k=" + str(k)
        else:
            V = self.kernel_optimization_for_V(A)
            name = "kernel_FDA_weighted_KNN/k=" + str(k)
        self.save_variable(variable=V, name_of_variable="V", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/V/')
        self.save_variable(variable=A, name_of_variable="A", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/A/')
        if not kernel_method:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs/")
            self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs_2/")
        self.save_image_of_cxc_variable(A, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/A_figs/")
        if not kernel_method:
            self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=False)
        else:
            self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=True)

    def train_FDA_weighted_CDM(self, LDA_or_QDA, kernel_method=False):
        self.calculate_M()
        if LDA_or_QDA == "LDA":
            clf = LDA()
        elif LDA_or_QDA == "QDA":
            clf = QDA()
        clf.fit(X=self.X.T, y=self.y)
        y_pred = clf.predict(self.X.T)
        conf_matrix = confusion_matrix(y_true=self.y, y_pred=y_pred)
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            for class_index_2 in range(self.n_classes):
                if class_index != class_index_2:
                    A[class_index_2, class_index_2, class_index] = conf_matrix[class_index, class_index_2] / self.N[class_index, class_index]
                else:
                    A[class_index_2, class_index_2, class_index] = 0
        if not kernel_method:
            V = self.optimization_for_V(A)
            V2 = self.optimization_for_V_notDirtySolution(A)
            name = "FDA_weighted_CDM"
        else:
            V = self.kernel_optimization_for_V(A)
            name = "kernel_FDA_weighted_CDM"
        self.save_variable(variable=V, name_of_variable="V", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/V/')
        self.save_variable(variable=A, name_of_variable="A", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/A/')
        if not kernel_method:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs/")
            self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs_2/")
        self.save_image_of_cxc_variable(A, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/A_figs/")
        if not kernel_method:
            self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=False)
        else:
            self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=True)

    def train_FDA_weighted_APAC(self, kernel_method=False):
        self.calculate_M()
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        for class_index in range(self.n_classes):
            for class_index_2 in range(self.n_classes):
                d = LA.norm(self.M[:, class_index_2, class_index])
                if class_index != class_index_2:
                    A[class_index_2, class_index_2, class_index] = (1 / (2 * (d ** 2))) * erf(d / (2 * (2**0.5)))
                else:
                    A[class_index_2, class_index_2, class_index] = 0
        if not kernel_method:
            V = self.optimization_for_V(A)
            V2 = self.optimization_for_V_notDirtySolution(A)
            name = "FDA_weighted_APAC"
        else:
            V = self.kernel_optimization_for_V(A)
            name = "kernel_FDA_weighted_APAC"
        self.save_variable(variable=V, name_of_variable="V", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/V/')
        self.save_variable(variable=A, name_of_variable="A", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/A/')
        if not kernel_method:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs/")
            self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs_2/")
        self.save_image_of_cxc_variable(A, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/A_figs/")
        if not kernel_method:
            self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=False)
        else:
            self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=True)

    def calculate_cosine(self, vec1, vec2):
        vec1 = vec1.reshape((-1, 1))
        vec2 = vec2.reshape((-1, 1))
        temp1 = vec1.T @ vec2
        temp2 = LA.norm(vec1) * LA.norm(vec2)
        return (1 / temp2) * temp1

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        k = (1 / np.sqrt(diag_kernel)).reshape((-1, 1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix

    def train_FDA_weighted_cosine(self, kernel_version, kernel_method=False):
        self.calculate_M()
        A = np.zeros((self.n_classes, self.n_classes, self.n_classes))
        if kernel_method and (kernel_version == 2):
            Kernel_classMeans_classMeans = pairwise_kernels(X=self.class_means.T, Y=self.class_means.T, metric=self.kernel)
            Kernel_classMeans_classMeans = self.normalize_the_kernel(kernel_matrix=Kernel_classMeans_classMeans)
        for class_index in range(self.n_classes):
            for class_index_2 in range(self.n_classes):
                if class_index != class_index_2:
                    if (not kernel_method) or (kernel_version == 1):
                        temp1 = self.calculate_cosine(self.class_means[:, class_index], self.class_means[:, class_index_2])
                        temp2 = 0.5 * (temp1 + 1)
                        A[class_index_2, class_index_2, class_index] = temp2
                    elif kernel_version == 2:
                        A[class_index_2, class_index_2, class_index] = Kernel_classMeans_classMeans[class_index, class_index_2]
                else:
                    A[class_index_2, class_index_2, class_index] = 0
        if not kernel_method:
            V = self.optimization_for_V(A)
            V2 = self.optimization_for_V_notDirtySolution(A)
            name = "FDA_weighted_cosine"
        else:
            V = self.kernel_optimization_for_V(A)
            if kernel_version == 1:
                name = "kernel_FDA_weighted_cosine_1"
            else:
                name = "kernel_FDA_weighted_cosine_2"
        self.save_variable(variable=V, name_of_variable="V", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/V/')
        self.save_variable(variable=A, name_of_variable="A", path_to_save='./saved_files/' + self.dataset_name + '/' + name + '/A/')
        if not kernel_method:
            self.save_image_of_dxd_variable(V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs/")
            self.save_image_of_dxd_variable(V2, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/V_figs_2/")
        self.save_image_of_cxc_variable(A, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/A_figs/")
        if not kernel_method:
            self.save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=False)
        else:
            self.kernel_save_projection_2D(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/projections/")
            self.classify_with_1NN(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test, V=V, path_to_save="./saved_files/" + self.dataset_name + '/' + name + "/KNN/", kernel_method=True)

    def calculate_objective_value_overall(self, V, A):
        S_B_weighted = self.calculate_S_B_weighted(A)
        if self.type_gradient == 1 or self.type_gradient == 2:
            f_value = -1 * np.trace((V.T) @ S_B_weighted @ V)
        elif self.type_gradient == 3:
            f_value = -1 * np.trace((V.T) @ S_B_weighted @ V)
            for class_index in range(self.n_classes):
                f_value += self.lambda_regularization * (LA.norm(A[:, :, class_index]) ** 2)
        return f_value

    def kernel_calculate_objective_value_overall(self, V, A):
        kernel_S_B_weighted = self.kernel_calculate_S_B_weighted(A)
        if self.type_gradient == 1 or self.type_gradient == 2:
            f_value = -1 * np.trace((V.T) @ kernel_S_B_weighted @ V)
        elif self.type_gradient == 3:
            f_value = -1 * np.trace((V.T) @ kernel_S_B_weighted @ V)
            for class_index in range(self.n_classes):
                f_value += self.lambda_regularization * (LA.norm(A[:, :, class_index]) ** 2)
        return f_value

    def calculate_objective_value_A(self, V, A, A_class):
        f_1 = self.calculate_objective_value_overall(V=V, A=A)
        # return f_1 + LA.norm(A_class)
        return f_1

    def kernel_calculate_objective_value_A(self, V, A, A_class):
        f_1 = self.kernel_calculate_objective_value_overall(V=V, A=A)
        # return f_1 + LA.norm(A_class)
        return f_1

    def simple_line_search_A(self, gradient, get_fvalue, **kwargs):
        fvalue_init = get_fvalue(V=kwargs['V'], A=kwargs['A'], A_class=kwargs['A_class'])
        step_size = 1.0
        while step_size > 0:
            A_class_new = kwargs['A_class'] - (step_size * gradient)
            A_new = kwargs['A'].copy()
            A_new[:, :, kwargs['class_index']] = A_class_new
            f_value = get_fvalue(V=kwargs['V'], A=A_new, A_class=A_class_new)
            if f_value <= fvalue_init:
                break
            step_size /= 2
        if f_value == fvalue_init:
            finish_optimization = True
        else:
            finish_optimization = False
        return A_class_new, f_value, finish_optimization

    def optimization_for_V(self, A):
        self.calculate_S_W()
        S_B_weighted = self.calculate_S_B_weighted(A)
        my_generalized_eigen_problem = My_generalized_eigen_problem(A=S_B_weighted, B=self.S_W)
        # eig_vec, eig_val = my_generalized_eigen_problem.solve()
        eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        eig_vec = (1 / LA.norm(eig_vec)) * eig_vec
        if self.n_components is not None:
            V = eig_vec[:, :self.n_components]
        else:
            V = eig_vec
        return V

    def optimization_for_V_notDirtySolution(self, A):
        self.calculate_S_W()
        S_B_weighted = self.calculate_S_B_weighted(A)
        my_generalized_eigen_problem = My_generalized_eigen_problem(A=S_B_weighted, B=self.S_W)
        eig_vec, eig_val = my_generalized_eigen_problem.solve()
        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        eig_vec = (1 / LA.norm(eig_vec)) * eig_vec
        if self.n_components is not None:
            V = eig_vec[:, :self.n_components]
        else:
            V = eig_vec
        return V

    def kernel_optimization_for_V(self, A):
        self.kernel_calculate_S_W()
        kernel_S_B_weighted = self.kernel_calculate_S_B_weighted(A)
        my_generalized_eigen_problem = My_generalized_eigen_problem(A=kernel_S_B_weighted, B=self.kernel_S_W)
        eig_vec, eig_val = my_generalized_eigen_problem.solve()   #--> used for ETH dataset
        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()   #--> used for ORL faces dataset
        eig_vec = (1 / LA.norm(eig_vec)) * eig_vec
        if self.n_components is not None:
            V = eig_vec[:, :self.n_components]
        else:
            V = eig_vec
        return V

    def set_some_values_to_zero(self, vector_, how_many_to_keep):
        if how_many_to_keep is None:
            return vector_
        n_dimensions = len(vector_)
        # how_many_to_keep = how_many_to_keep + 1  #--> because self is included here --> but it is wrong!
        how_many_to_keep = min(how_many_to_keep, n_dimensions)
        how_many_to_setToZero = n_dimensions - how_many_to_keep
        ascending_sort_indices = vector_.argsort()
        for index in range(how_many_to_setToZero):
            entry_index = ascending_sort_indices[index]
            vector_[entry_index] = 0
        return vector_

    def gradient_descent_A(self, class_index, V, A, A_class, max_iterations, kernel_method=False):
        assert self.M is not None and self.N is not None
        if self.report_progress_of_gradient_descent:
            f_value = self.calculate_objective_value_A(V=V, A=A, A_class=A_class)
            print("gradient descent for A, iteration " + str(-1) + ", loss: " + str(f_value))
        temp1 = self.N[class_index, class_index] * np.kron(self.M[:, :, class_index] @ (self.N.T), self.M[:, :, class_index])
        temp2 = -1 * (V @ (V.T))
        temp2 = temp2.reshape((-1, 1))
        temp8 = (temp1.T) @ temp2
        temp8_reshaped = temp8.reshape((self.n_classes, self.n_classes))
        for iteration_index in range(max_iterations):
            if self.type_gradient == 1:
                gradient = temp8_reshaped
            elif self.type_gradient == 2:
                temp4 = 1 / (LA.norm(A_class) ** 2)
                temp5 = (-2 * temp4) * np.kron(A_class.T, A_class)
                temp6 = np.eye(self.n_classes ** 2)
                temp7 = temp4 * (temp5 + temp6)
                temp9 = (temp7.T) @ temp8
                temp9_reshaped = temp9.reshape((self.n_classes, self.n_classes))
                gradient = temp9_reshaped
            elif self.type_gradient == 3:
                gradient = temp8_reshaped + (2 * self.lambda_regularization * LA.norm(A_class))
            # ---- fixed step size:
            # A_class = A_class - (self.learning_rate_A * gradient)
            # A_new = A.copy()
            # A_new[:, :, class_index] = A_class
            # f_value = self.calculate_objective_value_A(V=V, A=A_new, A_class=A_class)
            # ---- line-search step size:
            if not kernel_method:
                A_class, f_value, finish_optimization = self.simple_line_search_A(gradient=gradient, get_fvalue=self.calculate_objective_value_A,
                                                                                V=V, A=A, A_class=A_class, class_index=class_index)
            else:
                A_class, f_value, finish_optimization = self.simple_line_search_A(gradient=gradient, get_fvalue=self.kernel_calculate_objective_value_A,
                                                                                V=V, A=A, A_class=A_class, class_index=class_index)
            A[:, :, class_index] = A_class
            if self.report_progress_of_gradient_descent:
                print("gradient descent for A, class=" + str(class_index) + ", iteration " +
                        str(iteration_index) + ", loss: " + str(f_value))
            if finish_optimization:
                break
        # ---- l0 norm <= K_ell0:
        A_class_diag = np.diag(A_class).copy()
        A_class_diag = self.set_some_values_to_zero(vector_=A_class_diag, how_many_to_keep=self.K_ell0)
        # ----
        A_class = np.diagflat(A_class_diag)
        # ---- get f_value:
        A_new = A.copy()
        A_new[:, :, class_index] = A_class
        if not kernel_method:
            f_value = self.calculate_objective_value_A(V=V, A=A_new, A_class=A_class)
        else:
            f_value = self.kernel_calculate_objective_value_A(V=V, A=A_new, A_class=A_class)
        if self.report_progress_of_gradient_descent:
            print("gradient descent for A, class=" + str(class_index) + ", after projection" + ", loss: " + str(f_value))
        return A_class

    def calculate_M(self):
        assert self.X_classes != None
        class_means = [None] * self.n_classes
        self.class_means = np.zeros((self.n_dimensions, self.n_classes))
        for class_index in range(self.n_classes):
            X_class = self.X_classes[class_index]
            X_class_mean = (X_class.mean(axis=1)).reshape((-1, 1))
            class_means[class_index] = X_class_mean
            self.class_means[:, class_index] = X_class_mean.ravel()
        # calculating matrix M:
        self.M = np.zeros((self.n_dimensions, self.n_classes, self.n_classes))
        for class_index_1 in range(self.n_classes):
            for class_index_2 in range(self.n_classes):
                self.M[:, class_index_2, class_index_1] = (class_means[class_index_1] - class_means[class_index_2]).ravel()

    def kernel_calculate_M(self):
        assert self.X_classes != None
        M_c_classes = np.zeros((self.n_samples, self.n_classes))
        for class_index in range(self.n_classes):
            X_class = self.X_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            # ------ M_c:
            Kernel_allSamples_classSamples = pairwise_kernels(X=self.X.T, Y=X_class.T, metric=self.kernel)
            M_c = Kernel_allSamples_classSamples.sum(axis=1)
            M_c = M_c.reshape((-1, 1))
            M_c = (1 / n_samples_of_class) * M_c
            M_c_classes[:, class_index] = M_c.ravel()
        # calculating matrix M:
        self.M = np.zeros((self.n_samples, self.n_classes, self.n_classes))
        for class_index_1 in range(self.n_classes):
            for class_index_2 in range(self.n_classes):
                self.M[:, class_index_2, class_index_1] = (M_c_classes[:, class_index_1] - M_c_classes[:, class_index_2]).ravel()

    def calculate_S_W(self):
        assert self.X_classes != None
        S_W = 0
        for class_index in range(self.n_classes):
            X_class = self.X_classes[class_index]
            X_class_centered = self.center_the_matrix(the_matrix=X_class, mode="remove_mean_of_columns_from_columns")
            n_samples_of_class = X_class_centered.shape[1]
            S_W += n_samples_of_class * X_class_centered@(X_class_centered.T)
        self.S_W = S_W

    def kernel_calculate_S_W(self):
        assert self.X_classes != None
        N = np.zeros((self.n_samples, self.n_samples))
        for class_index in range(self.n_classes):
            X_class = self.X_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            Kernel_allSamples_classSamples = pairwise_kernels(X=self.X.T, Y=X_class.T, metric=self.kernel)
            K_c = Kernel_allSamples_classSamples
            H_c = np.eye(n_samples_of_class) - (1 / n_samples_of_class) * np.ones((n_samples_of_class, n_samples_of_class))
            N = N + K_c.dot(H_c).dot(K_c.T)
        self.kernel_S_W = N

    def calculate_S_B_weighted(self, A):
        self.calculate_M()
        assert self.M is not None and self.N is not None
        S_B_weighted = 0
        for class_index in range(self.n_classes):
            n_samples_of_class = self.N[class_index, class_index]
            S_B_weighted += n_samples_of_class * self.M[:, :, class_index] @ A[:, :, class_index] @ self.N \
                            @ self.M[:, :, class_index].T
        return S_B_weighted

    def kernel_calculate_S_B_weighted(self, A):
        self.kernel_calculate_M()
        assert self.M is not None and self.N is not None
        kernel_S_B_weighted = 0
        for class_index in range(self.n_classes):
            n_samples_of_class = self.N[class_index, class_index]
            kernel_S_B_weighted += n_samples_of_class * self.M[:, :, class_index] @ A[:, :, class_index] @ self.N \
                            @ self.M[:, :, class_index].T
        return kernel_S_B_weighted

    def save_image_of_dxd_variable(self, V, path_to_save="./"):
        # V --> self.n_dimensions * self.n_components
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        for component_index in range(V.shape[1]):
            V_component_in_image_form = V[:, component_index].reshape((self.image_height, self.image_width))
            plt.imshow(V_component_in_image_form, cmap='gray')
            plt.axis('off')
            # plt.colorbar()
            # plt.show()
            plt.savefig(path_to_save+"component_"+str(component_index)+".png")
        plt.clf()
        plt.close()

    def put_all_weights_in_a_matrix(self, A):
        # A --> self.n_classes * self.n_classes * self.n_classes
        weights_all_together = np.zeros((self.n_classes, self.n_classes))
        for class_index in range(A.shape[2]):
            weights_for_this_class_and_others = np.diag(A[:, :, class_index])
            weights_all_together[class_index, :] = weights_for_this_class_and_others
        return weights_all_together

    def save_image_of_cxc_variable(self, A, path_to_save="./"):
        # A --> self.n_classes * self.n_classes * self.n_classes
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        weights_all_together = np.zeros((self.n_classes, self.n_classes))
        for class_index in range(A.shape[2]):
            weights_for_this_class_and_others = np.diag(A[:, :, class_index])
            weights_all_together[class_index, :] = weights_for_this_class_and_others
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(weights_all_together, cmap='gray')
        plt.axis('on')
        tick_marks = np.arange(self.n_classes)
        plt.xticks(tick_marks, np.arange(1, self.n_classes + 1, 1), rotation=0)
        plt.yticks(tick_marks, np.arange(1, self.n_classes + 1, 1), rotation=0)
        plt.ylim([self.n_classes-1+0.5, -0.5])
        plt.colorbar()
        # plt.show()
        plt.savefig(path_to_save+"matrix"+".png")
        plt.clf()
        plt.close()

    def save_projection_2D(self, X, y, X_test, y_test, V, path_to_save="./"):
        # X: rows are features
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        X_projected = (V.T) @ X
        self.__plot_projected_X(X_projected=X_projected, y=y, path_to_save=path_to_save, name="train")
        if X_test is None:
            return
        X_test_projected = (V.T) @ X_test
        self.__plot_projected_X(X_projected=X_test_projected, y=y_test, path_to_save=path_to_save, name="test")
        # plot train and test together:
        _, ax = plt.subplots(1)
        classes = [str(class_index) for class_index in np.arange(1, self.n_classes + 1)]
        if np.min(y) == 1:
            y = y - 1
            y_test = y_test - 1
        # cmap='Spectral', 'tab10', 'tab20'
        plt.scatter(X_projected[0, :], X_projected[1, :], s=100, c=y, edgecolors="k", linewidths=0.5, cmap='Spectral', alpha=1)
        plt.scatter(X_test_projected[0, :], X_test_projected[1, :], s=100, c=y_test, marker="s", edgecolors="k", linewidths=0.5, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(self.n_classes + 1) - 0.5)
        cbar.set_ticks(np.arange(self.n_classes))
        cbar.set_ticklabels(classes)
        plt.savefig(path_to_save + "train_and_test" + ".png")
        plt.clf()
        plt.close()

    def kernel_save_projection_2D(self, X, y, X_test, y_test, V, path_to_save="./"):
        # X: rows are features
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        Kernel_allSamples_newSamples = pairwise_kernels(X=self.X.T, Y=X.T, metric=self.kernel)
        X_projected = (V.T) @ Kernel_allSamples_newSamples
        self.__plot_projected_X(X_projected=X_projected, y=y, path_to_save=path_to_save, name="train")
        if X_test is None:
            return
        Kernel_allSamples_newSamples = pairwise_kernels(X=self.X.T, Y=X_test.T, metric=self.kernel)
        X_test_projected = (V.T) @ Kernel_allSamples_newSamples
        self.__plot_projected_X(X_projected=X_test_projected, y=y_test, path_to_save=path_to_save, name="test")
        # plot train and test together:
        _, ax = plt.subplots(1)
        classes = [str(class_index) for class_index in np.arange(1, self.n_classes + 1)]
        if np.min(y) == 1:
            y = y - 1
            y_test = y_test - 1
        # cmap='Spectral', 'tab10', 'tab20'
        plt.scatter(X_projected[0, :], X_projected[1, :], s=100, c=y, edgecolors="k", linewidths=0.5, cmap='Spectral', alpha=1)
        plt.scatter(X_test_projected[0, :], X_test_projected[1, :], s=100, c=y_test, marker="s", edgecolors="k", linewidths=0.5, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(self.n_classes + 1) - 0.5)
        cbar.set_ticks(np.arange(self.n_classes))
        cbar.set_ticklabels(classes)
        plt.savefig(path_to_save + "train_and_test" + ".png")
        plt.clf()
        plt.close()

    def __plot_projected_X(self, X_projected, y, path_to_save, name):
        _, ax = plt.subplots(1)
        classes = [str(class_index) for class_index in np.arange(1, self.n_classes+1)]
        if np.min(y) == 1:
            y = y - 1
        # cmap='Spectral', 'tab10', 'tab20'
        plt.scatter(X_projected[0, :], X_projected[1, :], s=100, c=y, edgecolors="k", linewidths=0.5, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(self.n_classes + 1) - 0.5)
        cbar.set_ticks(np.arange(self.n_classes))
        cbar.set_ticklabels(classes)
        plt.savefig(path_to_save + name + ".png")
        plt.clf()
        plt.close()

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    def set_singular_values_one(self, matrix):
        Q, Sigma, Omega_h = LA.svd(matrix, full_matrices=False)
        Sigma = np.diag(np.ones((self.n_components, 1)).ravel())
        matrix_projected = Q.dot(Sigma).dot(Omega_h)
        return matrix_projected

    def prox_nuclear_norm(self, matrix, parameter_):
        assert matrix.shape[0] == matrix.shape[1]
        Q, Sigma, Omega_h = LA.svd(matrix, full_matrices=True)
        temp = parameter_ * np.eye(matrix.shape[0])
        Sigma = Sigma - temp
        matrix_projected = Q.dot(Sigma).dot(Omega_h)
        return matrix_projected

    def prox_l1_norm(self, matrix, parameter_):
        # https://math.stackexchange.com/questions/1766811/l-1-regularized-unconstrained-optimization-problem
        prox = np.zeros(matrix.shape)
        for dim1 in range(matrix.shape[0]):
            for dim2 in range(matrix.shape[1]):
                feature = matrix[dim1, dim2]
                prox[dim1, dim2] = np.sign(feature) * max(abs(feature) - parameter_, 0)
        return prox

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix

    def separate_samples_of_classes_2(self, X, y):  # it does not change the order of the samples within every class
        # X --> rows: features, columns: samples
        # return X_separated_classes --> each element of list --> rows: features, columns: samples
        y = np.asarray(y)
        y = y.reshape((-1, 1)).ravel()
        labels_of_classes = sorted(set(y.ravel().tolist()))
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        n_classes = len(labels_of_classes)
        X_separated_classes = [np.empty((n_dimensions, 0))] * n_classes
        original_index_in_whole_dataset = [[]] * n_classes
        for class_index in range(n_classes):
            for sample_index in range(n_samples):
                if y[sample_index] == labels_of_classes[class_index]:
                    X_separated_classes[class_index] = np.column_stack((X_separated_classes[class_index], X[:, sample_index].reshape((-1,1))))
                    original_index_in_whole_dataset[class_index].append(sample_index)
        return X_separated_classes, original_index_in_whole_dataset

    def classify_with_1NN(self, X, y, X_test, y_test, V, path_to_save, kernel_method=False):
        if not kernel_method:
            X_projected = (V.T) @ X
            X_test_projected = (V.T) @ X_test
        else:
            Kernel_allSamples_newSamples = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
            X_projected = (V.T) @ Kernel_allSamples_newSamples
            Kernel_allSamples_newSamples = pairwise_kernels(X=X.T, Y=X_test.T, metric=self.kernel)
            X_test_projected = (V.T) @ Kernel_allSamples_newSamples
        neigh = KNeighborsClassifier(n_neighbors=2)   #--> it includes itself too
        neigh.fit(X_projected.T, y)
        y_pred = neigh.predict(X_projected.T)
        accuracy_train = accuracy_score(y_true=y, y_pred=y_pred)
        conf_matrix_train = confusion_matrix(y_true=y, y_pred=y_pred)
        self.save_np_array_to_txt(variable=np.asarray(accuracy_train), name_of_variable="accuracy_train", path_to_save=path_to_save)
        self.save_variable(variable=accuracy_train, name_of_variable="accuracy_train", path_to_save=path_to_save)
        self.plot_confusion_matrix(confusion_matrix=conf_matrix_train, class_names=[str(class_index+1) for class_index in range(self.n_classes)],
                                   normalize=True, cmap="gray_r", path_to_save=path_to_save, name="train")
        y_pred = neigh.predict(X_test_projected.T)
        accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred)
        conf_matrix_test = confusion_matrix(y_true=y_test, y_pred=y_pred)
        self.save_np_array_to_txt(variable=np.asarray(accuracy_test), name_of_variable="accuracy_test", path_to_save=path_to_save)
        self.save_variable(variable=accuracy_test, name_of_variable="accuracy_test", path_to_save=path_to_save)
        self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index+1) for class_index in range(self.n_classes)],
                                   normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")

    def plot_confusion_matrix(self, confusion_matrix, class_names, normalize=False, cmap="gray", path_to_save="./", name="temp"):
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')
        # print(cm)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        # plt.colorbar()
        tick_marks = np.arange(len(class_names))
        # plt.xticks(tick_marks, class_names, rotation=45)
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names)
        # tick_marks = np.arange(len(class_names) - 1)
        # plt.yticks(tick_marks, class_names[1:])
        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.ylabel('true distortion type')
        plt.xlabel('predicted distortion type')
        plt.ylim([self.n_classes - 0.5, -0.5])
        plt.tight_layout()
        # plt.show()
        plt.savefig(path_to_save + name + ".png")
        plt.clf()
        plt.close()




