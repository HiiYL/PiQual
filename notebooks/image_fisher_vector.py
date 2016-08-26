import numpy as np
import h5py
import os
import cv2
from sklearn.decomposition import PCA
import ggmm.gpu as ggmm

from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve


import pandas as pd
from pandas import HDFStore, DataFrame
import pickle

class ImageFisherVector(object):
    dataset_dir = '../dataset_h5/'
    skipped_indices = []
    filename = 'images_224_delta_1.5.h5'
    test_filename = 'images_224.h5'

    classifier_filename = "classifier.p"

    gmm_filename = 'gmm.pkl'

    delta = 1.5

    gmm = ""
    def __init__(self):
        store = HDFStore('../dataset_h5/labels.h5')
        if(os.path.isfile(classifier_filename) and os.path.isfile(gmm_filename)):
            classifier =  pickle.load( open( "classifier.p", "rb" ) )
            print("Loaded Classifier!")
            gmm = load_gmm()
        else:
            try:
                labels_train = store['labels_train_trimmed']
                fv = np.load("fisher_vector_train.npy")
            except(FileNotFoundError,KeyError) as e:
                labels_to_train = store['labels_train']
                skipped_indices, fv,gmm = process_images(labels_to_train,delta=delta,is_training=True)
                labels_train = load_labels(skipped_indices,labels_to_train, True,delta)
                store['labels_train_trimmed'] = labels_train
                np.save("fisher_vector_train.npy",fv )
                np.save("skipped_indices.npy",skipped_indices)
            classifier = train(fv,labels_train.score)
            #classifier = train(fv,labels_train.good)
        try:
            labels_test = store['labels_test_trimmed']
            fv_test = np.load("fisher_vector_test.npy")
        except(FileNotFoundError,KeyError) as e:
            labels_to_test = store['labels_test']
            skipped_indices_test, fv_test = process_images(labels_to_test,is_training=False, input_gmm=gmm)
            labels_test = load_labels(skipped_indices_test,labels_to_test, False)
            store['labels_test_trimmed'] = labels_test
            np.save("fisher_vector_test.npy",fv_test )
            np.save("skipped_indices_test.npy",skipped_indices_test)


        accuracy_score(labels_test.good, [ 0 if label < 5 else 1 for label in classifier.predict(fv_test)])
        f1_score(labels_test.good, [ 0 if label < 0 else 1 for label in classifier.predict(fv_test)])
        roc_auc_score(labels_test.good, [ 0 if label < 0 else 1 for label in classifier.predict(fv_test)])
        #accuracy_score(labels_test.good, classifier.predict(fv_test))
        roc_auc_score(labels_test.good, classifierSVC.predict(fv_test))
        f1_score(labels_test.good, classifierSVC.predict(fv_test))




    def process_images(ava_table, is_training,gmm="",delta=0,input_gmm=None):
        ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
        ava_path = "../dataset/AVA/data/"
        ava_data_path = os.path.join(os.getcwd(), ava_path)

        skipped_indices = []
        image_features_list = []

        periodNum = ava_table.shape[0]

        image_descriptors_filename = "image_descriptors_{0}_{1}.pkl".format(is_training,periodNum)

        if(os.path.isfile(image_descriptors_filename)):
            print("Image Descriptors Loaded!!")
            image_features_list = pickle.load(open(image_descriptors_filename,"rb"))
        else:
            print("Image Descriptors Not Found, Generating....")
            i=0
            for index, row in ava_table.iterrows():
                if (i % 100) == 0:
                  print('Now Processing {0}/{1}'.format(i,periodNum))
                filename = "{0}.jpg".format(index)

                filepath = os.path.join(ava_data_path, filename)
                image = cv2.imread(filepath)
                image_features = extract_image_features(image)

                if image_features is not None and image_features.shape[0] >= 64:
                    image_features = reduce_features(image_features)
                    #print("{0} - {1}".format(index, image_features.shape))
                    image_features_list.append(image_features)
                else:
                    print(index)
                    skipped_indices.append(index)
                i = i + 1
            pickle.dump(image_features_list,open(image_descriptors_filename,"wb"))

        #[ reduce_features(image_features) for image_features in image_features_list]
        if input_gmm is None:
            print("Generating GMM")
            gmm = generate_gmm(image_features_list)
            #pickle.dump(gmm,open("gmm.pkl","wb"))
        else:
            print("Loading training GMM for test set")
            gmm = input_gmm
            #gmm = pickle.load(open("gmm.pkl", "rb"))

        print("Generating Fisher Vector")
        fv = [ fisher_vector(image,gmm) for image in image_features_list]

        if is_training:
            return skipped_indices, fv,gmm
        else:
            return skipped_indices, fv


        #return (skipped_indices, np.asarray(image_features_list))

        #return [extract_image_features(image,index, skipped_indices) for index, image in enumerate(images, 1)]

    def load_labels(skipped_indices, ava_table, is_training, delta=0):
        if is_training:
            ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

        #if n_labels < images_with_no_features[len(images_with_no_features) - 1]:
        #    images_with_no_features = [i for i in images_with_no_features if i <= n_labels]
        ava_table = ava_table.drop(ava_table.ix[skipped_indices].index)

        return ava_table

    def load_full_labels(skipped_indices, ava_table, is_training, delta=0):
        if is_training:
            ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

        #if n_labels < images_with_no_features[len(images_with_no_features) - 1]:
        #    images_with_no_features = [i for i in images_with_no_features if i <= n_labels]
        ava_table = ava_table.drop(ava_table.ix[skipped_indices].index)

        return ava_table




    def extract_image_features(image):
        sift = cv2.xfeatures2d.SIFT_create()
        _ , descriptors =  sift.detectAndCompute(image, None)
        return descriptors



    def reduce_features(image_descriptors):
        # if image_descriptors.shape[0] < 64:
        #     pca = decomposition.SparsePCA(n_components=64)# adjust yourself
        # else:
        pca = PCA(n_components=64)
        return (pca.fit_transform(image_descriptors))

    def generate_gmm(image_features_list):
        concatenated_descriptors = np.concatenate(image_features_list)
        gmm_filename = 'gmm.pkl'
        N, D = concatenated_descriptors.shape
        K=128

        print("The sizes are {0} and {1}".format(N,D))

        if(N > 3000000):
            batch_size = 3000000
        else:
            batch_size = N

        ggmm.init(batch_size * D)
        gmm = ggmm.GMM(K,D)

        thresh = 1e-3 # convergence threshold
        n_iter = 500 # maximum number of EM iterations
        init_params = 'wmc' # initialize weights, means, and covariances

        # train GMM
        converged = gmm.fit(concatenated_descriptors[:batch_size], thresh, n_iter, init_params=init_params)

        print("GMM converged? ... {0}".format(converged))

        pickle.dump((gmm.get_weights(), gmm.get_means(), gmm.get_covars()), open(gmm_filename,'wb'))

        return gmm

    def load_gmm():
        wmc = pickle.load(open("gmm.pkl","rb"))
        ggmm.init(wmc[0].shape[0] * 64)
        gmm = ggmm.GMM(wmc[0].shape[0],64)

        gmm.set_weights(wmc[0])
        gmm.set_means(wmc[1])
        gmm.set_covars(wmc[2])
        print("Loaded GMM Info!")
        return gmm

    def fisher_vector(xx, gmm):
        """Computes the Fisher vector on a set of descriptors.
        Parameters
        ----------
        xx: array_like, shape (N, D) or (D, )
            The set of descriptors
        gmm: instance of sklearn mixture.GMM object
            Gauassian mixture model of the descriptors.
        Returns
        -------
        fv: array_like, shape (K + 2 * D * K, )
            Fisher vector (derivatives with respect to the mixing weights, means
            and variances) of the given descriptors.
        Reference
        ---------
        J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
        Vectors for Image Categorization.  In ICCV, 2011.
        http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
        """
        xx = np.atleast_2d(xx)
        N = xx.shape[0]

        # Compute posterior probabilities.
        Q = gmm.compute_posteriors(xx)  # NxK
        
        Q = Q.asarray()

        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_xx = np.dot(Q.T, xx) / N
        Q_xx_2 = np.dot(Q.T, xx ** 2) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - gmm.get_weights()
        d_mu = Q_xx - Q_sum * gmm.get_means()
        d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.get_means() ** 2
            + Q_sum * gmm.get_covars()
            + 2 * Q_xx * gmm.get_means())

        # Merge derivatives into a vector.
        return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

    def train(features, labels):
        X = features
        Y = labels

        clf = svm.LinearSVC()

        #clf = linear_model.SGDRegressor()
        clf.fit(X, Y)

        return clf