from __future__ import print_function
import time
import torch.utils.data
import torch.utils.data
import logging
from sklearn.metrics import accuracy_score
import pandas
from scipy import stats
import torch.utils.data
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from joblib import Parallel, delayed
from dc import *
import classifier as ClSFR
import matplotlib.pyplot as plt
from vae import *
import data as Data
import os
import glob
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
estimate_prob_type = "OURS"
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers):
    C = 10  # num_classes
    Y = np.zeros((data_size, C), dtype=np.float32)
    D = np.zeros((data_size, C, len(source_domains)), dtype=np.float64)
    H = np.zeros((data_size, C, len(source_domains)), dtype=np.float32)

    i = 0
    precentage = int((i / data_size) * 100)
    print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    for target_domain, data_loader in data_loaders:

        for data, label in data_loader:

            data = data.to(device)
            N = len(data)
            # ---- Y (one-hot)
            y_vals = label.detach().cpu().numpy()
            one_hot = np.zeros((y_vals.size, C), dtype=np.float32)
            one_hot[np.arange(y_vals.size), y_vals] = 1.0
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    # Calculate h(x)
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i:i + N, :, k] = norm_output.cpu().detach().numpy()

                    # calculate log_p
                    x_hat, _, _ = models[source_domain](data)
                    log_p = models[source_domain].compute_log_probabitility_bernoulli(x_hat,
                                                                                      data.view(data.shape[0], -1))
                    # prob = torch.exp(log_p)
                    prob = torch.abs(log_p) # changed that
                    prob_tile = torch.tile(prob[:, None], (1, C))
                    D[i:i + N, :, k] = prob_tile.cpu().detach().numpy()

            i += N

        precentage = int((i / data_size) * 100)
        print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    for k, source_domain in enumerate(source_domains):
        # Make distribution
        D[:, :, k] = D[:, :, k] / D[:, :, k].sum()

    return Y, D, H


def DC_programming(seed, models, classifiers, source_domains, test_path, sgd_alpha, sgd_max_iter):
    ''' Calculate the distribution and hypothesis of the data (over the target data) '''
    logging.info("============== Build domain adaptation model ===================")

    data_size = 0
    data_loaders = []
    for k, domain in enumerate(source_domains):
        train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=1000)
        data_size += len(train_loader.dataset)
        data_loaders.append((domain, train_loader))

    Y, D, H = build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers)

    X = np.concatenate((D, H), axis=1)
    X = np.reshape(X, (data_size, -1))
    np.savetxt(test_path + r'/SGD_input.txt', X, delimiter=',')
    y = Y.argmax(axis=1)
    # clf = SGDClassifier(loss="log_loss", alpha=sgd_alpha, random_state=seed, tol=1e-4,
    #                     early_stopping=True, validation_fraction=0.4, max_iter=sgd_max_iter).fit(X, y)
    clf = SGDClassifier(loss="log_loss", alpha=sgd_alpha, random_state=seed,
                        early_stopping=False, max_iter=sgd_max_iter).fit(X, y)

    return clf


def test_DC_model(seed, models, classifiers, source_domains, target_domains, clf):
    data_size = 0
    data_loaders = []
    for domain in target_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=seed)
        data_size += len(test_loader.dataset)
        data_loaders.append((domain, test_loader))

    Y, D, H = build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers)

    X = np.concatenate((D, H), axis=1)
    X = np.reshape(X, (data_size, -1))
    y = clf.predict_proba(X)

    score = accuracy_score(y_true=Y.argmax(axis=1), y_pred=y.argmax(axis=1))
    print(score)
    logging.info("score = {}".format(score))

    return score


def run_domain_adaptation(alpha_pos, alpha_neg, vr_model_type, seed, test_path, classifiers,
                          source_domains, sgd_alpha, sgd_max_iter):
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    logging_filename = "domain_adaptation.log"
    logging.basicConfig(filename=logging_filename, level=logging.DEBUG)

    # Domains
    target_domains_sets = \
        [['MNIST', 'USPS', 'SVHN'],  # test set with all the domains
         ['MNIST', 'USPS'], ['MNIST', 'SVHN'], ['USPS', 'SVHN'],  # test set with pairs
         ['MNIST'], ['USPS'], ['SVHN']]  # test set with singles

    models = {}
    # for domain in source_domains:
    #     model = vr_model(alpha_pos, alpha_neg).to(device)
    #     model.load_state_dict(torch.load("./models_new/{}_{}_{}_{}_model.pt".format(
    #         vr_model_type, alpha_pos, alpha_neg, domain), map_location=torch.device(device)))
    #     models[domain] = model

    for domain in source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)

        filename = f"{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed1_model.pt"
        pattern = f"./models_{domain}_seed1*/{filename}"
        matches = glob.glob(pattern)

        if not matches:
            print(f"⚠️ Warning: Model file not found for pattern: {pattern}")
            continue

        model_path = matches[0]  # take the first match found

        print(f"✅ Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        models[domain] = model

    clf = DC_programming(seed, models, classifiers, source_domains, test_path, sgd_alpha, sgd_max_iter)

    with open(test_path + r'/SGD_accuracy_score_{}.txt'.format(seed), 'w') as fp:

        for target_domains in target_domains_sets:
            print(target_domains)

            score = test_DC_model(seed, models, classifiers, source_domains, target_domains, clf)
            logging.info("")
            target_domains_score = target_domains
            target_domains_score.append(str(score * 100))
            target_domains_score.append("\n")
            fp.write('\t'.join(target_domains_score))

def task_run(date, seed, sgd_alpha, sgd_max_iter,
             model_type, pos_alpha, neg_alpha,
             classifiers, source_domains):

    test_path = (
        f'./Results_____{date}/'
        f'seed_{seed}__sgd_alpha_{sgd_alpha}__sgd_max_iter_{sgd_max_iter}/'
        f'model_type_{model_type}___pos_alpha_{pos_alpha}___neg_alpha_{neg_alpha}'
    )
    os.makedirs(test_path, exist_ok=True)

    run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path,
                          classifiers, source_domains, sgd_alpha, sgd_max_iter)


# def main():

#     domains_accuracy_score = []
#     classifiers = {}
#     source_domains = ['MNIST', 'USPS', 'SVHN']
#     for domain in source_domains:
#         # Load classifiers
#         _, test_loader = Data.get_data_loaders(domain, seed=1)

#         classifier = ClSFR.Grey_32_64_128_gp().to(device)
#         classifier.load_state_dict(
#             torch.load("./classifiers_new/{}_classifier.pt".format(domain), map_location=torch.device(device)))
#         accuracy = ClSFR.test(classifier, test_loader)

#         domains_accuracy_score.append(domain + " = " + str(accuracy))
#         classifiers[domain] = classifier

#     with open(r'./domain_accuracy_score.txt', 'w') as fp:
#         fp.write('\n'.join(domains_accuracy_score))

#     date = '26_9'

#     for seed in [48]:
#         for sgd_alpha in [1e-5]:
#             for sgd_max_iter in [2000]:

#                 # -------- VRS-SGD --------
#                 model_type = 'vrs'
#                 for (pos_alpha, neg_alpha) in [(2, -2), (2, -0.5), (0.5, -2), (0.5, -0.5)]:
#                     test_path = './Results_____{}/seed_{}__sgd_alpha_{}__sgd_max_iter_{}/model_type_{}___pos_alpha_{}___neg_alpha_{}'.format(
#                         date, seed, sgd_alpha, sgd_max_iter, model_type, pos_alpha, neg_alpha)
#                     os.makedirs(test_path, exist_ok=True)
#                     run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path,
#                                         classifiers, source_domains, sgd_alpha, sgd_max_iter)

#                 # -------- VR-SGD --------
#                 model_type = 'vr'
#                 for (pos_alpha, neg_alpha) in [(2, -1), (0.5, -1)]:  
#                     test_path = './Results_____{}/seed_{}__sgd_alpha_{}__sgd_max_iter_{}/model_type_{}___pos_alpha_{}___neg_alpha_{}'.format(
#                         date, seed, sgd_alpha, sgd_max_iter, model_type, pos_alpha, neg_alpha)
#                     os.makedirs(test_path, exist_ok=True)
#                     run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path,
#                                         classifiers, source_domains, sgd_alpha, sgd_max_iter)

#                 # -------- VAE-SGD --------
#                 model_type = 'vae'
#                 pos_alpha, neg_alpha = 1, -1
#                 test_path = './Results_____{}/seed_{}__sgd_alpha_{}__sgd_max_iter_{}/model_type_{}___pos_alpha_{}___neg_alpha_{}'.format(
#                     date, seed, sgd_alpha, sgd_max_iter, model_type, pos_alpha, neg_alpha)
#                 os.makedirs(test_path, exist_ok=True)
#                 run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path,
#                                     classifiers, source_domains, sgd_alpha, sgd_max_iter)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = ", device)

    domains_accuracy_score = []
    classifiers = {}
    source_domains = ['MNIST', 'USPS', 'SVHN']
    for domain in source_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=1)
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        classifier.load_state_dict(torch.load(f"./classifiers_new/{domain}_classifier.pt",
                                              map_location=torch.device(device)))
        accuracy = ClSFR.test(classifier, test_loader)
        domains_accuracy_score.append(domain + " = " + str(accuracy))
        classifiers[domain] = classifier

    with open('./domain_accuracy_score.txt', 'w') as fp:
        fp.write('\n'.join(domains_accuracy_score))

    date = '26_9'
    seeds = [48]
    sgd_alphas = [1e-5]
    sgd_max_iters = [2000]

    tasks = []

    # VRS-SGD 
    for seed in seeds:
        for sgd_alpha in sgd_alphas:
            for sgd_max_iter in sgd_max_iters:
                for (pos_alpha, neg_alpha) in [(2, -2), (2, -0.5), (0.5, -2), (0.5, -0.5)]:
                    tasks.append((date, seed, sgd_alpha, sgd_max_iter,
                                  'vrs', pos_alpha, neg_alpha))

                # VR-SGD (VR2 VR0.5)
                for (pos_alpha, neg_alpha) in [(2, -1), (0.5, -1)]:
                    tasks.append((date, seed, sgd_alpha, sgd_max_iter,
                                  'vr', pos_alpha, neg_alpha))

                # VAE-SGD
                tasks.append((date, seed, sgd_alpha, sgd_max_iter,
                              'vae', 1, -1))

    Parallel(n_jobs=os.cpu_count(), backend="loky")(
        delayed(task_run)(*t, classifiers, source_domains) for t in tasks
    )



if __name__ == "__main__":
    main()
