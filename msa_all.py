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
from dc import *
import classifier as ClSFR
import matplotlib.pyplot as plt
from vae import *
import data as Data
import os
from joblib import Parallel, delayed
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors,
                           estimate_prob_type):
    C = 10  # num_classes
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))

    i = 0
    precentage = int((i / data_size) * 100)
    print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    for target_domain, data_loader in data_loaders:

        for data, label in data_loader:

            data = data.to(device)
            N = len(data)

            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            one_hot[np.arange(y_vals.size), y_vals] = 1
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    # Calculate h(x)
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i:i + N, :, k] = norm_output.cpu().detach().numpy()

                    # Calculate D(x)
                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = output.cpu().detach().numpy()
                    elif estimate_prob_type == "OURS-STD-SCORE" or estimate_prob_type == "OURS-KDE":
                        # calculate log_p
                        x_hat, _, _ = models[source_domain](data)
                        # log_p = models[source_domain].compute_log_probabitility_bernoulli(x_hat,
                        #                                                                   data.view(data.shape[0], -1))
                        log_p = models[source_domain].compute_log_probabitility_gaussian(x_hat,
                                                                                         data.view(data.shape[0], -1),
                                                                                         torch.zeros_like(x_hat))
                        log_p_tile = torch.tile(log_p[:, None], (1, C))
                        D[i:i + N, :, k] = log_p_tile.cpu().detach().numpy()

            i += N

        precentage = int((i / data_size) * 100)
        print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type == "GMSA" or estimate_prob_type == "OURS-KDE":
            # use grid search cross-validation to optimize the bandwidth

            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)

            data = D[:, :, k]
            shuffled_indices = np.random.permutation(len(data))  # return a permutation of the indices
            data_shuffle = data[shuffled_indices]
            grid.fit(data_shuffle[:2000])

            print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

            kde = grid.best_estimator_
            log_density = kde.score_samples(data)

            log_density_tile = np.tile(log_density[:, None], (1, C))
            D[:, :, k] = np.exp(log_density_tile)

        elif estimate_prob_type == "OURS-STD-SCORE" or estimate_prob_type == "OURS-STD-SCORE-WITH-KDE":
            log_p_mean, log_p_std = normalize_factors[source_domain]
            standard_score = (D[:, :, k] - log_p_mean.item()) / log_p_std.item()
            # D[:, :, k] = np.exp(-np.abs(standard_score))
            D[:, :, k] = np.exp(-np.abs(D[:, :, k]))
            if estimate_prob_type == "OURS-STD-SCORE-WITH-KDE":
                params = {"bandwidth": np.logspace(-2, 2, 40)}
                grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)

                data = standard_score
                shuffled_indices = np.random.permutation(len(data))  # return a permutation of the indices
                data_shuffle = data[shuffled_indices]
                grid.fit(data_shuffle[:2000])

                print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

                kde = grid.best_estimator_
                log_density = kde.score_samples(data)

                log_density_tile = np.tile(log_density[:, None], (1, C))
                D[:, :, k] = np.exp(log_density_tile)


        # Make distribution
        D[:, :, k] = D[:, :, k] / D[:, :, k].sum()

    return Y, D, H


def build_DP_model(data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors,
                   estimate_prob_type):
    C = 10  # num_classes
    Y = np.zeros((data_size))
    D = np.zeros((data_size, len(source_domains)))
    H = np.zeros((data_size, len(source_domains)))

    all_output = np.zeros((data_size, C, len(source_domains)))

    i = 0
    precentage = int((i / data_size) * 100)
    print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    for target_domain, data_loader in data_loaders:

        for data, label in data_loader:

            data = data.to(device)
            N = len(data)

            y_vals = label.cpu().detach().numpy()
            Y[i:i + N] = y_vals

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    # Calculate h(x)
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    y_pred = norm_output.data.max(1, keepdim=True)[1]
                    y_pred = y_pred.flatten().cpu().detach().numpy()
                    H[i:i + N, k] = y_pred

                    # Calculate D(x)
                    if estimate_prob_type == "GMSA":
                        all_output[i:i + N, :, k] = output.cpu().detach().numpy()
                    elif estimate_prob_type == "OURS-STD-SCORE" or estimate_prob_type == "OURS-KDE":
                        # calculate log_p
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_bernoulli(x_hat,
                                                                                          data.view(data.shape[0], -1))
                        D[i:i + N, k] = log_p.cpu().detach().numpy()

            i += N

        precentage = int((i / data_size) * 100)
        print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type == "GMSA" or estimate_prob_type == "OURS-KDE":
            # use grid search cross-validation to optimize the bandwidth
            if estimate_prob_type == "GMSA":
                data = all_output[:, :, k]
            elif estimate_prob_type == "OURS-KDE":
                data = D[:, k]
                data = data[:, None]

            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
            shuffled_indices = np.random.permutation(len(data))  # return a permutation of the indices
            data_shuffle = data[shuffled_indices]
            grid.fit(data_shuffle[:2000])

            print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

            kde = grid.best_estimator_
            log_density = kde.score_samples(data)

            D[:, k] = np.exp(log_density)

        elif estimate_prob_type == "OURS-STD-SCORE":
            log_p_mean, log_p_std = normalize_factors[source_domain]
            standard_score = (D[:, k] - log_p_mean.item()) / log_p_std.item()
            D[:, k] = np.exp(-np.abs(standard_score))

        # Make distribution
        D[:, k] = D[:, k] / D[:, k].sum()

    return Y, D, H


def DC_programming(seed, models, classifiers, source_domains, test_path, normalize_factors, estimate_prob_type,
                   init_z_method, multi_dim):
    ''' Calculate the distribution and hypothesis of the data (over the target data) '''
    logging.info("============== Build domain adaptation model ===================")

    data_size = 0
    data_loaders = []
    for k, domain in enumerate(source_domains):
        # if domain == "SVHN":
        #     train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=10000)
        # elif domain == "MNIST":
        #     train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=1000)
        # else:
        #     train_loader, _ = Data.get_data_loaders(domain, seed=seed)
        train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=1000)
        data_size += len(train_loader.dataset)
        data_loaders.append((domain, train_loader))

    if multi_dim:
        Y, D, H = build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path,
                                         normalize_factors, estimate_prob_type)
    else:
        Y, D, H = build_DP_model(data_loaders, data_size, source_domains, models, classifiers, test_path,
                                 normalize_factors, estimate_prob_type)

    DP = init_problem_from_model(Y, D, H, p=len(source_domains), C=10)
    prob = ConvexConcaveProblem(DP)
    solver = ConvexConcaveSolver(prob, seed, init_z_method)
    z_iter, o_iter, err_iter = solver.solve()

    return z_iter


def test_DC_model(seed, models, classifiers, source_domains, target_domains, learned_z, test_path,
                  normalize_factors, estimate_prob_type, multi_dim):
    data_size = 0
    data_loaders = []
    for domain in target_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=seed)
        data_size += len(test_loader.dataset)
        data_loaders.append((domain, test_loader))

    if multi_dim:
        Y, D, H = build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path,
                                         normalize_factors, estimate_prob_type)
    else:
        Y, D, H = build_DP_model(data_loaders, data_size, source_domains, models, classifiers, test_path,
                                 normalize_factors, estimate_prob_type)

    DP = init_problem_from_model(Y, D, H, p=len(source_domains), C=10)
    prob = ConvexConcaveProblem(DP)
    _, _, _, hz = prob.compute_DzJzKzhz(learned_z)

    if multi_dim:
        Y = Y.argmax(axis=1)
        hz = hz.argmax(axis=1)

    print("Hz : ", hz[:20])
    print("Y : ", Y[:20])

    print("\n============== Score : Multiple Domain Adaptation ===================")
    logging.info("============== Score : Multiple Domain Adaptation ===================")

    score = accuracy_score(y_true=Y, y_pred=hz)
    print(score)
    logging.info("score = {}".format(score))

    return score


def run_domain_adaptation(alpha_pos, alpha_neg, vr_model_type, seed, test_path, estimate_prob_type, init_z_method,
                          multi_dim, classifiers, source_domains):
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
    probabilities = torch.tensor([]).to(device)
    normalize_factors = {'MNIST': (0, 0), 'USPS': (0, 0), 'SVHN': (0, 0)}

    for domain in source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)
        model.load_state_dict(torch.load("./models_new/{}_{}_{}_{}_model.pt".format(
            vr_model_type, alpha_pos, alpha_neg, domain), map_location=torch.device(device)))
        models[domain] = model

        if estimate_prob_type == "OURS-STD-SCORE" or estimate_prob_type == "OURS-STD-SCORE-WITH-KDE":
            train_loader, test_loader = Data.get_data_loaders(domain, seed=seed)

            for data, _ in train_loader:
                with torch.no_grad():
                    data = data.to(device)

                    x_hat, mu, logstd = models[domain](data)
                    log_p = models[domain].compute_log_probabitility_bernoulli(x_hat, data.view(data.shape[0], -1))
                    probabilities = torch.cat((log_p, probabilities), 0)

            normalize_factors[domain] = (probabilities.mean(), probabilities.std())


    learned_z = DC_programming(seed, models, classifiers, source_domains, test_path,
                               normalize_factors, estimate_prob_type, init_z_method, multi_dim)
    cur_z_factors = {}
    cur_z_factors['MNIST'] = learned_z[0]
    cur_z_factors['USPS'] = learned_z[1]
    cur_z_factors['SVHN'] = learned_z[2]

    with open(test_path + r'/DC_accuracy_score_{}.txt'.format(seed), 'w') as fp:
        fp.write('\nz_MNIST = {}\tz_USPS = {}\tz_SVHN = {}\n'.format(learned_z[0], learned_z[1], learned_z[2]))

        for target_domains in target_domains_sets:
            print(target_domains)

            score = test_DC_model(seed, models, classifiers, source_domains, target_domains, learned_z, test_path,
                                  normalize_factors, estimate_prob_type, multi_dim)
            logging.info("")
            target_domains_score = target_domains
            target_domains_score.append(str(score * 100))
            target_domains_score.append("\n")
            fp.write('\t'.join(target_domains_score))


def task_run(date, seed, estimate_prob_type, init_z_method, multi_dim,
             model_type, pos_alpha, neg_alpha,
             classifiers, source_domains):

    test_path = (
        f'./{estimate_prob_type}_results_{date}/'
        f'init_z_{init_z_method}/use_multi_dim_{multi_dim}/seed_{seed}/'
        f'model_type_{model_type}___pos_alpha_{pos_alpha}___neg_alpha_{neg_alpha}'
    )
    os.makedirs(test_path, exist_ok=True)

    run_domain_adaptation(
        pos_alpha, neg_alpha, model_type, seed, test_path,
        estimate_prob_type, init_z_method, multi_dim,
        classifiers, source_domains
    )


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


#     # estimate_prob_types = ["GMSA", "OURS-KDE", "OURS-STD-SCORE"]
#     estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"]
#     # estimate_prob_types = ["OURS-STD-SCORE"]
#     # init_z_methods = ["err", "obj"]
#     init_z_methods = ["err"]
#     multi_dim_vals = [True]

#     date = '25_2'

#     for seed in [10]:
#         for estimate_prob_type in estimate_prob_types:
#             for init_z_method in init_z_methods:
#                 for multi_dim in multi_dim_vals:
#                     model_type = 'vrs'
#                     # for (pos_alpha, neg_alpha) in [(0.5, -0.5), (2, -2), (0.5, -2), (2, -0.5)]:
#                     for (pos_alpha, neg_alpha) in [(0.5, -0.5)]:

#                         test_path = './{}_results_{}/init_z_{}/use_multi_dim_{}/seed_{}/model_type_{}___pos_alpha_{}___neg_alpha_{}'.format(
#                             estimate_prob_type, date, init_z_method, multi_dim, seed, model_type, pos_alpha, neg_alpha)
#                         os.makedirs(test_path, exist_ok=True)
#                         run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path, estimate_prob_type,
#                                               init_z_method, multi_dim, classifiers, source_domains)

#                     # model_type = 'vr_pos'
#                     # for pos_alpha, neg_alpha in [(0.5, -2), (2, -0.5)]:
#                     #     test_path = './{}_results_{}/init_z_{}/use_multi_dim_{}/seed_{}/model_type_{}___pos_alpha_{}___neg_alpha_{}'.format(
#                     #         estimate_prob_type, date, init_z_method, multi_dim, seed, model_type, pos_alpha, neg_alpha)
#                     #     os.makedirs(test_path, exist_ok=True)
#                     #     run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path, estimate_prob_type,
#                     #                           init_z_method, multi_dim, classifiers, source_domains)
#                     #
#                     # model_type = 'vae'
#                     # pos_alpha = 2
#                     # neg_alpha = -0.5
#                     # test_path = './{}_results_{}/init_z_{}/use_multi_dim_{}/seed_{}/model_type_{}___pos_alpha_{}___neg_alpha_{}'.format(
#                     #     estimate_prob_type, date, init_z_method, multi_dim, seed, model_type, pos_alpha, neg_alpha)
#                     # os.makedirs(test_path, exist_ok=True)
#                     # run_domain_adaptation(pos_alpha, neg_alpha, model_type, seed, test_path, estimate_prob_type,
#                     #                       init_z_method, multi_dim, classifiers, source_domains)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = ", device)

    domains_accuracy_score = []
    classifiers = {}
    source_domains = ['MNIST', 'USPS', 'SVHN']
    for domain in source_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=1)

        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        classifier.load_state_dict(
            torch.load(f"./classifiers_new/{domain}_classifier.pt", map_location=torch.device(device))
        )
        accuracy = ClSFR.test(classifier, test_loader)
        domains_accuracy_score.append(domain + " = " + str(accuracy))
        classifiers[domain] = classifier

    with open('./domain_accuracy_score.txt', 'w') as fp:
        fp.write('\n'.join(domains_accuracy_score))

    estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"] 
    init_z_methods = ["err"]                          
    multi_dim_vals  = [True]                          

    date  = '26_9'
    seeds = [10]                                     

    alphas_by_model = {
        "vrs": [(2, -2), (2, -0.5), (0.5, -2), (0.5, -0.5)],
        "vr":  [(2, -1), (0.5, -1)],                 
        "vae": [(1, -1)],                       
    }

    tasks = []
    for seed in seeds:
        for estimate_prob_type in estimate_prob_types:
            for init_z_method in init_z_methods:
                for multi_dim in multi_dim_vals:
                    for model_type, pairs in alphas_by_model.items():
                        for (pos_alpha, neg_alpha) in pairs:
                            tasks.append((
                                date, seed, estimate_prob_type, init_z_method, multi_dim,
                                model_type, pos_alpha, neg_alpha
                            ))

    Parallel(n_jobs=os.cpu_count(), backend="loky")(
        delayed(task_run)(*t, classifiers, source_domains) for t in tasks
    )


if __name__ == "__main__":
    main()
