
import argparse
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
import os

from sklearn import metrics
from sklearn.model_selection import KFold
from scipy import interpolate

from retrieval_model import FOP

def read_data(ver, test_file_face, test_file_voice):
    if FLAGS.debug_prints:
        print('Reading Test Face')
    face_test = pd.read_csv(test_file_face, header=None)
    if FLAGS.debug_prints:
        print('Reading Test Voice') 
    voice_test = pd.read_csv(test_file_voice, header=None)
    
    face_test = np.asarray(face_test)
    face_test = face_test[:, :4096]
    voice_test = np.asarray(voice_test)
    voice_test = voice_test[:, :512]
    
    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    return face_test, voice_test

def test(ver, heard_lang, unheard_lang, face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard, compute_server_scores: bool =False):
    
    n_class = 64 if ver == 'v1' else 78
    model = FOP(FLAGS.cuda, FLAGS.fusion, FLAGS.dim_embed, FLAGS.mid_att_dim, face_test_heard.shape[1], voice_test_heard.shape[1], n_class)
    ckpt_path = f"./models/{ver}/{heard_lang}/best_checkpoint.pth.tar"
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint '{ckpt_path}' (epoch {checkpoint['epoch']})")
    model.eval()
    model.cuda()
    
    results = {"heard": {}, "unheard": {}}

    if FLAGS.cuda:
        face_test_heard, voice_test_heard = face_test_heard.cuda(), voice_test_heard.cuda()
        face_test_unheard, voice_test_unheard = face_test_unheard.cuda(), voice_test_unheard.cuda()

    face_test_heard, voice_test_heard = Variable(face_test_heard), Variable(voice_test_heard)
    face_test_unheard, voice_test_unheard = Variable(face_test_unheard), Variable(voice_test_unheard)
    
    with torch.no_grad():
        _, face_heard, voice_heard = model(face_test_heard, voice_test_heard)
        _, face_unheard, voice_unheard = model(face_test_unheard, voice_test_unheard)
                
        face_heard, voice_heard = face_heard.data, voice_heard.data
        face_unheard, voice_unheard = face_unheard.data, voice_unheard.data
        
        face_heard, voice_heard = face_heard.cpu().detach().numpy(), voice_heard.cpu().detach().numpy()
        face_unheard, voice_unheard = face_unheard.cpu().detach().numpy(), voice_unheard.cpu().detach().numpy()
        
        print('*'*30)
        print("Evaluation on heard language")
        print('-'*30)
        results["heard"]["ACC"], results["heard"]["AUC"], results["heard"]["ERR"] = eval_metrics(face_heard, voice_heard)
        print('*'*30)
        print("Evaluation on unheard language")
        print('-'*30)
        results["unheard"]["ACC"], results["unheard"]["AUC"], results["unheard"]["ERR"] = eval_metrics(face_unheard, voice_unheard)
        

        if compute_server_scores:
            print('Computing L2 scores for server submission:')
            scores_heard = np.linalg.norm(face_heard - voice_heard, axis=1, keepdims=True)
            scores_unheard = np.linalg.norm(face_unheard - voice_unheard, axis=1, keepdims=True)
            
            if FLAGS.debug_prints:
                print('Writing scores to files:')
            keys_heard = []
            keys_unheard = []
            
            with open(f"./face_voice_association_splits/{ver}/{heard_lang}_test.txt", 'r+') as f:
                for dat in f:
                    keys_heard.append(dat.split(' ')[0])
                
            with open(f"./face_voice_association_splits/{ver}/{unheard_lang}_test.txt", 'r+') as f:
                for dat in f:
                    keys_unheard.append(dat.split(' ')[0])
            
            if FLAGS.debug_prints:
                assert len(scores_heard) == len(keys_heard), f"Number of heard scores {len(scores_heard)} != number of heard samples in the split {len(keys_heard)}"
                assert len(scores_unheard) == len(keys_unheard), f"Number of unheard scores {len(scores_unheard)} != number of unheard samples in the split {len(keys_unheard)}"

            with open(f"./scores/sub_score_{ver}_{heard_lang}_heard.txt", 'w') as f:
                for i, dat in enumerate(scores_heard):
                    f.write(f"{keys_heard[i]} {dat}")
                print(f"Updated ./scores/sub_score_{ver}_{heard_lang}_heard.txt")
                    
            with open(f"./scores/sub_score_{ver}_{heard_lang}_unheard.txt", 'w') as f:
                for i, dat in enumerate(scores_unheard):
                    f.write(f"{keys_unheard[i]} {dat}")
                print(f"Updated ./scores/sub_score_{ver}_{heard_lang}_unheard.txt")

    return results

def eval_metrics(face, voice):
    feat_list = []
    
    for idx, sfeat in enumerate(face):
        feat_list.append(voice[idx])
        feat_list.append(sfeat)

    print('Total Number of Samples: ', len(feat_list))

    issame_lst = same_func(feat_list)
    feat_list = np.asarray(feat_list)

    tpr, fpr, accuracy, val, val_std, far = cross_eval(feat_list, issame_lst, 10)

    print(f"Accuracy: {np.mean(accuracy):.3f}+-{np.std(accuracy):.3f}")

    auc = metrics.auc(fpr, tpr)
    print(f"Area Under Curve (AUC):{auc: .3f}")
    fnr = 1-tpr
    abs_diffs = np.abs(fpr-fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(f"Equal Error Rate (EER): {eer:.3f}")

    return f"{np.mean(accuracy):.3f}+-{np.std(accuracy):.3f}", f"{auc: .3f}", f"{eer: .3f}"

def same_func(f):
    issame_lst = []
    for idx in range(len(f)):
        if idx % 2 == 0:
            issame = True
        else:
            issame = False
        issame_lst.append(issame)
    return issame_lst

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def cross_eval(embeddings, actual_issame, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    
    return tpr, fpr, accuracy, val, val_std, far

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to', type=str, help='Path to txt file for storing results')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--mid_att_dim', type=int, default=128,
                        help='Used only in case of gated fusion, it is Intermediate Embedding Size (Inside Attention Algorithm)')
    
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    parser.add_argument('--version', type=str, default='v1', help='Possible values: "v1" or "v2"')
    parser.add_argument('--heard_lang', type=str, default='English', help='Possible values: "English", "Hindi"(for v2), "Urdu"(for v1)')
    parser.add_argument('--all_langs', action='store_true', default=False, help='Running all possible language combinations')
    parser.add_argument('--compute_server_scores', action='store_true', default=False, help='Computing L2 scores for server submission')
    parser.add_argument('--debug_prints', action='store_true', default=False, help='Printing extra info helpful for debugging')
    
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.cuda = torch.cuda.is_available()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    if FLAGS.all_langs:
        vers = ['v1', 'v1', 'v2', 'v2']
        heard_langs = ['English', 'Urdu', 'English', 'Hindi']
    else:
        vers = [FLAGS.version]
        heard_langs = [FLAGS.heard_lang]

    results_dictionary = {
                        'v1':
                            {'English': {}, 'Urdu': {}},
                        'v2': 
                            {'English': {}, 'Hindi':{}}
                        }

    for i in range(len(vers)):
        ver = vers[i]
        heard_lang = heard_langs[i]

        if (ver == 'v1' and heard_lang == 'Hindi') or (ver == 'v2' and heard_lang == 'Urdu'):
            raise ValueError(f"Contradictory combination: ver='{ver}' and heard_lang='{heard_lang}'")

        assert ver == 'v1' or ver == 'v2', f"Invalid value for ver: {ver}"
        assert heard_lang == 'Urdu' or heard_lang == 'Hindi' or heard_lang == 'English', f"Invalid value for heard_lang: {heard_lang}"

        if ver == 'v1':
            unheard_lang = 'Urdu' if heard_lang == 'English' else 'English'
        if ver == 'v2':
            unheard_lang = 'Hindi' if heard_lang == 'English' else 'English'
        
        
        print(f"Heard_Language: {heard_lang}")
        print(f"Unheard Language: {unheard_lang}")
        print("-"*30)

        if FLAGS.debug_prints:
            print('Loading Heard Language Data')
        test_file_face = f"./pre_extracted_features/{ver}/{heard_lang}/{heard_lang}_faces_test.csv"
        test_file_voice = f"./pre_extracted_features/{ver}/{heard_lang}/{heard_lang}_voices_test.csv"
        face_test_heard, voice_test_heard = read_data(ver, test_file_face, test_file_voice)
        
        if FLAGS.debug_prints:
            print('Loading Unheard Language Data')
        test_file_face = f"./pre_extracted_features/{ver}/{heard_lang}/{unheard_lang}_faces_unheard_test.csv"
        test_file_voice = f"./pre_extracted_features/{ver}/{heard_lang}/{unheard_lang}_voices_unheard_test.csv"
        face_test_unheard, voice_test_unheard = read_data(ver, test_file_face, test_file_voice)
        
        results_dictionary[ver][heard_lang] = test(ver,
                                                   heard_lang, 
                                                   unheard_lang,
                                                   face_test_heard,
                                                   voice_test_heard,
                                                   face_test_unheard,
                                                   voice_test_unheard,
                                                   FLAGS.compute_server_scores)
        print("="*30)

    if FLAGS.save_to:
        results_path = FLAGS.save_to
    else:
        results_path = "_".join(["results", FLAGS.fusion, str(FLAGS.dim_embed), str(FLAGS.mid_att_dim)])+".txt"
    
    if not results_path.startswith("./results/"):
        results_path = os.path.join("results", results_path)

    if not results_path.endswith(".txt"):
        raise ValueError("Results are stored in .txt file, please provide corresponding path")

    print(f"Saving results into {results_path}")

    with open(results_path, 'w') as f:
        for metric in ["ACC", "AUC", "ERR"]:
            metric_full_name = "Accuracy (mean +- SD)"
            if metric == "AUC":
                metric_full_name = "AUC (Area Under the Curve)"
            elif metric == "ERR":
                metric_full_name = "ERR (Equal Error Rate)"
            print(f"{'***** ' + metric_full_name + ' *****': ^50}", file=f)
            print("+"*50, file=f)
            print(f"|{' '*16}{'English test': ^16}|{'Urdu test': ^15}|", file=f)
            print("-"*50, file=f)
            print(f"|{'English train': ^16}|{results_dictionary['v1']['English']['heard'][metric]: ^15}|{results_dictionary['v1']['English']['unheard'][metric]:^15}|", file=f)
            print("-"*50, file=f)
            print(f"|{'Urdu train': ^16}|{results_dictionary['v1']['Urdu']['unheard'][metric]: ^15}|{results_dictionary['v1']['Urdu']['heard'][metric]:^15}|", file=f)
            print("-"*50, file=f)
            print("+"*50, file=f)
            print(f"|{' '*16}{'English test': ^16}|{'Hindi test':^15}|", file=f)
            print("-"*50, file=f)
            print(f"|{'English train': ^16}|{results_dictionary['v2']['English']['heard'][metric]:^15}|{results_dictionary['v2']['English']['unheard'][metric]:^15}|", file=f)
            print("-"*50, file=f)
            print(f"|{'Hindi train': ^16}|{results_dictionary['v2']['Hindi']['unheard'][metric]:^15}|{results_dictionary['v2']['Hindi']['heard'][metric]:^15}|", file=f)
            print("-"*50, file=f)