
import argparse
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from retrieval_model import FOP

def read_data(ver, test_file_face, test_file_voice):
    print('Reading Test Face')
    face_test = pd.read_csv(test_file_face, header=None)
    print('Reading Test Voice')
    voice_test = pd.read_csv(test_file_voice, header=None)
    
    face_test = np.asarray(face_test)
    face_test = face_test[:, :4096]
    voice_test = np.asarray(voice_test)
    voice_test = voice_test[:, :512]
    
    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    return face_test, voice_test

def test(ver, heard_lang, unheard_lang, face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard):
    
    n_class = 64 if ver == 'v1' else 78
    model = FOP(FLAGS.cuda, FLAGS.fusion, FLAGS.dim_embed, face_test_heard.shape[1], voice_test_heard.shape[1], n_class)
    ckpt_path = f"./models/{ver}/{heard_lang}/best_checkpoint.pth.tar"
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint '{ckpt_path}' (epoch {checkpoint['epoch']})")
    model.eval()
    model.cuda()
    
    if FLAGS.cuda:
        face_test_heard, voice_test_heard = face_test_heard.cuda(), voice_test_heard.cuda()
        face_test_unheard, voice_test_unheard = face_test_unheard.cuda(), voice_test_unheard.cuda()

    face_test_heard, voice_test_heard = Variable(face_test_heard), Variable(voice_test_heard)
    face_test_unheard, voice_test_unheard = Variable(face_test_unheard), Variable(voice_test_unheard)
    print('Computing scores')
    with torch.no_grad():
        _, face_heard, voice_heard = model(face_test_heard, voice_test_heard)
        _, face_unheard, voice_unheard = model(face_test_unheard, voice_test_unheard)
                
        face_heard, voice_heard = face_heard.data, voice_heard.data
        face_unheard, voice_unheard = face_unheard.data, voice_unheard.data
        
        face_heard, voice_heard = face_heard.cpu().detach().numpy(), voice_heard.cpu().detach().numpy()
        face_unheard, voice_unheard = face_unheard.cpu().detach().numpy(), voice_unheard.cpu().detach().numpy()
        
        scores_heard = np.linalg.norm(face_heard - voice_heard, axis=1, keepdims=True)
        scores_unheard = np.linalg.norm(face_unheard - voice_unheard, axis=1, keepdims=True)
        
        print('Writing scores to files')
        
        keys_heard = []
        keys_unheard = []
        with open(f"./face_voice_association_splits/{ver}/{heard_lang}_test.txt", 'r+') as f:
            for dat in f:
                keys_heard.append(dat.split(' ')[0])
                
        with open(f"./face_voice_association_splits/{ver}/{unheard_lang}_test.txt", 'r+') as f:
            for dat in f:
                keys_unheard.append(dat.split(' ')[0])
        
        with open(f"./scores/sub_score_{ver}_{heard_lang}_heard.txt", 'w') as f:
            for i, dat in enumerate(scores_heard):
                f.write(f"{keys_unheard[i]} {dat}")
            print(f"Updated ./scores/sub_score_{ver}_{heard_lang}_heard.txt")
                
        with open(f"./scores/sub_score_{ver}_{heard_lang}_unheard.txt", 'w') as f:
            for i, dat in enumerate(scores_unheard):
                f.write(f"{keys_unheard[i]} {dat}")
            print(f"Updated ./scores/sub_score_{ver}_{heard_lang}_unheard.txt")
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    parser.add_argument('--version', type=str, default='v1', help='Possible values: "v1" or "v2"')
    parser.add_argument('--heard_lang', type=str, default='English', help='Possible values: "English", "Hindi"(for v2), "Urdu"(for v1)')
    parser.add_argument('--all_langs', action='store_true', default=False, help='Running all possible language combinations')
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

    for i in range(len(vers)):
        ver = vers[i]
        heard_lang = heard_langs[i]

        if (ver == 'v1' and heard_lang == 'Hindi') or (ver == 'v2' and heard_lang == 'Urdu'):
            raise ValueError(f"Contradictory combination: {ver=} and {heard_lang=}")

        assert ver == 'v1' or ver == 'v2', f"Invalid value for ver: {ver}"
        assert heard_lang == 'Urdu' or heard_lang == 'Hindi' or heard_lang == 'English', f"Invalid value for heard_lang: {heard_lang}"

        if ver == 'v1':
            unheard_lang = 'Urdu' if heard_lang == 'English' else 'English'
        if ver == 'v2':
            unheard_lang = 'Hindi' if heard_lang == 'English' else 'English'
        
        print("="*30)
        
        print(f"Heard_Language: {heard_lang}")
        print(f"Unheard Language: {unheard_lang}")
        print("-"*30)

        print('Loading Heard Language Data')
        test_file_face = f"./preExtracted_vggFace_utteranceLevel_Features/{ver}/{heard_lang}/{heard_lang}_faces_test.csv"
        test_file_voice = f"./preExtracted_vggFace_utteranceLevel_Features/{ver}/{heard_lang}/{heard_lang}_voices_test.csv"
        face_test_heard, voice_test_heard = read_data(ver, test_file_face, test_file_voice)
        print('Loading Unheard Language Data')
        test_file_face = f"./preExtracted_vggFace_utteranceLevel_Features/{ver}/{heard_lang}/{heard_lang}_faces_unheard_test.csv"
        test_file_voice = f"./preExtracted_vggFace_utteranceLevel_Features/{ver}/{heard_lang}/{heard_lang}_voices_unheard_test.csv"
        face_test_unheard, voice_test_unheard = read_data(ver, test_file_face, test_file_voice)
        test(ver, heard_lang, unheard_lang, face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard)
        
        print("="*30)
