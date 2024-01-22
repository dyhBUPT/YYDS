import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from os.path import join, dirname

from utils_new import extract_query_feat, extract_gall_feat

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')

parser.add_argument('--dataset', default='sysu', help='dataset name: llcm, regdb or sysu]')

parser.add_argument('--text_mode', default='', type=str)

parser.add_argument('--resume', '-r', default='xxx.t', type=str, help='resume from checkpoint')
parser.add_argument('--gpu', default='6', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu') # SYSU-MM01

parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--test_batch', default=32, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str, metavar='m', help='method type: base or awg')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--tvsearch', default=True, help='whether thermal to visible search on RegDB') # RegDB

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data1/dyh/data/SYSU-MM01/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
    pool_dim = 2048
elif dataset =='regdb':
    data_path = '/data1/dyh/data/RegDB/'
    n_class = 206
    test_mode = [1, 2]
    pool_dim = 1024
elif dataset =='llcm':
    data_path = '/data1/dyh/data/LLCM/'
    n_class = 713
    test_mode = [1, 2]
    pool_dim = 2048
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
print('==> Building model..')
net = embed_net(n_class, dataset, args=args)
net.to(device)
net = nn.DataParallel(net)
cudnn.benchmark = True

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(dataset, query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6, query_feat_txt, query_feat_joint = \
        extract_query_feat(net, query_loader, nquery, pool_dim, test_mode)

    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(dataset, gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6, gall_feat_txt, gall_feat_joint = \
            extract_gall_feat(net, trial_gall_loader, ngall, pool_dim, test_mode)

        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
        distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
        distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))

        a = 0.1
        distmat7 = distmat1 + distmat3 + distmat5 + distmat2 + distmat4 + distmat6
        distmat8 = a * (distmat1 + distmat3 + distmat5) + (1 - a) * (distmat2 + distmat4 + distmat6)

        cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_sysu(-distmat8, query_label, gall_label, query_cam, gall_cam)

        if args.text_mode in ('v1', 'v2'):
            b = 1
            distmat_txt = np.matmul(query_feat_txt, np.transpose(gall_feat_txt))
            distmat9 = distmat7 + b * distmat_txt
            cmc9, mAP9, mINP9 = eval_sysu(-distmat9, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat9 = distmat7
            cmc9, mAP9, mINP9 = cmc7, mAP7, mINP7

        if args.text_mode in ('v2',):
            c = 1
            distmat_joint = np.matmul(query_feat_joint, np.transpose(gall_feat_joint))
            distmat10 = distmat_joint
            distmat11 = distmat9 + c * distmat_joint
            cmc10, mAP10, mINP10 = eval_sysu(-distmat10, query_label, gall_label, query_cam, gall_cam)
            cmc11, mAP11, mINP11 = eval_sysu(-distmat11, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat11 = distmat9
            cmc10, mAP10, mINP10 = cmc9, mAP9, mINP9
            cmc11, mAP11, mINP11 = cmc9, mAP9, mINP9

        # save features
        SAVE_DIR = join(dirname(model_path), args.mode)
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(join(SAVE_DIR, 'dist_{}.npy'.format(trial)), distmat11)
        np.save(join(SAVE_DIR, 'gallery_pid_{}.npy'.format(trial)), gall_label)
        np.save(join(SAVE_DIR, 'gallery_cam_{}.npy'.format(trial)), gall_cam)
        np.save(join(SAVE_DIR, 'gallery_feat1_{}.npy'.format(trial)), gall_feat1)
        np.save(join(SAVE_DIR, 'gallery_feat2_{}.npy'.format(trial)), gall_feat2)
        np.save(join(SAVE_DIR, 'gallery_feat3_{}.npy'.format(trial)), gall_feat3)
        np.save(join(SAVE_DIR, 'gallery_feat4_{}.npy'.format(trial)), gall_feat4)
        np.save(join(SAVE_DIR, 'gallery_feat5_{}.npy'.format(trial)), gall_feat5)
        np.save(join(SAVE_DIR, 'gallery_feat6_{}.npy'.format(trial)), gall_feat6)
        np.save(join(SAVE_DIR, 'gallery_feat-txt_{}.npy'.format(trial)), gall_feat_txt)
        np.save(join(SAVE_DIR, 'gallery_feat-joint_{}.npy'.format(trial)), gall_feat_joint)
        with open(join(SAVE_DIR, 'gallery_path_{}.txt'.format(trial)), 'w') as f:
            for path in gall_img:
                f.write(path + '\n')
        if trial == 0:
            np.save(join(SAVE_DIR, 'query_pid.npy'), query_label)
            np.save(join(SAVE_DIR, 'query_cam.npy'), query_cam)
            np.save(join(SAVE_DIR, 'query_feat1.npy'), query_feat1)
            np.save(join(SAVE_DIR, 'query_feat2.npy'), query_feat2)
            np.save(join(SAVE_DIR, 'query_feat3.npy'), query_feat3)
            np.save(join(SAVE_DIR, 'query_feat4.npy'), query_feat4)
            np.save(join(SAVE_DIR, 'query_feat5.npy'), query_feat5)
            np.save(join(SAVE_DIR, 'query_feat6.npy'), query_feat6)
            np.save(join(SAVE_DIR, 'query_feat-txt.npy'), query_feat_txt)
            np.save(join(SAVE_DIR, 'query_feat-joint.npy'), query_feat_joint)
            with open(join(SAVE_DIR, 'query_path.txt'), 'w') as f:
                for path in query_img:
                    f.write(path + '\n')

        if trial == 0:
            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

            all_cmc8 = cmc8
            all_mAP8 = mAP8
            all_mINP8 = mINP8

            all_cmc9 = cmc9
            all_mAP9 = mAP9
            all_mINP9 = mINP9

            all_cmc10 = cmc10
            all_mAP10 = mAP10
            all_mINP10 = mINP10

            all_cmc11 = cmc11
            all_mAP11 = mAP11
            all_mINP11 = mINP11

        else:
            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7

            all_cmc8 = all_cmc8 + cmc8
            all_mAP8 = all_mAP8 + mAP8
            all_mINP8 = all_mINP8 + mINP8

            all_cmc9 = all_cmc9 + cmc9
            all_mAP9 = all_mAP9 + mAP9
            all_mINP9 = all_mINP9 + mINP9
            
            all_cmc10 = all_cmc10 + cmc10
            all_mAP10 = all_mAP10 + mAP10
            all_mINP10 = all_mINP10 + mINP10
            
            all_cmc11 = all_cmc11 + cmc11
            all_mAP11 = all_mAP11 + mAP11
            all_mINP11 = all_mINP11 + mINP11

        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc9[0], cmc9[4], cmc9[9], cmc9[19], mAP9, mINP9))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc10[0], cmc10[4], cmc10[9], cmc10[19], mAP10, mINP10))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc11[0], cmc11[4], cmc11[9], cmc11[19], mAP11, mINP11))

elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial + 1
        model_path = join(args.resume, 'regdb_deen_p4_n4_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial))
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
        else:
            print('==> no checkpoint found at {}'.format(model_path))

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        MODAL = {1: 'visible', 2: 'thermal'}
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal=MODAL[test_mode[1]])
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal=MODAL[test_mode[0]])

        gallset = TestData(dataset, gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(dataset, query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


        query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6, query_feat_txt, query_feat_joint = \
            extract_query_feat(net, query_loader, nquery, pool_dim, test_mode, flip=False)
        gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6, gall_feat_txt, gall_feat_joint = \
            extract_gall_feat(net, gall_loader, ngall, pool_dim, test_mode, flip=False)

        assert args.tvsearch
        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
        distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
        distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
        a = 0.1
        distmat7 = distmat1 + distmat3 + distmat5 + distmat2 + distmat4 + distmat6
        distmat8 = a * (distmat1 + distmat3 + distmat5) + (1 - a) * (distmat2 + distmat4 + distmat6)

        cmc7, mAP7, mINP7 = eval_regdb(-distmat7, gall_label, query_label)
        cmc8, mAP8, mINP8 = eval_regdb(-distmat8, gall_label, query_label)

        if args.text_mode in ('v1', 'v2'):
            b = 1
            distmat_txt = np.matmul(query_feat_txt, np.transpose(gall_feat_txt))
            distmat9 = distmat7 + b * distmat_txt
            cmc9, mAP9, mINP9 = eval_regdb(-distmat9, query_label, gall_label)
        else:
            distmat9 = distmat7
            cmc9, mAP9, mINP9 = cmc7, mAP7, mINP7

        if args.text_mode in ('v2',):
            c = 1
            distmat_joint = np.matmul(query_feat_joint, np.transpose(gall_feat_joint))
            distmat10 = distmat_joint
            distmat11 = distmat7 + c * distmat_joint
            cmc10, mAP10, mINP10 = eval_regdb(-distmat10, query_label, gall_label)
            cmc11, mAP11, mINP11 = eval_regdb(-distmat11, query_label, gall_label)
        else:
            distmat11 = distmat9
            cmc10, mAP10, mINP10 = cmc9, mAP9, mINP9
            cmc11, mAP11, mINP11 = cmc9, mAP9, mINP9

        SAVE_DIR = dirname(model_path)
        np.save(join(SAVE_DIR, 'dist_{}.npy'.format(trial)), distmat11)
        np.save(join(SAVE_DIR, 'gallery_pid_{}.npy'.format(trial)), gall_label)
        np.save(join(SAVE_DIR, 'gallery_feat1_{}.npy'.format(trial)), gall_feat1)
        np.save(join(SAVE_DIR, 'gallery_feat2_{}.npy'.format(trial)), gall_feat2)
        np.save(join(SAVE_DIR, 'gallery_feat3_{}.npy'.format(trial)), gall_feat3)
        np.save(join(SAVE_DIR, 'gallery_feat4_{}.npy'.format(trial)), gall_feat4)
        np.save(join(SAVE_DIR, 'gallery_feat5_{}.npy'.format(trial)), gall_feat5)
        np.save(join(SAVE_DIR, 'gallery_feat6_{}.npy'.format(trial)), gall_feat6)
        np.save(join(SAVE_DIR, 'gallery_feat-txt_{}.npy'.format(trial)), gall_feat_txt)
        np.save(join(SAVE_DIR, 'gallery_feat-joint_{}.npy'.format(trial)), gall_feat_joint)
        with open(join(SAVE_DIR, 'gallery_path_{}.txt'.format(trial)), 'w') as f:
            for path in gall_img:
                f.write(path + '\n')
        np.save(join(SAVE_DIR, 'query_pid_{}.npy'.format(trial)), query_label)
        np.save(join(SAVE_DIR, 'query_feat1_{}.npy'.format(trial)), query_feat1)
        np.save(join(SAVE_DIR, 'query_feat2_{}.npy'.format(trial)), query_feat2)
        np.save(join(SAVE_DIR, 'query_feat3_{}.npy'.format(trial)), query_feat3)
        np.save(join(SAVE_DIR, 'query_feat4_{}.npy'.format(trial)), query_feat4)
        np.save(join(SAVE_DIR, 'query_feat5_{}.npy'.format(trial)), query_feat5)
        np.save(join(SAVE_DIR, 'query_feat6_{}.npy'.format(trial)), query_feat6)
        np.save(join(SAVE_DIR, 'query_feat-txt_{}.npy'.format(trial)), query_feat_txt)
        np.save(join(SAVE_DIR, 'query_feat-joint_{}.npy'.format(trial)), query_feat_joint)
        with open(join(SAVE_DIR, 'query_path_{}.txt'.format(trial)), 'w') as f:
            for path in query_img:
                f.write(path + '\n')

        if trial == 0:
            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

            all_cmc8 = cmc8
            all_mAP8 = mAP8
            all_mINP8 = mINP8

            all_cmc9 = cmc9
            all_mAP9 = mAP9
            all_mINP9 = mINP9

            all_cmc10 = cmc10
            all_mAP10 = mAP10
            all_mINP10 = mINP10

            all_cmc11 = cmc11
            all_mAP11 = mAP11
            all_mINP11 = mINP11
        else:
            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7

            all_cmc8 = all_cmc8 + cmc8
            all_mAP8 = all_mAP8 + mAP8
            all_mINP8 = all_mINP7 + mINP8

            all_cmc9 = all_cmc9 + cmc9
            all_mAP9 = all_mAP9 + mAP9
            all_mINP9 = all_mINP9 + mINP9

            all_cmc10 = all_cmc10 + cmc10
            all_mAP10 = all_mAP10 + mAP10
            all_mINP10 = all_mINP10 + mINP10

            all_cmc11 = all_cmc11 + cmc11
            all_mAP11 = all_mAP11 + mAP11
            all_mINP11 = all_mINP11 + mINP11

        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc9[0], cmc9[4], cmc9[9], cmc9[19], mAP9, mINP9))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc10[0], cmc10[4], cmc10[9], cmc10[19], mAP10, mINP10))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc11[0], cmc11[4], cmc11[9], cmc11[19], mAP11, mINP11))

elif dataset == 'llcm':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(dataset, query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6, query_feat_txt, query_feat_joint = \
        extract_query_feat(net, query_loader, nquery, pool_dim, test_mode)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        trial_gallset = TestData(dataset, gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6, gall_feat_txt, gall_feat_joint = \
            extract_gall_feat(net, trial_gall_loader, ngall, pool_dim, test_mode)

        # fc feature
        assert test_mode == [1, 2]
        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
        distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
        distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
        a = 0.1
        distmat7 = distmat1 + distmat3 + distmat5 + distmat2 + distmat4 + distmat6
        distmat8 = a * (distmat1 + distmat3 + distmat5) + (1 - a) * (distmat2 + distmat4 + distmat6)

        cmc7, mAP7, mINP7 = eval_llcm(-distmat7, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_llcm(-distmat8, query_label, gall_label, query_cam, gall_cam)

        if args.text_mode in ('v1', 'v2'):
            b = 1
            distmat_txt = np.matmul(query_feat_txt, np.transpose(gall_feat_txt))
            distmat9 = distmat7 + b * distmat_txt
            cmc9, mAP9, mINP9 = eval_llcm(-distmat9, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat9 = distmat7
            cmc9, mAP9, mINP9 = cmc7, mAP7, mINP7

        if args.text_mode in ('v2',):
            c = 1
            distmat_joint = np.matmul(query_feat_joint, np.transpose(gall_feat_joint))
            distmat10 = distmat_joint
            distmat11 = distmat9 + c * distmat_joint
            cmc10, mAP10, mINP10 = eval_llcm(-distmat10, query_label, gall_label, query_cam, gall_cam)
            cmc11, mAP11, mINP11 = eval_llcm(-distmat11, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat11 = distmat9
            cmc10, mAP10, mINP10 = cmc9, mAP9, mINP9
            cmc11, mAP11, mINP11 = cmc9, mAP9, mINP9

        SAVE_DIR = dirname(model_path)
        np.save(join(SAVE_DIR, 'dist_{}.npy'.format(trial)), distmat11)
        np.save(join(SAVE_DIR, 'gallery_pid_{}.npy'.format(trial)), gall_label)
        np.save(join(SAVE_DIR, 'gallery_cam_{}.npy'.format(trial)), gall_cam)
        np.save(join(SAVE_DIR, 'gallery_feat1_{}.npy'.format(trial)), gall_feat1)
        np.save(join(SAVE_DIR, 'gallery_feat2_{}.npy'.format(trial)), gall_feat2)
        np.save(join(SAVE_DIR, 'gallery_feat3_{}.npy'.format(trial)), gall_feat3)
        np.save(join(SAVE_DIR, 'gallery_feat4_{}.npy'.format(trial)), gall_feat4)
        np.save(join(SAVE_DIR, 'gallery_feat5_{}.npy'.format(trial)), gall_feat5)
        np.save(join(SAVE_DIR, 'gallery_feat6_{}.npy'.format(trial)), gall_feat6)
        np.save(join(SAVE_DIR, 'gallery_feat-txt_{}.npy'.format(trial)), gall_feat_txt)
        np.save(join(SAVE_DIR, 'gallery_feat-joint_{}.npy'.format(trial)), gall_feat_joint)
        with open(join(SAVE_DIR, 'gallery_path_{}.txt'.format(trial)), 'w') as f:
            for path in gall_img:
                f.write(path + '\n')
        if trial == 0:
            np.save(join(SAVE_DIR, 'query_pid.npy'), query_label)
            np.save(join(SAVE_DIR, 'query_cam.npy'), query_cam)
            np.save(join(SAVE_DIR, 'query_feat1.npy'), query_feat1)
            np.save(join(SAVE_DIR, 'query_feat2.npy'), query_feat2)
            np.save(join(SAVE_DIR, 'query_feat3.npy'), query_feat3)
            np.save(join(SAVE_DIR, 'query_feat4.npy'), query_feat4)
            np.save(join(SAVE_DIR, 'query_feat5.npy'), query_feat5)
            np.save(join(SAVE_DIR, 'query_feat6.npy'), query_feat6)
            np.save(join(SAVE_DIR, 'query_feat-txt.npy'), query_feat_txt)
            np.save(join(SAVE_DIR, 'query_feat-joint.npy'), query_feat_joint)
            with open(join(SAVE_DIR, 'query_path.txt'), 'w') as f:
                for path in query_img:
                    f.write(path + '\n')

        if trial == 0:
            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

            all_cmc8 = cmc8
            all_mAP8 = mAP8
            all_mINP8 = mINP8

            all_cmc9 = cmc9
            all_mAP9 = mAP9
            all_mINP9 = mINP9

            all_cmc10 = cmc10
            all_mAP10 = mAP10
            all_mINP10 = mINP10

            all_cmc11 = cmc11
            all_mAP11 = mAP11
            all_mINP11 = mINP11

        else:
            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7

            all_cmc8 = all_cmc8 + cmc8
            all_mAP8 = all_mAP8 + mAP8
            all_mINP8 = all_mINP7 + mINP8

            all_cmc9 = all_cmc9 + cmc9
            all_mAP9 = all_mAP9 + mAP9
            all_mINP9 = all_mINP9 + mINP9

            all_cmc10 = all_cmc10 + cmc10
            all_mAP10 = all_mAP10 + mAP10
            all_mINP10 = all_mINP10 + mINP10

            all_cmc11 = all_cmc11 + cmc11
            all_mAP11 = all_mAP11 + mAP11
            all_mINP11 = all_mINP11 + mINP11

        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc9[0], cmc9[4], cmc9[9], cmc9[19], mAP9, mINP9))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc10[0], cmc10[4], cmc10[9], cmc10[19], mAP10, mINP10))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc11[0], cmc11[4], cmc11[9], cmc11[19], mAP11, mINP11))

cmc7 = all_cmc7 / 10
mAP7 = all_mAP7 / 10
mINP7 = all_mINP7 / 10

cmc8 = all_cmc8 / 10
mAP8 = all_mAP8 / 10
mINP8 = all_mINP8 / 10

cmc9 = all_cmc9 / 10
mAP9 = all_mAP9 / 10
mINP9 = all_mINP9 / 10

cmc10 = all_cmc10 / 10
mAP10 = all_mAP10 / 10
mINP10 = all_mINP10 / 10

cmc11 = all_cmc11 / 10
mAP11 = all_mAP11 / 10
mINP11= all_mINP11 / 10

print('All Average:')
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc9[0], cmc9[4], cmc9[9], cmc9[19], mAP9, mINP9))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc10[0], cmc10[4], cmc10[9], cmc10[19], mAP10, mINP10))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc11[0], cmc11[4], cmc11[9], cmc11[19], mAP11, mINP11))
