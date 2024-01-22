import argparse
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from loss import OriTripletLoss, CPMLoss, orthogonal_loss
from tensorboardX import SummaryWriter
from random_erasing import RandomErasing

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.cuda.amp import autocast, GradScaler

import logging
import logging.handlers

from utils_new import set_seed_ddp, mySampler, compute_kl_loss, \
    extract_query_feat, extract_gall_feat

import warnings
warnings.filterwarnings('ignore')


SAVE_DIR = '/data1/dyh/results/Refer-VIReID/'

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')

parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--max_epoch', default=80, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--batch_size', default=4, type=int, metavar='B', help='num of identities in each batch')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--use_amp', action='store_true', default=False)

parser.add_argument('--text_mode', default='', type=str)
parser.add_argument('--lambda_3', default=1, type=float)
parser.add_argument('--lambda_4', default=1, type=float)

parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--log_path', default='tmp', type=str, help='log save path')
parser.add_argument('--gpu', default='0,1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--test_batch', default=4, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--trial', default=-1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--lambda_1', default=0.8, type=float, help='lambda_1')
parser.add_argument('--lambda_2', default=0.01, type=float, help='lambda_2')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = 'True'

scaler = GradScaler(enabled=args.use_amp)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data1/dyh/data/SYSU-MM01/SYSU-MM01/'
    log_path = SAVE_DIR + 'SYSU-MM01/' + args.log_path + '/'
    test_mode = [1, 2]  # thermal to visible
    pool_dim = 2048
elif dataset == 'regdb':
    data_path = '/data1/dyh/data/RegDB/'
    log_path = SAVE_DIR + 'RegDB/' + args.log_path + '/'
    test_mode = [1, 2]  # thermal to visible
    pool_dim = 1024
elif dataset == 'llcm':
    data_path = '/data1/dyh/data/LLCM/'
    log_path = SAVE_DIR + 'LLCM/' + args.log_path + '/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    pool_dim = 2048

checkpoint_path = log_path
args.vis_log_path = log_path

os.makedirs(log_path, exist_ok=True)

suffix = dataset + '_deen_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
set_seed_ddp(args.seed, local_rank)
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
# logging
logger = logging.getLogger(__name__)
level = logging.DEBUG if local_rank in [-1, 0] else logging.ERROR
logger.setLevel(level)
formatter = logging.Formatter(fmt="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler1 = logging.StreamHandler()
handler1.setLevel(level)
handler1.setFormatter(formatter)
if dataset == 'regdb':
    handler2 = logging.FileHandler(filename='{}{}_trial_{}_os.txt'.format(log_path, suffix, args.trial), mode="w")
else:
    handler2 = logging.FileHandler(filename='{}{}_os.txt'.format(log_path, suffix), mode="w")
handler2.setLevel(level)
handler2.setFormatter(formatter)
logger.addHandler(handler1)
logger.addHandler(handler2)
# tensorboard
vis_log_dir = args.vis_log_path + '/'
if local_rank == 0:
    writer = SummaryWriter(vis_log_dir)
else:
    writer = None

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

logger.info("==========\nArgs:{}\n==========".format(args))

best_acc = 0  # best test accuracy
start_epoch = 0

logger.info('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_sysu = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.2, sh = 0.8, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_regdb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_llcm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_sysu)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_regdb)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    MODAL = {1: 'visible', 2: 'thermal'}
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=MODAL[test_mode[1]])
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal=MODAL[test_mode[0]])

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, transform=transform_llcm)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

gallset = TestData(dataset, gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(dataset, query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

logger.info('Dataset {} statistics:'.format(dataset))
logger.info('  ------------------------------')
logger.info('  subset   | # ids | # images')
logger.info('  ------------------------------')
logger.info('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
logger.info('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
logger.info('  ------------------------------')
logger.info('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
logger.info('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
logger.info('  ------------------------------')
logger.info('Data Loading Time:\t {:.3f}'.format(time.time() - end))

logger.info('==> Building model..')
net = embed_net(n_class, dataset, args=args)
net.to(device)

cudnn.benchmark = True

if (len(args.resume) > 0) and (local_rank == 0):
    model_path = args.resume
    if os.path.isfile(model_path):
        logger.info('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        logger.info('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        logger.info('==> no checkpoint found at {}'.format(args.resume))

net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
FLAG = True if args.text_mode else False
net = DDP(
    module=net,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=FLAG,
)

# define loss function
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_cpm= CPMLoss(margin=0.2)

criterion_id.to(device)
criterion_tri.to(device)
criterion_cpm.to(device)

module_lr_factor = dict(
    bottleneck=1,
    classifier=1,
    text_encoder=0,
    text_projection=1,
    visible_2_text=0.1,
    visible_projection=1,
)

special_params = list()
for module_name in module_lr_factor:
    if hasattr(net.module, module_name):
        special_params += list(map(id, getattr(net.module, module_name).parameters()))
base_params = filter(lambda p: id(p) not in special_params, net.module.parameters())

param_lr = [
    {'params': base_params, 'lr': 0.1 * args.lr},
]
for module_name, module_factor in module_lr_factor.items():
    if hasattr(net.module, module_name) and module_factor != 0:
        param_lr += [{
            'params': getattr(net.module, module_name).parameters(),
            'lr': args.lr * module_factor,
        }]

optimizer = optim.SGD(
    param_lr,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True,
)


def adjust_learning_rate(optimizer, epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    assert max_epoch in (150, 80)
    if max_epoch == 150:
        if epoch < 10:
            lr = args.lr * (epoch + 1) / 10
        elif 10 <= epoch < 20:
            lr = args.lr
        elif 20 <= epoch < 80:
            lr = args.lr * 0.1
        elif epoch >= 80:
            lr = args.lr * 0.01
        elif epoch >= 120:
            lr = args.lr * 0.001
    elif max_epoch == 80:
        if epoch < 10:
            lr = args.lr * (epoch + 1) / 10
        elif 10 <= epoch < 20:
            lr = args.lr
        elif 20 <= epoch < 50:
            lr = args.lr * 0.1
        elif epoch >= 50:
            lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch, args.max_epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    cpm_loss = AverageMeter()
    ort_loss = AverageMeter()
    kl_loss = AverageMeter()
    joint_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    total = 0

    net.train()
    end = time.time()

    for batch_idx, batch_data in enumerate(trainloader):

        input1, input2, text, label, input_ids, attention_mask = \
            batch_data.get('img1'), batch_data.get('img2'), batch_data.get('text'), \
            batch_data.get('target'), batch_data.get('input_ids'), batch_data.get('attention_mask')

        label2 = torch.cat((label, label), 0)
        label4 = torch.cat((label, label, label, label), 0)
        label6 = torch.cat((label, label, label, label, label, label), 0)

        input1 = input1.cuda()
        input2 = input2.cuda()
        inputs = {
            'visible_image': input1.cuda(),
            'thermal_image': input2.cuda(),
            'input_ids': input_ids.cuda(),
            'attention_mask': attention_mask.cuda(),
        }

        label2 = label2.cuda()
        label4 = label4.cuda()
        label6 = label6.cuda()
        data_time.update(time.time() - end)

        with autocast(enabled=args.use_amp):

            outputs = net(inputs)

            feat1, out1, txt_feat, v2t_feat, joint_feat, joint_logit = \
                outputs.get('feat'), outputs.get('logit'), \
                outputs.get('txt_feat'), outputs.get('v2t_feat'), \
                outputs.get('joint_feat'), outputs.get('joint_logit'),

            loss_id = criterion_id(out1, label6)

            loss_ort = orthogonal_loss(feat1)

            loss_tri = criterion_tri(feat1, label6)

            ft1, ft2, ft3 = torch.chunk(feat1, 3, 0)
            loss_cpm = (criterion_cpm(torch.cat((ft1, ft2), 0), label4) +
                        criterion_cpm(torch.cat((ft1, ft3), 0), label4)) *  args.lambda_1

            loss_ort = loss_ort *  args.lambda_2

            loss = loss_id + loss_tri + loss_cpm + loss_ort

            if args.text_mode in ('v1', 'v2'):
                if dataset == 'sysu':
                    loss_kl = (
                        compute_kl_loss(v2t_feat, txt_feat, label.cuda()) + \
                        compute_kl_loss(v2t_feat, v2t_feat, label.cuda()) + \
                        compute_kl_loss(txt_feat, txt_feat, label.cuda(), text=text, lambda_iou=1.)
                    ) * args.lambda_3 / 3
                elif dataset == 'regdb':
                    loss_kl = (
                        compute_kl_loss(v2t_feat, txt_feat, label.cuda(), text=text, lambda_iou=1.) + \
                        compute_kl_loss(v2t_feat, v2t_feat, label.cuda(), text=text, lambda_iou=1.) + \
                        compute_kl_loss(txt_feat, txt_feat, label.cuda(), text=text, lambda_iou=1.)
                    ) * args.lambda_3 / 3
                elif dataset == 'llcm':
                    loss_kl = (
                        compute_kl_loss(v2t_feat, txt_feat, label.cuda()) + \
                        compute_kl_loss(v2t_feat, v2t_feat, label.cuda()) + \
                        compute_kl_loss(txt_feat, txt_feat, label.cuda(), text=text, lambda_iou=1.)
                    ) * args.lambda_3 / 3
                loss += loss_kl
            else:
                loss_kl = torch.zeros([]).cuda()

            if args.text_mode in ('v2',):
                loss_joint = (
                    criterion_id(joint_logit, label2) +
                    criterion_tri(joint_feat, label2)
                ) * args.lambda_4 / 2
                loss += loss_joint
            else:
                loss_joint = torch.zeros([]).cuda()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        cpm_loss.update(loss_cpm.item(), 2 * input1.size(0))
        ort_loss.update(loss_ort.item(), 2 * input1.size(0))
        kl_loss.update(loss_kl.item(), input1.size(0))
        joint_loss.update(loss_joint.item(), input1.size(0))
        total += label6.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            logger.info(
                'Epoch: [{}][{}/{}] '
                'Loss:{train_loss.val:.3f} '
                'iLoss:{id_loss.val:.3f} '
                'TLoss:{tri_loss.val:.3f} '
                'CLoss:{cpm_loss.val:.3f} '
                'OLoss:{ort_loss.val:.3f} '
                'Text-KLLoss:{kl_loss.val:.3f} '
                'Text-JointLoss:{joint_loss.val:.3f} '.format(
                    epoch, batch_idx, len(trainloader),
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,
                    cpm_loss=cpm_loss, ort_loss=ort_loss, kl_loss=kl_loss, joint_loss=joint_loss,
                )
            )

    if local_rank == 0:
        writer.add_scalar('Loss/total_loss', train_loss.avg, epoch)
        writer.add_scalar('Loss/id_loss', id_loss.avg, epoch)
        writer.add_scalar('Loss/tri_loss', tri_loss.avg, epoch)
        writer.add_scalar('Loss/cpm_loss', cpm_loss.avg, epoch)
        writer.add_scalar('Loss/ort_loss', ort_loss.avg, epoch)
        writer.add_scalar('Loss/text_kl_loss', kl_loss.avg, epoch)
        writer.add_scalar('Train/lr', current_lr, epoch)


def test(epoch):
    # extract features
    gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6, gall_feat_txt, gall_feat_joint = \
        extract_gall_feat(net, gall_loader, ngall, pool_dim, test_mode, flip=False)

    query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6, query_feat_txt, query_feat_joint = \
        extract_query_feat(net, query_loader, nquery, pool_dim, test_mode, flip=False)

    start = time.time()
    # compute the similarity
    distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
    distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
    distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
    distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
    distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
    distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
    distmat7 = distmat1 + distmat2 + distmat3 + distmat4 + distmat5 + distmat6

    if args.text_mode in ('v1', 'v2'):
        distmat_txt = np.matmul(query_feat_txt, np.transpose(gall_feat_txt))
        distmat8 = distmat7 + distmat_txt
    else:
        distmat8 = distmat7

    if args.text_mode in ('v2',):
        dist_joint = np.matmul(query_feat_joint, np.transpose(gall_feat_joint))
        distmat9 = dist_joint
        if dataset == 'sysu':
            distmat10 = distmat8 + dist_joint
        elif dataset == 'regdb':
            distmat10 = distmat7 + dist_joint
        elif dataset == 'llcm':
            distmat10 = distmat8 + dist_joint
    elif args.text_mode in ('v1',):
        distmat9 = distmat8
        distmat10 = distmat8
    else:
        distmat9 = distmat7
        distmat10 = distmat7

    # evaluation
    if dataset == 'sysu':
        cmc1, mAP1, mINP1 = eval_sysu(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_sysu(-distmat8, query_label, gall_label, query_cam, gall_cam)
        cmc9, mAP9, mINP9 = eval_sysu(-distmat9, query_label, gall_label, query_cam, gall_cam)
        cmc10, mAP10, mINP10 = eval_sysu(-distmat10, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'regdb':
        cmc1, mAP1, mINP1 = eval_regdb(-distmat1, query_label, gall_label)
        cmc2, mAP2, mINP2 = eval_regdb(-distmat2, query_label, gall_label)
        cmc7, mAP7, mINP7 = eval_regdb(-distmat7, query_label, gall_label)
        cmc8, mAP8, mINP8 = eval_regdb(-distmat8, query_label, gall_label)
        cmc9, mAP9, mINP9 = eval_regdb(-distmat9, query_label, gall_label)
        cmc10, mAP10, mINP10 = eval_regdb(-distmat10, query_label, gall_label)
    elif dataset == 'llcm':
        cmc1, mAP1, mINP1 = eval_llcm(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_llcm(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_llcm(-distmat7, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_llcm(-distmat8, query_label, gall_label, query_cam, gall_cam)
        cmc9, mAP9, mINP9 = eval_llcm(-distmat9, query_label, gall_label, query_cam, gall_cam)
        cmc10, mAP10, mINP10 = eval_llcm(-distmat10, query_label, gall_label, query_cam, gall_cam)
    logger.info('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc1, mAP1, mINP1, \
           cmc2, mAP2, mINP2, \
           cmc7, mAP7, mINP7, \
           cmc8, mAP8, mINP8, \
           cmc9, mAP9, mINP9, \
           cmc10, mAP10, mINP10,


# training
logger.info('==> Start Training...')
for epoch in range(start_epoch, args.max_epoch + 1):

    logger.info('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label,
                              color_pos, thermal_pos, args.num_pos, args.batch_size)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    logger.info(epoch)
    logger.info(trainset.cIndex)
    logger.info(trainset.tIndex)

    train_sampler = mySampler(trainset, shuffle=False, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, sampler=train_sampler)
    # trainloader.sampler.set_epoch(epoch)

    # training
    train(epoch)

    if (epoch % 1 == 0) and (local_rank == 0):
        logger.info('Test Epoch: {}'.format(epoch))

        # testing
        cmc1, mAP1, mINP1, \
        cmc2, mAP2, mINP2, \
        cmc7, mAP7, mINP7, \
        cmc8, mAP8, mINP8, \
        cmc9, mAP9, mINP9, \
        cmc10, mAP10, mINP10 = test(epoch)

        # save model
        if cmc10[0] + mAP10 > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc10[0] + mAP10
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc10,
                'mAP': mAP10,
                'mINP': mINP10,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        writer.add_scalar('Test/Rank1', cmc10[0] * 100, epoch)
        writer.add_scalar('Test/mAP', mAP10 * 100, epoch)

        logger.info('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        logger.info('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        logger.info('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        logger.info('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))
        logger.info('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc9[0], cmc9[4], cmc9[9], cmc9[19], mAP9, mINP9))
        logger.info('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc10[0], cmc10[4], cmc10[9], cmc10[19], mAP10, mINP10))
        logger.info('Best Epoch [{}]'.format(best_epoch))