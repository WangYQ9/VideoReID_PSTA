from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

from torch.utils.data import DataLoader

import data_manager
from samplers import RandomIdentitySampler, RandomIdentitySamplerStrongBasaline, RandomIdentitySamplerV2
from video_loader import VideoDataset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model_analysis import MA

from lr_schedulers import WarmupMultiStepLR
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, CosineTripletLoss
from utils import AverageMeter, Logger, EMA, make_optimizer, DeepSupervision
from eval_metrics import evaluate_reranking
from config import cfg
from torch.optim import lr_scheduler

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="ReID Baseline Testing")
parser.add_argument("--config_file", default="./configs/softmax_triplet.yml", help="path to config file", type=str)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
parser.add_argument('--arch', type=str, default='STAM', choices=['ResNet50', 'tem_dense', 'STAM'])
parser.add_argument('--model_spatial_pool', type=str, default='avg', choices=['max','avg'], help='how to aggerate spatial feature map')
parser.add_argument('--model_temporal_pool', type=str, default='avg', choices=['max','avg'], help='how to aggerate temporal feaure vector')
parser.add_argument('--train_sampler', type=str, default='Random_interval', help='train sampler', choices=['random','Random_interval','Random_choice', 'Begin_interval'])
parser.add_argument('--test_sampler', type=str, default='Begin_interval', help='test sampler', choices=['dense', 'Begin_interval'])
parser.add_argument('--sampler',type=str,default='RandomIdentitySampler', choices=['RandomIdentitySampler', 'RandomIdentitySamplerStrongBasaline', 'RandomIdentitySamplerV2'])
parser.add_argument('--transform_method', type=str, default='consecutive',choices=['consecutive', 'interval'], help='transform method is tracklet level or frame level')
parser.add_argument('--sampler_method', type=str, default='random', choices=['random', 'fix'])
parser.add_argument('--triplet_distance', type=str, default='cosine', choices=['cosine','euclidean'])
parser.add_argument('--test_distance', type=str, default='cosine', choices=['cosine','euclidean'])
parser.add_argument('--is_cat', type=str, default='yes', choices=['yes','no'], help='gallery set = gallery set + query set')
parser.add_argument('--feature_method', type=str, default='cat', choices=['cat', 'final'])
parser.add_argument('--is_mutual_channel_attention', type=str, default='no', choices=['yes','no'])
parser.add_argument('--is_mutual_spatial_attention', type=str, default='yes', choices=['yes','no'])
parser.add_argument('--is_appearance_channel_attention', type=str, default='no', choices=['yes','no'])
parser.add_argument('--is_appearance_spatial_attention', type=str, default='yes', choices=['yes','no'])
parser.add_argument('--layer_num', type=int, default=3, choices=[1, 2, 3])
parser.add_argument('--seq_len', type=int, default=8, choices=[4, 8])
parser.add_argument('--split_id', type=int, default=0)
parser.add_argument('--is_down_channel', type=str, default='yes', choices=['yes', 'no'])
parser.add_argument('--dataset', type=str, default='mars', choices=['mars','prid','duke','ilidsvid'])

parser.add_argument('--test_path', type=str, default=None)
parser.add_argument('--print_heat', type=str, default='no', choices=['yes', 'no'])
parser.add_argument('--print_performance', type=bool, default=False)
parser.add_argument('--print_rank', type=str, default='no', choices=['yes', 'no'])
parser.add_argument('--print_gram', type=bool, default=False)
parser.add_argument('--layer_name', type=str, default='layer3', choices=['layer1','layer2', 'layer3', 'down_channel'])
parser.add_argument('--TSNE', type=bool, default=False)
parser.add_argument('--device_id', type=str, default='1')

args_ = parser.parse_args()

if args_.config_file != "":
    cfg.merge_from_file(args_.config_file)
cfg.merge_from_list(args_.opts)

tqdm_enable = False

def main():
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    print(cfg.OUTPUT_DIR)
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_test.txt'))

    print("=========================\nConfigs:{}\n=========================".format(cfg))
    s = str(args_).split(", ")
    print("Fine-tuning detail:")
    for i in range(len(s)):
        print(s[i])
    print("=========================")

    if use_gpu:
        print("Currently using GPU {}".format(cfg.MODEL.DEVICE_ID))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(cfg.DATASETS.NAME))

    dataset = data_manager.init_dataset(root=cfg.DATASETS.ROOT_DIR, name=args_.dataset, split_id = args_.split_id)
    print("Initializing model: {}".format(cfg.MODEL.NAME))

    model = models.init_model(name=args_.arch, num_classes=dataset.num_train_pids, pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
                              model_name=cfg.MODEL.NAME, seq_len = args_.seq_len,
                              layer_num=args_.layer_num,
                              is_mutual_channel_attention=args_.is_mutual_channel_attention,
                              is_mutual_spatial_attention=args_.is_mutual_spatial_attention,
                              is_appearance_channel_attention=args_.is_appearance_channel_attention,
                              is_appearance_spatial_attention=args_.is_appearance_spatial_attention,
                              is_down_channel = args_.is_down_channel,
                              )

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    transform_test = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pin_memory = True if use_gpu else False

    if args_.sampler == 'RandomIdentitySampler':
        video_sampler = RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE)
    elif args_.sampler == 'RandomIdentitySamplerStrongBasaline':
        video_sampler = RandomIdentitySamplerStrongBasaline(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE)
    elif args_.sampler == 'RandomIdentitySampler':
        video_sampler = RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE)

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args_.seq_len, sample=args_.train_sampler, transform=transform_test,
                     dataset_name=cfg.DATASETS.NAME, transform_method=args_.transform_method,
                     sampler_method=args_.sampler_method),
        sampler=video_sampler, shuffle=False,
        batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=True
    )

    if args_.print_gram:
        batchsize = 1
    else:
        batchsize = cfg.TEST.SEQS_PER_BATCH

    if args_.test_sampler == 'dense':
        print('Build dense sampler')
        queryloader = DataLoader(
            VideoDataset(dataset.query, seq_len=args_.seq_len, sample=args_.test_sampler, transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=1 , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False
        )

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, seq_len=args_.seq_len, sample=args_.test_sampler, transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=1 , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False,
        )
    else:
        queryloader = DataLoader(
            VideoDataset(dataset.query, seq_len=args_.seq_len, sample=args_.test_sampler,
                         transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=batchsize, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False
        )

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, seq_len=args_.seq_len, sample=args_.test_sampler,
                         transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=batchsize, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False,
        )

    model = nn.DataParallel(model)
    model.cuda()

    start_time = time.time()

    print("Loading checkpoint from '{}'".format(args_.test_path))
    model_analysis = MA(cfg.OUTPUT_DIR)

    print("load model... ")
    checkpoint = torch.load(args_.test_path)
    model.load_state_dict(checkpoint)

    print("Evaluate...")
    if args_.print_performance :
        q_g_dist = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu, cfg.DATASETS.NAME)

    if args_.print_heat == 'yes':
        print("print heat map!")
        if args_.anaysis_query == 'yes':
            model_analysis.heat_map(queryloader, model, 'query', args_.dataset)
        elif args_.anaysis_gallery == 'yes':
            model_analysis.heat_map(galleryloader, model, 'gallery')

    if args_.print_rank == 'yes':
        print("Build rank images!")
        model_analysis.visualize_ranked_results(q_g_dist, dataset, data_type="video")

    if args_.print_gram :
        print("Build gram picture!")
        model_analysis.Gram(model=model.module, loader = trainloader, layer_name = args_.layer_name)

    if args_.TSNE :
        pass

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def test(model, queryloader, galleryloader, pool, use_gpu, dataset, ranks=[1, 5, 10, 20]):

    with torch.no_grad():
        model.eval()
        qf, q_pids, q_camids = [], [], []
        query_pathes = []
        for batch_idx, (imgs, pids, camids, img_path) in enumerate(tqdm(queryloader)):
            query_pathes.append(img_path[0])
            del img_path
            if use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                camids = camids.cuda()

            if len(imgs.size()) == 6:
                method = 'dense'
                b, n, s, c, h, w = imgs.size()
                assert (b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
            else:
                method = None

            features, pids, camids = model(imgs, pids, camids)
            q_pids.extend(pids.data.cpu())
            q_camids.extend(camids.data.cpu())

            features = features.data.cpu()
            torch.cuda.empty_cache()
            features = features.view(-1, features.size(1))

            if method == 'dense':
                features = torch.mean(features, 0, keepdim=True)

            qf.append(features)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gallery_pathes = []
        for batch_idx, (imgs, pids, camids, img_path) in enumerate(tqdm(galleryloader)):
            gallery_pathes.append(img_path[0])
            if use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                camids = camids.cuda()

            if len(imgs.size()) == 6:
                method = 'dense'
                b, n, s, c, h, w = imgs.size()
                assert (b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
            else:
                method = None

            features, pids, camids = model(imgs, pids, camids)
            features = features.data.cpu()
            torch.cuda.empty_cache()
            features = features.view(-1, features.size(1))

            if method == 'dense':
                features = torch.mean(features, 0, keepdim=True)

            g_pids.extend(pids.data.cpu())
            g_camids.extend(camids.data.cpu())
            gf.append(features)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        if args_.is_cat == 'yes':
            # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
            gf = torch.cat((qf, gf), 0)
            g_pids = np.append(q_pids, g_pids)
            g_camids = np.append(q_camids, g_camids)


        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        cmc, q_g_dist = evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, args_.test_distance)
        return cmc, q_g_dist

if __name__ == '__main__':

    main()