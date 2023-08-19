import faiss
from mmcls.datasets.builder import build_dataset
from mmcv.runner.optimizer.builder import OPTIMIZERS
from torch import functional, nn
import torch
from mmcv.utils import Config, build_from_cfg, Registry
import logging
import sys
from mmcls.datasets import BaseDataset, PIPELINES, DATASETS, build_dataloader
from mmcls.datasets.pipelines import Compose
from mmcls.models import BACKBONES, HEADS, NECKS, build_loss
import numpy as np
import os
import argparse
from addict import Dict
import glob
from itertools import chain
import torch.optim.lr_scheduler
import time
import json
import datetime
from mmcv.utils import track_iter_progress
import pickle
from random import sample
from tqdm import tqdm
import math
from pdb import set_trace

def deleted(a:dict, b:str) ->dict:
    x = a.copy()
    del x[b]
    return x


SCHEDULER = Registry('scheduler')
for name in torch.optim.lr_scheduler.__dir__():
    klass = getattr(torch.optim.lr_scheduler, name)
    if type(klass) is not type:
        continue
    SCHEDULER._register_module(klass, name)


@DATASETS.register_module()
class ImageFolder(BaseDataset):
    def __init__(self, root:str, **kwargs):
        assert(os.path.exists(root))
        self.root = root
        if not root.endswith('/'):
            root = root + '/'
        root = root + '*'
        folders = glob.glob(root)
        folders.sort()
        class_map = {k:i for i,k in enumerate(folders)}
        self.records = []
        for f in folders:
            c = class_map[f]
            for file in glob.glob(f'{f}/*'):
                self.records.append(dict(
                    gt_label=c,
                    img_prefix=None,
                    img_info=dict(
                        filename=file,
                    )
                ))
        self.records = sorted(self.records, key=lambda k: k['img_info']['filename'])
        for i, rec in enumerate(self.records):
            rec['index'] = i
        super().__init__(None,  **kwargs)

    def load_annotations(self):
        return self.records


@PIPELINES.register_module()
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = Compose(base_transform)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        result = deleted(q, 'img')
        result.update(dict(
            img=q['img'],
            img_q=q['img'],
            img_k=k['img'],
            gt_label=x['gt_label'],
            meta_q=deleted(q, 'img'),
            meta_k=deleted(k, 'img'),
        ))
        return result

    def __repr__(self):
        return self.__class__.__name__ + self.base_transform.__repr__


class Denoise(nn.Module):
    def __init__(self, cfg:Config, inference_only=False) -> None:
        super().__init__()
        self.cfg = cfg
        # feature dim
        self.device = torch.device(cfg.device)
        self.encoder_q = nn.Sequential(
                build_from_cfg(cfg.backbone, BACKBONES),
                build_from_cfg(cfg.neck, NECKS),
        )
        self.mlp_q = nn.Sequential(
                nn.Linear(self.cfg.backbone_output_dim, self.cfg.backbone_output_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.backbone_output_dim, self.cfg.dim),
        )
        self.encoder_k = nn.Sequential(
                build_from_cfg(cfg.backbone, BACKBONES),
                build_from_cfg(cfg.neck, NECKS),
        )
        self.mlp_k = nn.Sequential(
                nn.Linear(self.cfg.backbone_output_dim, self.cfg.backbone_output_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.backbone_output_dim, self.cfg.dim),
        )
        self.supervised_mlp = torch.nn.Sequential(
            nn.Linear(self.cfg.backbone_output_dim,1000),
        )

        dim = self.cfg.dim
        K = self.cfg.K
        for param_q, param_k in zip(self.param_q, self.param_k):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("epoch", torch.zeros(1, dtype=torch.long))

        # build dataset
        if not inference_only:
            self.train_set = build_from_cfg(cfg.data.train, DATASETS)
            self.test_set = build_from_cfg(cfg.data.test, DATASETS)
            self.train_loader = build_dataloader(
                self.train_set,
                self.cfg.data.batch_size,
                self.cfg.data.workers,
                round_up=True,
                drop_last=True,
                dist=True,
                shuffle=True)
            self.train_eval_loader = build_dataloader(
                build_from_cfg(cfg.data.train_val, DATASETS),
                self.cfg.data.batch_size,
                self.cfg.data.workers,
                round_up=False,
                drop_last=False,
                dist=False,
                shuffle=False)
            self.test_loader = build_dataloader(
                self.test_set,
                8,
                self.cfg.data.workers,
                drop_last=False,
                shuffle=False,
                dist=False)

        opt_cfg =  self.cfg.optimizer.copy()
        opt_cfg['params'] = self.param_q
        self.optimizer = build_from_cfg(opt_cfg, OPTIMIZERS)
        self.criteria = nn.CrossEntropyLoss()
        lr_scheduler_cfg = self.cfg.lr_config.copy()
        lr_scheduler_cfg['optimizer'] = self.optimizer
        self.lr_scheduler = build_from_cfg(lr_scheduler_cfg, SCHEDULER)

        if self.cfg.load is not None and os.path.exists(self.cfg.load):
            print("Loading form ", self.cfg.load)
            static_dict = torch.load(open(self.cfg.load, 'rb'), map_location=self.device)
            print(self.load_state_dict(static_dict, strict=False))
            print('Loaded')
        # move me to the device
        self.to(self.device)

        self._c_hint_eye = torch.eye(self.cfg.n_class, dtype=torch.float32)
        self._c_hint_eye = torch.cat([
            self._c_hint_eye,
            torch.zeros((1, self.cfg.n_class), dtype=torch.float32)],
            dim=0)

        self.json_log_file = open(os.path.join(self.cfg.workdir, 'log.json'), 'w')

        formatter = logging.Formatter(
            fmt="%(levelname)6s [%(filename)15s:%(lineno)-3d %(asctime)s] %(message)s",
            datefmt='%H:%M:%S',
        )
        logger = logging.getLogger('fuck')
        if not logger.hasHandlers():
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler(os.path.join(self.cfg.workdir, 'log.log'))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
        self.logger = logger

    @property
    def param_q(self):
        return chain(self.encoder_q.parameters(), self.mlp_q.parameters())

    @property
    def param_k(self):
        return chain(self.encoder_k.parameters(), self.mlp_k.parameters())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.param_q, self.param_k):
            param_k.data = param_k.data * self.cfg.m + param_q.data * (1. - self.cfg.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert (int(self.cfg.K) % int(batch_size)) == 0, (self.cfg.K, batch_size)
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:(ptr + batch_size)] = keys.T
        ptr = (ptr + batch_size) % self.cfg.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None, cluster_result=None, index=None, label=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if label is None:
            label = torch.LongTensor([self.cfg.n_class] * im_q.size(0))
        hint = self._c_hint_eye[label].squeeze(dim=1).to(im_q.device)
        hfactor = self.cfg.cluster_hint_factor
        norm_val = math.sqrt(hfactor ** 2 + (1-hfactor) ** 2)
        ph1 = (1-hfactor) / norm_val
        ph2 = hfactor / norm_val

        if im_k is None:
            feat = self.encoder_q(im_q)
            query = self.mlp_q(feat)
            query = nn.functional.normalize(query, dim=1)
            query = torch.cat([query * ph1, hint * ph2], dim=1)
            logits = self.supervised_mlp(feat)
            result = dict(feat=feat, query=query, index=index, label=label, logits=logits)
            return result


        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.mlp_q(q)
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = self.mlp_k(k)
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.cfg.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        # self.logger.info(ph1)

        # self.logger.info(q)
        if cluster_result is not None:
            q = torch.cat([q * ph1, hint * ph2], dim=1)
            proto_labels = []
            proto_logits = []
            l_mse = 0
            for n, (im2cluster, prototypes, density) in enumerate(zip(
                    cluster_result['im2cluster'],
                    cluster_result['centroids'],
                    cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = set(im2cluster.squeeze().tolist())
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.squeeze().tolist())
                #sample r negative prototypes
                neg_proto_id = sample(neg_proto_id, min(self.cfg.neg_claster_sample, len(neg_proto_id)))
                neg_prototypes = prototypes[neg_proto_id]

                l_mse += functional.F.mse_loss(q, pos_prototypes)

                proto_selected = torch.cat([pos_prototypes, neg_prototypes],dim=0)

                # compute prototypical logits
                logits_proto = torch.mm(q, proto_selected.t())

                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().to(self.device)

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).to(self.device)],dim=0)]
                # logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels, l_mse

        return logits, labels, None, None, None

    def train_supervised_epoch(self):
        self.train()
        data_time_start = time.time()
        batch_time_start = time.time()
        for i_batch, data in enumerate(self.train_loader):
            data_time_end = time.time()
            self.optimizer.zero_grad()
            img_q, img_k = data['img_q'], data['img_k']
            index = data['index'].squeeze()
            img_q = img_q.to(self.device)
            # img_k = img_k.to(self.device)
            output = self.__call__(img_q)
            label = data['gt_label']
            label=label.squeeze().to(self.device)
            loss = torch.nn.functional.cross_entropy(output['logits'], label)
            loss.backward()
            self.optimizer.step()
            log_info = Dict(
                loss=float(loss.detach().cpu()),
                lr=float(self.optimizer.param_groups[0]['lr']),
                acc=float((output['logits'].detach().cpu().argmax(dim=1)==label.cpu()).float().mean()),
                data_time=data_time_end-data_time_start,
                batch_time=time.time()-batch_time_start,
                name='supervised_epoch',
            )
            batch_time_start = time.time()
            self.logger.info(f'[{int(self.epoch)},{i_batch}/{len(self.train_loader)}]'
                    + ' '.join([f'{k}:{v}' for k,v in log_info.items()]))
            log_info.epoch = int(self.epoch)
            log_info.batch = i_batch
            self.json_log_file.write(json.dumps(log_info) + '\n')
            data_time_start = time.time()
        self.lr_scheduler.step()
        self.epoch += 1
        if int(self.epoch) % self.cfg.snap_interval == 0:
            torch.save(self.state_dict(), os.path.join(self.cfg.workdir, f'{int(self.epoch):04d}.pth'))

    # TODO add eval here and add some things
    def eval_supervised(self):
        self.eval()
        for i_batch, data in enumerate(self.test_loader):
            data_time_end = time.time()
            img_q, img_k = data['img_q'], data['img_k']
            index = data['index'].squeeze()
            img_q = img_q.to(self.device)
            # img_k = img_k.to(self.device)
            output = self.__call__(img_q)
            label = data['gt_label']
            label=label.squeeze().to(self.device)
            loss = torch.nn.functional.cross_entropy(output['logits'], label)
            loss.backward()
            self.optimizer.step()
            log_info = Dict(
                loss=float(loss.detach().cpu()),
                lr=float(self.optimizer.param_groups[0]['lr']),
                acc=float((output['logits'].detach().cpu().argmax(dim=1)==label.cpu()).float().mean()),
                data_time=data_time_end-data_time_start,
                batch_time=time.time()-batch_time_start,
                name='supervised_epoch',
            )
            batch_time_start = time.time()
            self.logger.info(f'[{int(self.epoch)},{i_batch}/{len(self.train_loader)}]'
                    + ' '.join([f'{k}:{v}' for k,v in log_info.items()]))
            log_info.epoch = int(self.epoch)
            log_info.batch = i_batch
            self.json_log_file.write(json.dumps(log_info) + '\n')
            data_time_start = time.time()
        self.lr_scheduler.step()
        self.epoch += 1
        if int(self.epoch) % self.cfg.snap_interval == 0:
            torch.save(self.state_dict(), os.path.join(self.cfg.workdir, f'{int(self.epoch):04d}.pth'))


    def train_epoch(self):
        if int(self.epoch) > self.cfg.n_iter:
            return
        cluster = None
        if self.cfg.widh_pcl and int(self.epoch) >= self.cfg.warm_up:
            x, label = self.eval_all(self.train_eval_loader)
            all_feat = x['query']
            cluster = self.run_kmeans(all_feat, label, x['index'].squeeze())

        self.train()
        data_time_start = time.time()
        batch_time_start = time.time()
        for i_batch, data in enumerate(self.train_loader):
            # for _ in range(1024000):
            data_time_end = time.time()
            self.optimizer.zero_grad()
            img_q, img_k = data['img_q'], data['img_k']
            index = data['index'].squeeze()
            img_q = img_q.to(self.device)
            img_k = img_k.to(self.device)
            label = data['gt_label']

            output, target, output_proto, target_proto, l_mse = self.__call__(img_q, img_k, cluster, index, label)
            # only needs to calculate the feature part for moco loss
            loss = self.criteria(output[:,:self.cfg.dim], target)
            loss_proto = 0
            if output_proto is not None:
                loss_proto = 0
                for proto_out,proto_target in zip(output_proto, target_proto):
                    # TODO check if hint influence this part
                    # loss_proto += self.criteria(proto_out, proto_target)
                    # TODO add proto acc back
                    # accp = accuracy(proto_out, proto_target)[0]
                    pass
                # accp = accuracy(proto_out, proto_target)[0]
                    # accp = accuracy(proto_out, proto_target)[0]
                    # acc_proto.update(accp[0], images[0].size(0))
                # acc_proto.update(accp[0], images[0].size(0))
                    # acc_proto.update(accp[0], images[0].size(0))
                # average loss across all sets of prototypes
                # TODO add label acc
                # loss_proto /= len(self.cfg.num_cluster)
                loss_proto = l_mse
                loss += self.cfg.lambda_proto * loss_proto
            loss.backward()
            self.optimizer.step()
            log_info = Dict(
                loss=float(loss.detach().cpu()),
                loss_proto=float(loss_proto),
                lr=float(self.optimizer.param_groups[0]['lr']),
                acc=float((output.cpu().argmax(dim=1)==0).float().mean()),
                data_time=data_time_end-data_time_start,
                batch_time=time.time()-batch_time_start,
            )
            batch_time_start = time.time()
            self.logger.info(f'[{int(self.epoch)},{i_batch}/{len(self.train_loader)}]'
                    + ' '.join([f'{k}:{v:.4f}' for k,v in log_info.items()]))
            log_info.epoch = int(self.epoch)
            log_info.batch = i_batch
            self.json_log_file.write(json.dumps(log_info) + '\n')
            data_time_start = time.time()

        self.lr_scheduler.step()
        self.epoch += 1
        if int(self.epoch) % self.cfg.snap_interval == 0:
            torch.save(self.state_dict(), os.path.join(self.cfg.workdir, f'{int(self.epoch):04d}.pth'))

    def run_kmeans(self, x, label, img_index=None):
        """
        Args:
            x: data to be clustered
        """
        print('performing kmeans clustering')
        x = x.detach().cpu()
        x = x.numpy()
        results = {'im2cluster':[],'centroids':[],'density':[]}
        if(img_index is not None):
            img_index = img_index.squeeze()
        else:
            img_index = list(range(x.shape[0]))

        for seed, num_cluster in enumerate(self.cfg.num_cluster):
            k = int(num_cluster)
            d = x.shape[1]
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 10

            # res = faiss.StandardGpuResources()
            # cfg = faiss.GpuIndexFlatConfig()
            # cfg.useFloat16 = False
            # cfg.device = args.gpu
            # TODO change this to cosion similarity
            # TODO check this
            index = faiss.IndexFlatIP(d)
            # index = faiss.IndexFlatL2(d)

            clus.train(x, index)

            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im,i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i,dist in enumerate(Dcluster):
                if len(dist)>1:
                    d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                    density[i] = d

            #if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i,dist in enumerate(Dcluster):
                if len(dist)<=1:
                    density[i] = dmax

            #clamp extreme values for stability
            density = density.clip(np.percentile(density,10),np.percentile(density,90))
            density = self.cfg.T * density/density.mean()  #scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).to(self.device)
            centroids = nn.functional.normalize(centroids, p=2, dim=1)

            im2cluster = torch.LongTensor(im2cluster).to(self.device)
            density = torch.Tensor(density).to(self.device)

            results['centroids'].append(centroids)
            results['density'].append(density)
            img_index = torch.LongTensor(img_index)
            if img_index is not None:
                reverse_map = torch.zeros_like(img_index, dtype=torch.long)
                for i,j in enumerate(img_index):
                    reverse_map[j] = i
                # new_idx --> cluster
                # new_idx --> old_idx
                # want --> old_idx --> cluster
                results['im2cluster'].append(im2cluster[reverse_map])
                self.logger.info(reverse_map)
            else:
                results['im2cluster'].append(im2cluster)
        self.logger.info(f'cluster ACC: {self.calculate_cluster_acc(results, label)}')
        return results

    def calculate_cluster_acc(self, cluster_result, label):
            # calculate_major label ratio
        results = []
        for im2cluster in (cluster_result['im2cluster']):
            cluster_label = [[] for _ in range(im2cluster.max() + 1)]
            for i, cluster in enumerate(im2cluster):
                cluster_label[cluster].append(int(label[i]))
            avg = 0
            for c_labels in cluster_label:
                if len(c_labels) <= 0:
                    continue
                c_labels = np.array(c_labels, dtype=np.int32)
                c_labels = np.array(c_labels)
                common_pred = np.bincount(c_labels).argmax()
                avg += (common_pred == c_labels).mean() * len(c_labels) / len(im2cluster)
            # self.logger.info(f"AVG ACC: {avg}")
            results.append(avg)
        return results


    @torch.no_grad()
    def eval_all(self, dataloader, n=None):
        self.eval()
        feats = []
        labels = []
        cnt = 0
        if n is None:
            n=len(dataloader)
        for data in track_iter_progress(dataloader):
            n -= 1
            if n<0:
                break
            img = data['img']
            label = data['gt_label']
            img = img.to(self.device)
            feat = self.__call__(img, label=label)
            for k in feat.keys():
                if isinstance(feat[k], torch.Tensor):
                    feat[k] = feat[k].detach().cpu()
            feat.update(dict(index=data['index']))
            feats.append(feat)
            labels.append(label.clone())
        feats_rec = {}
        for k in feats[0].keys():
            feats_rec[k] = torch.cat([i[k] for i in feats])
        labels = torch.cat(labels).squeeze()
        return feats_rec, labels

    @torch.no_grad()
    def eval_knn(self, k=32):
        self.eval()
        dfeats, dlabels = self.eval_all(self.train_eval_loader)
        tfeats, tlabels = self.eval_all(self.test_loader)
        feature = dfeats['query']
        result = self.run_kmeans(feature[:, :self.cfg.dim])
        pickle.dump((dfeats, dlabels, tfeats, tlabels),
            open(os.path.join(self.cfg.workdir, f'eval_dump_{int(self.epoch)}.pkl'), 'wb'))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fuck you!')
    parser.add_argument('config', help="your fucking configure file")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workdir', default=None)
    parser.add_argument('-i', default=False, help='interactive mode',  action='store_true')
    parser.add_argument('-eval', default=False, help='eval mode',  action='store_true')
    parser.add_argument('-supervised', default=False, help='eval mode',  action='store_true')
    parser.add_argument('--load', default=None, help='load model')
    args = parser.parse_args()
    if args.workdir is None:
        time_now = datetime.datetime.now().strftime("%Y_%m_%d.%H_%M_%S")
        workdir = os.path.join('workdir', time_now + os.path.splitext(os.path.split(args.config)[1])[0])
    else:
        workdir = args.workdir
    if args.eval:
        workdir = os.path.join(workdir, 'eval')
    if args.supervised:
        workdir = os.path.join(workdir, 'supervised')
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.dump(os.path.join(workdir, 'config.py'))
    cfg.device = args.device
    cfg.workdir = workdir
    cfg.load = args.load
    denoise =Denoise(cfg)
    if args.i:
        pass
    elif args.eval:
        res = denoise.eval_knn()
    elif args.supervised:
        for i in range(cfg.n_iter):
            denoise.train_supervised_epoch()
    else:
        for i in range(cfg.n_iter):
            denoise.train_epoch()
