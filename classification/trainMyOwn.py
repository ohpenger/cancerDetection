import argparse
import datetime
import sys

import torch.cuda
import torch.nn as nn
import numpy as np

from cancerDetection.util.logconf import logging
from model import LunaModel
from torch.utils.data import DataLoader
from dsets import LunaDataset
from cancerDetection.util.util import enumerateWithEstimate

logger = logging.getLogger(__name__)
METRICS_lABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

class LunaTrainingApp:

    def __init__(self,sys_args=None):
        if sys_args is None:
            sys_args = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument("--num_workers",
                            help="the number of working process at the background data loading",
                            default=3,
                            type=int)
        parser.add_argument("--batch_size",
                            default=32,
                            help="the number of samples in a batch",
                            type=int)
        parser.add_argument("--epoch",
                            default=1,
                            help="epoch",
                            type=int)
        parser.add_argument("--balanced",
                            help="Balance the training data to a ration of negtive to positive",
                            default=True,
                            type=bool)
        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=True,
        )
        parser.add_argument('--augment-flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true',
                            default=False,
        )
        parser.add_argument('--augment-offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true',
                            default=False,
        )
        parser.add_argument('--augment-scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true',
                            default=True,
        )
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=True,
        )
        parser.add_argument('--augment-noise',
                            help="Augment the training data by randomly adding noise to the data.",
                            action='store_true',
                            default=False,
        )
        self.cli_args = parser.parse_args(sys_args)
        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0
        self.time_str = datetime.datetime.now().strftime("%Y-%M-%D_%H:%M:%S")
        self.use_cuda = torch.cuda.is_available()
        self.device = ("cuda" if self.use_cuda else "cpu")
        self.totalTrainingSamples_count = 0
        self.model = self.initModel()
        self.optimizer = self.initOptim()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            logger.info("using cuda, {} devices".format(torch.cuda.device_count()))
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def initOptim(self):
        return torch.optim.SGD(self.model.parameters(),lr=0.001, momentum=0.99)

    def initTrainDL(self):
        train_ds = LunaDataset(
            val_stride= 10,
            isValSet_bool=False,
            ratio_int=self.cli_args.balanced,
            augmentation_dict=self.augmentation_dict
        )
        train_ds.shuffleSamples()
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(
            train_ds,
            batch_size= batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=True
        )
        return train_dl

    def initValDL(self):
        Val_ds = LunaDataset(
            val_stride= 10,
            isValSet_bool=True
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        Val_dl = DataLoader(
            Val_ds,
            batch_size= batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=True
        )
        return Val_dl

    def doTraining(self,train_dl,epoch_ndx):
        self.model.train()
        trnMetrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers
        )

        for ndx, batch_data in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                ndx,
                batch_data,
                train_dl.batch_size,
                trnMetrics
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return  trnMetrics.to('cpu')

    def doValidation(self,val_dl,epoch_ndx):
        with torch.no_grad():
            self.model.eval()

            valMetrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device
            )

            val_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers
            )

            for ndx, val_data in val_iter:
                self.computeBatchLoss(
                    ndx,
                    val_data,
                    val_dl.batch_size,
                    valMetrics
                )
            return valMetrics.to('cpu')

    def computeBatchLoss(self,batch_ndx,batch_data,batch_size,metrics):
        input_t,label_t,*_ = batch_data

        input_g = input_t.to(self.device,non_blocking=True) # asychronously transfer the data in cpu to GPU
        label_g = label_t.to(self.device,non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss()
        loss_val = loss_func(
            logits_g,
            label_g[:,1]
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + batch_size

        metrics[METRICS_lABEL_NDX,start_ndx:end_ndx]= label_g[:,1].detach()
        metrics[METRICS_PRED_NDX,start_ndx:end_ndx] = probability_g[:,1].detach()
        metrics[METRICS_LOSS_NDX,start_ndx:end_ndx] = loss_val.detach()

        return loss_val.mean()

    def logMetrics(self,
                   epoch_ndx,
                   mode_str,
                   metrics,
                   classificationThreshold = 0.5):
        logger.info("E{} {}".format(epoch_ndx,type(self).__name__))

        posLabel_mask = metrics[METRICS_lABEL_NDX] >= classificationThreshold
        posPred_mask = metrics[METRICS_PRED_NDX] >= classificationThreshold

        negLabel_mask = ~posLabel_mask
        negPred_mask = ~posPred_mask

        pos_count = int(posLabel_mask.sum())
        neg_count = int(negLabel_mask.sum())

        true_pos = int((posLabel_mask & posPred_mask).sum())
        true_neg = int((negPred_mask & negLabel_mask).sum())
        false_pos = int(pos_count - true_pos)
        false_neg = int(neg_count - true_neg)

        recall = np.float32(true_pos) / (true_pos + false_neg)
        precision = np.float32(true_pos) / (true_pos + false_pos)
        f1_score = np.float32(2*recall*precision)/(recall+precision)

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics[METRICS_LOSS_NDX].mean()
        metrics_dict["loss/pos"] = metrics[METRICS_LOSS_NDX,posLabel_mask].mean()
        metrics_dict["loss/neg"] = metrics[METRICS_LOSS_NDX,negLabel_mask].mean()

        metrics_dict["correct/all"] = np.float32(true_pos+true_neg) / metrics.shape[1] * 100
        metrics_dict["correct/pos"] = np.float32(true_pos) / pos_count * 100
        metrics_dict["correct/neg"] = np.float32(true_neg) / neg_count * 100

        metrics_dict["pr/precision"] = precision
        metrics_dict["pr/recall"] = recall
        metrics_dict["pr/f1_score"] = f1_score

        logger.info("E{} {:8} loss/all: {loss/all:.4f} correct/all: {correct/all:5.1f}%  "
                    "precision: {pr/precision} recall {pr/recall} f1_socre: {pr/f1_score}".format(
            epoch_ndx,
            mode_str,
            **metrics_dict
        ))

        logger.info(("E{} {:8} loss/pos: {loss/pos:.4f}"
                    "correct/pos: {correct/pos:5.1f}"
                    "true_pos: {true_pos:5f} out of pos_count {pos_count:5f}").format(
            epoch_ndx,
            mode_str+"_pos",
            true_pos = true_pos,
            pos_count = pos_count,
            **metrics_dict
        ))
        logger.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({true_neg:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                true_neg=true_neg,
                neg_count=neg_count,
                **metrics_dict,
            )
        )

    def main(self):
        logger.info("{} {} starting".format(type(self).__name__,self.cli_args))

        train_dl = self.initTrainDL()
        val_dl = self.initValDL()

        for epoch_ndx in range(1,self.cli_args.epoch + 1):
            logger.info("Epoch {} of {} {}/{} batch_size {}*{}".format(
                epoch_ndx,
                self.cli_args.epoch,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1)
            ))

            trnMetrics = self.doTraining(train_dl,epoch_ndx)
            self.logMetrics(epoch_ndx,"trn",trnMetrics)

            valMetrics = self.doValidation(val_dl,epoch_ndx)
            self.logMetrics(epoch_ndx,"val",valMetrics)

if __name__ == "__main__":
    LunaTrainingApp().main()