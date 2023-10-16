import numpy as np
from sklearn.metrics import confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes, buffer_size=1):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.buffer_size = buffer_size

    def update(self, label_trues, label_preds):
        if self.buffer_size>1:
            label_trues,  label_preds= self.label_buffer(label_trues, label_preds,  buffer_size=self.buffer_size)
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())  # 只更新混淆矩阵


    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)  # mask 为uint8或者布尔类型
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def label_buffer(self, label, pred, buffer_size=1):  # BCHW
        h, w = label.shape[-2:]
        label = label[0][0]
        pred = pred[0][0]

        pred_temp = np.array(pred.flatten()).astype(bool)
        label_temp = np.array(label.flatten()).astype(bool)
        index_all = np.array(range(len(pred_temp)))
        index_pred = index_all[pred_temp]
        index_lable = index_all[label_temp]
        x_list_pred = [i%h for i in index_pred]
        y_list_pred = [i//h for i in index_pred]
        x_list_label = [i % h for i in index_lable]
        y_list_label = [i // h for i in index_lable]

        step = buffer_size-1
        padding_h = np.zeros((h, step))
        padding_w = np.zeros((step, w+2*step))
        temp_label = np.concatenate((padding_h, label, padding_h), axis=1)
        label_padded = np.concatenate((padding_w, temp_label, padding_w), axis=0)
        temp_pred = np.concatenate((padding_h, label, padding_h), axis=1)
        pred_padded = np.concatenate((padding_w, temp_pred, padding_w), axis=0)

        for y, x in zip(y_list_pred, x_list_pred):
            if np.sum(label_padded[y-step:y+step, x-step:x+step]) >= 1:
                label[y-step, x-step] = 1.
        for y, x in zip(y_list_label, x_list_label):
            if np.sum(pred_padded[y-step:y+step, x-step:x+step]) >= 1:
                pred[y-step, x-step] = 1.

        label = np.array([[label]])  # BCHW
        pred = np.array([[pred]])
        return label, pred


    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        recall_cls = np.diag(hist) / hist.sum(axis=1)  # 查全
        precision_cls = np.diag(hist) / hist.sum(axis=0)  # 查准
        # acc_cls = np.nanmean(acc_cls)  # 用于计算忽略nan值的数组平均值
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # 0是背景，1是目标
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))  # dict(zip())结合使用
        # cal Kappa
        p0 = acc
        pe = ((hist.sum(axis=0)/hist.sum())*(hist.sum(axis=1)/hist.sum())).sum()
        kappa = (p0-pe)/(1-pe)


        return {
            "Overall Acc": acc,
            # "Mean Acc": acc_cls,
            "Precision_back": precision_cls[0],
            "Precision_fore": precision_cls[1],
            "Recall_back": recall_cls[0],
            "Recall_fore": recall_cls[1],
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Kappa":kappa
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]