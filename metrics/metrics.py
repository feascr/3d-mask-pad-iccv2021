import numpy as np
import logging

#real_preds -> tar
#spoof_preds -> imp
class MetricCalculator:
    def _get_fr_fa_at_threshold(self, tar, imp, threshold=0.5):
        fr = self._get_fr_at_threshold(tar, threshold=threshold)
        fa = self._get_fa_at_threshold(imp, threshold=threshold)
        return fr, fa

    def _get_fa_at_threshold(self, imp, threshold=0.5):
        fa = np.NaN
        if len(imp) > 0:
            fa = len(np.where(imp > threshold)[0])
            fa = fa * 100.0 / len(imp)
        return fa

    def _get_fr_at_threshold(self, tar, threshold=0.5):
        fr = np.NaN
        if len(tar) > 0:
            fr = len(np.where(tar < threshold)[0])
            fr = fr * 100.0 / len(tar)
        return fr

    def _get_eer(self, tar, imp):
        return self._compute_eer(tar, imp)

    def _compute_eer(self, tar, imp):
        tar_imp, fr, fa = self.compute_frr_far(tar, imp)
        index_min = np.argmin(np.abs(fr - fa))
        eer = 100.0 * np.mean((fr[index_min], fa[index_min]))
        threshold = tar_imp[index_min]
        return eer, threshold

    def compute_frr_far(self, tar, imp):
        tar_unique, tar_counts = np.unique(tar, return_counts=True)
        imp_unique, imp_counts = np.unique(imp, return_counts=True)
        thresholds = np.unique(np.hstack((tar_unique, imp_unique)))
        pt = np.hstack(
            (tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=np.int))
        )
        pi = np.hstack(
            (np.zeros(len(thresholds) - len(imp_counts), dtype=np.int), imp_counts)
        )
        pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
        pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]
        fr = np.zeros(pt.shape[0] + 1, dtype=np.int)
        fa = np.zeros(pi.shape[0] + 1, dtype=np.int)
        for i in range(1, len(pt) + 1):
            fr[i] = fr[i - 1] + pt[i - 1]
        for i in range(len(pt) - 1, -1, -1):
            fa[i] = fa[i + 1] + pi[i]
        frr = fr / len(tar)
        far = fa / len(imp)
        thresholds = np.hstack((thresholds, thresholds[-1] + 1e-6))
        return thresholds, frr, far


    def calculate_metrcis(self, outputs, targets):
        threshold = 0.5
        metrics_dict = {'target': None}
        
        tar = outputs[targets == 1]
        imp = outputs[targets == 0]
        bpcer, apcer = self._get_fr_fa_at_threshold(tar=tar, imp=imp, threshold=threshold)
        acer = (bpcer + apcer) / 2.0
        eer, threshold = self._get_eer(tar=tar, imp=imp)
        metrics_dict['target'] = {'EER': eer, 'ACER': acer, 'BPCER': bpcer, 'APCER': apcer}
        return metrics_dict
