"""Train-only covariance features and classical classifiers for the new study.

CSP is an independent covariance-domain implementation, not a new algorithm.
TangentSpace and MDM use pyRiemann with evaluation-batch updates disabled.
"""
from __future__ import annotations

import copy
import itertools
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from .config import digest
from .frontier_data import shrink_covariances


def candidates():
    result = []

    def add(family, channels=16, window="late", bands=(1,), **kwargs):
        spec = {"family": family, "channels": channels, "window": window,
                "bands": list(bands), **kwargs}
        spec["name"] = family + "_" + digest(spec)[:12]
        result.append(spec)

    for channels, window, band, components, reg in itertools.product(
            [16,64], ["late","full"], [0,1], [4,8], [0.01,0.1,0.3]):
        add("csp", channels, window, [band], components=components, reg=reg,
            classifier="lda", normalize=False)
    for channels, window, band, C in itertools.product([16,64], ["late","full"], [0,1], [0.1,1,10]):
        add("csp", channels, window, [band], components=4, reg=0.1,
            classifier="svm", C=C, normalize=False)
    for channels, window, components, classifier, select in itertools.product(
            [16,64], ["late","full"], [2,4], ["lda","logistic","svm"], [False,True]):
        add("fbcsp", channels, window, range(2,8), components=components, reg=0.1,
            classifier=classifier, C=1.0, normalize=True,
            select=(8 if components==2 else 12) if select else None)
    for metric, normalize, C in itertools.product(["riemann","logeuclid"], [False,True], [.01,.1,1,10]):
        add("tangent", 16, "late", [1], metric=metric, normalize=normalize,
            classifier="logistic", C=C)
    for normalize, C in itertools.product([False,True], [.1,1]):
        add("tangent", 64, "late", [1], metric="logeuclid", normalize=normalize,
            classifier="logistic", C=C, pca=128)
    for window, metric, C in itertools.product(["full","late"], ["riemann","logeuclid"], [.1,1]):
        add("fbtangent", 16, window, range(3,8), metric=metric, normalize=True,
            classifier="logistic", C=C)
    for channels, band, normalize in itertools.product([16,64], [0,1], [False,True]):
        for metric in (["riemann","logeuclid"] if channels==16 else ["logeuclid"]):
            add("mdm", channels, "late", [band], metric=metric, normalize=normalize)
    for channels, window in itertools.product([16,64], ["late","full"]):
        add("power", channels, window, range(2,8), classifier="lda")
        for classifier,C in itertools.product(["logistic","svm"],[.1,1,10]):
            add("power", channels, window, range(2,8), classifier=classifier, C=C)
        for classifier in ["extra_trees","hist_boost"]:
            add("power", channels, window, range(2,8), classifier=classifier)
    for architecture, channels in itertools.product(["eegnet","shallow","attention"], [16,64]):
        add("neural", channels, "full", [0], architecture=architecture,
            augmentation="none", dropout=.5)
    for architecture in ["eegnet","shallow","attention"]:
        add("neural", 16, "full", [0], architecture=architecture,
            augmentation="mixup", dropout=.5)
    for architecture in ["eegnet","shallow"]:
        add("neural", 16, "full", [0], architecture=architecture,
            augmentation="segment_recombine", dropout=.5)
    result.append({"family":"raw_csp", "channels":16,"window":"late","bands":[1],
                   "name":"csp16_baseline"})
    result.append({"family":"neural", "channels":16,"window":"late","bands":[1],
                   "name":"eegnet16_baseline", "architecture":"eegnet_baseline",
                   "augmentation":"none","dropout":.5,"fixed_epochs":25})
    calibration=[]
    for spec in result:
        family=spec["family"]
        include=(
            family=="csp" and spec["window"]=="late" and spec["components"]==4
            and spec["reg"]==.1 and spec["classifier"]=="lda"
            or family=="fbcsp" and spec["window"]=="late" and spec["components"]==4
            and spec["classifier"]=="logistic" and spec["select"] is None
            or family=="tangent" and spec["metric"]=="logeuclid" and spec["normalize"]
            and spec["C"] in [.1,1]
            or family=="fbtangent" and spec["window"]=="late" and spec["metric"]=="logeuclid"
            or family=="power" and spec["window"]=="late" and spec["classifier"]=="logistic"
            and spec["C"]==1
            or family=="neural" and spec["architecture"]=="eegnet" and spec["channels"]==16
            and spec["augmentation"]=="none")
        if include:
            calibrated=copy.deepcopy(spec)
            calibrated.pop("name")
            calibrated["arm"]="rest_calibrated"
            calibrated["name"]=family+"_rest_"+digest(calibrated)[:12]
            calibration.append(calibrated)
    result.extend(calibration)
    assert len({s["name"] for s in result}) == len(result)
    return result


def feature_identity(spec):
    return {k:v for k,v in spec.items() if k not in {"name","classifier","C","pca","select"}}


class CovarianceFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, spec):
        self.spec = spec

    def _cov(self, C):
        C = np.asarray(C, dtype=np.float64)
        if self.spec.get("normalize",False):
            C = C / np.trace(C,axis1=-2,axis2=-1)[...,None,None]
        return C

    def fit(self, C, y):
        C = self._cov(C)
        family = self.spec["family"]
        self.fit_trials_ = len(C)
        self.fit_classes_ = np.unique(y).tolist()
        if self.fit_classes_ != [0,1]:
            raise ValueError("Both classes required for training.")
        self.transformers_ = []
        for b in range(C.shape[1]):
            if family in {"csp","fbcsp"}:
                means = [C[np.asarray(y)==label,b].mean(axis=0) for label in [0,1]]
                means = [shrink_covariances(m, self.spec["reg"]) for m in means]
                eigenvalues, eigenvectors = eigh(means[0],means[0]+means[1])
                n = self.spec["components"]
                ordering = np.asarray([item for pair in zip(range(len(eigenvalues)-1,-1,-1),
                    range(len(eigenvalues))) for item in pair])[:n]
                self.transformers_.append(eigenvectors[:,ordering].T)
            elif family in {"tangent","fbtangent"}:
                from pyriemann.tangentspace import TangentSpace
                mapper = TangentSpace(metric=self.spec["metric"], tsupdate=False)
                mapper.fit(shrink_covariances(C[:,b]), y)
                self.transformers_.append(mapper)
            elif family == "power":
                break
            else:
                raise ValueError(f"Unsupported feature family {family}.")
        return self

    def transform(self, C):
        C = self._cov(C)
        if self.spec["family"] == "power":
            power = np.maximum(np.diagonal(C,axis1=-2,axis2=-1),1e-15)
            relative = power / power.sum(axis=1,keepdims=True)
            return np.concatenate([np.log(power).reshape(len(C),-1),
                                   np.log(relative).reshape(len(C),-1)],axis=1)
        features = []
        for b, mapper in enumerate(self.transformers_):
            if isinstance(mapper,np.ndarray):
                variance = np.einsum("kc,ncd,kd->nk",mapper,C[:,b],mapper,optimize=True)
                if self.spec.get("normalize",False):
                    variance = variance / np.maximum(variance.sum(axis=1,keepdims=True),1e-15)
                features.append(np.log(np.maximum(variance,1e-15)))
            else:
                features.append(mapper.transform(shrink_covariances(C[:,b])))
        return np.concatenate(features,axis=1)


def classifier(spec, seed=7):
    kind = spec["classifier"]
    if kind == "lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
    elif kind == "logistic":
        estimator = LogisticRegression(C=spec.get("C",1),max_iter=700,class_weight="balanced",random_state=seed)
    elif kind == "svm":
        estimator = SVC(C=spec.get("C",1),kernel="rbf",gamma="scale",probability=True,
                        class_weight="balanced",random_state=seed)
    elif kind == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=250,max_features="sqrt",min_samples_leaf=4,
                                         class_weight="balanced",n_jobs=1,random_state=seed)
    elif kind == "hist_boost":
        estimator = HistGradientBoostingClassifier(max_iter=160,learning_rate=.05,max_leaf_nodes=15,
                                                   l2_regularization=1,random_state=seed)
    else:
        raise ValueError(kind)
    steps = [("scale",StandardScaler())]
    if spec.get("select"):
        steps.append(("select",SelectKBest(f_classif,k=spec["select"])))
    if spec.get("pca"):
        steps.append(("pca",PCA(n_components=spec["pca"],svd_solver="randomized",random_state=seed)))
    return Pipeline(steps+[("classifier",estimator)])


class CovarianceModel:
    def __init__(self, spec, features=None, classifier_model=None):
        self.spec = spec
        self.features = features
        self.classifier_model = classifier_model

    def fit(self, C, y):
        if self.spec["family"] == "mdm":
            from pyriemann.classification import MDM
            self.classifier_model = MDM(metric=self.spec["metric"],n_jobs=1)
            self.classifier_model.fit(self._mdm_cov(C),y)
        else:
            self.features = CovarianceFeatures(self.spec).fit(C,y)
            self.classifier_model = classifier(self.spec).fit(self.features.transform(C),y)
        return self

    def _mdm_cov(self,C):
        C = np.asarray(C[:,0],dtype=np.float64)
        if self.spec.get("normalize",False):
            C = C / np.trace(C,axis1=-2,axis2=-1)[:,None,None]
        return shrink_covariances(C)

    def predict_proba(self, C):
        if self.spec["family"] == "mdm":
            return self.classifier_model.predict_proba(self._mdm_cov(C))
        return self.classifier_model.predict_proba(self.features.transform(C))
