import numpy as np
from collections import Counter

def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.
    """
    # sort index 
    idx = np.argsort(feature_vector)
    f_sorted = feature_vector[idx]
    t_sorted = target_vector[idx]
    
    
    unique_feats, unique_idx = np.unique(f_sorted, return_index=True)
    
    if len(unique_feats) < 2:
        return np.array([]), np.array([]), None, None
    
    thresholds = (unique_feats[:-1] + unique_feats[1:]) / 2.0
    
    n = len(target_vector)
    n1_total = np.sum(t_sorted == 1)
    
    split_idx = unique_idx[1:]
    
    cumulative_sum = np.cumsum(t_sorted == 1)
    n1_l = cumulative_sum[split_idx - 1]
    
    n_l = split_idx
    n_r = n - n_l
    
    n1_r = n1_total - n1_l
    n0_l = n_l - n1_l
    n0_r = n_r - n1_r
    
    gini_l = 1.0 - (n1_l / n_l)**2 - (n0_l / n_l)**2
    gini_r = 1.0 - (n1_r / n_r)**2 - (n0_r / n_r)**2
    
    ginis = - (n_l / n) * gini_l - (n_r / n) * gini_r
    
    best_i = np.argmax(ginis)
    
    return thresholds, ginis, thresholds[best_i], ginis[best_i]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        # validate feature types loop
        for ft in feature_types:
            if ft != "real" and ft != "categorical":
                raise ValueError("There is unknown feature type")

        self._tree = {}
        self._f_types = feature_types
        self._max_depth = max_depth
        self._min_split = min_samples_split
        self._min_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # check recursion depth limit
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        # proverka na min 
        if self._min_split is not None and len(sub_y) < self._min_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        best_f, best_thr, best_gini, best_split = None, None, None, None
        
        n_feats = sub_X.shape[1]
        for i in range(n_feats):
            f_type = self._f_types[i]
            cat_map = {}
            f_vec = None

            if f_type == "real":
                f_vec = sub_X[:, i]
            elif f_type == "categorical":
                counts = Counter(sub_X[:, i])
                clicks = Counter(sub_X[sub_y == 1, i])
                
                ratio = {}
                for k, cnt in counts.items():
                    clk = clicks[k] if k in clicks else 0
                    ratio[k] = cnt / clk 
                
                # sort map by values
                sorted_cats = sorted(ratio.items(), key=lambda item: item[1])
                sorted_keys = [x[0] for x in sorted_cats]
                
                # create index map
                cat_map = {k: idx for idx, k in enumerate(sorted_keys)}
                
                # transform vector via map
                f_vec = np.array([cat_map[x] for x in sub_X[:, i]])
            else:
                raise ValueError

            # check unique values count
            if len(np.unique(f_vec)) < 2:
                continue

            _, _, thr, gini = find_best_split(f_vec, sub_y)
            
            if gini is None: 
                continue

            # check min samples leaf constraint
            curr_split = f_vec < thr
            if self._min_leaf is not None:
                if np.sum(curr_split) < self._min_leaf or np.sum(~curr_split) < self._min_leaf:
                    continue

            if best_gini is None or gini > best_gini:
                best_f = i
                best_gini = gini
                best_split = curr_split
                
                if f_type == "real":
                    best_thr = thr
                elif f_type == "categorical":
                    best_thr = [k for k, v in cat_map.items() if v < thr]
                else:
                    raise ValueError

        if best_f is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_f
        
        if self._f_types[best_f] == "real":
            node["threshold"] = best_thr
        elif self._f_types[best_f] == "categorical":
            node["categories_split"] = best_thr
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        
        # recursivniy vyzov dlya postroeniya dereva
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~best_split], sub_y[~best_split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        idx = node["feature_split"]
        f_type = self._f_types[idx]
        
        if f_type == "real":
            if x[idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else: 
            # categorical branch
            if x[idx] in node["categories_split"]:
                 return self._predict_node(x, node["left_child"])
            else:
                 return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        # vydelenie pamyati pod masiv
        predicted = []
        # loop through all samples
        for i in range(len(X)):
            predicted.append(self._predict_node(X[i], self._tree))
        return np.array(predicted)
