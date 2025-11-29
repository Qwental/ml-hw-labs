from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np

# Функция для отрисовки
def plot_surface(clf, X, y):
    step = 0.01
    n_classes = len(np.unique(y))
    pal = sns.color_palette("magma", n_colors=n_classes)
    cmap = ListedColormap(pal)
    
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
    
    pt_colors = np.array(pal)[y]
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.7,
                edgecolors=pt_colors, linewidths=2)

# --- Эксперимент 1: Изменение max_depth ---

d_params = [1, 3, 5, None] 
n_rows = len(d_params)
n_cols = len(datasets)

plt.figure(figsize=(20, 5 * n_rows))
plt_idx = 1

for i in range(n_rows):
    depth = d_params[i]
    for j in range(n_cols):
        X, y = datasets[j]
        
        # разбиение выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # инициализация и фит модели
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred_tr = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_tr)
        
        y_pred_te = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_te)
        
        plt.subplot(n_rows, n_cols, plt_idx)
        plot_surface(clf, X_train, y_train)
        
        d_str = str(depth) if depth is not None else "None"
        t_str = f"датасетик {j+1} (depth={d_str}): точность train = {train_acc:.4f}, точность test = {test_acc:.4f}"
        plt.title(t_str)
        print(t_str)
        plt_idx += 1

plt.suptitle("Зависимость от max_depth", fontsize=16, y=1.02)
plt.show()


# --- Эксперимент 2: Изменение min_samples_leaf ---

l_params = [1, 5, 20]
n_rows = len(l_params)
n_cols = len(datasets)

plt.figure(figsize=(20, 5 * n_rows))
plt_idx = 1

for i in range(n_rows):
    leaf = l_params[i]
    for j in range(n_cols):
        X, y = datasets[j]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train, y_train)
        
        # считаем аккураси
        tr_acc = accuracy_score(y_train, clf.predict(X_train))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        
        plt.subplot(n_rows, n_cols, plt_idx)
        plot_surface(clf, X_train, y_train)
        
        t_str = f"датасетик {j+1} (leaf={leaf}): точность train = {tr_acc:.4f}, точность test = {te_acc:.4f}"
        plt.title(t_str)
        print(t_str)
        
        plt_idx += 1

plt.suptitle("Зависимость от min_samples_leaf", fontsize=16, y=1.02)
plt.show()
