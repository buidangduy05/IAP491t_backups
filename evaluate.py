from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from numpy import load

X_test = load("data_X_test.npy")
y_test = load("data_y_test.npy")


acc = []
f1 = []
precision = []
recall = []
roc_auc = []
conf_mat = []
cl = []
model_name = []

for filename in sorted(os.listdir("dumped_models/classification")):
    if not filename.endswith(".pkl"):
        continue
    # model_dbow.pkl / feature_scaler.pkl không phải classifier -> bỏ qua
    if filename in ("model_dbow.pkl", "feature_scaler.pkl"):
        continue
    with open('dumped_models/classification/{}'.format(filename), 'rb') as f:
        model = pickle.load(f)
        y_pred = model.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        roc_auc.append(roc_auc_score(y_test, y_pred))
        conf_mat.append(confusion_matrix(y_test, y_pred))
        cl.append(model)
        model_name.append(filename.replace(".pkl", ""))

result = pd.DataFrame({'Model': model_name, 'Accuracy': acc,
                      'F1-score': f1, 'Precision': precision, 'Recall': recall,
                      'ROC-AUC': roc_auc})
print(result)
result.to_csv("evaluation_results.csv", index=False)

plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(20, 10))
for i in range(0, len(cl)):
    c = cl[i]
    # SỬA LỖI bản gốc: model_name[i][0] chỉ lấy ký tự đầu tiên của tên file
    # (vd "Logistic Regression" -> "L"), khiến mọi nhãn trên biểu đồ ROC bị
    # sai. Ở đây dùng đúng model_name[i].
    model = model_name[i]
    y_pred_proba = c.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    plot_map = sns.lineplot(x=fpr, y=tpr, label=model +
                            ': AUC=' + str(round(auc, 4)))
plot_map.set_xlabel('False Positive Rate')
plot_map.set_ylabel('True Positive Rate')
fig.savefig("ROCs")
plt.close(fig)

fig = plt.figure(figsize=(20, 50))
for i in range(0, len(conf_mat)):
    cm_con = conf_mat[i]
    model = model_name[i]
    sub_fig_title = fig.add_subplot(5, 2, i + 1).set_title(model)
    plot_map = sns.heatmap(cm_con, annot=True, cmap='Greens_r', fmt='g')
    plot_map.set_xlabel('Predicted_Values')
    plot_map.set_ylabel('Actual_Values')
fig.savefig("confusion-matrix")
plt.close(fig)

print("\nSaved: evaluation_results.csv, ROCs.png, confusion-matrix.png")
