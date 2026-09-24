from load_data import *
from preprocess import *
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import os
import pickle
from numpy import asarray
from numpy import save

# 0 = IMBALANCED (giữ nguyên tỉ lệ thật, Enron >> phishing)
# 1 = BALANCED   (downsample Enron xuống bằng số lượng email phishing)
BALANCED_MODE = 0


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(
        *[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors


def training():
    # ---- 1. Load dữ liệu thô (Nazario mbox + SpamAssassin + Enron maildir) --
    data = pd.DataFrame()
    data = load_data_phishing(data)

    data_enron = pd.DataFrame()
    # SpamAssassin: tự phân loại ham (legit) / spam (phishing) theo tên thư
    # mục con, gộp thẳng vào 2 pool data/data_enron -- gọi TRƯỚC
    # load_data_enron để n_phishing bên dưới tính đủ cả phần spam vừa thêm.
    data, data_enron = load_data_spamassassin(data, data_enron)

    data_enron = load_data_enron(
        data_enron, BALANCED_MODE, n_phishing=len(data))

    data = pd.concat([data, data_enron], ignore_index=True)

    # ---- 1b. Khử trùng lặp / gần trùng (chống data leakage) ---------------
    # Enron corpus có rất nhiều email bị forward cho nhiều người -> nằm rải
    # rác ở nhiều mailbox, nội dung body gần như giống hệt nhau. Nếu không
    # loại bỏ, train_test_split ngẫu nhiên rất dễ chia 2 bản gần-trùng vào
    # cả train lẫn test -- model "nhìn thấy" đáp án test ngay lúc train,
    # điểm test (đặc biệt Accuracy/F1 gần 100%) sẽ KHÔNG đáng tin.
    before_dedup = len(data)
    print(f"Khử trùng lặp/gần trùng theo nội dung body trên {before_dedup} email ...")
    data['_dedup_key'] = data['Text'].apply(compute_dedup_key)
    data = data.drop_duplicates(subset='_dedup_key', keep='first').drop(columns=['_dedup_key'])
    data = data.reset_index(drop=True)
    after_dedup = len(data)
    removed = before_dedup - after_dedup
    print(f"  -> Loại {removed} email trùng/gần trùng ({removed/before_dedup*100:.1f}%), "
          f"còn lại {after_dedup} email.")

    cnt_pro = data['Class'].value_counts()
    print("Phân bố lớp SAU dedup (0=legit, 1=phishing):\n", cnt_pro)

    if data['Class'].nunique() < 2:
        raise SystemExit(
            "Dataset chỉ có 1 lớp duy nhất (Class distribution ở trên) -- "
            "kiểm tra lại thư mục Dataset/phishing/ và Dataset/enron/ có "
            "thật sự chứa email không, và đường dẫn có đúng không (mặc định "
            "là './Dataset/phishing/' và './Dataset/enron/' tính từ nơi bạn "
            "chạy `python train.py`)."
        )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=cnt_pro.index, y=cnt_pro.values, alpha=0.8)
    plt.ylabel('Number of emails', fontsize=12)
    plt.xlabel('Class (0=legit, 1=phishing)', fontsize=12)
    plt.xticks(rotation=90)
    plt.savefig("data")
    plt.close()

    # ---- 2. Trích xuất ĐẦY ĐỦ đặc trưng kỹ thuật TRÊN TEXT GỐC ------------
    # Bắt buộc chạy TRƯỚC cleanText(), vì cleanText sẽ xoá URL/markup/ký tự
    # đặc biệt -- các nhóm URL/HTML/Attachment/Header cần dữ liệu thô.
    print("Trích xuất engineered features (url/header/html/attachment/"
          "social-engineering) trên", len(data), "email ...")
    reset_extraction_stats()
    data['Features'] = data['Text'].apply(extract_engineered_features)
    stats = get_extraction_stats()
    if stats["failed"]:
        pct = stats["failed"] / max(stats["total"], 1) * 100
        print(f"      -> {stats['failed']}/{stats['total']} email ({pct:.1f}%) "
              f"bị lỗi khi trích feature, đã fallback về vector 0 cho các email đó.")

    # ---- 3. Làm sạch text (giữ nguyên các bước gốc của repo) --------------
    data['Text'] = data['Text'].apply(cleanText)
    data['Text'] = data['Text'].apply(lambda x: remove_stopwords(x))
    data['Text'] = data['Text'].apply(lambda x: ' '.join(
        [WordNetLemmatizer().lemmatize(word, pos=wordnet.VERB) for word in x.split()]))

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # ## Training model Doc2Vec (giống bản gốc)
    tqdm.pandas(desc="progress-bar")

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Class]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Class]), axis=1)

    model_dbow = Doc2Vec()
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(
            train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    y_train, X_train_doc2vec = vec_for_learning(model_dbow, train_tagged)
    y_test, X_test_doc2vec = vec_for_learning(model_dbow, test_tagged)

    # ---- 4. Ghép [Doc2Vec vector | 42 engineered features] ----------------
    # Chuẩn hoá (StandardScaler) riêng khối engineered features vì thang đo
    # của chúng (vd char_length có thể tới hàng nghìn) rất khác thang của
    # vector Doc2Vec (thường quanh [-1, 1]) -- nếu không chuẩn hoá, các model
    # khoảng cách/tuyến tính (KNN, SVM, LogisticRegression) sẽ bị lệch bởi
    # riêng vài đặc trưng có giá trị lớn.
    train_engineered = np.array(train['Features'].tolist())
    test_engineered = np.array(test['Features'].tolist())

    feature_scaler = StandardScaler()
    train_engineered_scaled = feature_scaler.fit_transform(train_engineered)
    test_engineered_scaled = feature_scaler.transform(test_engineered)

    X_train = np.hstack([np.array(X_train_doc2vec), train_engineered_scaled])
    X_test = np.hstack([np.array(X_test_doc2vec), test_engineered_scaled])

    print(f"Kích thước feature cuối cùng: Doc2Vec(100) + Engineered({len(FEATURE_NAMES)}) "
          f"= {X_train.shape[1]} chiều")

    # ## Classification -- giữ đúng model_pip 10 classifier như bản gốc
    model_pip = []
    model_pip.append(('Logistic Regression', LogisticRegression(max_iter=2000)))
    model_pip.append(('Random Forest', RandomForestClassifier()))
    model_pip.append(('XGBoost', XGBClassifier()))
    model_pip.append(('SVM', SVC(probability=True)))
    model_pip.append(('KNN', KNeighborsClassifier()))
    model_pip.append(('Naive Bayes', GaussianNB()))
    model_pip.append(('Decision Tree', DecisionTreeClassifier()))
    model_pip.append(('MLP', MLPClassifier(max_iter=500)))
    model_pip.append(('AdaBoost', AdaBoostClassifier()))
    model_pip.append(('Gradient Boosting', GradientBoostingClassifier()))

    return X_test, y_test, model_pip, X_train, y_train, model_dbow, feature_scaler


if __name__ == "__main__":
    X_test, y_test, model_pip, X_train, y_train, model_dbow, feature_scaler = training()

    data_X_test = asarray(X_test)
    data_y_test = asarray(y_test)
    save("data_X_test.npy", data_X_test)
    save("data_y_test.npy", data_y_test)

    # Write the model to a file
    if not os.path.isdir("dumped_models/classification/"):
        os.makedirs("dumped_models/classification", exist_ok=True)

    # model_dbow + feature_scaler PHẢI được lưu lại: model_deployment.py cần
    # đúng 2 file này để dựng lại vector đặc trưng khi suy luận email mới
    # (bản gốc thiếu bước lưu model_dbow.pkl trong train.py).
    pickle.dump(model_dbow, open('dumped_models/classification/model_dbow.pkl', 'wb'))
    pickle.dump(feature_scaler, open('dumped_models/classification/feature_scaler.pkl', 'wb'))

    for classifier in model_pip:
        print(f"Training {classifier[0]} ...")
        classifier[1].fit(X_train, y_train)
        pickle.dump(classifier[1], open(
            'dumped_models/classification/' + classifier[0] + '.pkl', 'wb'))

    print("Done. Models saved to dumped_models/classification/")
