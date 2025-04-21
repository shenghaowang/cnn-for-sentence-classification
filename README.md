# 📚 CNN for Sentence Classification

This repository contains PyTorch Lightning implementations of KimCNN models for benchmarking text classification tasks. The CNN model is based on [Kim (2014)](https://arxiv.org/abs/1408.5882).

---

## 🚀 Features

- ✅ KimCNN and BiLSTM implementations using PyTorch Lightning
- ✅ Pretrained word embeddings (Google News word2vec)
- ✅ Experiments on benchmark datasets: TREC, MR, BBC News
- ✅ Stratified data splitting and reproducible evaluation
- ✅ Early stopping, checkpointing, and runtime logging

---

## 📦 Installation

While Docker is the recommended method for setting up the environment, the following steps provide a quick alternative using a Python virtual environment:

1. **Clone the repository**
```bash
git clone https://github.com/your-username/text-classification-cnn-lstm.git
cd text-classification-cnn-lstm
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the `pre-commit` hooks**
```bash
pre-commit install
pre-commit run --all-files
```

---

## 📂 Datasets

The following datasets and the Google News word vectors need to be available in the [data](data/) folder.

* [TREC](https://cogcomp.seas.upenn.edu/Data/QA/QC/): Question classification by topic (e.g., person, location) (Li & Roth, 2002).
* [MR](https://www.cs.cornell.edu/people/pabo/movie-review-data/): Single-sentence movie reviews (Pang & Lee, 2005).
* [BBC News](https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv): News article classification by topic (business, entertainment, politics, sport, tech) (Greene & Cunningham, 2005).
* [Google News word vectors](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)(Mikolov et al., 2013)

---

## Usage

1. **Preprocess text data**

```bash
export PYTHONPATH=src
python src/preprocess/main.py dataset=<dataset_name>
```

Supported dataset names:

* trec (Question classification)
* mr (Movie review sentiment analysis)
* bbc (News topic classification)

2. **Train and evaluate KimCNN or Bidirectional LSTM model**

```bash
export PYTHONPATH=src
python src/train/main.py dataset=<dataset_name> model=<model_type>
```

Supported model types:

* cnn: KimCNN
* lstm: Bidirectional LSTM

## KimCNN model architecture

An example architecture of the KimCNN model based on the TREC dataset is as follows.

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ConvNet                                  [1, 6]                    --
├─ModuleList: 1-1                        --                        --
│    └─Conv1d: 2-1                       [1, 100, 8]               90,100
│    └─Conv1d: 2-2                       [1, 100, 7]               120,100
│    └─Conv1d: 2-3                       [1, 100, 6]               150,100
├─Dropout: 1-2                           [1, 300]                  --
├─Linear: 1-3                            [1, 6]                    1,806
==========================================================================================
Total params: 362,106
Trainable params: 362,106
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 2.46
==========================================================================================
```

---

## References

* Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv. https://arxiv.org/abs/1408.5882
* Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space (arXiv preprint arXiv:1301.3781). https://arxiv.org/abs/1301.3781
* Pang, B., & Lee, L. (2005). Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of ACL 2005.
* Li, X., & Roth, D. (2002). Learning Question Classifiers. In Proceedings of ACL 2002. https://dl.acm.org/doi/10.3115/1072228.1072378
* Greene, D., & Cunningham, P. (2005). Producing accurate interpretable clusters from high-dimensional data (Technical Report TCD-CS-2005-42). Department of Computer Science, Trinity College Dublin.
