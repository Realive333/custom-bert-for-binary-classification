# custom-bert-for-binary-classification

Custom BERT for binary classification is a Huggingface-based BERT for Japanese document classification. It aimed to address the problem of Transformer-based long document classification: the tranning result will decline due to the input limitation. Combining with our Kakuyomu-Clipper project, our modify BERT is adjusted in able to accompany our datasets.

## Custom BERT

We will provide several customed BERT in the future.

1. BERT-CLS-AVG model

Based on Sparsified Firstmatch/Nearest-K clipping method, our BERT-CLS-AVG model is able to train the data withe sparsified paragraphs from one document. The model will train on an averaged classification vectors from the batched dataset text.


## Author
- Verniy "Realive" Akatsuki
- realive333@gmail.com