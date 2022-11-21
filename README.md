# Persian Emotion Detection using ParsBERT and Imbalanced Data Handling Approaches

## Abstract
Emotion recognition is one of the machine learning applications which can be done using text, speech, or image data gathered from social media spaces. Detecting emotion can help us in different fields, including opinion mining. With the spread of social media, different platforms like Twitter have become data sources, and the language used in these platforms is informal, making the emotion detection task difficult. EmoPars and ArmanEmo are two new human-labeled emotion datasets for the Persian language. These datasets, especially EmoPars, are suffering from inequality between several samples between two classes. In this paper, we evaluate EmoPars and compare them with ArmanEmo. Throughout this analysis, we use data augmentation techniques, data re-sampling, and class-weights with Transformer-based Pretrained Language Models(PLMs) to handle the imbalance problem of these datasets. Moreover, feature selection is used to enhance the models' performance by emphasizing the text's specific features. In addition, we provide a new policy for selecting data from EmoPars, which selects the high-confidence samples; as a result, the model does not see samples that do not have specific emotion during training. Our model reaches a Macro-averaged F1-score of 0.81 and 0.76 on ArmanEmo and EmoPars, respectively, which are new state-of-the-art results in these benchmarks.

## Files
```
|
|__ augmentation: notebook used for data augmentation
|__ augmented datasets: datasets with augmented samples
|__ data analysis: data analysis notebook
|__ dataset modifier: notebook used to create datasets using thresholds or removing uncertain samples
|__ main dataset: includes EmoPars and ArmanEmo datasets
|__ modified datasets: result of dataset modifier notebook
|__ models: files to create binary classifiers
   |
   |__ data: dictionary used to detect mispelled words
   |__ multilabel: files to train multilabel classifier
```

## Binary Classifer
If you want to make any changes in training the model including using F1CE loss function or using different hyperparameteres, change the related files which in this instance, they are `hyperparameteres.py` and `f1ce_loss.py`.

Furthermore, the feature extraction is not embedded in the main model and you need to use methods in `feature_extraction.py` file to add the features at the end of each sample. Preprocess in embedded in the file.

Finally to train the model use the following command:
```
python3 binary_classifier.py \\
   [target_emotion_label(Anger/Happiness/Hatred/Wonder/Sadness/Fear)] [data_address] [model_name]
```

For help you can use:
```
python3 main.py -h
```


## Multilabel Classifer
Again you may change file like `hyperparameteres.py` for some minor changes. Preprocess in embedded in the file.

Finally to train the model use the following command:
```
python3 binary_classifier.py \\
   [data_address] [model_name] [threshold]
```

For help you can use:
```
python3 main.py -h
```

## Address to PLM Used in This Study
* HooshvareLab/bert-base-parsbert-uncased
* xlm-roberta-base
* xlm-emo-t

## Results
![image](https://user-images.githubusercontent.com/50926437/202441958-805b8386-090f-47a4-85f8-b8052a87385e.png)

![image](https://user-images.githubusercontent.com/50926437/202441962-347fe51a-5ff1-423d-94c9-77e562be6b7c.png)

![image](https://user-images.githubusercontent.com/50926437/202441973-0a05d069-e208-4a00-9f03-5dbbae657171.png)


## Citation
```
@misc{https://doi.org/10.48550/arxiv.2211.08029,
  doi = {10.48550/ARXIV.2211.08029},
  url = {https://arxiv.org/abs/2211.08029},
  author = {Abaskohi, Amirhossein and Sabri, Nazanin and Bahrak, Behnam},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Persian Emotion Detection using ParsBERT and Imbalanced Data Handling Approaches},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
