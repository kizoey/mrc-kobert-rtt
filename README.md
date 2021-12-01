# mrc-kobert-rtt
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"/></a>
</p>
<b><i>MRC kobert RTT</i></b> is a research project on 'improving Korean reading comprehension model using round-trip translation (data augmentation method) for KorQuAD v2 dataset'. <b>KorQuAD v2</b> dataset is a Korean Machine Reading Comprehension dataset consisting of a total of 100,000+ pairs, including 20,000+ pairs of question and answer data from KorQuAD v1. KorQuAD v2 dataset can be found at their <a href=https://korquad.github.io/>leaderboard website</a>. The dataset are very long passages with tables and ordered/unordered lists included. Therefore, it was essential to understand the structure of the document through HTML tag parsing.<br>
<br>
<b>Round-trip translation</b>, also known as back-and-forth translation, is the data augmentation technique of translating a word, phrase or text into another language (forward translation), then translating the result back into the original language (back translation), using machine translation utils. Its effectiveness to enhance performance for reading comprehension models is proven in many papers as it allows the model to learn rich expressions to a greater extent.<br>
<br>
For the baseline model, we employed <a href=https://github.com/SKTBrain/KoBERT>KoBERT(Korean BERT)</a> pre-trained model to deal with Korean dataset. We then compared the baseline performance with data augmented models.


<h2> major Contributions </h2>

- Applied data augmentation technique to enhance **KoBERT** model performance
- Analysis of the structure and distribution of the **KorQuAD v2** dataset
- Research on previous papers related to reading comprehension and KorQuAD dataset

<h2> Requirements </h2>

```
torch>=1.1.0
transformers==2.9.1
tensorboardX>=2.0
numpy~=1.21.2
tqdm~=4.62.2
setproctitle~=1.2.2
```

<h2> Directory </h2>

```
MRC_KoBERT
┖ bert_models       
  ┖ src
┖ config        
  ┖ no_rtt
    ┖ kobert_config.json
  ┖ rtt
    ┖ kobert_config.json
┖ data
  ┖ no_rtt
    ┖ korquadv2_dev_ori.json
    ┖ korquadv2_train_ori.json
  ┖ rtt
    ┖ korquadv2_dev_ori.json
    ┖ korquadv2_train_rtt.json
┖ eval_results                  
┖ utils
  ┖ requirements.txt
  ┖ utils.py
data.py
dataloader.py
run_no_rtt.py
run_rtt.py
train_no_rtt.sh
train_rtt.sh
```

### _files_
- **bert_models**: QA에 사용되는 pre-trained 모델들의 configuration과 tokenization이 저장된 디렉토리
- **config**
    - **kobert_config.json**
    : 직접 변경할수 있도록 모델 학습/성능 평가 과정에서 사용되는 argument 저장된 집합

- **data**
    - **korquadv2_train_ori.json**: KorQuAD v2 학습 데이터
    - **korquadv2_dev_ori.json**: KorQuAD v2 검증 데이터
    - **korquadv2_train_rtt.json**: RTT를 적용해서 2배로 증강시킨 학습 데이터

- **eval_results**: 학습, 검증한 뒤에 검증 결과 txt 파일을 저장하는 디렉토리
- **utils**
    - **requirements.txt**: 프로젝트 돌릴 시 설치해야 되는 패키지 리스트
    - **utils.py**: pre-trained configuration, tokenizer, model이 저장된 디렉토리

- **data.py**: pre-trained된 SquadExample과 데이터에서 각 구성요소를 불러올 수 있는 Dataset 클래스를 inherit한 다음, KorQuAD v2 데이터에 맞춰 그 구조를 변형한 뒤, KorQuAD v2 데이터를 불러오는 파일

- **dataloader.py**: data.py를 통해 불러와진 데이터를 데이터로더를 이용해 torch 데이터셋으로 변환해주는 파일

<h2> Train </h2>

```
sh train_no_rtt.sh
sh train_rtt.sh
```
