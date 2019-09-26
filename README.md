# Towards Scalable and Reliable Capsule Networks for Challenging NLP Applications

Accepted in ACL-19: https://arxiv.org/abs/1906.02829

Requirements: Code is written in Python 3 and requires Pytorch.

# Preparation
For quick start, please refer to the [link](https://drive.google.com/open?id=1gPYAMyYo4YLrmx_Egc9wjCqzWx15D7U8) to download EUR-Lex dataset and saved model.

# Code Explanation 
The data_helpers implements the functions for data processing.

The layers.py implements all the main functions of capsule network, including KDE routing, Adaptive KDE routing, Primary Capsule layer and etc.

The network.py provides the wrapper of our model as well as baseline models for the comparison.

The utils.py provides all the evaluation functions such as Precision@1,3,5 and NDCG@1,3,5.

The EUR_Cap.py and EUR_eval.py are for training and inference, respectively.
# Quick start

```bash
SET_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python EUR_eval.py

SET_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python EUR_Cap.py
```

# Performance on EUR-Lex dataset

```bash
NLP-Capsule with Adaptive KDE routing:

Epoch: 20 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.33459
Tst Prec@1,3,5:  [0.7948253557567917, 0.65605864596808838, 0.53666235446312649]  
Tst NDCG@1,3,5:  [0.7948253557567917, 0.70826730037244034, 0.6843311797551882]

Epoch: 21 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.24704
Tst Prec@1,3,5:  [0.79301423027166884, 0.6552824493316064, 0.53666235446312793]  
Tst NDCG@1,3,5:  [0.79301423027166884, 0.70672871614554134, 0.68443643153244704]

Epoch: 22 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.24949
Tst Prec@1,3,5:  [0.79404915912031049, 0.65554118154376773, 0.53800776196636135] 
Tst NDCG@1,3,5:  [0.79404915912031049, 0.70816714976829975, 0.68780244631961929]

Epoch: 23 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.25533
Tst Prec@1,3,5:  [0.8046571798188874, 0.65890470030185422, 0.53604139715394228]  
Tst NDCG@1,3,5:  [0.8046571798188874, 0.71380071010660562, 0.69040247647419262]

Epoch: 24 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.26880
Tst Prec@1,3,5:  [0.80620957309184993, 0.65614489003880982, 0.53661060802069527]  
Tst NDCG@1,3,5:  [0.80620957309184993, 0.7133596479633022, 0.69571103238443532]

Epoch: 25 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.25847
Tst Prec@1,3,5:  [0.80155239327296246, 0.65329883570504454, 0.53448900388098108]  
Tst NDCG@1,3,5:  [0.80155239327296246, 0.7096033706441367, 0.69201706652281636]

Epoch: 26 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.26063
Tst Prec@1,3,5:  [0.80000000000000004, 0.65381630012936431, 0.53350582147477121]  
Tst NDCG@1,3,5:  [0.80000000000000004, 0.71043623399753963, 0.69499344732549306]

Epoch: 27 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.26004
Tst Prec@1,3,5:  [0.79689521345407499, 0.65398878827080587, 0.53376455368693132]  
Tst NDCG@1,3,5:  [0.79689521345407499, 0.71269493382033577, 0.69812854866301688]

Epoch: 28 Iteration: 120/121 (99.2%)  Loss: 0.00000 0.27287
Tst Prec@1,3,5:  [0.79818887451487708, 0.65588615782664883, 0.53500646830530163]  
Tst NDCG@1,3,5:  [0.79818887451487708, 0.71429911265714374, 0.70057615675866636]


XML-CNN:
Epoch: 31 Iteration: 45/46 (97.8%)  Loss: 0.00006 0.15460
Tst Prec@1,3,5:  [0.7583441138421734, 0.6164726175075479, 0.5073738680465716]  
Tst NDCG@1,3,5:  [0.7583441138421734, 0.6661232856458101, 0.644838787586548]

Epoch: 32 Iteration: 45/46 (97.8%)  Loss: 0.00005 0.15354
Tst Prec@1,3,5:  [0.759379042690815, 0.6143165157395448, 0.5062871927554978]  
Tst NDCG@1,3,5:  [0.759379042690815, 0.6648180435110952, 0.6434396675410785]

Epoch: 33 Iteration: 45/46 (97.8%)  Loss: 0.00005 0.15399
Tst Prec@1,3,5:  [0.757567917205692, 0.6169038378611481, 0.507373868046571]  
Tst NDCG@1,3,5:  [0.757567917205692, 0.666160785036582, 0.6440332351720106]

Epoch: 34 Iteration: 45/46 (97.8%)  Loss: 0.00004 0.15153
Tst Prec@1,3,5:  [0.7573091849935317, 0.616645105648988, 0.5099094437257432]  
Tst NDCG@1,3,5:  [0.7573091849935317, 0.6659194956789641, 0.6458294426678642]

Epoch: 35 Iteration: 45/46 (97.8%)  Loss: 0.00005 0.15212
Tst Prec@1,3,5:  [0.7552393272962484, 0.6153514445881856, 0.5092367399741262]  
Tst NDCG@1,3,5:  [0.7552393272962484, 0.6648419426927356, 0.6453632713906606]

Epoch: 36 Iteration: 45/46 (97.8%)  Loss: 0.00004 0.15231
Tst Prec@1,3,5:  [0.7596377749029755, 0.6157826649417857, 0.5093402328589907]  
Tst NDCG@1,3,5:  [0.7596377749029755, 0.6661452963066051, 0.646133349811576]

Epoch: 37 Iteration: 45/46 (97.8%)  Loss: 0.00006 0.15357
Tst Prec@1,3,5:  [0.7570504527813713, 0.6175937904269097, 0.5088227684346699]  
Tst NDCG@1,3,5:  [0.7570504527813713, 0.6670823259018512, 0.6455866525334287]

Epoch: 38 Iteration: 45/46 (97.8%)  Loss: 0.00006 0.16400
Tst Prec@1,3,5:  [0.7583441138421734, 0.6162138852953867, 0.5085122897800777]  
Tst NDCG@1,3,5:  [0.7583441138421734, 0.6658377730303046, 0.6448260229129755]

Epoch: 39 Iteration: 45/46 (97.8%)  Loss: 0.00004 0.15555
Tst Prec@1,3,5:  [0.7578266494178525, 0.6173350582147488, 0.509029754204398]  
Tst NDCG@1,3,5:  [0.7578266494178525, 0.6667396690496684, 0.645590263852396]

Epoch: 40 Iteration: 45/46 (97.8%)  Loss: 0.00004 0.15414
Tst Prec@1,3,5:  [0.7565329883570504, 0.61811125485123, 0.5087192755498058]  
Tst NDCG@1,3,5:  [0.7565329883570504, 0.6674559324640292, 0.6452839523583206]

```

# Reference
If you find our source code useful, please consider citing our work.
```
@inproceedings{zhao2019capsule,
    title = "Towards Scalable and Reliable Capsule Networks for Challenging {NLP} Applications",
    author = "Zhao, Wei and Peng, Haiyun and Eger, Steffen and Cambria, Erik and Yang, Min",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1150",
    doi = "10.18653/v1/P19-1150",
    pages = "1549--1559"
}
```
