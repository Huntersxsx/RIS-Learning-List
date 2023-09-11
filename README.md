<<<<<<< HEAD
# **RIS-Learning-List**

## **Introduction**
This repository introduces **Referring Image Segmentation** task, and collects some related works.
## **Content**

- [Definition](#Definition)
- [Dataset](#Datsets)
- [Evaluation Metric](#Evaluation-Metric)
- [Related Works](#Related-Works)
- [Performance](#Performance)
- [Reference](#Reference)

## **Definition**

**Referring Image Segmentation (RIS)** is a challenging problem at the intersection of computer vision and natural language processing. Given an image and a natural language expression, the goal is to produce a segmentation mask in the image corresponding to the objects referred by the the natural language expression.
![](https://github.com/Huntersxsx/RIS-Learning-List/blob/main/img/definition.png)

## **Datsets**
- [**RefCOCO**](https://arxiv.org/pdf/1608.00272): It contains **19,994 images** with **142,210 referring expressions** for **50,000 objects**, which are collected from the MSCOCO via a two-player game. The dataset is split into 120,624 train, 10,834 validation, 5,657 test A, and 5,095 test B samples, respectively. 
- [**RefCOCO+**](https://arxiv.org/pdf/1608.00272): It contains **141,564 language expressions** with **49,856 objects** in **19,992 images**. The datasetis split into train, validation, test A, and test B with 120,624, 10,758, 5,726, and 4,889 samples, respectively. Compared with RefCOCO dataset, some kinds of **absolute-location words are excluded** from the RefCOCO+ dataset.
- [**G-Ref**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf): It includes **104,560 referring expressions** for **54,822 objects** in **26,711 images**. 
- Expressions in RefCOCO and RefCOCO+ are very succinct (containing 3.5 words on average). In contrast, **expressionsin G-Ref are more complex** (containing 8.4 words on average). Conversely, **RefCOCO and RefCOCO+ tend to have more objects of the same category per image** (3.9 on average) compared to G-Ref (1.6 on average). 

## **Evaluation Metric**
- **overall IoU:** It is the total intersection area divided by the total union area, where both intersection area and union area are accumulated over all test samples (each test sample is an image and a referential expression).
- **mean IoU:** It is the IoU between the prediction and ground truth averaged across all test samples.
- **Precision@X:** It measures the percentage of test images with an IoU score higher than the threshold X ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.

## **Related Works**

- **Shatter and Gather:** [Shatter and Gather: Learning Referring Image Segmentation with Text Supervision](https://arxiv.org/pdf/2308.15512v1.pdf). *in ICCV 2023*.
- **Group-RES:** [Advancing Referring Expression Segmentation Beyond Single Image](https://arxiv.org/pdf/2305.12452.pdf). *in ICCV 2023*. [code](https://github.com/yixuan730/group-res)
- **ETRIS:** [Bridging Vision and Language Encoders: Parameter-Efficient Tuning for Referring Image Segmentation](https://arxiv.org/pdf/2307.11545.pdf). *in ICCV 2023*. [code](https://github.com/kkakkkka/ETRIS)
- **TRIS:** [Referring Image Segmentation Using Text Supervision](https://arxiv.org/pdf/2308.14575.pdf). *in ICCV 2023*. [code](https://github.com/fawnliu/TRIS)
- **RIS-DMMI:** [Beyond One-to-One: Rethinking the Referring Image Segmentation](https://arxiv.org/pdf/2308.13853.pdf). *in ICCV 2023*. [code](https://github.com/toggle1995/RIS-DMMI)
- **BKINet:** [Bilateral Knowledge Interaction Network for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/10227590). *in TMM 2023*. [code](https://github.com/dhding/BKINet)
- **SLViT:** [SLViT: Scale-Wise Language-Guided Vision Transformer for Referring Image Segmentation](https://www.ijcai.org/proceedings/2023/0144.pdf). *in IJCAI 2023*. [code](https://github.com/NaturalKnight/SLViT)
- **WiCo:** [WiCo: Win-win Cooperation of Bottom-up and Top-down Referring Image Segmentation](https://www.ijcai.org/proceedings/2023/0071.pdf). *in IJCAI 2023*.
- **CM-MaskSD:** [CM-MaskSD: Cross-Modality Masked Self-Distillation for Referring Image Segmentation](https://arxiv.org/pdf/2305.11481.pdf). *in Arxiv 2023*.
- **CGFormer:** [Contrastive Grouping with Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Contrastive_Grouping_With_Transformer_for_Referring_Image_Segmentation_CVPR_2023_paper.pdf). *in CVPR 2023*. [code](https://github.com/Toneyaya/CGFormer)
- **Partial-RES:** [Learning to Segment Every Referring Object Point by Point](https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_Learning_To_Segment_Every_Referring_Object_Point_by_Point_CVPR_2023_paper.pdf). *in CVPR 2023*. [code](https://github.com/qumengxue/Partial-RES.git)
- **Zero-shot RIS:** [Zero-shot Referring Image Segmentation with Global-Local Context Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.pdf). *in CVPR 2023*. [code](https://github.com/Seonghoon-Yu/Zero-shot-RIS)
- **MCRES:** [Meta Compositional Referring Expression Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Meta_Compositional_Referring_Expression_Segmentation_CVPR_2023_paper.pdf). *in CVPR 2023*. 
- **PolyFormer:** [PolyFormer: Referring Image Segmentation as Sequential Polygon Generation](https://arxiv.org/pdf/2302.07387.pdf). *in CVPR 2023*. [project](https://polyformer.github.io/)
- **GRES:** [Generalized Referring Expression Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.pdf). *in CVPR 2023*. [project](https://henghuiding.github.io/GRES)
- **SADLR:** [Semantics-Aware Dynamic Localization and Refinement for Referring Image Segmentation](https://arxiv.org/pdf/2303.06345.pdf). *in AAAI 2023*.
- **PCAN:** [Position-Aware Contrastive Alignment for Referring Image Segmentation](https://arxiv.org/pdf/2212.13419.pdf). *in Arxiv 2022*.
- **CoupAlign:** [CoupAlign: Coupling Word-Pixel with Sentence-Mask Alignments for Referring Image Segmentation](https://arxiv.org/pdf/2212.01769.pdf). *in NeurIPS 2022*. [code](https://gitee.com/mindspore/models/tree/master/research/cv/CoupAlign)
- **CRSCNet:** [Cross-Modal Recurrent Semantic Comprehension for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9998537). *in TCSVT 2022*.
- **LGCT:** [Local-global coordination with transformers for referring image segmentation](https://www.sciencedirect.com/science/article/pii/S0925231222015119). *in Neurocomputing 2022*. 
- **RES&REG:** [A Unified Mutual Supervision Framework for Referring Expression Segmentation and Generation](https://arxiv.org/pdf/2211.07919.pdf). *in Arxiv 2022*.
- **VLT:** [VLT: Vision-Language Transformer and Query Generation for Referring Segmentation](https://arxiv.org/pdf/2210.15871.pdf). *in TPAMI 2022*. [code](https://github.com/henghuiding/Vision-Language-Transformer)
- [Learning From Box Annotations for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9875225). *in TNNLS 2022*. [code](https://github.com/fengguang94/Weakly-Supervised-RIS)
- [Instance-Specific Feature Propagation for Referring Segmentation](https://ieeexplore.ieee.org/abstract/document/9745353). *in TMM 2022*. 
- **SeqTR:** [SeqTR: A Simple Yet Universal Network for Visual Grounding](https://arxiv.org/pdf/2203.16265.pdf). *in ECCV 2022*. [code](https://github.com/sean-zhuh/SeqTR)
- **LAVT:** [LAVT: Language-Aware Vision Transformer for Referring Image Segmentation](https://arxiv.org/abs/2112.02244). *in CVPR 2022*. [code](https://github.com/yz93/LAVT-RIS)
- **CRIS:** [CRIS: CLIP-Driven Referring Image Segmentation](https://arxiv.org/abs/2111.15174). *in CVPR 2022*. [code](https://github.com/DerrickWang005/CRIS.pytorch)
- **ReSTR:** [ReSTR: Convolution-free Referring Image Segmentation Using Transformers](https://www.microsoft.com/en-us/research/uploads/prod/2022/03/01404.pdf). *in CVPR 2022*. [project](http://cvlab.postech.ac.kr/research/restr/)
- [Bidirectional relationship inferring network for referring image localization and segmentation](https://ieeexplore.ieee.org/document/9526878). *in TNNLS 2021*. 
- **RefTR:** [Referring Transformer: A One-step Approach to Multi-task Visual Grounding](https://openreview.net/pdf?id=J64lDCrYGi). *in NeurIPS 2021*. 
- **TV-Net:** [Two-stage Visual Cues Enhancement Network for Referring Image Segmentation](https://arxiv.org/abs/2110.04435). *in ACM MM 2021*. [code](https://github.com/sxjyjay/tv-net)
- **VLT:** [Vision-Language Transformer and Query Generation for Referring Segmentation](https://arxiv.org/abs/2108.05565). *in ICCV 2021*. [code](https://github.com/henghuiding/Vision-Language-Transformer)
- **MDETR:** [MDETR - Modulated Detection for End-to-End Multi-Modal Understanding](https://arxiv.org/abs/2104.12763). *in ICCV 2021*. [code](https://github.com/ashkamath/mdetr)
- **EFNet:** [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Encoder_Fusion_Network_With_Co-Attention_Embedding_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf). *in CVPR 2021*. [code](https://github.com/fengguang94/CEFNet)
- **BUSNet:** [Bottom-Up Shift and Reasoning for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Bottom-Up_Shift_and_Reasoning_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf). *in CVPR 2021*. [code](https://github.com/incredibleXM/BUSNet)
- **LTS:** [Locate then Segment: A Strong Pipeline for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Locate_Then_Segment_A_Strong_Pipeline_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf). *in CVPR 2021*. 
- **CGAN:** [Cascade Grouped Attention Network for Referring Expression Segmentation](https://dl.acm.org/doi/abs/10.1145/3394171.3414006). *in ACM MM 2020*.
- **LSCM:** [Linguistic Structure Guided Context Modeling for Referring Image Segmentation](http://colalab.org/media/paper/Linguistic_Structure_Guided_Context_Modeling_for_Referring_Image_Segmentation.pdf). *in ECCV 2020*. 
- **CMPC-Refseg:** [Referring Image Segmentation via Cross-Modal Progressive Comprehension](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Referring_Image_Segmentation_via_Cross-Modal_Progressive_Comprehension_CVPR_2020_paper.pdf). *in CVPR 2020*. [code](https://github.com/spyflying/CMPC-Refseg)
- **BRINet:** [Bi-directional Relationship Inferring Network for Referring Image Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Bi-Directional_Relationship_Inferring_Network_for_Referring_Image_Segmentation_CVPR_2020_paper.pdf). *in CVPR 2020*. [code](https://github.com/fengguang94/CVPR2020-BRINet)
- **PhraseCut:** [PhraseCut: Language-based Image Segmentation in the Wild](https://people.cs.umass.edu/~smaji/papers/phrasecut+supp-cvpr20.pdf). *in CVPR 2020*. [code](https://github.com/ChenyunWu/PhraseCutDataset)
- **MCN:** [Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation](https://arxiv.org/abs/2003.08813). *in CVPR 2020*. [code](https://github.com/luogen1996/MCN)
- [Dual Convolutional LSTM Network for Referring Image Segmentation](https://arxiv.org/abs/2001.11561). *in TMM 2020*. 
- **lang2seg:** [Referring Expression Object Segmentation with Caption-Aware Consistency](https://arxiv.org/pdf/1910.04748.pdf). *in BMVC 2019*. [code](https://github.com/wenz116/lang2seg)
- **STEP:** [See-Through-Text Grouping for Referring Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_See-Through-Text_Grouping_for_Referring_Image_Segmentation_ICCV_2019_paper.pdf). *in ICCV 2019*. 
- **CMSA-Net:** [Cross-Modal Self-Attention Network for Referring Image Segmentation](https://arxiv.org/pdf/1904.04745.pdf). *in CVPR 2019*. [code](https://github.com/lwye/CMSA-Net)
- **KWA:** [Key-Word-Aware Network for Referring Expression Image Segmentation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengcan_Shi_Key-Word-Aware_Network_for_ECCV_2018_paper.pdf). *in ECCV 2018*. [code](https://github.com/shihengcan/key-word-aware-network-pycaffe)
- **DMN:** [Dynamic Multimodal Instance Segmentation Guided by Natural Language Queries](http://openaccess.thecvf.com/content_ECCV_2018/papers/Edgar_Margffoy-Tuay_Dynamic_Multimodal_Instance_ECCV_2018_paper.pdf). *in ECCV 2018*. [code](https://github.com/BCV-Uniandes/DMS)
- **RRN:** [Referring Image Segmentation via Recurrent Refinement Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Referring_Image_Segmentation_CVPR_2018_paper.pdf). *in CVPR 2018*. [code](https://github.com/liruiyu/referseg_rrn)
- **MAttNet:** [MAttNet: Modular Attention Network for Referring Expression Comprehension](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_MAttNet_Modular_Attention_CVPR_2018_paper.pdf). *in CVPR 2018*. [code](https://github.com/lichengunc/MAttNet)
- **RMI:** [Recurrent Multimodal Interaction for Referring Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Recurrent_Multimodal_Interaction_ICCV_2017_paper.pdf). *in ICCV 2017*. [code](https://github.com/chenxi116/TF-phrasecut-public)
- **LSTM-CNN:** [Segmentation from natural language expressions](https://arxiv.org/pdf/1603.06180.pdf). *in ECCV 2016*. [code](https://github.com/ronghanghu/text_objseg)


## Performance


## Reference
[MarkMoHR / Awesome-Referring-Image-Segmentation](https://github.com/MarkMoHR/Awesome-Referring-Image-Segmentation)
=======
# **RIS-Learning-List**

## **Introduction**
This repository introduces **Referring Image Segmentation** task, and collects some related works.
## **Content**

- [Definition](#Definition)
- [Dataset](#Datsets)
- [Evaluation Metric](#Evaluation-Metric)
- [Related Works](#Related-Works)
- [Performance](#Performance)
- [Reference](#Reference)

## **Definition**

**Referring Image Segmentation (RIS)** is a challenging problem at the intersection of computer vision and natural language processing. Given an image and a natural language expression, the goal is to produce a segmentation mask in the image corresponding to the objects referred by the the natural language expression.
![](https://github.com/Huntersxsx/RIS-Learning-List/blob/main/img/definition.png)

## **Datsets**
- [**RefCOCO**](https://arxiv.org/pdf/1608.00272): It contains **19,994 images** with **142,210 referring expressions** for **50,000 objects**, which are collected from the MSCOCO via a two-player game. The dataset is split into 120,624 train, 10,834 validation, 5,657 test A, and 5,095 test B samples, respectively. 
- [**RefCOCO+**](https://arxiv.org/pdf/1608.00272): It contains **141,564 language expressions** with **49,856 objects** in **19,992 images**. The datasetis split into train, validation, test A, and test B with 120,624, 10,758, 5,726, and 4,889 samples, respectively. Compared with RefCOCO dataset, some kinds of **absolute-location words are excluded** from the RefCOCO+ dataset.
- [**G-Ref**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf): It includes **104,560 referring expressions** for **54,822 objects** in **26,711 images**. 
- Expressions in RefCOCO and RefCOCO+ are very succinct (containing 3.5 words on average). In contrast, **expressionsin G-Ref are more complex** (containing 8.4 words on average). Conversely, **RefCOCO and RefCOCO+ tend to have more objects of the same category per image** (3.9 on average) compared to G-Ref (1.6 on average). 

## **Evaluation Metric**
- **overall IoU:** It is the total intersection area divided by the total union area, where both intersection area and union area are accumulated over all test samples (each test sample is an image and a referential expression).
- **mean IoU:** It is the IoU between the prediction and ground truth averaged across all test samples.
- **Precision@X:** It measures the percentage of test images with an IoU score higher than the threshold X ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.

## **Related Works**

- **CM-MaskSD:** [CM-MaskSD: Cross-Modality Masked Self-Distillation for Referring Image Segmentation](https://arxiv.org/pdf/2305.11481.pdf). *in Arxiv 2023*.
- **CGFormer:** [Contrastive Grouping with Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Contrastive_Grouping_With_Transformer_for_Referring_Image_Segmentation_CVPR_2023_paper.pdf). *in CVPR 2023*. [code](https://github.com/Toneyaya/CGFormer)
- **Partial-RES:** [Learning to Segment Every Referring Object Point by Point](https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_Learning_To_Segment_Every_Referring_Object_Point_by_Point_CVPR_2023_paper.pdf). *in CVPR 2023*. [code](https://github.com/qumengxue/Partial-RES.git)
- **Zero-shot RIS:** [Zero-shot Referring Image Segmentation with Global-Local Context Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.pdf). *in CVPR 2023*. [code](https://github.com/Seonghoon-Yu/Zero-shot-RIS)
- **MCRES:** [Meta Compositional Referring Expression Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Meta_Compositional_Referring_Expression_Segmentation_CVPR_2023_paper.pdf). *in CVPR 2023*. 
- **PolyFormer:** [PolyFormer: Referring Image Segmentation as Sequential Polygon Generation](https://arxiv.org/pdf/2302.07387.pdf). *in CVPR 2023*. [project](https://polyformer.github.io/)
- **GRES:** [Generalized Referring Expression Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.pdf). *in CVPR 2023*. [project](https://henghuiding.github.io/GRES)
- **SADLR:** [Semantics-Aware Dynamic Localization and Refinement for Referring Image Segmentation](https://arxiv.org/pdf/2303.06345.pdf). *in AAAI 2023*.
- **PCAN:** [Position-Aware Contrastive Alignment for Referring Image Segmentation](https://arxiv.org/pdf/2212.13419.pdf). *in Arxiv 2022*.
- **CoupAlign:** [CoupAlign: Coupling Word-Pixel with Sentence-Mask Alignments for Referring Image Segmentation](https://arxiv.org/pdf/2212.01769.pdf). *in NeurIPS 2022*. [code](https://gitee.com/mindspore/models/tree/master/research/cv/CoupAlign)
- **CRSCNet:** [Cross-Modal Recurrent Semantic Comprehension for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9998537). *in TCSVT 2022*.
- **LGCT:** [Local-global coordination with transformers for referring image segmentation](https://www.sciencedirect.com/science/article/pii/S0925231222015119). *in Neurocomputing 2022*. 
- **RES&REG:** [A Unified Mutual Supervision Framework for Referring Expression Segmentation and Generation](https://arxiv.org/pdf/2211.07919.pdf). *in Arxiv 2022*.
- **VLT:** [VLT: Vision-Language Transformer and Query Generation for Referring Segmentation](https://arxiv.org/pdf/2210.15871.pdf). *in TPAMI 2022*. [code](https://github.com/henghuiding/Vision-Language-Transformer)
- [Learning From Box Annotations for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9875225). *in TNNLS 2022*. [code](https://github.com/fengguang94/Weakly-Supervised-RIS)
- [Instance-Specific Feature Propagation for Referring Segmentation](https://ieeexplore.ieee.org/abstract/document/9745353). *in TMM 2022*. 
- **SeqTR:** [SeqTR: A Simple Yet Universal Network for Visual Grounding](https://arxiv.org/pdf/2203.16265.pdf). *in ECCV 2022*. [code](https://github.com/sean-zhuh/SeqTR)
- **LAVT:** [LAVT: Language-Aware Vision Transformer for Referring Image Segmentation](https://arxiv.org/abs/2112.02244). *in CVPR 2022*. [code](https://github.com/yz93/LAVT-RIS)
- **CRIS:** [CRIS: CLIP-Driven Referring Image Segmentation](https://arxiv.org/abs/2111.15174). *in CVPR 2022*. [code](https://github.com/DerrickWang005/CRIS.pytorch)
- **ReSTR:** [ReSTR: Convolution-free Referring Image Segmentation Using Transformers](https://www.microsoft.com/en-us/research/uploads/prod/2022/03/01404.pdf). *in CVPR 2022*. [project](http://cvlab.postech.ac.kr/research/restr/)
- [Bidirectional relationship inferring network for referring image localization and segmentation](https://ieeexplore.ieee.org/document/9526878). *in TNNLS 2021*. 
- **RefTR:** [Referring Transformer: A One-step Approach to Multi-task Visual Grounding](https://openreview.net/pdf?id=J64lDCrYGi). *in NeurIPS 2021*. 
- **TV-Net:** [Two-stage Visual Cues Enhancement Network for Referring Image Segmentation](https://arxiv.org/abs/2110.04435). *in ACM MM 2021*. [code](https://github.com/sxjyjay/tv-net)
- **VLT:** [Vision-Language Transformer and Query Generation for Referring Segmentation](https://arxiv.org/abs/2108.05565). *in ICCV 2021*. [code](https://github.com/henghuiding/Vision-Language-Transformer)
- **MDETR:** [MDETR - Modulated Detection for End-to-End Multi-Modal Understanding](https://arxiv.org/abs/2104.12763). *in ICCV 2021*. [code](https://github.com/ashkamath/mdetr)
- **EFNet:** [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Encoder_Fusion_Network_With_Co-Attention_Embedding_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf). *in CVPR 2021*. [code](https://github.com/fengguang94/CEFNet)
- **BUSNet:** [Bottom-Up Shift and Reasoning for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Bottom-Up_Shift_and_Reasoning_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf). *in CVPR 2021*. [code](https://github.com/incredibleXM/BUSNet)
- **LTS:** [Locate then Segment: A Strong Pipeline for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Locate_Then_Segment_A_Strong_Pipeline_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf). *in CVPR 2021*. 
- **CGAN:** [Cascade Grouped Attention Network for Referring Expression Segmentation](https://dl.acm.org/doi/abs/10.1145/3394171.3414006). *in ACM MM 2020*.
- **LSCM:** [Linguistic Structure Guided Context Modeling for Referring Image Segmentation](http://colalab.org/media/paper/Linguistic_Structure_Guided_Context_Modeling_for_Referring_Image_Segmentation.pdf). *in ECCV 2020*. 
- **CMPC-Refseg:** [Referring Image Segmentation via Cross-Modal Progressive Comprehension](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Referring_Image_Segmentation_via_Cross-Modal_Progressive_Comprehension_CVPR_2020_paper.pdf). *in CVPR 2020*. [code](https://github.com/spyflying/CMPC-Refseg)
- **BRINet:** [Bi-directional Relationship Inferring Network for Referring Image Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Bi-Directional_Relationship_Inferring_Network_for_Referring_Image_Segmentation_CVPR_2020_paper.pdf). *in CVPR 2020*. [code](https://github.com/fengguang94/CVPR2020-BRINet)
- **PhraseCut:** [PhraseCut: Language-based Image Segmentation in the Wild](https://people.cs.umass.edu/~smaji/papers/phrasecut+supp-cvpr20.pdf). *in CVPR 2020*. [code](https://github.com/ChenyunWu/PhraseCutDataset)
- **MCN:** [Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation](https://arxiv.org/abs/2003.08813). *in CVPR 2020*. [code](https://github.com/luogen1996/MCN)
- [Dual Convolutional LSTM Network for Referring Image Segmentation](https://arxiv.org/abs/2001.11561). *in TMM 2020*. 
- **lang2seg:** [Referring Expression Object Segmentation with Caption-Aware Consistency](https://arxiv.org/pdf/1910.04748.pdf). *in BMVC 2019*. [code](https://github.com/wenz116/lang2seg)
- **STEP:** [See-Through-Text Grouping for Referring Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_See-Through-Text_Grouping_for_Referring_Image_Segmentation_ICCV_2019_paper.pdf). *in ICCV 2019*. 
- **CMSA-Net:** [Cross-Modal Self-Attention Network for Referring Image Segmentation](https://arxiv.org/pdf/1904.04745.pdf). *in CVPR 2019*. [code](https://github.com/lwye/CMSA-Net)
- **KWA:** [Key-Word-Aware Network for Referring Expression Image Segmentation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengcan_Shi_Key-Word-Aware_Network_for_ECCV_2018_paper.pdf). *in ECCV 2018*. [code](https://github.com/shihengcan/key-word-aware-network-pycaffe)
- **DMN:** [Dynamic Multimodal Instance Segmentation Guided by Natural Language Queries](http://openaccess.thecvf.com/content_ECCV_2018/papers/Edgar_Margffoy-Tuay_Dynamic_Multimodal_Instance_ECCV_2018_paper.pdf). *in ECCV 2018*. [code](https://github.com/BCV-Uniandes/DMS)
- **RRN:** [Referring Image Segmentation via Recurrent Refinement Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Referring_Image_Segmentation_CVPR_2018_paper.pdf). *in CVPR 2018*. [code](https://github.com/liruiyu/referseg_rrn)
- **MAttNet:** [MAttNet: Modular Attention Network for Referring Expression Comprehension](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_MAttNet_Modular_Attention_CVPR_2018_paper.pdf). *in CVPR 2018*. [code](https://github.com/lichengunc/MAttNet)
- **RMI:** [Recurrent Multimodal Interaction for Referring Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Recurrent_Multimodal_Interaction_ICCV_2017_paper.pdf). *in ICCV 2017*. [code](https://github.com/chenxi116/TF-phrasecut-public)
- **LSTM-CNN:** [Segmentation from natural language expressions](https://arxiv.org/pdf/1603.06180.pdf). *in ECCV 2016*. [code](https://github.com/ronghanghu/text_objseg)


## Performance


## Reference
[MarkMoHR / Awesome-Referring-Image-Segmentation](https://github.com/MarkMoHR/Awesome-Referring-Image-Segmentation)
>>>>>>> b9a3a4c91e17b067a52b1562390e2e675608f444
