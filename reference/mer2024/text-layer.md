## PDF page 1

MER 2024: Semi-Supervised Learning, Noise Robustness, and
Open-Vocabulary Multimodal Emotion Recognition
Zheng Lian
Institute of Automation, Chinese
Academy of Sciences (CAS)
Beijing, China
Haiyang Sun
Institute of Automation, CAS
Beijing, China
Licai Sun
Institute of Automation, CAS
Beijing, China
Zhuofan Wen
Institute of Automation, CAS
Beijing, China
Siyuan Zhang
Institute of Automation, CAS
Beijing, China
Shun Chen
Institute of Automation, CAS
Beijing, China
Hao Gu
Institute of Automation, CAS
Beijing, China
Jinming Zhao
Renmin University of China
Beijing, China
Ziyang Ma
Shanghai Jiao Tong University
Shanghai, China
Xie Chen
Shanghai Jiao Tong University
Shanghai, China
Jiangyan Yi
Institute of Automation, CAS
Beijing, China
Rui Liu
Inner Mongolia University
Inner Mongolia, China
Kele Xu
National University of Defense
Technology
Beijing, China
Bin Liu
Institute of Automation, CAS
Beijing, China
Erik Cambria
Nanyang Technological University
Singapore
Guoying Zhao
University of Oulu
Oulu, Finland
Björn W. Schuller
Imperial College London
London, United Kingdom
Jianhua Tao
Tsinghua University
Beijing, China
ABSTRACT
Multimodal emotion recognition is an important research topic
in artificial intelligence. Over the past few decades, researchers
have made remarkable progress by increasing the dataset size and
building more effective algorithms. However, due to problems such
as complex environments and inaccurate annotations, current sys-
tems are hard to meet the demands of practical applications. There-
fore, we organize the MER series of competitions to promote the
development of this field. Last year, we launched MER20231, fo-
cusing on three interesting topics: multi-label learning, noise ro-
bustness, and semi-supervised learning. In this year’s MER20242,
besides expanding the dataset size, we further introduce a new
track around open-vocabulary emotion recognition. The main pur-
pose of this track is that existing datasets usually fix the label
1http://merchallenge.cn/mer2023
2https://zeroqiaoba.github.io/MER2024-website
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
ACM MM, 2024, Melbourne, Australia
© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
space and use majority voting to enhance the annotator consis-
tency. However, this process may lead to inaccurate annotations,
such as ignoring non-majority or non-candidate labels. In this track,
we encourage participants to generate any number of labels in any
category, aiming to describe emotional states as accurately as pos-
sible. Our baseline code relies on MERTools3 and is available at:
https://github.com/zeroQiaoba/MERTools/tree/master/MER2024.
CCS CONCEPTS
• Computing methodologies →Neural networks; Artifcial
intelligence; Computer vision; Natural language processing.
KEYWORDS
MER 2024, multimodal emotion recognition, semi-supervised learn-
ing, noise robustness, open-vocabulary emotion recognition
ACM Reference Format:
Zheng Lian, Haiyang Sun, Licai Sun, Zhuofan Wen, Siyuan Zhang, Shun
Chen, Hao Gu, Jinming Zhao, Ziyang Ma, Xie Chen, Jiangyan Yi, Rui Liu,
Kele Xu, Bin Liu, Erik Cambria, Guoying Zhao, Björn W. Schuller, and Jian-
hua Tao. 2024. MER 2024: Semi-Supervised Learning, Noise Robustness,
and Open-Vocabulary Multimodal Emotion Recognition. In Proceedings of
the 32nd ACM International Conference on Multimedia (MM’24), October 28-
November 1, 2024, Melbourne, Australia. ACM, New York, NY, USA, 8 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
3https://github.com/zeroQiaoba/MERTools
arXiv:2404.17113v4  [cs.LG]  18 Jul 2024


## PDF page 2

ACM MM, 2024, Melbourne, Australia
author name and author name, et al.
1
INTRODUCTION
Multimodal emotion recognition plays an important role in human-
computer interaction. Recently, researchers have made significant
progress in this field. However, this task is still not well solved
and its performance still cannot meet the requirements of practical
applications [1]. To this end, last year, we launched MER2023 [2],
focusing on three important topics: multi-label learning, noise ro-
bustness, and semi-supervised learning. This year, we continue the
latter two tracks and introduce a new track around open-vocabulary
emotion recognition, aiming to describe emotional states accurately.
First, it is hard to collect samples with emotion labels. On the
one hand, the collected samples are often emotionless (i.e. neutral)
[3]. On the other hand, researchers usually hire multiple annotators
and use majority voting to improve label consistency [4], greatly
increasing the annotation cost. To address the sparsity of emotional
data, previous works used unlabeled data and focused on unsu-
pervised or semi-supervised learning. Recently, MERBench [1] has
conducted a systematic analysis, pointing out the necessity of using
unlabeled data from the same domain as labeled data. Therefore,
we provide a large number of human-centric unlabeled videos in
MER-SEMI and encourage participants to explore more effective
unsupervised or semi-supervised strategies.
Secondly, in real scenarios, we cannot ensure that every video
is free of audio noise and every frame is in high resolution. To
copy with complex environments, emotion recognition systems
should have a certain degree of noise robustness. Therefore, we
organize MER-NOISE to fairly evaluate the noise robustness of
different systems. Although there are many types of noise, we
only consider the two most common ones: audio additive noise
and image blur noise. In this track, we encourage participants to
use data augmentation [5] or other effective techniques [6, 7] to
improve performance under noisy conditions.
Thirdly, to improve label consistency, existing datasets usually
restrict the label space to a few discrete categories, then employ
multiple annotators and use majority voting to select the most
likely label. However, this process may cause some correct but
non-candidate or non-majority labels to be ignored. Therefore, we
introduce MER-OV, centered around open-vocabulary emotion
recognition. We encourage participants to generate any number
of labels in any category, trying to describe emotional states accu-
rately.
In summary, MER2024 consists of three tracks: MER-SEMI, MER-
NOISE, and MER-OV. In MER-SEMI, we encourage participants to
use unlabeled data during training; in MER-NOISE, we focus on
noise robustness; in MER-OV, we require participants to describe
emotional states as accurately as possible. The MER series of chal-
lenges aims to provide participants with a common platform to
fairly compare the performance of different techniques. In the rest
of this paper, we will introduce the datasets, baselines, evaluation
metrics, and experimental results in detail.
2
CHALLENGE DATASET
MER2024 is an extended version of MER2023 [2] and its construc-
tion process is summarized in Figure 1. Specifically, MER2023 con-
tains four subsets: Train&Val, MER-MULTI, MER-NOISE, and MER-
SEMI. In the last subset, in addition to the labeled data, it also
Unlabeled data
Train&Val
MER-MULTI
MER-NOISE
MER-SEMI
Train&Val
Unlabeled data
MER-SEMI
MER-NOISE
MER-OV
OV Annotation
Merge
MER2023
MER2024
Figure 1: Dataset construction pipeline of MER2024.
contains a large amount of unlabeled data. In MER2024, we merge
all labeled samples and obtain the updated Train&Val. Meanwhile,
we collect more unlabeled data and select a subset for annotation,
getting MER-SEMI and MER-NOISE. For MER-OV, we select 332
samples from Train&Val and provide open-vocabulary labels. Table
1 summarizes the statistics of these datasets.
2.1
Data Collection
As shown in Figure 1, MER2024 expands the dataset size. Its data
collection process is borrowed from MER2023 and includes two key
steps: video cutting and video filtering.
Video Cutting. The raw data in MER2024 comes from movies
and TV series, which are close to real scenarios. However, these
videos are usually long and have many characters, so they need to
be segmented into video clips. In this process, we require that the
content in these video clips is relatively complete. For videos with
subtitles, the timestamps in the subtitles provide accurate boundary
information, and we use them for video segmentation. For videos
without subtitles, we use the voice activity detection toolkit, Silero
VAD4, to segment the video and the speaker identification toolkit,
Deep Speaker5, to merge the consecutive clips if they are likely to
be from the same speaker.
Video Filtering. Through the above process, we can generate
video clips containing relatively complete content from the same
speaker. However, it only guarantees that the audio is from the
same person, but cannot guarantee that there is only one person
in the visual frames. Therefore, we further filter these video clips.
Specifically, we use the face detection toolkit, YuNet6, to ensure
that most frames contain only one face. Then, we use the face recog-
nition toolkit, face.evoLVe7, to ensure that these faces belong to the
same person. Meanwhile, the length of video clips is also important.
Video clips that are too short may not convey the complete mean-
ing; video clips that are too long may contain emotional changes,
4https://github.com/snakers4/silero-vad
5https://github.com/philipperemy/deep-speaker
6https://github.com/ShiqiYu/libfacedetection
7https://github.com/ZhaoJ9014/face.evoLVe


## PDF page 3

MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition
ACM MM, 2024, Melbourne, Australia
Table 1: Statistics of MER2023 and MER2024. Here, “A/B”
means that there are B samples in the subset and we only
annotate A samples for performance evaluation.
MER2023
# of samples
(labeled/whole) MER2024
# of samples
(labeled/whole)
Train&Val
3373
Train&Val
5030
MER-MULTI
411
MER-SEMI
1169/115595
MER-NOISE
412
MER-NOISE
1170/115595
MER-SEMI
834/73982
MER-OV
322
making it difficult to describe the emotional state. Therefore, we
only select video clips between 2∼16 seconds.
2.2
MER-SEMI and MER-NOISE
Annotating all unlabeled data requires a lot of labor costs. To reduce
the cost, we only select samples with a high probability of explicit
emotions. Specifically, we use the top-16 results in last year’s chal-
lenge and calculate the proportion of primary labels as the basis for
selection. For example, if a sample is predicted as happy 10 times
and neutral 6 times, its score is 𝑣= max(6, 10)/16. Then, taking into
account the calculated score and class balance, we select a total of
6,000 samples for labeling. During the annotation process, we hire
5 annotators and only select samples in which at least 4 annotators
assign the same label, resulting in 2,339 labeled samples. All these
samples are equally divided into two parts, one for MER-SEMI and
the other for MER-NOISE.
For MER-NOISE, we additionally add noise to videos. This paper
considers two types of noise: audio additive noise and image blur
noise. For the audio, we select noise from the MUSAN dataset [32],
which contains three subsets: music, noise, and speech. The noise in
the first two subsets may convey emotions and affect the emotion
of the raw audio. For example, rain and thunder may lead to a neg-
ative sentiment, while a pleasant breeze may generate a positive
sentiment. Therefore, we only use the noise in the last subset. Specif-
ically, we randomly select a speech-to-noise ratio (SNR) between
5dB∼10dB and randomly select noise from the speech subset. For
the video, image blur is a common noise. To generate blurry images,
we downsample the image to lose some details and then upsample
the low-resolution image to keep the size unchanged. This paper
randomly selects a downsampling factor from 𝑟= {1, 2, 4}.
2.3
MER-OV
Unlike MER-SEMI and MER-NOISE which predict the most likely
emotion within a fixed label space, MER-OV needs to predict any
number of emotions in any category. Previously, researchers made
an initial attempt at this task [8]. They first annotated emotion-
related acoustic and visual clues. Then, they relied on the reasoning
ability of LLMs to disambiguate subtitles using these clues. This
process can generate descriptions with rich emotions. After that,
they used GPT-3.5 (“gpt-3.5-turbo-16k-0613”) to extract all labels
using the following prompt: Please assume the role of an expert in
the field of emotions. We provide clues that may be related to the emo-
tions of the characters. Based on the provided clues, please identify
the emotional states of the main characters. Please separate different
emotional categories with commas and output only the clearly iden-
tifiable emotional categories in a list format. If none are identified,
please output an empty list.
Finally, they performed the manual check and got the ground
truth 𝑌𝑔𝑡. Through the above process, each sample can have an aver-
age of 3 labels. However, due to the high cost, they only annotated
332 samples [8]. For MER-OV, participants can borrow some basic
ideas from this process. Meanwhile, we encourage participants to
use MLLMs or to further conduct instruction fine-tuning.
2.4
Challenge Protocol
To download the dataset, participants should fill out an End User
License Agreement (EULA)8. It asks participants to use this dataset
only for academic research and not to edit or upload it to the Inter-
net. For MER-SEMI and MER-NOISE, each team should predict the
most likely label among 6 categories (i.e., worried, happy, neutral,
angry, surprised, and sad). For MER-OV, each team can submit any
number of labels in any category. Meanwhile, participants cannot
use closed-source models (such as GPT or Claude) in MER-OV.
For all tracks, participants should predict results for 115,595 un-
labeled data, although we only evaluate a small subset of them.
It requires participants to focus on the generalization ability and
develop systems that do not require a lot of inference time. Addition-
ally, participants cannot manually annotate samples in MER2024.
We will ask them to submit the code for further checking. Finally,
each team should submit a paper describing their method. For other
requirements, please refer to our official website9.
3
BASELINES AND EVALUATION METRICS
In this section, we first introduce the baselines of three tracks. Then,
we illustrate implementation details and evaluation metrics.
3.1
MER-SEMI and MER-NOISE
For MER-SEMI and MER-NOISE, we build baselines based on MER-
Tools10. Feature selection and fusion are crucial for emotion recogni-
tion systems. For feature selection, we adopt the recommendations
of MERBench [1]. Language compatibility is important for lexical
and acoustic features, so we mainly select encoders trained on Chi-
nese corpora. Domain compatibility is important for visual features.
Therefore, besides encoders trained on action or caption datasets,
we also select encoders trained on human-centric videos. Table 2
lists the model cards of some representative encoders.
Regarding the fusion strategy, MERBench points out that the
attention mechanism can achieve relatively good performance [1].
The reason lies in that the labeled samples are limited in emotion
recognition. Complex architectures may cause overfitting problems,
which will affect the model’s generalization ability on unseen data.
Assume that acoustic, lexical, and visual features are 𝑓𝑎∈R𝑑𝑎,
𝑓𝑙∈R𝑑𝑙, 𝑓𝑣∈R𝑑𝑣, respectively. During the fusion process, we first
map them to a fixed dimension 𝑑:
ℎ𝑚= ReLU (𝑓𝑚𝑊𝑚+ 𝑏𝑚) ,𝑚∈{𝑎,𝑙, 𝑣},
(1)
8https://drive.google.com/file/d/1cXNfKHyJzVXg_7kWSf_nVKtsxIZVa517/view?usp=sharing
9https://zeroqiaoba.github.io/MER2024-website
10https://github.com/zeroQiaoba/MERTools


## PDF page 4

ACM MM, 2024, Melbourne, Australia
author name and author name, et al.
Table 2: Model cards for some representative feature extractors. Here, “CH”, “EN”, and “MULTI” are the abbreviations of
Chinese, English, and multilingualism, respectively.
Feature
Training Data (Language)
Link
Visual Modality
VideoMAE-base
Kinetics-400
huggingface.co/MCG-NJU/videomae-base
VideoMAE-large
Kinetics-400
huggingface.co/MCG-NJU/videomae-large
CLIP-base
WIT
huggingface.co/openai/clip-vit-base-patch32
CLIP-large
WIT
huggingface.co/openai/clip-vit-large-patch14
EVA-02-base
ImageNet-22k
huggingface.co/timm/eva02_base_patch14_224.mim_in22k
DINOv2-large
LVD-142M
huggingface.co/facebook/dinov2-large
VideoMAE-base (VoxCeleb2)
VoxCeleb2
github.com/zeroQiaoba/MERTools
VideoMAE-base (MER2023)
MER2023
github.com/zeroQiaoba/MERTools
Acoustic Modality
emotion2vec
Mix (EN)
github.com/ddlBoJack/emotion2vec
Whisper-base
Internet (MULTI, mainly EN)
huggingface.co/openai/whisper-base
Whisper-large
Internet (MULTI, mainly EN)
huggingface.co/openai/whisper-large-v2
wav2vec 2.0-base
WenetSpeech (CH)
huggingface.co/TencentGameMate/chinese-wav2vec2-base
wav2vec 2.0-large
WenetSpeech (CH)
huggingface.co/TencentGameMate/chinese-wav2vec2-large
HUBERT-base
WenetSpeech (CH)
huggingface.co/TencentGameMate/chinese-hubert-base
HUBERT-large
WenetSpeech (CH)
huggingface.co/TencentGameMate/chinese-hubert-large
Lexical Modality
PERT-base
EXT Data (CH)
huggingface.co/hfl/chinese-pert-base
PERT-large
EXT Data (CH)
huggingface.co/hfl/chinese-pert-large
LERT-base
EXT Data (CH)
huggingface.co/hfl/chinese-lert-base
LERT-large
EXT Data (CH)
huggingface.co/hfl/chinese-lert-large
XLNet-base
EXT Data (CH)
huggingface.co/hfl/chinese-xlnet-base
MacBERT-base
EXT Data (CH)
huggingface.co/hfl/chinese-macbert-base
MacBERT-large
EXT Data (CH)
huggingface.co/hfl/chinese-macbert-large
RoBERTa-base
EXT Data (CH)
huggingface.co/hfl/chinese-roberta-wwm-ext
RoBERTa-large
EXT Data (CH)
huggingface.co/hfl/chinese-roberta-wwm-ext-large
ELECTRA-base
EXT Data (CH)
huggingface.co/hfl/chinese-electra-180g-base-discriminator
ELECTRA-large
EXT Data (CH)
huggingface.co/hfl/chinese-electra-180g-large-discriminator
BLOOM-7B
ROOTS (MULTI)
huggingface.co/bigscience/bloom-7b1
MOSS-7B
Mix (MULTI, mainly EN and CH)
huggingface.co/fnlp/moss-base-7b
ChatGLM2-6B
Mix (MULTI, mainly EN and CH)
huggingface.co/THUDM/chatglm2-6b
Baichuan-13B
Mix (MULTI, mainly EN and CH)
huggingface.co/baichuan-inc/Baichuan-13B-Base
Table 3: Prompt for generating emotion-related descriptions using MLLMs (drawn from previous works [8]).
Models
Prompt
Audio LLM
As an expert in the field of emotions, please focus on the acoustic information in the audio to discern
clues related to the emotions of the individual. Please provide a detailed description and ultimately
predict the emotional state of the individual.
Video LLM
As an expert in the field of emotions, please focus on the facial expressions, body movements, envi-
ronment, etc., in the video to discern clues related to the emotions of the individual. Please provide a
detailed description and ultimately predict the emotional state of the individual in the video.
Audio-Video LLM
As an expert in the field of emotions, please focus on the facial expressions, body movements, environ-
ment, acoustic information, etc., in the video to discern clues related to the emotions of the individual.
Please provide a detailed description and ultimately predict the emotional state of the individual in the
video.
where 𝑊𝑚∈R𝑑𝑚×𝑑and 𝑏𝑚∈R𝑑are trainable parameters. Then,
we calculate the attention score for each modality:
ℎ= Concat (ℎ𝑎,ℎ𝑙,ℎ𝑣) ,
(2)
𝛼= softmax

ℎ𝑇𝑊𝛼+ 𝑏𝛼

,
(3)
where 𝑊𝛼∈R𝑑×1 and 𝑏𝛼∈R1 are trainable parameters. Here,
ℎ∈R𝑑×3 and 𝛼∈R3×1. Finally, the fused multimodal features
𝑧= ℎ𝛼∈R𝑑are used for emotion recognition.


## PDF page 5

MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition
ACM MM, 2024, Melbourne, Australia
Table 4: Unimodal results (%). Besides five-fold cross-validation results on Train&Val, we also report the results on MER-SEMI
and MER-NOISE. The values in the gray columns are used for the final ranking.
Feature
Train&Val
MER-SEMI
MER-NOISE
Average
WAF (↑)
ACC (↑)
WAF (↑)
ACC (↑)
WAF (↑)
ACC (↑)
WAF (↑) ACC (↑)
Visual Modality
VideoMAE-base [9]
52.86±0.23
53.27±0.21
47.37±0.41
52.04±0.16
48.53±0.39
51.81±0.26
49.59
52.37
EmoNet [10]
51.76±0.31
53.18±0.10
52.45±0.18
54.94±0.15
51.22±0.21
53.56±0.43
51.81
53.90
VideoMAE-large [9]
57.04±0.25
57.49±0.30
51.05±0.39
55.73±0.41
50.81±0.28
55.22±0.62
52.97
56.15
DINOv2-large [11]
58.44±0.12
59.57±0.15
53.41±0.26
57.47±0.22
52.37±0.31
56.10±0.14
54.74
57.71
SENet-FER2013 [12]
57.67±0.10
58.79±0.16
54.62±0.16
56.36±0.24
52.31±0.24
54.79±0.16
54.87
56.65
ResNet-FER2013 [13]
58.73±0.30
59.66±0.16
53.07±0.20
55.13±0.27
53.26±0.29
55.54±0.20
55.02
56.78
MANet-RAFDB [14]
59.91±0.12
61.10±0.11
56.59±0.31
58.87±0.43
55.15±0.30
57.49±0.23
57.22
59.15
EVA-02-base [15]
61.41±0.20
62.28±0.22
55.25±0.21
58.80±0.24
55.36±0.14
58.14±0.15
57.34
59.74
CLIP-base [16]
61.74±0.18
62.56±0.20
57.16±0.14
60.69±0.24
57.43±0.16
61.25±0.19
58.78
61.50
VideoMAE-base (VoxCeleb2) [1]
63.33±0.13
63.84±0.14
57.98±0.20
61.56±0.23
57.18±0.28
60.06±0.19
59.50
61.82
VideoMAE-base (MER2023) [1]
64.50±0.17
64.93±0.19
58.34±0.32
62.10±0.35
57.32±0.20
61.07±0.31
60.05
62.70
CLIP-large [16]
66.66±0.12
67.17±0.12 63.27±0.29
65.45±0.30 60.04±0.19
62.78±0.13
63.32
65.13
Acoustic Modality
eGeMAPS [17]
39.68±0.53
42.88±0.38
30.27±0.58
33.85±0.79
28.92±0.39
32.05±0.43
32.96
36.26
VGGish [18]
48.60±0.10
50.20±0.09
45.44±0.37
48.52±0.23
40.70±0.31
43.69±0.23
44.91
47.47
Whisper-base [19]
56.65±0.13
57.08±0.24
59.70±0.16
60.60±0.12
41.26±0.41
42.52±0.23
52.54
53.40
emotion2vec [20]
56.08±0.09
56.48±0.03
58.98±0.33
59.68±0.26
45.66±0.31
46.97±0.31
53.57
54.38
Whisper-large [19]
63.23±0.20
63.27±0.20
69.24±0.87
70.62±0.62
53.26±0.86
55.34±0.57
61.91
63.08
wav2vec 2.0-base [21]
64.89±0.26
65.14±0.30
68.12±0.34
69.29±0.25
53.91±0.14
56.87±0.29
62.31
63.76
wav2vec 2.0-large [21]
65.50±0.29
65.83±0.27
68.12±0.30
70.06±0.22
56.41±0.73
59.12±0.63
63.35
65.00
HUBERT-base [22]
69.26±0.13
69.43±0.15
78.70±0.20
79.28±0.20
59.55±0.70
59.76±0.82
69.17
69.49
HUBERT-large [22]
73.02±0.13
73.10±0.15 83.42±0.37
84.00±0.32 73.21±0.36
74.03±0.31
76.55
77.05
Lexical Modality
XLNet-base [23]
48.59±0.30
48.95±0.29
49.13±0.07
48.64±0.04
46.14±0.16
46.44±0.10
47.96
48.01
ELECTRA-large [24]
50.45±0.15
50.83±0.09
52.07±0.16
51.81±0.12
47.34±0.18
47.14±0.12
49.95
49.93
MOSS-7B
51.25±0.23
51.64±0.19
51.62±0.22
51.64±0.14
47.87±0.19
48.42±0.29
50.25
50.57
PERT-large [25]
50.36±0.23
50.69±0.10
53.47±0.17
53.26±0.13
48.76±0.23
48.83±0.22
50.86
50.93
PERT-base [25]
50.26±0.08
50.62±0.12
53.64±0.21
53.50±0.18
49.29±0.18
49.44±0.18
51.06
51.19
LERT-large [26]
52.22±0.18
52.38±0.19
52.40±0.20
52.18±0.28
49.17±0.44
49.14±0.49
51.26
51.24
ELECTRA-base [24]
50.98±0.07
51.30±0.07
54.50±0.33
54.35±0.30
49.35±0.15
49.32±0.16
51.61
51.65
LERT-base [26]
52.37±0.13
52.72±0.10
54.84±0.21
54.62±0.20
47.98±0.07
48.15±0.15
51.73
51.83
RoBERTa-base [27]
51.84±0.22
52.24±0.18
54.37±0.24
53.99±0.34
49.49±0.51
49.56±0.41
51.90
51.93
MacBERT-base [28]
51.40±0.13
51.75±0.13
55.11±0.29
54.74±0.37
49.68±0.24
49.74±0.25
52.06
52.08
RoBERTa-large [27]
52.66±0.18
52.92±0.13
55.14±0.31
54.88±0.31
49.06±0.21
49.23±0.28
52.29
52.34
ChatGLM2-6B [29]
53.04±0.23
53.28±0.22
55.52±0.56
55.06±0.63
50.39±0.43
50.40±0.40
52.98
52.91
MacBERT-large [28]
52.19±0.14
52.47±0.12
56.99±0.20
56.81±0.31
50.24±0.20
50.34±0.23
53.14
53.21
BLOOM-7B [30]
53.13±0.24
53.30±0.24
56.01±0.36
55.99±0.40
50.38±0.21
50.58±0.13
53.18
53.29
Baichuan-13B [31]
54.86±0.12
55.11±0.06
56.63±0.18
56.17±0.34
52.01±0.30
51.90±0.21
54.50
54.39
3.2
MER-OV
OV emotion recognition is a new task and the lack of large-scale
datasets makes it difficult to conduct supervised training. There-
fore, we choose pre-trained MLLMs as baselines because they can
handle various multimodal tasks without further training. To solve
OV emotion recognition using MLLMs, we borrow the dataset con-
struction pipeline in Section 2.3. Specifically, we first use MLLMs
and the prompts in Table 3 to extract multifaceted and multimodal
emotion-related clues. Then, we use these clues to disambiguate
the subtitle and generate descriptions with rich emotions. Finally,
we extract all emotion labels using the prompt in Section 2.3.
3.3
Implementation Details
For MER-SEMI and MER-NOISE, we use the attention mechanism
for multimodal fusion. This process involves one hyper-parameter,
the dimension of the hidden representation 𝑑, and we choose it
from {64, 128, 256}. During training, we use the Adam optimizer
and choose the learning rate from {10−3, 10−4}. To alleviate the
overfitting problem, we also use Dropout and select the rate from
{0.2, 0.3, 0.4, 0.5}. Therefore, a total of 3 hyper-parameters need to
be adjusted. To find the optimal hyper-parameters, we randomly
select 50 parameter combinations and choose the best-performing
combination in five-fold cross-validation. Finally, we report its
average result and standard deviation.
For MER-OV, we directly use pretrained MLLMs. Due to limited
GPU memory, we use their 7B weights by default. All models are


## PDF page 6

ACM MM, 2024, Melbourne, Australia
author name and author name, et al.
Table 5: Multimodal results (%). “A”, “V”, and “T” represent acoustic, visual, and textual modalities, respectively. “TopN” means
that we select the top-N features for each modality and their ranking is based on the average WAF in Table 4.
# Top
Train&Val
MER-SEMI
MER-NOISE
Average
WAF (↑)
ACC (↑)
WAF (↑)
ACC (↑)
WAF (↑)
ACC (↑)
WAF (↑)
ACC (↑)
A+V
Top1
78.86±0.17
78.98±0.13
84.07±0.17
84.60±0.16
78.14±0.31
79.43±0.22
80.36
81.00
Top2
78.92±0.04
78.97±0.11
84.20±0.15
84.73±0.13
77.33±0.14
78.55±0.18
80.15
80.75
Top3
79.16±0.11
79.18±0.16
84.09±0.15
84.67±0.15
77.26±0.38
78.70±0.33
80.17
80.85
Top4
79.11±0.08
79.23±0.09
84.12±0.11
84.72±0.11
77.16±0.15
78.55±0.15
80.13
80.83
Top5
79.17±0.07
79.25±0.10
84.38±0.23
84.92±0.22
77.05±0.17
78.46±0.32
80.20
80.88
Top6
79.12±0.09
79.25±0.12
84.14±0.20
84.72±0.19
77.54±0.34
78.78±0.22
80.27
80.91
A+T
Top1
73.67±0.13
73.82±0.09
84.06±0.35
84.57±0.32
74.74±0.26
75.61±0.23
77.49
78.00
Top2
73.75±0.19
73.84±0.17
84.50±0.36
85.01±0.35
73.38±0.46
74.25±0.59
77.21
77.70
Top3
73.97±0.11
73.97±0.05
84.64±0.16
85.10±0.17
73.67±0.39
74.32±0.39
77.42
77.80
Top4
73.82±0.22
73.87±0.22
84.37±0.57
84.85±0.54
73.90±0.30
74.36±0.29
77.36
77.69
Top5
73.73±0.19
73.80±0.18
84.08±0.30
84.64±0.27
73.53±0.14
74.20±0.15
77.11
77.55
Top6
73.68±0.12
73.70±0.11
84.31±0.36
84.80±0.31
73.99±0.49
74.62±0.40
77.32
77.71
V+T
Top1
72.28±0.16
72.47±0.11
77.34±0.30
77.74±0.28
71.98±0.42
73.38±0.31
73.87
74.53
Top2
73.88±0.08
74.03±0.07
77.17±0.43
77.89±0.39
71.99±0.21
73.72±0.11
74.34
75.21
Top3
74.29±0.09
74.35±0.10
77.98±0.37
78.60±0.39
72.61±0.34
74.26±0.35
74.96
75.74
Top4
74.06±0.12
74.22±0.09
78.31±0.34
79.08±0.33
72.51±0.18
74.26±0.13
74.96
75.85
Top5
74.04±0.10
74.20±0.10
77.78±0.23
78.50±0.25
73.12±0.41
74.81±0.33
74.98
75.84
Top6
74.33±0.17
74.48±0.21
77.83±0.26
78.53±0.26
73.03±0.36
74.70±0.27
75.07
75.90
A+V+T
Top1
79.31±0.03
79.40±0.04
86.73±0.13
87.09±0.16
79.47±0.29
80.67±0.19
81.84
82.39
Top2
80.05±0.06
80.07±0.09
85.94±0.25
86.44±0.24
79.11±0.33
80.33±0.29
81.70
82.28
Top3
80.28±0.16
80.33±0.12
86.27±0.22
86.74±0.18
78.75±0.20
80.17±0.19
81.77
82.41
Top4
80.02±0.11
80.10±0.11
86.19±0.14
86.67±0.16
79.62±0.27
80.82±0.28
81.94
82.53
Top5
80.17±0.13
80.21±0.12
85.93±0.15
86.45±0.17
78.91±0.28
80.36±0.22
81.67
82.34
Top6
80.34±0.10
80.43±0.12
86.32±0.22
86.86±0.19
79.24±0.40
80.33±0.28
81.97
82.54
implemented with PyTorch and all inference processes are carried
out with a 32G NVIDIA Tesla V100 GPU.
3.4
Evaluation Metrics
For MER-SEMI and MER-NOISE, we choose two widely used met-
rics in emotion recognition for performance evaluation: accuracy
and weighted average f1-score (WAF) [33]. Considering the inher-
ent class imbalance, we choose WAF for the final ranking.
For MER-OV, we use set-level accuracy and recall for perfor-
mance evaluation, consistent with previous works [8]. Specifically,
assume that the true label set is 𝑌𝑔𝑡= {𝑦𝑖}𝑀
𝑖=1 and the predicted
label set is ˆ𝑌𝑝= { ˆ𝑦𝑖}𝑁
𝑖=1, where 𝑀and 𝑁are the number of la-
bels. Since we do not fix the label space, there may be synonyms,
i.e., labels with different expressions but the same meaning. There-
fore, we first group all labels using GPT-3.5: Please assume the role
of an expert in the field of emotions. We provide a set of emotions.
Please group the emotions, with each group containing emotions with
the same meaning. Directly output the results. The output format
should be a list containing multiple lists. Next, we use the GPT-based
grouping results 𝐺(·) to map all labels to their group IDs:
𝑌𝑚
𝑔𝑡= {𝐺(𝑥)|𝑥∈{𝑦𝑖}𝑀
𝑖=1}, ˆ𝑌𝑚
𝑝= {𝐺(𝑥)|𝑥∈{ ˆ𝑦𝑖}𝑁
𝑖=1}.
(4)
We then calculate set-level accuracy and recall and use their average
for the final ranking, consistent with previous works [8].
Accuracy𝑠=
|𝑌𝑚
𝑔𝑡∩ˆ𝑌𝑚
𝑝|
| ˆ𝑌𝑚
𝑝|
, Recall𝑠=
|𝑌𝑚
𝑔𝑡∩ˆ𝑌𝑚
𝑝|
|𝑌𝑚
𝑔𝑡|
.
(5)
Avg = Accuracy𝑠+ Recall𝑠
2
.
(6)
4
RESULTS AND DISCUSSION
This section reports baseline results for three tracks. For MER-SEMI
and MER-NOISE, we report unimodal and multimodal results. For
MER-OV, we report the results of various MLLMs.
4.1
MER-SEMI and MER-NOISE
Table 4 shows the unimodal results. From this table, we observe that
models that perform well on Train&Val generally perform well on
MER-SEMI and MER-NOISE. These results suggest that the models
trained on our dataset have a good generalization ability.
For the visual modality, unsupervised or semi-supervised mod-
els (e.g., CLIP-large) generally outperform supervised models (e.g.,
SENet-FER2013). This phenomenon suggests that unsupervised or
semi-supervised strategies can capture universal representations,
which are also helpful for emotion recognition. Meanwhile, previ-
ous works have emphasized the importance of domain compatibility


## PDF page 7

MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition
ACM MM, 2024, Melbourne, Australia
Table 6: Baseline results on MER-OV (taken from previous
works [8]). The values in the gray column are used for the
final ranking. Here, “L”, “V”, and “A” indicate whether lexical,
visual, and acoustic information are used during inference.
Model
L V A
Avg
Accuracys
Recalls
Empty
× × × 0.00±0.00
0.00±0.00
0.00±0.00
Random
× × × 19.13±0.06 24.85±0.15 13.42±0.04
Qwen-Audio [34]
√× √40.23±0.09 49.42±0.18 31.04±0.00
OneLLM [35]
√× √43.04±0.06 45.92±0.05 40.15±0.06
Otter [36]
√√× 44.40±0.09 50.71±0.10 38.09±0.09
VideoChat [37]
√√× 45.70±0.09 42.90±0.27 48.49±0.10
Video-LLaMA [38]
√√× 44.74±0.14 44.14±0.13 45.34±0.15
PandaGPT [39]
√√√46.21±0.17 50.03±0.01 42.38±0.33
SALMONN [40]
√× √48.06±0.04 50.20±0.04 45.92±0.04
Video-LLaVA [41]
√√× 47.12±0.15 48.58±0.02 45.66±0.29
VideoChat2 [42]
√√× 49.60±0.28 54.72±0.41 44.47±0.15
OneLLM [35]
√√× 50.99±0.08 55.93±0.09 46.06±0.06
LLaMA-VID [43]
√√× 51.29±0.09 52.71±0.18 49.87±0.00
mPLUG-Owl [44]
√√× 52.79±0.13 54.54±0.13 51.04±0.13
Video-ChatGPT [45] √√× 50.73±0.06 54.03±0.04 47.44±0.07
Chat-UniVi [46]
√√× 53.09±0.01 53.68±0.00 52.50±0.02
GPT-4V [47]
√√× 56.69±0.04 48.52±0.07 64.86±0.00
[1]. Therefore, we take VideoMAE-base as an example and further
train it on in-domain corpora, such as VoxCeleb2 and MER2023. This
process leads to significant performance improvements compared
to the original model. Consequently, we recommend participants
train other visual encoders on in-domain corpora.
For the acoustic and lexical modalities, we primarily choose en-
coders trained on the Chinese corpus, as suggested by MERBench
[1]. For the acoustic modality, we observe that unsupervised or
semi-supervised models (e.g., HUBERT-large) generally perform
better than traditional encoders (e.g., eGeMAPS), which is consis-
tent with the findings in the visual modality. For the lexical modality,
we observe that LLMs generally outperform pretrained language
models (PLMs). This suggests that by increasing the training data
and model size, we can build more powerful lexical encoders.
Table 5 shows the multimodal results. In this table, we choose the
top-N features for each modality and use the attention mechanism
for multimodal fusion. Their ranking is based on the average WAF
in Table 4. We observe that different modality combinations prefer
distinct 𝑁. Therefore, we should adjust it for each combination.
Meanwhile, the trimodal results generally perform best, reflecting
the effectiveness of multimodal fusion and the necessity of each
modality. Overall, in terms of WAF scores, our baseline system can
reach 86.73% in MER-SEMI and 79.62% in MER-NOISE.
4.2
MER-OV
This section discusses the performance of different methods on
MER-OV. In addition to MLLMs, we introduce two heuristic base-
lines: Empty and Random. For the former, we do not assign any
label. For the latter, we randomly select a label from six candidate
categories (i.e., worried, happy, neutral, angry, surprised, and sad).
Experimental results are shown in Table 6. We observe that MLLMs
generally outperform heuristic baselines, indicating that they can
solve this task to some extent. However, there is still a significant
gap between MLLMs and ground truth, indicating the difficulty of
this task. We recommend participants test other MLLMs or use su-
pervised fine-tuning, which may bring performance improvement.
5
CONCLUSIONS
This paper introduces MER2024, an extension of the MER2023 com-
petition. Besides including more samples, we also introduce a new
track called open-vocabulary emotion recognition, requiring par-
ticipants to predict any number of labels in any category, aiming to
achieve more accurate emotion recognition. In this paper, we intro-
duce the datasets, baselines, evaluation metrics, and experimental
results. We also open-source the code to ensure reproducibility. We
hope that the MER series of challenges can provide researchers with
a common platform to fairly evaluate their emotion recognition
systems and further promote the development of this field.
REFERENCES
[1] Zheng Lian, Licai Sun, Yong Ren, Hao Gu, Haiyang Sun, Lan Chen, Bin Liu, and
Jianhua Tao. Merbench: A unified evaluation benchmark for multimodal emotion
recognition. arXiv preprint arXiv:2401.03429, 2024.
[2] Zheng Lian, Haiyang Sun, Licai Sun, Kang Chen, Mngyu Xu, Kexin Wang, Ke Xu,
Yu He, Ying Li, Jinming Zhao, et al. Mer 2023: Multi-label learning, modal-
ity robustness, and semi-supervised learning. In Proceedings of the 31st ACM
International Conference on Multimedia, pages 9610–9614, 2023.
[3] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik
Cambria, and Rada Mihalcea. Meld: A multimodal multi-party dataset for emo-
tion recognition in conversations. In Proceedings of the 57th Conference of the
Association for Computational Linguistics, pages 527–536, 2019.
[4] Shan Li, Weihong Deng, and JunPing Du. Reliable crowdsourcing and deep
locality-preserving learning for expression recognition in the wild. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pages 2852–2861, 2017.
[5] Devamanyu Hazarika, Yingting Li, Bo Cheng, Shuai Zhao, Roger Zimmermann,
and Soujanya Poria. Analyzing modality robustness in multimodal sentiment
analysis. In Proceedings of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, pages 685–696, 2022.
[6] Licai Sun, Zheng Lian, Bin Liu, and Jianhua Tao. Efficient multimodal transformer
with dual-level feature restoration for robust multimodal sentiment analysis. IEEE
Transactions on Affective Computing, 2023.
[7] Zheng Lian, Lan Chen, Licai Sun, Bin Liu, and Jianhua Tao. Gcnet: Graph
completion network for incomplete multimodal learning in conversation. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 45(07):8419–8432, 2023.
[8] Zheng Lian, Licai Sun, Mingyu Xu, Haiyang Sun, Ke Xu, Zhuofan Wen, Shun
Chen, Bin Liu, and Jianhua Tao. Explainable multimodal emotion reasoning.
arXiv preprint arXiv:2306.15401, 2023.
[9] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked au-
toencoders are data-efficient learners for self-supervised video pre-training. In
Proceedings of the Advances in Neural Information Processing Systems, pages
10078–10093, 2022.
[10] Samira Ebrahimi Kahou, Xavier Bouthillier, Pascal Lamblin, Caglar Gulcehre, Vin-
cent Michalski, Kishore Konda, Sébastien Jean, Pierre Froumenty, Yann Dauphin,
Nicolas Boulanger-Lewandowski, et al. Emonets: Multimodal deep learning ap-
proaches for emotion recognition in video. Journal on Multimodal User Interfaces,
10:99–111, 2016.
[11] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec,
Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-
Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv
preprint arXiv:2304.07193, 2023.
[12] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
7132–7141, 2018.
[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning
for image recognition. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 770–778, 2016.
[14] Zengqun Zhao, Qingshan Liu, and Shanmin Wang. Learning deep global multi-
scale and local attention features for facial expression recognition in the wild.
IEEE Transactions on Image Processing, 30:6544–6556, 2021.
[15] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and
Yue Cao. Eva-02: A visual representation for neon genesis. arXiv preprint
arXiv:2303.11331, 2023.


## PDF page 8

ACM MM, 2024, Melbourne, Australia
author name and author name, et al.
[16] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al. Learning transferable visual models from natural language supervision. In
Proceedings of the International Conference on Machine Learning, pages 8748–8763.
PMLR, 2021.
[17] Florian Eyben, Klaus R Scherer, Björn W Schuller, Johan Sundberg, Elisabeth
André, Carlos Busso, Laurence Y Devillers, Julien Epps, Petri Laukka, Shrikanth S
Narayanan, et al. The geneva minimalistic acoustic parameter set (gemaps) for
voice research and affective computing. IEEE Transactions on Affective Computing,
7(2):190–202, 2016.
[18] Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren
Jansen, R Channing Moore, Manoj Plakal, Devin Platt, Rif A Saurous, Bryan
Seybold, et al. Cnn architectures for large-scale audio classification. In IEEE
International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages
131–135. IEEE, 2017.
[19] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey,
and Ilya Sutskever. Robust speech recognition via large-scale weak supervision.
In Proceedings of the International Conference on Machine Learning, pages 28492–
28518. PMLR, 2023.
[20] Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang,
and Xie Chen. emotion2vec: Self-supervised pre-training for speech emotion
representation. arXiv preprint arXiv:2312.15185, 2023.
[21] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec
2.0: A framework for self-supervised learning of speech representations. In
Proceedings of the Advances in Neural Information Processing Systems, pages
12449–12460, 2020.
[22] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan
Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech
representation learning by masked prediction of hidden units. IEEE/ACM Trans-
actions on Audio, Speech, and Language Processing, 29:3451–3460, 2021.
[23] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov,
and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language
understanding. Proceedings of the Advances in Neural Information Processing
Systems, pages 5754–5764, 2019.
[24] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra:
Pre-training text encoders as discriminators rather than generators. In Proceedings
of the International Conference on Learning Representations, pages 1–18, 2020.
[25] Yiming Cui, Ziqing Yang, and Ting Liu. Pert: pre-training bert with permuted
language model. arXiv preprint arXiv:2203.06906, 2022.
[26] Yiming Cui, Wanxiang Che, Shijin Wang, and Ting Liu. Lert: A linguistically-
motivated pre-trained language model. arXiv preprint arXiv:2211.05344, 2022.
[27] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly
optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
[28] Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping
Hu. Revisiting pre-trained models for chinese natural language processing. In
Findings of the Association for Computational Linguistics: EMNLP 2020, pages
657–668, 2020.
[29] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang,
and Jie Tang. Glm: General language model pretraining with autoregressive
blank infilling. In Proceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022.
[30] BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie
Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni,
François Yvon, et al. Bloom: A 176b-parameter open-access multilingual language
model. arXiv preprint arXiv:2211.05100, 2022.
[31] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin,
Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale
language models. arXiv preprint arXiv:2309.10305, 2023.
[32] David Snyder, Guoguo Chen, and Daniel Povey. Musan: A music, speech, and
noise corpus. arXiv preprint arXiv:1510.08484, 2015.
[33] Zheng Lian, Bin Liu, and Jianhua Tao. Ctnet: Conversational transformer network
for emotion recognition. IEEE/ACM Transactions on Audio, Speech, and Language
Processing, 29:985–1000, 2021.
[34] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang
Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding
via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919,
2023.
[35] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua
Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all
modalities with language. arXiv preprint arXiv:2312.03700, 2023.
[36] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei
Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv
preprint arXiv:2305.03726, 2023.
[37] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang,
Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv
preprint arXiv:2305.06355, 2023.
[38] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-
visual language model for video understanding. arXiv preprint arXiv:2306.02858,
2023.
[39] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt:
One model to instruction-follow them all. In Proceedings of the 1st Workshop on
Taming Large Language Models: Controllability in the era of Interactive Assistants,
pages 11–23, 2023.
[40] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu,
MA Zejun, and Chao Zhang. Salmonn: Towards generic hearing abilities for
large language models. In The Twelfth International Conference on Learning
Representations, 2023.
[41] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava:
Learning united visual representation by alignment before projection. arXiv
preprint arXiv:2311.10122, 2023.
[42] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan
Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive
multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024.
[43] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens
in large language models. arXiv preprint arXiv:2311.17043, 2023.
[44] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Jun-
yang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modular-
ization empowers large language models with multimodality. arXiv preprint
arXiv:2304.14178, 2023.
[45] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan.
Video-chatgpt: Towards detailed video understanding via large vision and lan-
guage models. arXiv preprint arXiv:2306.05424, 2023.
[46] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-
univi: Unified visual representation empowers large language models with image
and video understanding. arXiv preprint arXiv:2311.08046, 2023.
[47] OpenAI. Gpt-4v(ision) system card, 2023.
