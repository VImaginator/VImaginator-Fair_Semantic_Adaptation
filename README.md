
# [CVPR 2023] [Fairness Domain Adaptation Approach to Semantic Scene Understanding: FREDOM](https://arxiv.org/abs/2304.02135)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fredom-fairness-domain-adaptation-approach-to/domain-adaptation-on-synthia-to-cityscapes)](https://paperswithcode.com/sota/domain-adaptation-on-synthia-to-cityscapes?p=fredom-fairness-domain-adaptation-approach-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fredom-fairness-domain-adaptation-approach-to/domain-adaptation-on-gta5-to-cityscapes)](https://paperswithcode.com/sota/domain-adaptation-on-gta5-to-cityscapes?p=fredom-fairness-domain-adaptation-approach-to)

> [Fairness Domain Adaptation Approach to Semantic Scene Understanding](https://arxiv.org/abs/2304.02135)<br>
> VImaginator and Collaborators<br>
> University of Arkansas, Computer Vision and Image Understanding Lab, CVIU

## Abstract

Although Domain Adaptation in Semantic Scene Segmentation has shown impressive improvement in recent years,
the fairness concerns in the domain adaptation have yet to be well defined and addressed. In addition, fairness is one of the most critical aspects when deploying the segmentation models into human-related real-world applications, e.g., autonomous driving, as any unfair predictions could influence human safety. In this paper, we propose a novel Fairness Domain Adaptation (FREDOM) approach to semantic scene segmentation. In particular, from the proposed formulated fairness objective, a new adaptation framework will be introduced based on the fair treatment of class distributions. Moreover, to generally model the context of structural dependency, a new conditional structural constraint is introduced to impose the consistency of predicted segmentation. Thanks to the proposed Conditional Structure Network, the self-attention mechanism has sufficiently modeled the structural information of segmentation. Through the ablation studies, the proposed method has shown the performance improvement of the segmentation models and promoted fairness in the model predictions. The experimental results on the two standard benchmarks, i.e., SYNTHIA to Cityscapes and GTA5 to Cityscapes, have shown that our method achieved State-of-the-Art (SOTA) performance.

<a href="https://youtu.be/Feo4UMd1eac" target="_blank">
 <img src="http://img.youtube.com/vi/Feo4UMd1eac/mqdefault.jpg" alt="FREDOM" width="960" height="540" border="10" />
</a>

## Installation
This repo requires Python 3.6+, Pytorch >= 1.4.0, and CUDA 10.0+.
```bash
pip install torch torchvision
```

## Testing
We have released our inference code of the DeepLabV2 backbone.
```
python inference.py --checkpoint [path to checkpoint] --input_path [path to input image] --output_path [path to output image]
```

## Training
The training code will be available soon.

## Acknowledgment

Our work is supported by NSF Data Science, Data Analytics that are Robust and Trusted (DART), NSF WVAR-CRESH, and Googler Initiated Research Grant.
We also thank the Arkansas High Performance Computing Center for providing GPUs.

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{truong2023fredom,
  title={FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding},
  author={VImaginator and Collaborators},
  booktitle={IEEE/CVF Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```