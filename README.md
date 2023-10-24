# ARNIQA (WACV 2024)

### Learning Distortion Manifold for Image Quality Assessment

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.14918)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/ARNIQA?style=social)](https://github.com/miccunifi/ARNIQA)

This is the **official repository** of the [**paper**](https://arxiv.org/abs/2310.14918) "*ARNIQA: Learning Distortion Manifold for Image Quality Assessment*".

## Overview

### Abstract

No-Reference Image Quality Assessment (NR-IQA) aims to develop methods to measure image quality in alignment with human perception without the need for a high-quality reference image. In this work, we propose a self-supervised approach named ARNIQA (leArning distoRtion maNifold for Image Quality Assessment for modeling the image distortion manifold to obtain quality representations in an intrinsic manner. First, we introduce an image degradation model that randomly composes ordered sequences of consecutively applied distortions. In this way, we can synthetically degrade images with a large variety of degradation patterns. Second, we propose to train our model by maximizing the similarity between the representations of patches of different images distorted equally, despite varying content. Therefore, images degraded in the same manner correspond to neighboring positions within the distortion manifold. Finally, we map the image representations to the quality scores with a simple linear regressor, thus without fine-tuning the encoder weights. The experiments show that our approach achieves state-of-the-art performance on several datasets. In addition, ARNIQA demonstrates improved data efficiency, generalization capabilities, and robustness compared to competing methods.

<p align="center">
  <img src="assets/arniqa_teaser.png" width="75%" alt="Comparison between our approach and the State of the Art for NR-IQA">
</p>

Comparison between our approach and the State of the Art for NR-IQA. While the SotA maximizes the similarity between the representations of crops from the same image, we propose to consider crops from different images degraded equally to learn the image distortion manifold. The t-SNE visualization of the embeddings of the [KADID](http://database.mmsp-kn.de/kadid-10k-database.html) dataset shows that, compared to [Re-IQA](https://arxiv.org/abs/2304.00451), ARNIQA yields more discernable clusters for different distortions. In the plots, a higher alpha value corresponds to a stronger degradation intensity.

## Citation

```bibtex
 @misc{agnolucci2023arniqa,
      title={ARNIQA: Learning Distortion Manifold for Image Quality Assessment}, 
      author={Lorenzo Agnolucci and Leonardo Galteri and Marco Bertini and Alberto Del Bimbo},
      year={2023},
      eprint={2310.14918},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## TO-DO:
- [ ] Pre-trained models and regressors
- [ ] Testing code
- [ ] Training code


## Authors

* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)
* [**Leonardo Galteri**](https://scholar.google.com/citations?user=_n2R2bUAAAAJ&hl=en)
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.
