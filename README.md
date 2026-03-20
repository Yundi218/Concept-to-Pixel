# Concept-to-Pixel: Prompt-Free Universal Medical Image Segmentation

<p align="center">
  <img src="figures/teaser_C2P.png" alt="C2P Teaser" width="90%">
</p>


<div align="center">
    <span class="author-block">
    <a target="_blank">Haoyun Chen</a><sup>1,2*</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=x1pODsMAAAAJ&hl=en" target="_blank">Fenghe Tang</a><sup>1,2*</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=r0-tZ8cAAAAJ&hl=en" target="_blank">Wenxin Ma</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ&hl=en" target="_blank">S. Kevin Zhou</a><sup>1,2†</sup>
    </span>
</div>
<div align="center">
    <p><sup>*</sup>Equal contribution. <sup>†</sup>Corresponding author.</p>
</div>

<br>

<div align="center">
    <sup>1</sup>
    <a target='_blank'>School of Biomedical Engineering, University of Science and Technology of China</a>&emsp;
    <br>
    <sup>2</sup> <a target='_blank'>Suzhou Institute for Advanced Research, University of Science and Technology of China</a>&emsp;
    <br>
</div>


<br>
<br>

[![arXiv](https://img.shields.io/badge/arxiv-2603.17746-b31b1b.svg)](https://arxiv.org/abs/2603.17746) [![github](https://img.shields.io/badge/github-Concept--to--Pixel-white)](https://github.com/Yundi218/Concept-to-Pixel)    <a href="#LICENSE--citation"><img alt="License: Apache2.0" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue.svg"/></a>


## 📖 Introduction


Universal medical image segmentation seeks to use a single foundational model to handle diverse tasks across multiple imaging modalities. However, existing approaches often rely heavily on manual visual prompts or retrieved reference images, limiting their automation and robustness. Furthermore, naive joint training often fails to address large domain shifts, leading to negative transfer.

To address these limitations, we propose **Concept-to-Pixel (C2P)**, a novel prompt-free universal segmentation framework. C2P explicitly separates anatomical knowledge into two components:

* **Semantic Tokens (`[SEM]`):** Distilled from Multimodal Large Language Models (MLLMs, e.g., Qwen3-VL-Plus) to capture abstract, high-level medical concepts (like margin definitions or internal textures).
* **Geometric Tokens (`[GEO]`):** Explicitly supervised to regress physical attributes (e.g., area, centroid, bounding box) to enforce universal physical and structural constraints.

These disentangled tokens interact deeply with image features through a **Token-Guided Dynamic Head (TGDH)** to generate input-specific dynamic kernels for precise mask prediction. Additionally, we introduce a **Geometry-Aware Inference Consensus** mechanism, which utilizes the explicitly regressed geometric priors to assess prediction reliability and suppress outliers during inference.

## ✨ Key Features

* **Prompt-Free & Zero-Reference:** Operates as a single universal model without requiring any manual visual prompts or support reference images during inference.
* **Mitigates Negative Transfer:** Explicitly disentangles modality-agnostic geometric constraints from VLM-distilled semantic concepts, successfully overcoming feature space collapse in heterogeneous joint training.
* **SOTA Performance:** Achieves an impressive **88.22%** average Dice across 8 diverse datasets spanning 7 modalities, outperforming task-specialized baselines (e.g., nnU-Net V2) and state-of-the-art universal methods (e.g., Spider, SR-ICL).
* **Strong Zero-Shot Generalization:** Demonstrates robust generalization to unseen datasets and achieves competitive performance even on entirely unseen macroscopic modalities (e.g., X-Ray) without relying on references or prompts.

## 🏗️ Framework Overview

<p align="center">
  <img src="figures/framework.png" alt="C2P Framework" width="100%">
</p>

The C2P framework consists of four key components:

1. **Token Formulation & Modality Injection:** Initializes universal `[GEO]` and MLLM-distilled `[SEM]`, utilizing a Style-Content Fusion Module (SCFM) to resolve cross-modality ambiguities.
2. **Bidirectional Token-Image Interaction:** Enables deep interaction via Cross Attention, where concept tokens query image features while simultaneously guiding them with high-level priors.
3. **Token-Guided Dynamic Head (TGDH):** Synthesizes instance-specific convolutional kernels driven by the aggregated tokens and global context for precise mask prediction.
4. **Geometry-Aware Inference Consensus:** An intrinsic evaluation mechanism leveraging the physical self-consistency of `[GEO]` to filter unreliable predictions.

## 🚀 Getting Started

```
