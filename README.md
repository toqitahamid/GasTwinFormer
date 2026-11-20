<div align="center">

<h1>GasTwinFormer</h1>
<h3>A Hybrid Vision Transformer for Livestock Methane Emission Segmentation and Dietary Classification in Optical Gas Imaging</h3>

Toqi Tahamid Sarker<sup>1</sup>, Mohamed Embaby<sup>2</sup>, Taminul Islam<sup>1</sup>, Amer AbuGhazaleh<sup>1</sup>, Khaled R. Ahmed<sup>1</sup>

<sup>1</sup> Southern Illinois University Carbondale, <sup>2</sup> University of California, Davis

**ICCV 2025 Workshops**

[Paper (arXiv)](https://arxiv.org/abs/2508.15057) | [CVF](https://openaccess.thecvf.com/content/ICCV2025W/SEA/html/Sarker_GasTwinFormer_A_Hybrid_Vision_Transformer_for_Livestock_Methane_Emission_Segmentation_ICCVW_2025_paper.html) | [IEEE Xplore](https://ieeexplore.ieee.org/document/11374624) | [Project Page](https://gastwinformer.github.io/) | [Dataset](https://huggingface.co/datasets/toqi/beef-cattle-methane-ogi) | [Weights](https://huggingface.co/toqi/GasTwinFormer)

</div>

## Abstract

Livestock methane emissions represent 32% of human-caused methane production, making automated monitoring critical for climate mitigation strategies. We introduce GasTwinFormer, a hybrid vision transformer for real-time methane emission segmentation and dietary classification in optical gas imaging through a novel Mix Twin encoder alternating between spatially-reduced global attention and locally-grouped attention mechanisms. Our architecture incorporates a lightweight LR-ASPP decoder for multi-scale feature aggregation and enables simultaneous methane segmentation and dietary classification in a unified framework. We contribute the first comprehensive beef cattle methane emission dataset using OGI, containing 11,694 annotated frames across three dietary treatments. GasTwinFormer achieves 74.47% mIoU and 83.63% mF1 for segmentation while maintaining exceptional efficiency with only 3.348M parameters, 3.428G FLOPs, and 114.9 FPS inference speed. Additionally, our method achieves perfect dietary classification accuracy (100%), demonstrating the effectiveness of leveraging diet-emission correlations. Extensive ablation studies validate each architectural component, establishing GasTwinFormer as a practical solution for real-time livestock emission monitoring.

## Installation

The code is built on MMSegmentation 1.2 and needs Python 3.8 or newer and PyTorch 2.0 or newer.

```bash
git clone https://github.com/toqitahamid/GasTwinFormer.git
cd GasTwinFormer
pip install -r requirements.txt
pip install -e .
```

## Dataset

The data are on Hugging Face: [toqi/beef-cattle-methane-ogi](https://huggingface.co/datasets/toqi/beef-cattle-methane-ogi). It has 11,694 labelled frames from 19 FLIR recordings. Each frame has a binary plume mask, a multi-class mask and the diet label (0 Mixed Diet, 1 High Forage, 2 High Grain). To write the PNG folders and split lists that the config reads, run

```bash
hf download toqi/beef-cattle-methane-ogi --repo-type dataset --local-dir gastwin_hf
pip install pyarrow
python tools/unpack_dataset.py gastwin_hf data
```

This writes the official split (`train.txt`, `val.txt`, `test.txt`) and a by-animal split (`animal_train.txt`, `animal_val.txt`, `animal_test.txt`). We recommend the by-animal split for new work, because the official split puts frames of the same animal in training and test. To use it, change the `ann_file` names in `gastwinformer/configs/_base_/datasets/beef_cattle.py`.

## Pretrained Weights

The checkpoint is on Hugging Face: [toqi/GasTwinFormer](https://huggingface.co/toqi/GasTwinFormer).

```bash
hf download toqi/GasTwinFormer gastwinformer_best_mIoU_iter_80000.pth --local-dir weights
python tools/test.py gastwinformer/configs/gastwinformer_80k.py weights/gastwinformer_best_mIoU_iter_80000.pth
```

## Training

```bash
python tools/train.py gastwinformer/configs/gastwinformer_80k.py
bash tools/dist_train.sh gastwinformer/configs/gastwinformer_80k.py 4   # four GPUs
```

## Results

| Model | Backbone | mIoU (%) | mF1 (%) | Diet Acc (%) | Params (M) | FLOPs (G) | FPS |
|-------|----------|----------|---------|--------------|------------|-----------|-----|
| SegFormer | MiT-B0 | 72.11 | 81.57 | 100.0 | 3.782 | 7.885 | 119.66 |
| Twins | PCPVT-S | 74.05 | 83.25 | 100.0 | 27.906 | 44.34 | 61.60 |
| Twins | SVT-S | 72.06 | 81.62 | 100.0 | 27.846 | 38.471 | 51.64 |
| GasFormer | MiT-B0 | 72.25 | 81.69 | 100.0 | 3.716 | 9.913 | 102.29 |
| DeepLabV3 | ResNet-50 | 70.36 | 80.03 | 100.0 | 68.625 | 270.0 | 91.79 |
| DDRNet | DDRNet | 68.91 | 78.65 | 99.94 | 5.766 | 4.56 | 156.38 |
| **GasTwinFormer** | **MixTwin** | **74.47** | **83.63** | **100.0** | **3.348** | **3.428** | **114.9** |

FPS is measured on an NVIDIA A100 GPU.

## Citation

If you use this code, the dataset or the checkpoint, you must cite the paper below.

```bibtex
@inproceedings{sarker2025gastwinformer,
  author    = {Sarker, Toqi Tahamid and Embaby, Mohamed and Islam, Taminul and AbuGhazaleh, Amer and Ahmed, Khaled},
  title     = {GasTwinFormer: A Hybrid Vision Transformer for Livestock Methane Emission Segmentation and Dietary Classification in Optical Gas Imaging},
  booktitle = {2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2025},
  pages     = {2808--2817},
  doi       = {10.1109/ICCVW69036.2025.00293}
}
```

## License

The code is released under the [MIT License](LICENSE). The dataset and the checkpoint are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use of the data is not permitted.

## Acknowledgments

This work is supported by the project ["Artificial Intelligence for Greener Livestock: Educational and Research"](https://ai-greener-livestock.github.io/), award no. 2022-70001-37404, from the U.S. Department of Agriculture's National Institute of Food and Agriculture.

The code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [SegFormer](https://github.com/NVlabs/SegFormer) and [Twins](https://github.com/Meituan-AutoML/Twins). We thank the authors for their work.
