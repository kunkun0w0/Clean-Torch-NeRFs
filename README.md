# Clean-Torch-NeRFs

This repo aims to provide simplified NeRFs' code to help you to learn NeRFs quickly. 

> **NOTE: I am not sure my implementations have the same accuracy as the official, so please use them with caution in some important cases like research.**


## NeRF
> NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis <br> [[official project]](https://www.matthewtancik.com/nerf)

NeRF: Chinese Tutorial [[theory]](https://zhuanlan.zhihu.com/p/481275794) , [[code]](https://zhuanlan.zhihu.com/p/482154458)

In this repo, I implement the `Hierarchical volume sampling` but I have not used coarse-to-fine strategy.

**360° Lego result**

<img src='/img/nerf.gif' alt="sym" width="120px">

## pixelNeRF
> pixelNeRF: Neural Radiance Fields from One or Few Images <br> [[official project]](https://alexyu.net/pixelnerf/)

pixelNeRF: Chinese Tutorial [[theory]](https://zhuanlan.zhihu.com/p/550890576)

In this repo, I have not used the `Hierarchical volume sampling` and coarse-to-fine strategy.

**3-views Lego result**

<img src='/img/pixelnerf.gif' alt="sym" width="120px">
