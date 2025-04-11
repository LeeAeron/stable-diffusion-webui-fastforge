## Sd-Forge-TeaCache: Speed up Your Diffusion Models

**Introduction**

Timestep Embedding Aware Cache (TeaCache) is a revolutionary training-free caching approach that leverages the
fluctuating differences between model outputs across timesteps. This acceleration technique significantly boosts
inference speed for various diffusion models, including Image, Video, and Audio.

 TeaCache's integration into SD Forge WebUI for Flux only. Installation is as
straightforward as any other extension:

* **Clone:**  `git clone https://github.com/likelovewant/sd-forge-teacache.git`

into extensions directory ,relauch the system .


**Speed Up Your Diffusion Generation**

TeaCache can accelerate FLUX inference by up to 2x with minimal visual quality degradation, all without requiring any training. 

Within the Forge WebUI, you can easily adjust the following settings:

* **Relative L1 Threshold:** Controls the sensitivity of TeaCache's caching mechanism.
* **Steps:**  Matches the number of sampling steps used in TeaCache.

**Performance Tuning**

Based on [TeaCache4FLUX](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX), you can achieve different
speedups:

* 0.25 threshold for 1.5x speedup
* 0.4 threshold for 1.8x speedup
* 0.6 threshold for 2.0x speedup
* 0.8 threshold for 2.25x speedup

**Important Notes:**

* **Maintain Consistency:** Keep the sampling steps in TeaCache aligned with the steps used in your Flux Sampling steps .Discrepancies can lead to lower quality outputs.
* **LoRA Considerations:** When utilizing LoRAs, adjust the steps or scales based on your GPU's capabilities. A recommended starting point is 28 steps or more.

To ensure smooth operation, remember to:

1. **Clear Residual Cache (optional):** When changing image sizes or disabling the TeaCache extension, always click "Clear Residual Cache" within the Forge WebUI. This prevents potential conflicts and maintains optimal performance.
2. **Disable TeaCache Properly:**  Ensure disable the TeaCache extension if you don't need it in your Forge WebUI. If not proper `Clear Residual Cache`, you may encounter unexpected behavior and require a full relaunch.


Several AI assistants has assisting with code generation and refinement for this extension based on the below resources.

**Credits and Resources**

This adaptation leverages [TeaCache4FLUX](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX)
From ali-vilab TeaCache repository:[TeaCache](https://github.com/ali-vilab/TeaCache).

For additional information and other integrations, explore:

* [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache)

