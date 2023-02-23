# NeRF PyTorch

## 1 组件

### 1.1 NeRF模型

NeRF由位置编码$x$和视角编码$d$得到颜色值$c$和体密度$\sigma$。实现上，体密度$\sigma$仅与$x$有关，颜色值$c$由$x$和$d$共同决定。在8层MLP之后，将$d$输入网络中，通过后续的几层线性层降维到128维最后得到一个3维的RGB颜色值。

> A model trained without
view dependence (only x as input) has difficulty representing specularities.

论文中提到，也可以仅仅使用输入$x$训练模型，这时令`d_viewdirs`和`viewdir`均为`None`，直接使用输出层得到一个4维向量，代表颜色值和体密度。这种训练方式的缺点是不能很好地渲染镜面反射。

### 1.2 位置编码器

> Deep networks are biased towards learning lower frequency functions.

位置编码旨在为MLP网络显式地提供高维信息。NeRF采用了对数空间下的位置编码方式（详见论文中4式），将输入的三维信息$(x, y, z)$编码为不同频率的信息$(x_i, y_i, z_i)_{i=1}^N$。

位置编码在论文中用$\gamma$表示，对$x$使用10阶编码，对$d$使用4阶编码。

### 1.3 体渲染器



### 1.4 光线采样

均匀采样会有下面的问题：

> Free space and occluded
regions that do not contribute to the rendered image are still sampled repeatedly.

解决方法是更加合理地分配样本位置。为了解决这个问题，NeRF才采用了粗糙（coarse）网络和精细（fine）网络，其中粗糙网络就是用于更加合理地分配样本。

粗糙网络结构上和精细网络保持一致，因此仍然会输出颜色值和体密度，应用体渲染公式，可以得到每个样本点颜色值的权重。这个权重从某种意义上代表令每个样本点贡献大小，归一化后即可作为概率密度函数，接下来再次按照这个概率密度进行采样，就可以得到更加合理的采样点位置。

这里实现了两个函数：
- `stratified_sample`

    分层采样本质上仍然是均匀采样，均匀采样可以看做线性插值，而该函数提供了`inverse_depth`用于控制使用线性插值或调和插值，还提供了`perturb`用于控制采样点是否进行微小扰动。插值结果保存在`z_vals`中，对光线进行插值即可获取采样点。

    - z_vals：[64]，表示采样点插值，随后需要扩展为[2500, 64]，接着扩展为[2500, 64, 1]
    - rays_o：[2500, 3]，表示每条光线的原点，随后需要扩展为[2500, 1, 3]
    - rays_d：[2500, 3]，表示每条光线的方向，随后需要扩展为[2500, 1, 3]

    ``` py
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    ```

    这样得到的`pts`形状就是[2500, 64, 3]，表示64个采样点的3维位置。这些样本点构成集合$N_c$。
    

- `hierarchical_sample`

    在此之前先解释一下按照概率密度采样的函数`sample_pdf`。根据粗糙网络计算得到的权重`weight`，可以计算概率密度`pdf`和分布函数`cdf`。

    以下是`sample_pdf`函数中形状：
    - weights：[2500, 62]（后续解释）
    - pdf：[2500, 62]
    - cdf：[2500, 62]，接着扩展为[2500, 63]

    `sample_pdf`将分布函数和均匀分布进行比较，得到那些概率密度大的地方，进行采样。采样点可能不是整数的下标，则使用插值方法得到`samples`。

    再来看`hierarchical_sample`，它根据传入的`weight`进行采样，得到新的样本点。这些新的样本点构成集合$N_f$。

## 2 训练

### 2.1 训练集

训练集来自[TinyNeRF](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz)。

### 2.2 前向传播

前向传播的输入中包含了所有的光线信息，接着就是光线采样。`stratified_sample`采样得到$N_c$并进行位置编码。类似地，视角方向也需要进行位置编码。

第一轮是粗糙模型的前向传播。使用chunk的目的是对所有的采样点进行切分，限制网络输入的最大容量，避免显存不足。粗糙模型每一轮的输出是一个[input_chunksize, 4]的矩阵，包含了颜色值和体密度值。这些结果最终被合并成[160000, 4]的输出，重新构建形状为[2500, 64, 4]，代表了2500条光线，每条光线64个采样点上的颜色值和体密度值。

第二轮是精细模型的前向传播。我们由之前得到的概率密度进行`hierarchical_sample`得到$N_f$，同样得到[2500, 64, 4]的输出。这些结果被保存在`outputs`中。

### 2.3 训练过程

``` bash
torchrun --nproc_per_node=NUM_GPUS src/main.py
```

随机选取一张图片生成光线，进行一次前向传播得到`outputs`，其中`rgb_predicted`是预测的颜色值，与原始图片相比较进行误差反向传播。

## 3 实验

