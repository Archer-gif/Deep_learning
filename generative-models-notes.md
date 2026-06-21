# 生成式模型复习笔记

> 依据第九章《生成式模型》课件、重点截图，以及本轮关于 VAE 隐变量分布、KL 正则项、重参数化技巧、随机采样生成、文生图条件控制的问答整理。红色重点详写；黑色内容以“知道是什么、为什么出现”为主；课件中出现但重点图未列出的内容简单带过。

## 0. 复习优先级

| 优先级 | 内容 | 复习目标 |
|---|---|---|
| 重点 | VAE 数学原理、KL 正则项、重参数化技巧、Diffusion 的去噪模型设计与训练流程 | 能讲清每个分布的含义、为什么要引入 Encoder、为什么 KL 是正则项、为什么重参数化后能反传、扩散模型到底训练什么 |
| 了解 | GAN 的模型架构、生成器与判别器、损失函数与优化策略、Diffusion 的隐空间扩散 | 能说清基本流程、训练目标、为什么 Stable Diffusion 要在隐空间里做扩散 |
| 简单带过 | Flow Matching、Stable Diffusion 的应用意义、VAE/GAN/Diffusion/FM 的发展脉络 | 知道名字、核心思想、和扩散模型的大致区别即可 |

## 1. 生成模型到底在学什么【了解】

生成模型的目标不是给图片分类，而是学习真实数据的分布。假设真实图像来自一个复杂分布：

$$
p_{\text{data}}(x)
$$

模型希望学到一个近似分布：

$$
p_\theta(x)\approx p_{\text{data}}(x)
$$

训练好以后，从模型分布中采样，就能得到新图像。课件这一章主要讲四类生成模型：

- VAE：把图像编码到隐空间，再从隐空间解码生成图像。
- GAN：生成器和判别器对抗训练，让生成图像越来越像真实图像。
- Diffusion Model：从真实图像逐步加噪，再学习从噪声逐步去噪生成图像。
- Flow Matching：学习从噪声分布到图像分布的连续传输路径。

这章重点是 VAE 和 Diffusion。GAN 知道基本架构和博弈目标即可；Flow Matching 只需要知道它也是“从噪声到数据”的生成路线。

## 2. VAE 的核心动机【重点】

图像维度很高。例如一张 `256 x 256 x 3` 的彩色图像接近二十万维。直接在像素空间里随机采样，大多数结果都没有意义。

VAE 的思路是：不要直接在高维像素空间里采样，而是学习一个低维隐空间。

普通自编码器是：

$$
x \rightarrow \text{Encoder} \rightarrow z \rightarrow \text{Decoder} \rightarrow x'
$$

其中 `x` 是输入图像，`z` 是低维隐变量，`x'` 是重建图像。普通自编码器主要追求：

$$
x'\approx x
$$

问题是，普通自编码器通常把一张图压成一个确定的点：

$$
x_1\rightarrow z_1,\quad x_2\rightarrow z_2,\quad x_3\rightarrow z_3
$$

这样隐空间可能是破碎的。Decoder 只见过训练样本对应的少数点，随机采一个没见过的 `z`，可能落在空白区域，生成结果就容易崩。

VAE 的关键改动是：

> Encoder 不输出一个确定向量，而是输出一个分布。

也就是：

$$
q_\phi(z|x)=N(\mu,\sigma^2)
$$

这里 $q_\phi(z|x)$ 表示：给定图片 `x`，Encoder 认为它可能对应哪些隐变量 `z`。

### `q(z|x)` 是二维分布吗

不一定。它是几维，取决于你设定的隐变量 `z` 是几维。

如果 latent dim = 2：

$$
z=[z_1,z_2]
$$

那么 `q(z|x)` 是二维高斯分布，采样一次就是一个二维点。

如果 latent dim = 128：

$$
z\in R^{128},\quad \mu\in R^{128},\quad \sigma\in R^{128}
$$

那么采样一次得到的是一个 128 维向量。课件中画二维通常只是为了方便可视化。

实际 VAE 常用对角高斯：

$$
q_\phi(z|x)=N(\mu,\operatorname{diag}(\sigma^2))
$$

意思是每个维度独立采样：

$$
z_i\sim N(\mu_i,\sigma_i^2)
$$

## 3. VAE 的数学原理【重点】

VAE 的生成假设是：

$$
z\sim p(z),\qquad x\sim p_\theta(x|z)
$$

其中：

- `p(z)`：先验分布，通常取标准正态分布 `N(0,I)`。
- $p_\theta(x|z)$：Decoder 分布，表示给定隐变量 `z` 后生成图像 `x` 的概率。
- $p_\theta(x)$：模型整体认为图像 `x` 出现的概率。

真正想最大化的是训练集中真实图像的似然：

$$
\max_\theta \log p_\theta(x)
$$

而：

$$
p_\theta(x)=\int p(z)p_\theta(x|z)\,dz
$$

这句话的意思是：某张图 `x` 的概率，等于所有可能的 `z` 生成它的概率加起来。

问题是这个积分很难算。`z` 可能是几十维或几百维的连续变量，直接从整个 `p(z)` 里取遍所有值几乎不可行。

### 为什么要引入 $q_\phi(z|x)$

你之前理解的“缩小范围”是对的，但要再精确一点：

> $q_\phi(z|x)$ 是训练时针对当前图片 `x` 的搜索分布，用来告诉模型哪些 `z` 更可能和这张图有关。

原本要从整个先验分布里考虑：

$$
z\sim p(z)
$$

但对一张猫图来说，大部分随机 `z` 可能和这张猫没有关系。Encoder 于是输出：

$$
q_\phi(z|x)
$$

它相当于在说：当前这张图比较可能来自隐空间的这一小片区域。

所以训练时变成：

$$
z\sim q_\phi(z|x)
$$

再把 `z` 交给 Decoder 重建 `x`。

不过注意：$q_\phi(z|x)$ 不是生成时用来采样的最终分布。生成时仍然用公共先验：

$$
z\sim p(z)=N(0,I)
$$

$q_\phi(z|x)$ 的作用是在训练中帮助 Decoder 学会隐空间到图像的对应关系。

### $p_\theta(z|x)$ 和 $q_\phi(z|x)$ 的区别

真实后验是：

$$
p_\theta(z|x)=\frac{p_\theta(x|z)p(z)}{p_\theta(x)}
$$

它表示：给定图片 `x`，最可能对应哪些隐变量 `z`。

但它里面含有 $p_\theta(x)$，而 $p_\theta(x)$ 又是那个算不动的积分，所以 $p_\theta(z|x)$ 也算不动。

于是 VAE 引入一个可训练的近似分布：

$$
q_\phi(z|x)\approx p_\theta(z|x)
$$

这就是“变分推断”，也是 VAE 中“变分”的来源。

<details>
<summary>展开：ELBO 推导看不懂时怎么抓主线</summary>

我们想最大化：

$$
\log p_\theta(x)
$$

但它不好直接算。引入任意分布 $q_\phi(z|x)$ 后，可以得到一个下界：

$$
\log p_\theta(x)\ge
E_{q_\phi(z|x)}[\log p_\theta(x|z)]-
KL(q_\phi(z|x)\|p(z))
$$

这个下界叫 ELBO。最大化 ELBO，就等价于间接提高 $\log p_\theta(x)$。

其中第一项：

$$
E_{q_\phi(z|x)}[\log p_\theta(x|z)]
$$

要求 Decoder 从采样到的 `z` 还原 `x`，所以对应重建项。

第二项：

$$
KL(q_\phi(z|x)\|p(z))
$$

要求 Encoder 输出的分布靠近标准正态先验，所以对应 KL 正则项。

训练时通常最小化负 ELBO：

$$
L_{\text{VAE}}
=
-E_{q_\phi(z|x)}[\log p_\theta(x|z)]
+
KL(q_\phi(z|x)\|p(z))
$$

所以可以记成：

$$
\text{VAE Loss}=\text{重建损失}+\text{KL 正则项}
$$

</details>

## 4. VAE 损失函数：重建项和 KL 正则项【重点】

VAE 损失有两部分：

$$
L_{\text{VAE}}=L_{\text{recon}}+L_{\text{KL}}
$$

### 重建损失管什么

重建损失负责让生成图像像原图：

$$
x'\approx x
$$

实际图像任务中常用 MSE：

$$
L_{\text{recon}}=\|x-x'\|^2
$$

它追求的是“重建精度”。如果只有这项，模型会尽量把每张图的信息塞进 `z`，使 Decoder 还原得越像越好。

### KL 散度为什么叫正则项

KL 正则项负责约束 Encoder 输出的分布：

$$
KL(q_\phi(z|x)\|p(z))
$$

通常：

$$
p(z)=N(0,I)
$$

所以 KL 项的意思是：

$$
q_\phi(z|x)\quad \text{不要离}\quad N(0,I)\quad \text{太远}
$$

这就是课件里“正则项”的标准含义。更准确地说：

- 生成精度项 = 重建损失。
- 正则项 = KL 散度。

如果课件某处让人感觉“生成精度也是正则项”，可以理解为它想表达“总损失里有两个约束”，但机器学习术语里真正的正则项通常指 KL 散度这一项。

### 为什么不能只要重建损失

如果只追求重建精度，模型会倾向于让：

$$
\sigma\rightarrow 0
$$

也就是每张图的分布退化成一个点。这样 VAE 又变回普通自编码器：

$$
x\rightarrow z\rightarrow x'
$$

隐空间重新变得破碎，随机采样时容易落到 Decoder 没见过的区域。

KL 项的作用是防止这种退化：

$$
q_\phi(z|x)\approx N(0,I)
$$

这样训练集中不同图像对应的分布会被规整到同一个公共隐空间里，区域之间有更多重叠和过渡。

### 训练后不用 Encoder，那为什么还能生成

训练阶段：

$$
x_i\rightarrow q_\phi(z|x_i)\rightarrow z_i\rightarrow \text{Decoder}\rightarrow x_i'
$$

每张图确实对应不同的分布。但是 KL 项不断把这些分布往同一个标准正态分布拉：

$$
q_\phi(z|x_i)\approx N(0,I)
$$

所以 Decoder 在训练中学到的是：

> 标准正态隐空间里的不同区域，对应不同类型的图像。

训练结束后，Encoder 的整理工作已经完成。生成时直接：

$$
z\sim N(0,I),\qquad x=\text{Decoder}(z)
$$

这不是说“所有图片都来自同一个具体样本的分布”，而是说：从公共隐空间分布里随机抽一个坐标点，Decoder 把这个点翻译成图像。

### 抽歪一点会不会完全变成别的东西

理想情况下不会。VAE 的训练目标会推动隐空间变得平滑：

$$
z\text{ 稍微变化}\rightarrow x\text{ 稍微变化}
$$

原因有两个：

- Encoder 输出的是分布，不是点。同一张图附近的一团 `z` 都要能重建相似图像。
- KL 项让不同样本分布靠近公共正态分布，减少孤立点和空白区域。

但这不是绝对保证。VAE 训练不好、KL 太弱或隐空间组织不理想时，仍然可能采到奇怪区域。

<details>
<summary>展开：对角高斯 KL 的解析式</summary>

若：

$$
q_\phi(z|x)=N(\mu,\operatorname{diag}(\sigma^2)),\qquad p(z)=N(0,I)
$$

则 KL 有解析解：

$$
KL(q_\phi(z|x)\|p(z))
=
\frac{1}{2}\sum_i
(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1)
$$

代码里常用 `log_var=log(sigma^2)`，写成：

$$
L_{\text{KL}}
=
-\frac{1}{2}\sum_i
(1+\log\sigma_i^2-\mu_i^2-\sigma_i^2)
$$

两种写法本质一样。

</details>

## 5. 重参数化技巧【重点】

VAE 训练时需要从 Encoder 输出的分布中采样：

$$
z\sim N(\mu,\sigma^2)
$$

问题是，直接采样像一个黑盒：

$$
(\mu,\sigma)\rightarrow \text{sample}\rightarrow z
$$

反向传播时很难知道 $\mu$ 或 $\sigma$ 变化一点，采样结果 `z` 会怎么变化。

重参数化技巧把采样改写为：

$$
\epsilon\sim N(0,I)
$$

$$
z=\mu+\sigma\cdot \epsilon
$$

随机性被移动到外部噪声 $\epsilon$ 上，而 $\epsilon$ 不依赖网络参数。

### 为什么这样就能算梯度

一次前向传播中，先随机采到一个 $\epsilon$。采到之后，在这一次计算图里它就是一个固定常数。

例如：

$$
\epsilon=0.7
$$

则：

$$
z=\mu+0.7\sigma
$$

这就是普通可导函数：

$$
\frac{\partial z}{\partial \mu}=1,\qquad
\frac{\partial z}{\partial \sigma}=\epsilon
$$

你之前问得很关键：$\partial z/\partial \sigma=\epsilon$，而 $\epsilon$ 是随机的，怎么梯度下降？

答案是：**随机梯度不是不能用，它是对真实期望梯度的随机估计。**

训练真正优化的是：

$$
E_{z\sim N(\mu,\sigma^2)}[L(z)]
$$

重参数化后等价于：

$$
E_{\epsilon\sim N(0,I)}[L(\mu+\sigma\epsilon)]
$$

每次采一个或几个 $\epsilon$，就像 mini-batch SGD 每次只采一小批样本一样，用随机估计来更新参数。单次梯度会有噪声，但长期平均方向是对的。

<details>
<summary>展开：一个一维例子说明随机梯度为什么可用</summary>

假设：

$$
L(z)=z^2,\qquad z=\mu+\sigma\epsilon
$$

则：

$$
L=(\mu+\sigma\epsilon)^2
$$

对 $\mu$ 和 $\sigma$ 求导：

$$
\frac{\partial L}{\partial \mu}=2(\mu+\sigma\epsilon)
$$

$$
\frac{\partial L}{\partial \sigma}=2(\mu+\sigma\epsilon)\epsilon
$$

它们确实包含随机的 $\epsilon$。但取期望后：

$$
E[2(\mu+\sigma\epsilon)]=2\mu
$$

$$
E[2(\mu+\sigma\epsilon)\epsilon]=2\sigma
$$

这正好是：

$$
E[z^2]=\mu^2+\sigma^2
$$

的真实梯度：

$$
\frac{\partial}{\partial\mu}=2\mu,\qquad
\frac{\partial}{\partial\sigma}=2\sigma
$$

所以重参数化后的梯度虽然每次随机，但在统计意义上是正确的。

</details>

## 6. VAE 的生成能力与局限【了解】

VAE 的生成流程是：

$$
z\sim N(0,I)
$$

$$
x=\text{Decoder}(z)
$$

它能生成训练集中没见过的新图像，是因为训练时 Decoder 已经学习了公共隐空间到图像空间的映射。

但普通 VAE 生成不可控。你不知道哪个 `z` 精确对应“白猫”“汽车”“某种风格”。如果要控制生成，需要额外条件，例如类别标签或文本：

$$
p_\theta(x|z,c)
$$

其中 `c` 是条件信息。

VAE 另一个常见缺点是图像容易模糊。原因是像素级 MSE 倾向于平均多种可能结果。比如两张语义上都是猫的图，像素位置可能差很多，MSE 会鼓励模型生成平滑的折中结果。

这也是课件随后引出 GAN 的原因：GAN 不只看像素误差，而是通过判别器学习“像不像真实图像”的分布级信号。

## 7. GAN 基本架构【了解】

GAN 包含两个网络：

- 生成器 `G`：输入随机噪声 `z`，输出假图像 `G(z)`。
- 判别器 `D`：输入一张图，判断它是真图还是假图。

训练目标是对抗博弈：

$$
\min_G\max_D V(D,G)
$$

经典损失写作：

$$
V(D,G)
=
E_{x\sim p_{\text{data}}}[\log D(x)]
+
E_{z\sim p(z)}[\log(1-D(G(z)))]
$$

判别器希望：

$$
D(x)\rightarrow 1,\qquad D(G(z))\rightarrow 0
$$

生成器希望：

$$
D(G(z))\rightarrow 1
$$

训练时交替优化：

- 固定 `G`，训练 `D` 区分真图和假图。
- 固定 `D`，训练 `G` 让假图骗过 `D`。

当生成分布和真实分布完全一致时：

$$
p_G(x)=p_{\text{data}}(x)
$$

此时判别器无法区分真假：

$$
D(x)=\frac{1}{2}
$$

课件里还提到理论上最优判别器会把生成器目标转化为真实分布与生成分布之间的 JS 散度。复习时知道结论即可：GAN 的目标是让生成分布逼近真实图像分布。

## 8. Diffusion Model 总览【重点】

扩散模型的核心是两条过程：

- 前向过程：从真实图像开始，逐步加噪，最后接近纯高斯噪声。
- 反向过程：从纯噪声开始，逐步去噪，最后生成图像。

直观流程：

$$
x_0\rightarrow x_1\rightarrow x_2\rightarrow \cdots \rightarrow x_T
$$

其中 `x_0` 是真实图像，`x_T` 接近纯噪声。

生成时反过来：

$$
x_T\rightarrow x_{T-1}\rightarrow \cdots \rightarrow x_0
$$

前向加噪可以直接人为定义，因为我们知道怎么加高斯噪声。难点是反向去噪：模型要学会每一步该去掉什么噪声。

所以 Diffusion 的本质可以记成：

> 训练一个噪声预测网络，让它在任意时间步 `t` 看见含噪图 `x_t` 后，预测当初加进去的噪声。

## 9. Diffusion 去噪模型设计【重点】

课件给出两个思路。

思路一：直接预测加噪前图像。

$$
x_t\rightarrow \text{Model}\rightarrow x_{t-1}\ \text{或}\ x_0
$$

问题是图像分布复杂，猫、车、人脸、风景差异很大，直接预测干净图像学习难度高。

思路二：预测噪声。

$$
x_t,t\rightarrow \epsilon_\theta(x_t,t)
$$

然后用预测噪声去还原更干净的图像。

为什么预测噪声更容易？因为无论原图是什么，加入的噪声都来自高斯分布：

$$
\epsilon\sim N(0,I)
$$

噪声的统计规律比图像本身简单，模型更容易学习。

## 10. Diffusion 训练流程【重点】

朴素做法是从 `x_0` 开始一步步加噪，同时训练模型预测每一步噪声。但这样有两个问题：

- 训练顺序固定，前期只学低噪声，后期只学高噪声，优化不稳定。
- 如果每次随机时间步 `t` 都要真的加噪 `t` 次，计算很浪费。

改进做法是：随机采样时间步 `t`，并用闭式公式一步得到 `x_t`。

常见前向加噪公式是：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中：

- `x_0`：干净图像。
- `t`：随机采样的时间步。
- $\epsilon\sim N(0,I)$：真实加入的噪声。
- $\bar{\alpha}_t$：由噪声强度累计得到的系数，控制第 `t` 步保留多少原图信息。

训练时，模型输入：

$$
(x_t,t)
$$

模型输出：

$$
\epsilon_\theta(x_t,t)
$$

损失函数是预测噪声和真实噪声的 L2 损失：

$$
L_{\text{diffusion}}
=
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$

### 训练流程一页版

1. 从训练集中随机取一张图 `x_0`。
2. 随机采样时间步 `t`。
3. 随机采样高斯噪声 $\epsilon$。
4. 用公式一步构造含噪图 `x_t`。
5. 让模型根据 `x_t` 和 `t` 预测噪声。
6. 用 $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$ 更新模型。

<details>
<summary>展开：为什么可以一步得到任意时间步的含噪图</summary>

每一步加噪可写成：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

不断递归展开，可以得到：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中：

$$
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

关键点是：多个独立高斯噪声的线性组合仍然是高斯噪声，所以可以把多步噪声合并成一个新的 $\epsilon\sim N(0,I)$。

这就是课件里“多步加噪变成单步加噪即可得到训练数据”的数学原因。

</details>

## 11. Diffusion 推理与条件生成【重点】

推理阶段从纯噪声开始：

$$
x_T\sim N(0,I)
$$

然后循环执行：

$$
x_t\rightarrow x_{t-1}
$$

每一步模型预测当前噪声：

$$
\epsilon_\theta(x_t,t)
$$

再根据预测结果去掉一部分噪声。课件还提到，去噪后通常会加入一定随机噪声，这样可以增加生成多样性，也能缓解误差累积。

### 为什么普通扩散生成不可控

如果输入只有随机噪声：

$$
x_T\sim N(0,I)
$$

模型只能随机生成图像。它不知道你想要猫、车还是风景。

要可控生成，就要加入条件：

$$
\epsilon_\theta(x_t,t,c)
$$

其中 `c` 可以是类别标签、文本向量、边缘图、深度图等。

文生图时，文本不是直接变成一个 VAE 向量，而是作为条件参与每一步去噪：

$$
\text{prompt}\rightarrow \text{text encoder}\rightarrow c
$$

$$
\epsilon_\theta(x_t,t,c)
$$

也就是说：

> 文本指导模型每一步“应该往哪个方向去噪”，最后随机噪声逐渐变成符合文字描述的图像。

这正好回答之前那个疑问：模型不是先知道某个隐空间向量精确对应“白猫”，而是在大量图文配对训练中学会“文本条件如何改变去噪方向”。

## 12. 隐空间扩散与 Stable Diffusion【重点/了解】

普通扩散如果直接在像素空间做去噪，计算量很大。例如 `512 x 512` 图像有 262,144 个像素位置，还要多步迭代去噪，速度和显存压力都很高。

Stable Diffusion 使用 Latent Diffusion 的思想：先用 VAE 把图像压到低维隐空间，再在隐空间里做扩散。

大致流程：

$$
x\rightarrow \text{VAE Encoder}\rightarrow z
$$

在训练扩散模型时，对隐变量 `z` 加噪、预测噪声、计算损失。

生成时：

$$
z_T\sim N(0,I)
$$

$$
\text{text prompt}\rightarrow \text{text encoder}\rightarrow c
$$

$$
z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_0
$$

最后：

$$
z_0\rightarrow \text{VAE Decoder}\rightarrow x
$$

所以 Stable Diffusion 里，VAE 主要负责：

$$
\text{像素图像}\leftrightarrow \text{压缩隐空间}
$$

真正决定“根据文字生成什么”的，是文本编码器和条件去噪网络。

这也解释了之前的问题：

- 普通 VAE：从 `N(0,I)` 抽一个 `z`，Decoder 随机生成，不好控制。
- 文生图扩散：从随机隐空间噪声开始，文本条件持续指导去噪，最终得到符合 prompt 的隐变量，再交给 VAE Decoder 解码。

## 13. Flow Matching【简单带过】

Flow Matching 也是从噪声生成图像，但它强调学习一条从噪声分布到数据分布的连续路径。

课件中的直观说法是：

$$
x_0\rightarrow x_{0.25}\rightarrow x_{0.5}\rightarrow x_{0.75}\rightarrow x_1
$$

其中 `x_0` 来自噪声分布，`x_1` 来自图像分布。模型学习的是沿着路径把噪声搬运到图像的速度场或方向。

只需要记住：

- Diffusion：常理解为多步加噪、再多步去噪。
- Flow Matching：直接学习从噪声到数据的连续传输路径。
- 现在一些前沿生成模型会使用 Flow Matching 或相关思想。

## 14. 一页速记

| 模型 | 核心思想 | 重点记忆 |
|---|---|---|
| VAE | Encoder 输出分布，Decoder 从隐变量生成图像 | `Loss = 重建损失 + KL 正则项`；`q(z|x)` 是训练时针对当前样本的近似后验；生成时从 `N(0,I)` 采样 |
| GAN | 生成器和判别器对抗训练 | `G` 想骗过 `D`，`D` 想区分真假；交替优化；理论目标是让 `p_G` 逼近 `p_data` |
| Diffusion | 先把图像加噪成高斯噪声，再学习反向去噪 | 模型主要预测噪声；训练随机采样时间步；损失是预测噪声和真实噪声的 L2 |
| Stable Diffusion | 在 VAE 隐空间中做条件扩散 | 文本不是直接变成图像向量，而是指导每一步去噪；最后用 VAE Decoder 解码 |
| Flow Matching | 学习从噪声到数据的传输路径 | 知道概念即可：从噪声分布沿连续路径走向图像分布 |

最容易混的几个点：

- $q_\phi(z|x)$ 是 Encoder 给当前图片预测的隐变量分布，不是生成时最终使用的分布。
- 生成时不用 Encoder，是因为训练时 KL 已经把各样本的隐变量分布规整到公共 `N(0,I)` 隐空间。
- KL 散度才是 VAE 的正则项；重建损失负责生成精度。
- 重参数化后 $\epsilon$ 虽然随机，但一次前向中可当常数；梯度是随机估计，可以用于 SGD。
- Diffusion 文生图不是“文字直接找到某个 z”，而是“文字条件指导随机噪声逐步去噪成目标图像”。
