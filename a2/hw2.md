1.  Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss between $\boldsymbol{y}$ and $\hat{\boldsymbol{y}} ;$ i.e., show that:


$$
-\sum_{w \in V o c a b} y_{w} \log \left(\hat{y}_{w}\right)=-\log \left(\hat{y}_{o}\right)
$$
答：

​	因为$y_m$是一个one hot 向量，只有当前在真实的上下文词o的位置上，才是1。所以
$$
\begin{array}{l}
\sum_{w \in \operatorname{Vocab}} y_{w} \log \left(\hat{y_{w}}\right) \\
=1 \times \log \left(\hat{y_{o}}\right) \\
=\log P(O=o \mid C=c)
\end{array}
$$

2. (5 points) Compute the partial derivative of $\boldsymbol{J}_{\text {naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)$ with respect to $\boldsymbol{v}_{c}$. Please write your answer in terms of $\boldsymbol{y}, \hat{\boldsymbol{y}},$ and $\boldsymbol{U}$.

答：
$$
\begin{array}{l}
\frac{\partial J_{\text {naive }-\text { softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}} \\
=-\frac{\partial \log (P(O=o \mid C=c))}{\partial \boldsymbol{v}_{c}} \\
=\frac{\partial \log \left(\sum_{w=1}^{V} \exp \left(\boldsymbol{u}_{w}^{T} \boldsymbol{v}_{c}\right)\right)}{\partial \boldsymbol{v}_{c}}-\frac{\partial \log \left(\exp \left(\boldsymbol{u}_{o}^{T} \boldsymbol{v}_{c}\right)\right)}{\partial \boldsymbol{v}_{c}} \\
=\sum_{w}^{V} \boldsymbol{u}_{w} \hat{y}_{w}-\boldsymbol{u}_{o} \\
=\boldsymbol{U}^{T}(\hat{\boldsymbol{y}}-\boldsymbol{y})
\end{array}
$$

3. (5 points) Compute the partial derivatives of $\boldsymbol{J}_{\text {naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)$ with respect to each of the 'outside' word vectors, $\boldsymbol{u}_{w}$ 's. There will be two cases: when $w=o,$ the true 'outside' word vector, and $w \neq o,$ for all other words. Please write you answer in terms of $\boldsymbol{y}, \hat{\boldsymbol{y}},$ and $\boldsymbol{v}_{c}$.

答：
$$
\begin{array}{l}
\frac{\partial J_{\text {naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{w}} \\
=\frac{\partial \log \left(\sum_{w=1}^{V} \exp \left(\boldsymbol{u}_{w}^{T} \boldsymbol{v}_{c}\right)\right)}{\partial \boldsymbol{u}_{w}}-\frac{\partial \log \left(\exp \left(\boldsymbol{u}_{o}^{T} \boldsymbol{v}_{c}\right)\right)}{\partial \boldsymbol{u}_{w}}
\end{array}
$$
当 $w = o$时：
$$
\begin{array}{l}
\frac{\partial J_{\text {naive }-\text { softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{w}} \\
=\frac{\boldsymbol{v}_{c} \exp \left(\boldsymbol{u}_{o}^{T} \boldsymbol{v}_{c}\right)}{\sum_{i}^{V} \exp \left(\boldsymbol{u}_{i}^{T} \boldsymbol{v}_{c}\right)}-\boldsymbol{v}_{c} \\
=\left(\hat{y}_{o}-1\right) \boldsymbol{v}_{c}
\end{array}
$$
当$w != o$时：
$$
\begin{array}{l}
\frac{\partial J_{\text {naive }-\operatorname{softmax}}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{w}} \\
=\frac{\boldsymbol{v}_{c} \exp \left(\boldsymbol{u}_{w}^{T} \boldsymbol{v}_{c}\right)}{\sum_{i}^{V} \exp \left(\boldsymbol{u}_{i}^{T} \boldsymbol{v}_{c}\right)} \\
=\hat{y}_{w} \boldsymbol{v}_{c}
\end{array}
$$
两种情况可以合并为：
$$
\begin{array}{l}
\frac{\partial J_{\text {naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{w}} \\
=\left(\hat{y}_{w}-y_{w}\right) \boldsymbol{v}_{c} \\
y_{w}=\left\{\begin{array}{l}
1, \mathrm{w}=o \\
0, \mathrm{w} \neq o
\end{array}\right.
\end{array}
$$

4. (3 Points) The sigmoid function is given by Equation 4 :
   $$
   \sigma(\boldsymbol{x})=\frac{1}{1+e^{-\boldsymbol{x}}}=\frac{e^{\boldsymbol{x}}}{e^{\boldsymbol{x}}+1}
   $$
   Please compute the derivative of $\sigma(x)$ with respect to $\boldsymbol{x},$ where $\boldsymbol{x}$ is a vector.

   答：
   $$
   \begin{aligned}
   \frac{\partial \sigma\left(x_{i}\right)}{\partial x_{i}} &=\frac{1}{\left(1+\exp \left(-x_{i}\right)\right)^{2}} \exp \left(-x_{i}\right)=\sigma\left(x_{i}\right)\left(1-\sigma\left(x_{i}\right)\right) \\
   \frac{\partial \sigma(x)}{\partial x} &=\left[\frac{\partial \sigma\left(x_{j}\right)}{\partial x_{i}}\right]_{d \times d} \\
   &=\left[\begin{array}{cccc}
   \sigma^{\prime}\left(x_{1}\right) & 0 & \cdots & 0 \\
   0 & \sigma^{\prime}\left(x_{2}\right) & \cdots & 0 \\
   \vdots & \vdots & \vdots & \vdots \\
   0 & 0 & \cdots & \sigma^{\prime}\left(x_{d}\right)
   \end{array}\right] \\
   &=\operatorname{diag}\left(\sigma^{\prime}(x)\right)
   \end{aligned}
   $$
   

5. (4 points) Now we shall consider the Negative Sampling loss, which is an alternative to the Naive Softmax loss. Assume that $K$ negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as $w_{1}, w_{2}, \ldots, w_{K}$ and their outside vectors as $\boldsymbol{u}_{1}, \ldots, \boldsymbol{u}_{K} .$ Note that $o \notin\left\{w_{1}, \ldots, w_{K}\right\}$. For a center word $c$ and an outside word $o,$ the negative sampling loss function is given by:
   $$
   \boldsymbol{J}_{\text {neg-sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)=-\log \left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)\right)-\sum_{k=1}^{K} \log \left(\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\right)
   $$
   

   for a sample $w_{1}, \ldots w_{K},$ where $\sigma(\cdot)$ is the sigmoid function. $^{3}$
   Please repeat parts (b) and (c), computing the partial derivatives of $\boldsymbol{J}_{\text {neg-sample }}$ with respect to $\boldsymbol{v}_{c},$ with respect to $\boldsymbol{u}_{o},$ and with respect to a negative sample $\boldsymbol{u}_{k} .$ Please write your answers in terms of the vectors $\boldsymbol{u}_{o}, \boldsymbol{v}_{c},$ and $\boldsymbol{u}_{k},$ where $k \in[1, K] .$ After you've done this, describe with one sentence why this loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able to use your solution to part (d) to help compute the necessary gradients here.

   答：

   对中心词向量 $\boldsymbol{v}_{c}$ 求导
   $$
   \begin{array}{l}
   \frac{\partial J_{n e g-\text { sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}} \\
   =-\left(1-\sigma\left(\boldsymbol{u}_{o}^{T} \boldsymbol{v}_{c}\right)\right) \boldsymbol{u}_{o}+\sum_{k=1}^{K}\left(1-\sigma\left(-\boldsymbol{u}_{k}^{T} \boldsymbol{v}_{c}\right)\right) \boldsymbol{u}_{k}
   \end{array}
   $$
   对上下文词向量 $\boldsymbol{u}_{o}$ 求导：
   $$
   \begin{array}{c}
   \frac{\partial J_{\text {neg }-\text { sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{k}} \\
   =-\left(1-\sigma\left(\boldsymbol{u}_{o}^{T} \boldsymbol{v}_{c}\right)\right) \boldsymbol{v}_{c}=-\sigma\left(-\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{v}_{c}
   \end{array}
   $$
   对上下文词向量 $\boldsymbol{u}_{k}$ 求导：
   $$
   \begin{array}{c}
   \frac{\partial J_{\text {neg }-\text { sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{k}} \\
   =\left(1-\sigma\left(-\boldsymbol{u}_{k}^{T} \boldsymbol{v}_{c}\right)\right) \boldsymbol{v}_{c}=\sigma\left(\boldsymbol{-u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{v}_{c}
   \end{array}
   $$
   

   从求得的偏导数中我们可以看出，原始的softmax函数每次对 vcvc 进行反向传播时，需要与 output vector matrix 进行大量且复杂的矩阵运算，而负采样中的计算复杂度则不再与词表大小 V 有关，而是与采样数量 K 有关。

6. (3 points) Suppose the center word is $c=w_{t}$ and the context window is $\left[w_{t-m}, \ldots, w_{t-1}, w_{t}, w_{t+1}, \ldots,\right.$ $\left.w_{t+m}\right],$ where $m$ is the context window size. Recall that for the skip-gram version of word2vec, the total loss for the context window is:
   $$
   \boldsymbol{J}_{\text {skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)=\sum_{-m \leq j \leq m \atop j \neq 0} \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)
   $$
   Here, $\boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)$ represents an arbitrary loss term for the center word $c=w_{t}$ and outside word $w_{t+j} \cdot \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)$ could be $\boldsymbol{J}_{\text {naive-softmax }}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)$ or $\boldsymbol{J}_{\text {neg-sample }}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right),$ depending on your
   implementation.
   Write down three partial derivatives:
   (i) $\partial \boldsymbol{J}_{\text {skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial \boldsymbol{U}$
   (ii) $\partial \boldsymbol{J}_{\text {skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial \boldsymbol{v}_{c}$

答：
$$
\frac{\partial J_{s g}}{\partial U}=\sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial U}\\ \frac{\partial J_{s g}}{\partial v_{c}}=\sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial v_{c}}\\ \frac{\partial J_{s g}}{\partial v_{w}}=0(\text { when } w \neq c)
$$
