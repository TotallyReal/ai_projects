# Decomposition of skew-symmetric $\times$ orthogonal

During my journey to understand the mathematics behind [computer vision algorithms](points_of_view.md),
I came across the problem of trying to calibrate between the positions of two cameras. This is an interesting
problem, since once we know the position and orientation of the two cameras, we can try to combine the 2D 
images into a 3D scene.

To define the position of a second camera relative to the first, we need:
- The relative position $0\neq P\in \mathbb{R}^3$,
- The relative orientation $K \in \mathrm{SO}_3(\mathbb{R})$.

A previous interesting mathematical step lets us discover the matrix $[P_\times] \cdot K$ up to a nonzero 
scalar multiplication, where $[P_\times]$ is the matrix representing the cross product by $P$, namely:
$$[P_\times]v=P\times v\;\;\;,\;\;\;[P_\times]=\pmatrix{0 & -P_3 & P_2 \\P_3 & 0 & -P_1 \\-P_2 & P_1 & 0 }.$$

Now the problem is to decompose it into $[P_\times]$ and $K$.

### Problem:
> Let $P\in \mathbb{R}^3$ and $K\in \mathrm{SO}_3(\mathbb{R})$. Given the product
$[P_\times]K$ up to a nonzero scalar multiplication, find $P$ and $K$.

### Remark:
> In case you never noticed this, like me after a couple of math postdocs, every $3\times 3$ real skew symmetric matrix
has the form $[P_\times]$ for some vector $P$, namely, it "represents" a direction\perpendicular plane in $\mathbb{R}^3$.

# The skew symmetric part

Finding $P$ is mostly easy. Since $P\neq 0$, we have that $rank([P_\times])=2$ and therefore
$rank([P_\times]K)=2$, meaning that the null space has dimension 1.

It is not hard to check that $P$ itself is in the left null space, namely $P^T \cdot [P_\times] K = 0$,
and it is even simpler once you notice that for any vector:
$$v^T [P_\times] = ([P_\times]^T v)^T = -([P_\times]v)^T=-(P\times v)^T.$$
Thus, $P$ generates the left null space of $[P_\times]K$ so we can find it up to a scalar multiplication,
which is the best that we can do.

If $[P_\times]$ was invertible, then we could have easily found $A=[P_\times]^{-1}([P_\times]K)$. However, this doesn't hold here,
so we need some way to extract the rank 2 part from $F$, while keeping $A$ intact. Luckily for us, both $[P_\times]$ and
$A$ have nice geometric structure that we can exploit.

# A bit of spectral decomposition

Computing the characteristic polynomial of $[P_\times]$ we get:
$$\det(xI-[P_\times]) = x(x^2+|P|^2).$$
Hence we have 3 eigenvalues: 0 and $\pm i |P|$. Since $P\neq 0$, these eigenvalues are distinct, and therefore 
$[P_\times]$ is diagonalizable.

Using our ever most helpful and stronger theorem about spectral decomposition for normal operators, we actually know that
we can diagonalize it with a unitary matrix:
$$[P_\times] = UDU^*,\; UU^*=I,\; D=\pmatrix{\lambda i & 0 & 0 \\ 0 & -\lambda i & 0 \\ 0 & 0 & 0}.$$

So now, a product of skew-symmetric with orthogonal looks like:
$$[P_\times] K = U \pmatrix{\lambda & 0 & 0 \\ 0 & \lambda & 0 \\ 0 & 0 & 0} \pmatrix{i & 0 & 0 \\ 0 & -i & 0 \\ 0 & 0 & 1} U^* K,$$
which is a unitary $\times$ nonegative diagonal $\times$ unitary. If we are given this decomposition $[P_\times] K = U\Sigma V^T$, 
then we can find $K$ by:
$$K=U \pmatrix{-i & 0 & 0 \\ 0 & i & 0 \\ 0 & 0 & 1} V^T.$$

Luckily for us, we have just the technique to find such decompositions: The singular value decomposition. However,
the problem with this theorem is that while the diagonal part is more or less unique, the two unitary matrices are not.
For example, no one promises us that the $K$ defined above is even a real matrix. But, if we can choose the decomposition 
is such a way where $K$ is real orthogonal, then $([P_\times]K)K^{-1}$ will have to be real skew symmetric, and we have 
already seen that it is unique, which will complete our decomposition.

With this idea in mind, we need to change the decomposition above to only use real matrices, and the first step is to 
get rid of the $\pm i$. We can do it using:

$$\pmatrix{\lambda i & 0 \\ 0 &-\lambda i} =\overbrace{\frac{1}{\sqrt{2}}\pmatrix{1 & -i \\ 1 &i}}^E\pmatrix{0 & \lambda \\ -\lambda & 0} \overbrace{\frac{1}{\sqrt{2}}\pmatrix{1 & 1 \\ i &-i}}^{E^*}$$

This means that we can push $E$ (or its $3\times 3$ version) into $U$ above to get:

$$[P_\times] K = (UE) \pmatrix{0 & \lambda & 0 \\ -\lambda & 0 & 0 \\ 0 & 0 & 0} (UE)^* K = (UE) \pmatrix{\lambda & 0 & 0 \\ 0 & \lambda & 0 \\ 0 & 0 & 0} \pmatrix{0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 1} (UE)^* K.$$

This is much better, since if we know that $UE$ is a real matrix, then so will $K$.

# Formal Proof

## Construction:
Let $F=[P_\times]K$ and write its singular value decomposition $F=U\Sigma V^T$, where:
- Since $F$ is real, we can find such decomposition where both $U,V$ are real.
- $\Sigma$ is diagonal with nonegative entries.
- To find the entries in $\Sigma$, we have:
  $$U\Sigma^2 U^T=FF^T=-[P_\times]^2.$$
  Conjugate matrices have the same eigenvalues, which on the right hand side are $0, \lambda^2, \lambda^2$.
  The eigenvalues of $\Sigma$ are the nonnegative square roots of these, so $0, \lambda, \lambda$, and up to reordering,
  we can assume that:
  $$\Sigma = \pmatrix{\lambda&0&0\\0&\lambda&0\\0&0&0}$$

Define $W_\pm$ to be
$$W = \pmatrix{0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 1},$$
and note that $W^{-1}=W$, so it is orthogonal. We now have the decomposition
$$F=U(\Sigma W) W^T V^T=U\pmatrix{0 & \lambda & 0 \\ -\lambda & 0 & 0 \\ 0 & 0 & 0} (VW)^T,$$
which is exactly the form that we expected to. Let us define
$$\begin{align}\tilde{K} & =U(VW)^T \in \mathrm{O}_3(\mathbb{R})\\
S & = F\tilde{K}^{-1}=U(\Sigma W) U^T = U \pmatrix{0 & \lambda & 0 \\ -\lambda & 0 & 0 \\ 0 & 0 & 0} U^T\end{align}.$$

Since $\Sigma W$ is skew-symmetric, so is $S$, and we have found a decomposition $F=S\tilde{K}$ as a product
of skew-symmetric and orthogonal.

## Uniqueness:

As mentioned before, we can write $S=[\tilde{P}_\times]$ for some $\tilde{P}\in\mathbb{R}^3$, so that
$$[P_\times]K = [\tilde{P}_\times]\tilde{K}.$$
We have already shown that this implies that $\alpha P= \tilde{P}$ for some $\alpha \neq 0$, so that
$$[P_\times]K = [P_\times]\alpha\tilde{K}\; \Rightarrow\; [P_\times] =[P_\times]\alpha\tilde{K}K^T.$$

The matrix $[P_\times]$ has rank two, so it contains two independent (nonzero) rows $w_1^T, w_2^T$, for which 
$$w_i^T = w_i^T(\alpha \tilde{K}K^T).$$
Using the fact that orthogonal matrices do not change the norm, we get that
$$|w_i^T| = |w_i^T(\alpha \tilde{K}K^T)| = |w_i^T||\alpha|,$$
so that $\alpha = \pm 1$.

The matrix $\alpha\tilde{K}K^T$ is orthogonal and already fixes two directions. Note that these
two directions are exactly the vectors which span $P^\perp$, so we only need to find out what the 
matrix does to $P$ itself. Since it is orthogonal it must send it to $\pm P$.

Let $T_P$ be the reflection through the plane perpendicular to $P$, namely it fixes that plane,
and switch the sign of $P$, or formally:
$$T_P(v) = v-2\left\langle v, \frac{P}{|P|} \right\rangle \frac{P}{|P|}.$$
It follows that $\alpha\tilde{K}K^T$ is either the identity or $T_P$, or generally:

### Claim:
> Let $P,\tilde{P}\in \mathbb{R}^3$ and $K,\tilde{K} \in \mathrm{O}_3(\mathbb{R})$ such that $[P_\times]K = [\tilde{P}_\times]\tilde{K}$.
> 
> Then there is $\alpha \in \{\pm 1\}$ and $\epsilon \in \{0,1\}$ such that
> $$\begin{align}\tilde{P} & = \alpha P \\ \tilde{K} & = \alpha T_P^\epsilon K\end{align}$$

We see that the decomposition is not exactly unique, but only unique up to the choices of $\alpha$ and $\epsilon$.

If we fix our choice of $P$, then it fixes the choice of $\alpha$. If we further work only
with orientation preserving matrices, namely $\det(K)=\det(\tilde{K})=1$, then
we can extract whether $\epsilon$ should be zero or one, and we get full
uniqueness.

# Some final vision intuition

Originally, this problem came up when trying to determine a position of a camera. So is this lack of total uniqueness
means that we cannot really find the relative position and orientation?

Yes and No:
- First of all, I ignored here completely the fact that we don't really know $[P_\times]K$ but rather 
  $\lambda [P_\times]K$ for some $\lambda \neq 0$. This is actually a scaling that we can never realy find:
  If you increase everything in your scene time 2, and also the distance between your cameras, then you will 
  see exactly the same images. Another way of thinking about it, is that you can measure your distances in 
  meters, centimeters, kilometers (or if you are a real masochist feet and miles), and it doesn't really change
  what you see - only your units of measurements.
- But what about the $\alpha,\epsilon$ choices in the last claim?
  - The sign in $\alpha$ is part of the explanation above, but instead of just scaling everything we flip all
    the directions (e.g. forward and backward switch, right and left switch, etc.).
  - The $epsilon$ choice is a bit of mind and physics breaking case. You can have two worlds, each with two camera
    which see the same two images, and the second camera has the same relative position with respect to the first camera,
    but not the same relative orientation. However, this means that the second camera looks on the whole world through
    a reflection in a mirror (which is exactly what $T_P$ does, and some time I will probably add a nice picture
    that shows what happens.)
