# Decomposition of skew-symmetric $\times$ orthogonal

During my journey to understand the mathematics behind [computer vision algorithms](points_of_view.md),
I came across the problem of trying to calibrate between the positions of two cameras. This is an interesting
problem, since once we know the position and orientation of the two cameras, we can try to combine the 2D 
images into a 3D scene.

To define the position of a second camera relative to the first, we need:
- The relative position $0\neq P\in \mathbb{R}^3$,
- The relative orientation $K \in \mathrm{SO}_3(\mathbb{R})$.

A previous interesting mathematical step lets us discover the matrix $[P_\times] \cdot K$ up to a nonzero 
scalar multiplication, where $[P_\times]$ is the skew-symmetric matrix representing the cross product by $P$, namely:
$$[P_\times]v=P\times v\;\;\;,\;\;\;[P_\times]=\pmatrix{0 & -P_3 & P_2 \\P_3 & 0 & -P_1 \\-P_2 & P_1 & 0 }.$$

Now the problem is to decompose it into $[P_\times]$ and $K$.

### Problem:
> Let $P\in \mathbb{R}^3$ and $K\in \mathrm{SO}_3(\mathbb{R})$. Given $\alpha [P_\times]K$ for some
> unknown nonzero $\alpha\in\mathbb{R}^\times$, find $P$ and $K$.

### Remark:
> In case you never noticed this, like me after a couple of math postdocs, every $3\times 3$ real skew symmetric matrix
has the form $[P_\times]$ for some vector $P$, namely, it "represents" a direction\perpendicular plane in $\mathbb{R}^3$.

# The skew symmetric part

Finding $P$ is mostly easy. Since $P\neq 0$, we have that $rank([P_\times])=2$ and therefore
$rank(\alpha[P_\times]K)=2$, meaning that the null space has dimension 1, and furthermore, it 
is independent of the choice of $\alpha$.


It is not hard to check that $P$ itself is in the left null space, namely $P^T \cdot [P_\times] K = 0$,
indeed, we have:
$$v^T [P_\times] = ([P_\times]^T v)^T = -([P_\times]v)^T=-(P\times v)^T.$$
Thus, $P$ generates the left null space of $\lambda[P_\times]K$ so we can find it up to a scalar multiplication,
which is the best that we can do.

If $[P_\times]$ was invertible, then we could have easily found $K=[P_\times]^{-1}([P_\times]K)$. However, this doesn't hold here,
so we need some way to extract the rank 2 part from $[P_\times]K$, while keeping $A$ intact. Luckily for us, both $[P_\times]$ and
$A$ have nice geometric structure that we can exploit.

# A bit of spectral decomposition

Let's understand $[P_\times]$ a little better.
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

Luckily for us, we have just the technique to find such decompositions: **The singular value decomposition**. However,
the problem with this theorem is that while the diagonal part is more or less unique, the two unitary matrices are not.
For example, no one promises us that the $K$ defined above is even a real matrix. But, if we can choose the decomposition 
is such a way where $K$ is real orthogonal, then $([P_\times]K)K^{-1}$ will have to be real skew symmetric, and we have 
already seen that it is unique, which will complete our decomposition.

With this idea in mind, we need to change the decomposition above to only use real matrices, and the first step is to 
get rid of the $\pm i$. We can do it using:

$$\pmatrix{\lambda i & 0 \\ 0 &-\lambda i} =\overbrace{\frac{1}{\sqrt{2}}\pmatrix{1 & -i \\ 1 &i}}^Q\pmatrix{0 & \lambda \\ -\lambda & 0} \overbrace{\frac{1}{\sqrt{2}}\pmatrix{1 & 1 \\ i &-i}}^{Q^*}$$

This means that we can push $Q$ (or its $3\times 3$ version) into $U$ above to get:

$$[P_\times] K = (UQ) \pmatrix{0 & \lambda & 0 \\ -\lambda & 0 & 0 \\ 0 & 0 & 0} (UQ)^* K = (UQ) \pmatrix{\lambda & 0 & 0 \\ 0 & \lambda & 0 \\ 0 & 0 & 0} \pmatrix{0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 1} (UQ)^* K.$$

This is much better, since if we know that $UQ$ is a real matrix, then so will $K$.

# Formal Proof

## Construction:
Let $E=\alpha[P_\times]K$ and write its singular value decomposition $E=U\Sigma V^T$, where:
- Since $E$ is real, we can find such decomposition where both $U,V$ are real.
- $\Sigma$ is diagonal with nonegative entries.
- To find the entries in $\Sigma$, we have:
  $$U\Sigma^2 U^T=EE^T=-\alpha^2[P_\times]^2.$$
  Conjugate matrices have the same eigenvalues, which on the right hand side are $0, (\alpha\lambda)^2, (\alpha\lambda)^2$.
  The eigenvalues of $\Sigma$ are the nonnegative square roots of these, so $0, \alpha\lambda, \alpha\lambda$, and up to reordering,
  we can assume that:
  $$\Sigma = \alpha\lambda\pmatrix{1&0&0\\0&1&0\\0&0&0}$$

As in the intuition above, define $W_\pm$ to be
$$W_\pm = \pmatrix{0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & \pm1},$$
and note that $W_\pm^{-1}=W_\pm^T$, so it is orthogonal. We now have the decomposition

$$E=U(\Sigma W_\pm) W_\pm^T V^T=\alpha\lambda \cdot U\pmatrix{0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0} (VW_\pm)^T,$$

which is exactly the form that we expected to. Let us define
$$\begin{align}\tilde{K}_\pm & =U(VW_\pm)^T \in \mathrm{O}_3(\mathbb{R})\\
S & = F\tilde{K}_\pm^{-1}=U(\Sigma W_\pm) U^T = \alpha\lambda \cdot U \pmatrix{0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0} U^T\;.\end{align}$$

Since $\Sigma W_\pm$ is skew-symmetric, so is $S$, and note that it is independent of $\pm$ choice in $W_\pm$.

To summarize, we found a decomposition $E=S\cdot \tilde{K}_\pm$ as a skew symmetric times an orthogonal matrix.

## Uniqueness:

As mentioned before, we can write $S=[\tilde{P}_\times]$ for some $\tilde{P}\in\mathbb{R}^3$, so we just managed 
to find that
$$\alpha[P_\times]K = E = [\tilde{P}_\times]\tilde{K}_\pm.$$
We have already shown that $P$ and $\tilde{P}$ are on the same line, namely $\tilde{P}=\beta P$ for some $\beta\in \mathbb{R}^\times$.
It follows that 
$$[P_\times]K = [P_\times]\frac{\beta}{\alpha}\tilde{K}_\pm\; \Rightarrow\; [P_\times] =[P_\times]\frac{\beta}{\alpha}\tilde{K}_\pm K^T.$$

The matrix $[P_\times]$ has rank two, so it contains two independent (nonzero) rows $w_1^T, w_2^T$ (which generate $P^\perp$), for which 
$$w_i^T = w_i^T(\frac{\beta}{\alpha} \tilde{K}_\pm K^T).$$
Using the fact that orthogonal matrices do not change the norm, we get that
$$|w_i^T| = |w_i^T(\frac{\beta}{\alpha} \tilde{K}_\pm K^T)| = |w_i^T|\left|\frac{\beta}{\alpha}\right|,$$
so that $\epsilon:=\frac{\beta}{\alpha} \in \{\pm 1\}$.

The matrix $\epsilon\tilde{K}_\pm K^T$ is orthogonal and already fixes two directions which generate $P^\perp$.
Hence, it must send the remaining vector $P$ to one of $\pm P$. As an exercise, you should check that this new $\pm$ sign
is basically the same $\pm$ sign that we have been carrying inside the $W_\pm$ matrix. Indeed, we didn't care about that 
sign precisely because it "disappeared" when multiplied by the skew-symmetric part $S$.

More formally, let $T_P$ be the reflection through the plane perpendicular to $P$, namely it fixes that plane,
and switch the sign of $P$:
$$T_P(v) = v-2\left\langle v, \frac{P}{|P|} \right\rangle \frac{P}{|P|}.$$

### Exercise:
> With the notations from the previous section:
> $$T_p=U\pmatrix{1&0&0 \\ 0&1&0 \\ 0&0&-1}U^T.$$

It follows that $\epsilon\tilde{K}_\pm K^T$ is either the identity or $T_P$. We can now put everything together.

### Claim:
> Let $P,\tilde{P}\in \mathbb{R}^3$ be nonzero vectors, $K,\tilde{K} \in \mathrm{O}_3(\mathbb{R})$ orientations, and 
> $\alpha\in \mathbb{R}^\times$ such that 
> $$\alpha[P_\times]K = [\tilde{P}_\times]\tilde{K}.$$
> 
> 1) The vectors $P$, $\tilde{P}$ are on the same line, namely $\tilde{P}=\beta P$, $\beta\neq 0$.
> 2) $\tilde{K}= (-1)^\varepsilon T_p^\nu K$ where $\varepsilon,\nu \in \{0,1\}$.
> 3) If we further assume that $K,\tilde{K} \in \mathrm{SO}_3(\mathbb{R})$, then $\varepsilon = \nu$. 

We see that the decomposition is not exactly unique, but only unique up to the choices of length and 
direction of the positions, and the choice of $\varepsilon,\nu$.

If we only use orientation preserving matrices, then once the position is chosen, we only have 
the two orientation choices, namely $K, -T_PK$.

# Some final vision intuition

Originally, this problem came up when trying to determine a position of a camera relative to another one (which we think 
of as being at the origin with standard orientation). So is this lack of total uniqueness
means that we cannot really find the relative position and orientation?

Yes and No:
- First of all, the lack of uniqueness in the scaling of $P$ is intrinsic to this problem: 
  If you increase everything in your scene times 2, and also the distance between your cameras, then you will 
  see exactly the same images! Another way of thinking about it, is that you can measure your distances in 
  meters, centimeters, kilometers (or if you are a real masochist feets and miles), and it doesn't really change
  what you see - only your units of measurements.
- But what about the $\varepsilon, \nu$ choices in the last claim?
  - **The $(-1)^\varepsilon$ sign**: If you take an image, and flip between its right and left, and also between up and down,
    you will get a new different image. More specifically, it will be the original image, but rotated 180 degrees. 
    However, if you also switch between forward and backwards, you actually get back your original image. Mathematically, 
    this happens because we can only see projections of points, and $v$ and $-v$ have the same projection.
    While the math works here, it is impossible to do in our 3D space (unlike just the 180 degrees rotation), and this is 
    indicated by the negative determinant of $\det(-I)=-1$.
  - The $\nu$ choice is a bit of mind and physics breaking case. You can have two worlds, each with two cameras
    which see the same two images, and the second camera has the same relative position with respect to the first camera,
    but not the same relative orientation. However, this means that the second camera looks on the whole world through
    a reflection in a mirror (which is exactly what $T_P$ does, and some time I will probably add a nice picture
    that shows what happens.)
  - Each of these choices separately are not possible in the 3D world (without using mirrors), but if we take both together
    it is actually possible (or formally, the product of two determinant $-1$ matrices has determinant 1).

All in all, if we fix the scale of the world so that $|P|=1$ and choose $K\in\mathrm{SO}_3(\mathbb{R})$, there are 4 
ways to write $[P_\times]K$ as $[\tilde{P}_\times]\tilde{K}$ with $|\tilde{P}|=1$ and $\tilde{K}\in\mathrm{SO}_3(\mathbb{R})$:

- $[P_\times]K$
- $[P_\times](-T_P\cdot K)$
- $[(-P)_\times]K$
- $[(-P)_\times](-T_P\cdot K)$

According to [wikipedia](https://en.wikipedia.org/wiki/Essential_matrix#Finding_all_solutions), while you do get 4 solutions,
if you extract the 3D positions of the points from the 2D images, in only 1 solution all the points will be in front 
of the two cameras. One day, I might even write (if and) how this works here... 
