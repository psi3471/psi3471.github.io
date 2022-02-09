# Regressão linear

Whether you write your book's content in Jupyter Notebooks (`.ipynb`) or
in regular markdown files (`.md`), you'll write in the same flavor of markdown
called **MyST Markdown**.

## Regressão linear univariada

Em engenharia, é muito comum buscar um modelo para dados experimentais previamente observados. Buscar um modelo adequado é útil para prever um valor a partir de dados não observados, se tratando de um problema de *regressão*. Um dos modelos mais simples é o ajuste de uma reta a dados conhecidos, o que leva à solução conhecida como *regressão linear univariada*.


Considere que

$$
\{(x_1,d_1),(x_2,d_2),\cdots, (x_N,d_N)\},
$$

seja um conjunto de $N$ pontos conhecidos previamente que podem ser fruto de um experimento. Vamos obter a melhor reta que se ajusta a esses pontos. Quando dizemos \emph{melhor}, precisamos especificar em que sentido. Como, em geral, os pontos são experimentais, raramente se consegue obter uma reta
 que se ajuste exatamente aos pontos. Por isso, vamos buscar uma aproximação que melhor se ajusta aos dados, considerando o critério dos mínimos quadrados. Assim, deseja-se obter uma relação matemática do tipo
 
$$
d=wx+b
$$

entre as variáveis $x$ e $d$, em que $w$ e $b$ são constantes que se deseja determinar. É comum  chamar $d$ de sinal desejado ou rótulo, $x$ de entrada, $w$ de peso e $b$ de viés (ou \emph{bias}).

Quando os pontos experimentais são colineares, a reta passa exatamente por todos os $n$ pontos e as constantes desconhecidos $w$ e $b$ satisfazem

$$
\begin{array}{c}
  d_1=w\;x_1+b \\
  d_2=w\;x_2+b \\
  \vdots \\
  d_N=w\;x_N+b.
\end{array}
$$

Podemos reescrever esse sistema de equações na forma matricial, ou seja,

$$
\underbrace{\left[
  \begin{array}{c}
    d_1 \\
    d_2 \\
    \vdots \\
   d_N \\
  \end{array}
\right]}_{\mathbf{d}}=
\underbrace{\left[
  \begin{array}{cc}
    1&x_1 \\
    1&x_2 \\
    \vdots&\vdots \\
    1&x_N
  \end{array}
\right]}_{\mathbf{X}}
\underbrace{\left[
  \begin{array}{c}
    b \\
    w   \end{array}
\right]}_{\mathbf{w}}.
$$

Neste caso, como os pontos experimentais são colineares, vale $\mathbf{d}-\mathbf{X}\mathbf{w}=\mathbf{0}$.

Caso os pontos não sejam colineares, o que acontece na maior parte dos casos, $\mathbf{d}-\mathbf{X}\mathbf{w}\neq\mathbf{0}$. Dessa forma, para encontrar a reta que melhor se ajusta aos dados, vamos representar a diferença entre os dois vetores por meio do vetor de erros, ou seja, 

$$
\mathbf{e}=\mathbf{d}-\mathbf{X}\mathbf{w}.
$$

Os elementos desse vetor de erros, $e_i=d_i-b-wx_i$ para $i=1,\cdots,n$, representam as distâncias verticais da reta $d=wx+b$ aos  pontos experimentais $(x_i,d_i)$, como ilustrado na {numref}`MV2`, considerando $n=3$.

```{figure} ./images/mmq.png
---
height: 250px
name: MV2
---
Distância de um conjunto de pontos a uma determinada reta, considerando $n=3$
```

A *melhor* reta segundo o critério dos mínimos quadrados deve minimizar a norma Euclidiana ao quadrado do vetor de erros, ou seja

$$
\|\mathbf{e}\|^2=\sum_{i=1}^n e_i^2=\|\mathbf{d}-\mathbf{X}\mathbf{w}\|^2=\sum_{i=1}^n(y_i-b-wx_i)^2.
$$

Para minimizar essa norma quadrática, devemos derivá-la em relação às constantes $w$ e $b$ que se deseja determinar e igualar essas derivadas a zero. Assim, obtemos as seguintes derivadas

$$
 \begin{array}{cccc}
   \displaystyle\frac{\displaystyle\partial\sum_{i=1}^n e_i^2}{\partial w} & = & 2\displaystyle\sum_{i=1}^n e_i\displaystyle\frac{\partial e_i}{\partial w} &
  = -2\displaystyle\sum_{i=1}^n e_i x_i\\
   \displaystyle\frac{\displaystyle\partial\sum_{i=1}^n e_i^2}{\partial b} & = & 2\displaystyle\sum_{i=1}^n e_i\displaystyle\frac{\partial e_i}{\partial b} &
  = - 2\displaystyle\sum_{i=1}^n e_i,
 \end{array}
$$

que podem ser escritas de forma compacta como

$$
\displaystyle\frac{\displaystyle\partial\sum_{i=1}^n e_i^2}{\partial \mathbf{w}}=
 -2\displaystyle\sum_{i=1}^n \left[
                                  \begin{array}{c}
                                    x_i \\
                                    1 \\
                                  \end{array}
                                \right] e_i=-2\mathbf{X}^{{\rm T}}\mathbf{e}=-2\mathbf{X}^{{\rm T}}(\mathbf{d}-\mathbf{X}\mathbf{w}),
$$

em que $(\cdot)^{{\rm T}}$ representa a operação de transposição da matriz $\mathbf{X}$. Igualando essa derivada ao vetor nulo, obtemos

$$
-2\mathbf{X}^{{\rm T}}(\mathbf{d}-\mathbf{X}\mathbf{w}^{\rm o})=\mathbf{0},
$$

ou ainda

$$
\mathbf{X}^{{\rm T}}\mathbf{X}\mathbf{w}^{\rm o}=\mathbf{X}^{{\rm T}}\mathbf{d}.
$$

Portanto, o vetor de coeficientes $\mathbf{w}$ que satisfaz  essa equação, denotado como $\mathbf{w}^{\rm o}=[\,b^{\rm o}\;\;w^{\rm o}\,]^{{\rm T}}$, minimiza a norma quadrática do vetor do erros e $$d=w^{\rm o}x+b^{\rm o}$$ é a melhor reta que se ajusta aos pontos experimentais segundo o critério dos mínimos quadrados.

Se $\mathbf{X}^{{\rm T}}\mathbf{X}$ for invertível, $\mathbf{w}^{\rm o}$

$$
\mathbf{w}^{\rm o}=(\mathbf{X}^{{\rm T}}\mathbf{X})^{-1}\mathbf{X}^{{\rm T}}\mathbf{d}\nonumber.
$$

Essa equação expressa a unicidade da solução. Assim, se os pontos experimentais são não colineares, existe uma única reta que se ajusta a esses pontos segundo o critério dos mínimos quadrados.

Observações importantes:

* O modelo $d=w^{\rm o}x+b^{\rm o}$ é de fato linear apenas quando $b^{\rm o}\neq 0$, pois neste caso $x=0$ leva a $d=0$. No entanto, o termo \emph{linear} é frequentemente usado na literatura neste caso para se referir ao modelo dado por uma reta.

* Os dados $\{(x_1,d_1),(x_2,d_2),\cdots, (x_N,d_N)\}$ conhecidos previamente foram totalmente usados aqui para se obter o modelo da reta. Neste caso, eles podem ser chamados de dados de {\rm T}extbf{treinamento} do modelo.

* A matriz $(\mathbf{X}^{{\rm T}}\mathbf{X})^{-1}\mathbf{X}^{{\rm T}}$ é conhecida na literatura como a pseudoinversa de $\mathbf{X}$.
