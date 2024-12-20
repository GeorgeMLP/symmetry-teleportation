# Symmetry Teleportation in Optimization

This repository contains part of my course project for **EE 127BT: Convex Optimization** at UC Berkeley. The project explores the use of symmetry teleportation to accelerate optimization. The repository is organized as follows:

- **`conv_opt.py`**: Demonstrates the application of symmetry teleportation to optimize a convex function, $f(x)=x^2+9y^2$.
- **`relu_mlp.py`**: Applies symmetry teleportation to optimize a 3-layer ReLU neural network.
- **`teleportation.py`**: Contains the core implementation of symmetry teleportation.
- **`project_writeup.pdf`**: Contains the writeup of our project.

## Attribution

This repository is based on the [Symmetry Teleportation repository](https://github.com/Rose-STL-Lab/Symmetry-Teleportation/tree/main). The following modifications were made:

1. **Convex Optimization Task**
   - Used a different objective function, $f(x)=x^2+9y^2$, designed to exhibit both fast and slow optimization paths.  
   - The initial point was chosen near the slow optimization path, and symmetry teleportation was employed to "teleport" the optimization process to a faster convergence path.

2. **Neural Network Optimization Task**
   - Explored a new symmetry for ReLU neural networks, specifically the [positive-scale invariance](https://openreview.net/pdf?id=SyxfEn09Y7).
   - Experimental results show that leveraging this symmetry outperforms the original approach in the reference repository. The original method required computing the pseudo-inverse of the neural network input and considered symmetries limited to the last two layers.

## References

For more information on symmetry teleportation, refer to the following publications:

- Meng, Q., Zheng, S., Zhang, H., Chen, W., Ma, Z. M., Liu, T. Y. (2018). $\mathcal{G}$*-SGD: Optimizing ReLU Neural Networks in its Positively Scale-Invariant Space*. International Conference on Learning Representations.
- Godfrey, C., Brown, D., Emerson, T., Kvinge, H. (2022). *On the Symmetries of Deep Learning Models and their Internal Representations*. Advances in Neural Information Processing Systems.
- Zhao, B., Dehmamy, N., Walters, R., Yu, R. (2022). *Symmetry Teleportation for Accelerated Optimization*. Advances in Neural Information Processing Systems.
- Zhao, B., Gower, R. M., Walters, R., Yu, R. (2024). *Improving Convergence and Generalization Using Parameter Symmetries*. International Conference on Learning Representations.
