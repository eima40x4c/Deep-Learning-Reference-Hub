# Contributing to Deep Learning Reference Hub

Thank you for considering contributing to this project!  
The **Deep Learning Reference Hub** aims to provide **clean, well-documented, and educational implementations** of core deep learning concepts.

Please follow these guidelines to keep the repository consistent and high-quality.

---

## 📌 Table of Contents

- [Contributing to Deep Learning Reference Hub](#contributing-to-deep-learning-reference-hub)
  - [Table of Contents](#-table-of-contents)
  - [Code of Conduct](#-code-of-conduct)
  - [Getting Started](#️-getting-started)
  - [Coding Standards](#-coding-standards)
  - [Documentation Standards](#-documentation-standards)
    - [Module-Level Docstrings](#module-level-docstrings)
    - [Class and Function Docstrings](#class-and-function-docstrings)
  - [Adding New Resources](#-adding-new-resources)
  - [Pull Requests](#-pull-requests)

---

## ✅ Code of Conduct

Be respectful and constructive. Discussions should stay technical and educational.

---

## ⚙️ Getting Started

1. Fork the repository and clone it locally.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a new branch for your changes:
```bash
git checkout -b feature/my-new-feature
```

--- 

## 📝 Coding Standards

- Follow PEP 8 for Python code style.
- Use Black for auto-formatting:
```bash
black .
```
- Run pydocstyle to check docstrings:
```bash
pydocstyle .
```

--- 

## 📖 Documentation Standards

We use NumPy style docstrings across the repository.  
Each module must include a module-level docstring, and each public function or class should have clear parameter/return documentation.

### Module-Level Docstrings

Use the following format for all module-level docstrings:
```python 
"""
<Module Title>
=========================

A brief but clear description of what this module does, its purpose, and context 
in deep learning. Mention if it's an implementation, utility, or theoretical demonstration.

References
----------
- <Author(s)>. <Title of Paper or Book>. <Publisher/Conference>, <Year>.
  <URL if applicable> 

Author
------
<Your Name or Team Name> (Deep Learning Reference Hub)

License
-------
MIT License

Notes
-----
Any special considerations, numerical stability warnings, or implementation notes.
"""
```
Example:
```python 
"""
Adam Optimizer
==============

Implements Adaptive Moment Estimation (Adam) for stochastic optimization.

References
----------
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
  https://arxiv.org/abs/1412.6980

Author
------
Deep Learning Reference Hub

License
-------
MIT License
"""
```

### Class and Function Docstrings

Example (NumPy style):
```python 
class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) Optimizer.

    Combines the benefits of AdaGrad and RMSProp by computing adaptive learning
    rates using first and second moment estimates.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size for parameter updates.
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    """

def update_parameters(params: Dict, grads: Dict, t: int) -> Dict:
    """
    Update model parameters using Adam optimization.

    Parameters
    ----------
    params : dict
        Model parameters to be updated.
    grads : dict
        Gradients for each parameter.
    t : int
        Timestep for bias correction.

    Returns
    -------
    dict
        Updated parameters after applying Adam step.
    """
```

---

## 🔗 Adding New Resources
To add books, papers, or courses:
1. Open `Resources.md`.
2. Add the resource under the appropriate section.
3. Use this format:
```markdown
- **Title (Author, Year)** – [link](https://example.com)
```
Only add well-established, high-quality resources.

---

## ✅ Pull Requests

1. Ensure code passes formatting and style checks:
```bash
black .
pydocstyle .
```
2. Write clear commit messages.
3. Reference any related issue in your PR description.
4. PRs will be reviewed for:
    - Correctness
    - Code clarity
    - Proper documentation (docstrings, updated `Resources.md` if needed)

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard to ensure a clean and meaningful commit history.

- Use the appropriate prefix, such as:
  - `feat:` for new features or implementations
  - `fix:` for bug fixes
  - `docs:` for documentation updates (including docstrings)
  - `test:` for adding or updating tests
  - `chore:` for maintenance, dependency updates, or repo structure changes

Example:
```
feat: add RMSprop optimizer with full documentation
```
Keeping a consistent commit style helps maintainers review PRs efficiently and improves changelog generation.


**Thank you for helping improve this project! 🚀**