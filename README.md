<div align="center">

# AdversariaLeak: External Information Leakage Attack Using Adversarial Samples on Face Recognition Systems (ECCV'2024)

[![Python](https://img.shields.io/badge/Python-3.9.15-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3915/)
[![PyTorch](https://img.shields.io/badge/PyTorch-red?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Sklearn](https://img.shields.io/badge/Sklearn-orange?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-blue?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![ART](https://img.shields.io/badge/ART-lightgrey?style=flat&logo=python&logoColor=white)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)




</div>

---

## 📄 Description

This repository contains the code for the paper **"AdversariaLeak: External Information Leakage Attack Using Adversarial Samples on Face Recognition Systems"** published in ECCV 2024 conference.
AdversariaLeak is a novel and practical attack targeting face recognition systems to infer sensitive information about the training data by utilizing adversarial samples.

---

## 🔑 Key Features
- **Attack Type:** External Information Leakage (EIL) attack using adversarial samples.
- **Objective:** Infers statistical properties of the training set in face verification models.
- **Datasets:** Tested on MAAD-Face and CelebA datasets.
- **Approach:** Utilizes adversarial samples to determine properties that characterize the training data.

---

## 🔗 Resources

Here are some useful links related to this project:

- [**📄 Paper**](https://link.springer.com/chapter/10.1007/978-3-031-73226-3_17): Access the published paper.
- [**📑 Supplementary Material**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09603-supp.pdf): View the supplementary materials (PDF).
- [**🌐 Project Page**](https://eccv.ecva.net/virtual/2024/poster/324): Find additional information and project details on the ECCV 2024 virtual platform.

---

## 📁 Project Structure

- `Attacks/` - Contains scripts and methods to generate adversarial samples.
- `FR_System/` - Holds modules for face recognition and verification.
- `Run_scripts/` - Scripts for running and evaluating attacks.
- `Demo.py` - Demonstration of the attack pipeline.
- `guidelines` - Instructions and guidelines for setting up and running the project.
- `requirements.txt` - Lists dependencies required for this project.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.9.15+ installed. Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Adjusting the code
Read the guindlines and adjust the code accordingly.


### Running the Demo
To see the attack pipeline in action, run:

```bash
python Demo.py
```

---

## ✍️ Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{katzav2025adversarialeak,
  title={AdversariaLeak: External Information Leakage Attack Using Adversarial Samples on Face Recognition Systems},
  author={Katzav, Roye and Giloni, Amit and Grolman, Edita and Saito, Hiroo and Shibata, Tomoyuki and Omino, Tsukasa and Komatsu, Misaki and Hanatani, Yoshikazu and Elovici, Yuval and Shabtai, Asaf},
  booktitle={European Conference on Computer Vision},
  pages={288--303},
  year={2025},
  organization={Springer}
}
