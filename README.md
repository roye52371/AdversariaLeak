<div align="left">

### AdversariaLeak: External Information Leakage Attack Using Adversarial Samples on Face Recognition Systems

**Description:**
This repository contains the code for the paper **"AdversariaLeak: External Information Leakage Attack Using Adversarial Samples on Face Recognition Systems"**. AdversariaLeak is a novel and practical attack targeting face recognition systems to infer sensitive information about the training data by utilizing adversarial samples.

**Key Features:**
- **Attack Type:** External Information Leakage (EIL) attack using adversarial samples.
- **Objective:** Infers statistical properties of the training set in face verification models.
- **Datasets:** Tested on MAAD-Face and CelebA datasets.
- **Approach:** Utilizes adversarial samples to determine properties that characterize the training data.

**Project Structure:**
- `Attacks/` - Contains scripts and methods to generate adversarial samples.
- `FR_System/` - Holds modules for face recognition and verification.
- `Run_scripts/` - Scripts for running and evaluating attacks.
- `Demo.py` - Demonstration of the attack pipeline.
- `guidlines` - Instructions and guidelines for setting up and running the project.
- `requirements.txt` - Lists dependencies required for this project.

**Instructions:**
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `Demo.py` to execute a sample attack on a pre-defined model.

**Authors:**
- Roye Katzav, Amit Giloni, Edita Grolman, Hiroo Saito, Tomoyuki Shibata, Tsukasa Omino, Misaki Komatsu, Yoshikazu Hanatani, Yuval Elovici, and Asaf Shabtai.

**Link to the paper:**
[https://link.springer.com/chapter/10.1007/978-3-031-73226-3_17](https://link.springer.com/chapter/10.1007/978-3-031-73226-3_17)

**Citation:**
If you use this code in your research, please cite the following paper:

@inproceedings{katzav2025adversarialeak,
  title={AdversariaLeak: External Information Leakage Attack Using Adversarial Samples on Face Recognition Systems},
  author={Katzav, Roye and Giloni, Amit and Grolman, Edita and Saito, Hiroo and Shibata, Tomoyuki and Omino, Tsukasa and Komatsu, Misaki and Hanatani, Yoshikazu and Elovici, Yuval and Shabtai, Asaf},
  booktitle={European Conference on Computer Vision},
  pages={288--303},
  year={2025},
  organization={Springer}
}
