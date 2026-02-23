# L2CP: Test-Time Training with Local Contrast-Preserving Copy-Paste for Domain Generalization in Retinal Vessel Segmentation

This repository contains the official implementation of **L2CP**, a **test-time training (TTT)** strategy for **domain generalization** in **retinal vessel segmentation**.

> **Paper title:** Test-Time Training with Local Contrast-Preserving Copy-Pasted Image for Domain Generalization in Retinal Vessel Segmentation

---

## 🔥 Motivation

Deep learning models for retinal vessel segmentation often achieve strong performance when test images follow the same distribution as training images.  
However, in real-world clinical scenarios (e.g., images with lesions, different devices, lighting), **domain shift** causes a significant performance drop.

Existing TTT methods (e.g., TENT, CoTTA) are general-purpose and **not tailored to the thin, low-contrast vessel structures**, resulting in suboptimal adaptation on cross-domain retinal vessel segmentation.

---

## ✨ Key Idea

We generate **synthetic target-style images** during testing as a bridge between source and target domains, and then fine-tune the model at test time.

### Step 1: Build a Target-Style Vessel-Free Background
Leveraging the fact that retinal vessels are thin and elongated, we apply **morphological closing** on the test image to **remove vessels**, producing a **vessel-free image that preserves the target domain style**.

### Step 2: Local Contrast-Preserving Copy-Paste (L2CP)
Instead of directly pasting vessel grayscale values from source images (which causes severe intensity mismatch across domains), we copy-paste the **local contrast map** of vessels onto the target-style background.

This **mitigates foreground/background grayscale distribution mismatch**, making the synthetic images more realistic and closer to the target domain.

### Step 3: Test-Time Training with Synthetic Images
We fine-tune the segmentation model using these synthetic images during testing, improving robustness to out-of-distribution (OOD) samples.

---

## 🧠 Method Overview

**Input:**
- Source-domain images with vessels (during training or from a source pool)
- Target test image (unlabeled)

**Output:**
- Synthetic target-style image with realistic vessels
- Updated model adapted to the target test domain

---

## ✅ Main Contributions

1. **L2CP copy-paste strategy:**  
   A simple yet effective method that pastes **local contrast** rather than grayscale, alleviating large domain gaps in both foreground and background distributions.

2. **Synthetic bridge for TTT:**  
   We use L2CP-generated images to fine-tune models at test time, consistently improving domain generalization across multiple datasets and networks.

---

## 📊 Experiments

We conduct extensive cross-domain retinal vessel segmentation experiments using multiple classical networks and three public datasets:

- **DRIVE**  
- **STARE**
- **CHASE_DB1**

Our L2CP-based test-time training consistently improves performance under domain shift.

---

## 📁 Datasets

Please download datasets from their official sources:

- DRIVE: https://drive.grand-challenge.org/
- STARE: http://cecas.clemson.edu/~ahoover/stare/
- CHASE_DB1: https://blogs.kingston.ac.uk/retinal/chasedb1/

