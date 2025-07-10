# BreathGardien

**BreathGardien** is a Python desktop application designed for **automated lung tumor segmentation from 3D CT scans**. Using deep learning models built with **PyTorch** and **MONAI**, it allows clinicians and researchers to **visualize CT data in both 2D and 3D**, predict segmentation masks, and overlay results—all in an intuitive PyQt5-based GUI.

---

## Project Scope: Lung Tumor Segmentation

This application is focused on **segmenting lung tumors** in volumetric CT data. It leverages a trained neural network to predict tumor masks and provides both visual and interactive tools to inspect the results for clinical and research validation.

---

## Features

- Load 3D CT volumes stored in `.npy` format  
- Predict lung tumor segmentation masks using **PyTorch + MONAI**  
- View 2D axial slices (original, predicted, and overlay)  
- Render 3D CT volumes and predicted masks using VTK  
- Switch between 2D/3D views and control visualization  
- Dark-mode, responsive UI for comfortable review  

---

## Project Structure

```

breathgardien/
    ├── GUI.py                  # Main GUI application
    ├── requirements.txt        # Dependency list
    ├── .gitignore              # Git ignore rules
    └── src/
        └── backend/
            ├── find_peak.py
            ├── load_tensor.py
            ├── predect.py
            ├── Show2D.py
            ├── Show3DData.py
            ├── Superpose2D.py
            └── model_params/
                ├── UNETR_part0_530.pth
                ├── UNETR_part1_817.pth
                └── UNETR_part2_150.pth

````

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/breathgardien.git
cd breathgardien
````

### 2. Create and Activate a Virtual Environment (Required)

Create the environment:

```bash
python -m venv venv
```

Activate the environment:

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python GUI.py
```

---

## Data Format

The application expects CT volumes in `.npy` format with the following naming convention:

```
data-<patient_index>-<slice_index>.npy
```

### Example:

```
data-0-0.npy  
data-0-1.npy  
...
data-1-0.npy
...
```

These files are combined into a full 3D NumPy tensor of shape `[depth, height, width]`.

---

## Segmentation Process

The `predict()` function in `src/backend/predect.py` runs the lung tumor segmentation model.

The model used is **UNETR**, a hybrid architecture that combines the strengths of U-Net and Transformer networks. It is built using MONAI and PyTorch and returns a binary mask.

Predictions can be visualized in 2D side-by-side with the original data, or in full 3D using VTK overlays.

You can check the code used to train the model and the dataset directly on Kaggle:

* 📘 **Training Code**: [UNETR Training Notebook](https://www.kaggle.com/code/raedbenromdhane/unter-raed4)
* 📂 **Dataset**: [Lung Cancer Dataset](https://www.kaggle.com/datasets/raedbenromdhane/lung-cancer-data-set)

You can also check the associated paper in this repository: **`Breath Gardien.pdf`**.

---

## User Interface Guide

🎥 **Demo Video**: [Watch how the app works](https://drive.google.com/file/d/1L1hx5bACR7EwJu9556mJs7gTceno7tRa/view?usp=drive_link)

| Button      | Description                                       |
| ----------- | ------------------------------------------------- |
| Open Folder | Load a folder of `.npy` CT files                  |
| Show 2D     | View axial slices of the CT volume                |
| Predict     | Run tumor segmentation and show mask              |
| Overlay     | Superimpose predicted mask on CT image (2D or 3D) |
| Show 3D     | Visualize CT volume and/or masks in 3D            |

---

## .gitignore

Make sure your `.gitignore` file includes the following to avoid committing local environments and caches:

```
/venv/
/__pycache__/
```

---

## Credits

Developed by **Raed Ben Romdhane** and **Asma Mhatli**.

Model powered by **PyTorch** and **MONAI**.

GUI built with **PyQt5**.

---
