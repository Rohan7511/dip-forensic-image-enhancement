# 🔍 Digital Image Forensics

**Enhancing CCTV Footage for Criminal Investigation using Image Processing Techniques**

---

## 📌 About

A Digital Image Processing case study that demonstrates how traditional image enhancement techniques can improve the quality of surveillance footage for forensic investigation. No Machine Learning or Deep Learning is used — only classical DIP methods.

## 👥 Team

- SH Nihil Mukkesh
- Nikhil Jones A
- Rohan Gubba
- Siddartha

## 🛠️ Techniques Used

| Technique | Purpose |
|---|---|
| Histogram Equalization | Contrast enhancement |
| Average / Gaussian / Median Filters | Noise reduction |
| Laplacian Sharpening | Edge enhancement |
| Unsharp Masking | Detail recovery |
| Bit-Plane Slicing | Structural analysis & steganography detection |
| Sobel & Canny Edge Detection | Object boundary extraction |

## 📂 Project Structure

```
├── 4.png                          # Input CCTV image
├── Digital_Image_Forensics.ipynb  # Main notebook
├── outputs/                       # Generated output images
│   ├── grayscale.png
│   ├── histogram_equalized.png
│   ├── gaussian.png
│   ├── median.png
│   ├── laplacian.png
│   ├── unsharp.png
│   ├── bitplane7.png
│   └── canny.png
└── README.md
```

## 📦 Requirements

```
opencv-python
numpy
matplotlib
```

## ▶️ How to Run

1. Clone the repo
2. Make sure `4.png` is in the root directory
3. Open `Digital_Image_Forensics.ipynb` in Jupyter Notebook / VS Code
4. Run all cells top to bottom

## 📊 Dataset

The input image is sourced from the [Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset) on Kaggle.

## 📄 License

This project is for academic purposes only.
