# 🧠 MNIST Digit Generator using Conditional GAN (CGAN)

This project demonstrates a Conditional Generative Adversarial Network (CGAN) trained on the MNIST dataset to generate realistic-looking handwritten digits from 0 to 9. The model is deployed as an interactive web app using Streamlit.

## 🌐 Live Demo

👉 [Click here to try the app](https://your-streamlit-app-link)  
Generate MNIST-style handwritten digits just by selecting a number!

---

## 🚀 Features

- Trained a CGAN model from scratch using PyTorch
- Generates 28x28 grayscale digit images conditioned on digit labels
- Interactive web interface using Streamlit
- Deployed seamlessly with Streamlit Cloud

---

## 🧰 Tech Stack

- **Model**: Conditional GAN (PyTorch)
- **Dataset**: MNIST (handwritten digits)
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Cloud
- **Visualization**: Matplotlib, Torchvision

---

## 📁 Project Structure

```
.
├── models/
│   └── cgan_generator.pth        # Trained generator model
├── streamlit_digit_generator.py  # Streamlit app script
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

---

## ⚙️ Getting Started

### 🔧 Prerequisites
Make sure you have Python 3.7+ installed. Then:

```bash
git clone https://github.com/your-username/digit-generator-app.git
cd digit-generator-app
pip install -r requirements.txt
```

### ▶️ Run Locally

```bash
streamlit run streamlit_digit_generator.py
```

---

## 🖼️ Sample Output

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="MNIST Sample" width="400"/>
</p>

---

## 🙌 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)

---

## 📬 Feedback

Feel free to share feedback or suggestions by opening an [issue](https://github.com/your-username/digit-generator-app/issues) or messaging me directly!

---

## ⭐ Like this project?

Give it a star ⭐ on GitHub and try the app live!
