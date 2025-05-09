<p align="center">
  <img src="https://hitechnectar.com/wp-content/uploads/2022/06/Here-are-the-Proven-Real-world-Applications-of-Artificial-Neural-Network-jpg-webp.webp" alt="Neural Network Logo" width="200"/>
</p>

<h1 align="center">London Property Value Neural Network 🏙️🧠</h1>

<p align="center">
  A machine learning project that predicts property values in London using a custom-built neural network.<br/>
  <a href="https://github.com/adilhafiz123/PropValueML">🔗 View on GitHub</a>
</p>

---

## 🚀 Overview

This project is a neural network-based regression model built to predict **property prices in London**. It takes structured property data and uses deep learning techniques to generate estimated market values.

Ideal for:
- Real estate analysts
- Property investors
- Data science enthusiasts

---

## 🧠 Key Features

- Custom feed-forward neural network using PyTorch
- Cleaned and feature-engineered London housing dataset
- Train/test split and RMSE performance tracking
- Model checkpoint saving and loading
- Interactive prediction interface (coming soon)

---

## 📊 Sample Predictions

| Feature                         | Value                     |
|---------------------------------|---------------------------|
| Bedrooms                        | 3                         |
| Location (Borough)              | Camden                    |
| Property Type                   | Flat                      |
| Year Built                      | 2001                      |
| **Predicted Price (£)**         | **£612,000**              |

---

## 🛠️ Tech Stack

- 🐍 Python 3.x  
- 🔥 PyTorch  
- 📊 Pandas, NumPy, Matplotlib  
- 🧪 Scikit-learn  
- 💾 Pickle for model saving  

---

## 🧪 How to Run

```bash
# Clone the repo
git clone https://github.com/adilhafiz123/PropValueML
cd PropValueML

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training script
python train_model.py
```
## 📈 Model Architecture
Input layer: Normalised property features

2x Hidden layers: 64 neurons each with ReLU activation

Output layer: Single predicted price value (linear)

css
Copy
Edit
[Input] → [Dense 64, ReLU] → [Dense 64, ReLU] → [Output]
## 🔮 Future Work
Incorporate geospatial coordinates

Add interactive web interface using Streamlit

Improve accuracy with ensemble models

## 🤝 Contributing
Contributions are welcome! If you have suggestions for new features or performance improvements, feel free to fork the repo and submit a pull request.

## 📄 License
This project is licensed under the MIT License.
