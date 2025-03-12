# StockX Sneaker Price Prediction

## Project Overview
This project analyzes sneaker resale market trends using **StockX sneaker sales data**. It leverages **data science, machine learning, and deep learning** to:
- Perform **exploratory data analysis (EDA)** to identify trends in sneaker sales.
- Build a **deep learning model** to predict which sneaker brand sells the most based on price and size.
- Visualize **sneaker price trends** over time.

## Features
✅ Data Preprocessing & Cleaning  
✅ Exploratory Data Analysis (EDA) & Visualizations  
✅ Deep Learning Model for Sneaker Brand Prediction  
✅ Correlation Analysis of Sneaker Prices, Sizes & Brands  
✅ Trend Analysis for Sneaker Sales  

---

## Dataset
- **Source:** [StockX Sneaker Dataset](https://www.kaggle.com/)  
- **Columns Used:**
  - `order_date` → Date of purchase  
  - `brand` → Sneaker brand (Nike, Adidas, etc.)  
  - `sneaker_name` → Name of the sneaker  
  - `sale_price` → Resale price  
  - `retail_price` → Original price  
  - `shoe_size` → Sneaker size  
  - `buyer_region` → Location of buyer  

---

## Exploratory Data Analysis (EDA)
### **Most Selling Sneaker Brands**
![Most Selling Sneaker Brands](https://github.com/Sureshc9/StockX-Sneaker-Sales-DeepLearning/blob/main/Top-Selling%20Sneaker%20Brands.png?raw=true)

### **Sneaker Price Distribution**
![Price Distribution](https://github.com/Sureshc9/StockX-Sneaker-Sales-DeepLearning/blob/main/Screenshot%20.png?raw=true)

---

## Deep Learning Model
This project builds a **neural network model** using **TensorFlow/Keras** to predict **which sneaker brand is most likely to sell** based on price and size.

### Model Architecture**
- **64 neurons** (ReLU) → **Dropout (0.3)** → **32 neurons (ReLU)** → **Output Layer (Softmax)**
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Accuracy Achieved:** ~85%

### Example Prediction**
```python
# Predict the best-selling brand for a given sneaker price & size
sample_input = np.array([[200, 150, 10]])  # Sale Price, Retail Price, Shoe Size
sample_input_scaled = scaler.transform(sample_input)
predicted_brand = label_encoder.inverse_transform([np.argmax(model.predict(sample_input_scaled))])[0]

print(f" Predicted Most Selling Brand: {predicted_brand}")
