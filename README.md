# Paddy Leaf Disease Classification and Identification using Machine Learning

## Project Overview
This project focuses on using Machine Learning to classify and identify paddy leaf diseases, helping farmers detect diseases early, optimize crop management, and improve yield prediction. The model utilizes Python with libraries such as `scikit-learn` and `OpenCV` to preprocess images, extract features, and train machine learning models.

## Authors
- **Nathaly Cobo Piza**
- **Sai Chaitanya Yerramsetty**

## Technologies Used
- **Programming Language**: Python
- **Libraries & Frameworks**:
  - `scikit-learn` (Machine Learning)
  - `OpenCV` (Image Processing)
  - `matplotlib` (Data Visualization)
  - `pandas`, `numpy` (Data Handling)

## Project Workflow
1. **Data Collection**:
   - Paddy leaf images were gathered from multiple sources to create a dataset.
   - Images were preprocessed using OpenCV for noise reduction and enhancement.

2. **Feature Extraction**:
   - Local Binary Pattern (LBP) features were used for better classification accuracy.
   - Histogram and Correlation Heatmaps were used to analyze feature relevance.

3. **Machine Learning Models**:
   - Several ML models were trained and tested:
     - Logistic Regression (Accuracy: 0.528)
     - Support Vector Machine (SVM) (Accuracy: 0.724)
     - K-Nearest Neighbors (KNN) (Accuracy: 0.731)
     - Decision Tree (Accuracy: 0.736)
     - Random Forest (Best Model: Accuracy 0.819)
     - AdaBoost & Gradient Boost (Accuracy: 0.752, 0.795)

4. **Model Evaluation**:
   - Accuracy scores and performance metrics were used to assess model effectiveness.
   - The Random Forest model demonstrated the highest classification accuracy.

## Importance in Agriculture
- Early disease detection prevents the spread of infections.
- Farmers can apply targeted treatments, reducing waste and increasing efficiency.
- Machine Learning improves agricultural productivity and sustainability.

## Optimizations & Future Work
- Expanding the dataset with geographically diverse images.
- Integrating environmental data (weather, humidity) to improve predictions.
- Implementing deep learning models such as Convolutional Neural Networks (CNNs) for higher accuracy.

## Conclusion
This project highlights how Machine Learning can revolutionize agricultural disease detection. With an accuracy of 81.9%, the Random Forest model proved most effective. Future improvements can further enhance disease classification and support sustainable farming practices.

## References
- Kaggle Rice Disease Dataset: [Link](https://www.kaggle.com/datasets/yenugularajeev/rice-disease?resource=download)
- Research papers on leaf disease detection and classification techniques.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repo-link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
4. Train the model:
   ```bash
   python train.py
   ```
5. Test the model:
   ```bash
   python test.py
   ```
6. View results and accuracy metrics in the output directory.

## License
This project is open-source and available for academic and research purposes.
