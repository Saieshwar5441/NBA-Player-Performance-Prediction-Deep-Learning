# NBA-Player-Performance-Prediction-Deep-Learning

This project presents a deep learning-based approach for predicting NBA player scoring performance—specifically **Points Per Game (PPG)**—using state-of-the-art neural network architectures. The goal is to explore and compare the effectiveness of various models like **LSTM**, **BiLSTM**, and **Transformer** in accurately forecasting player performance.

---

## Project Overview

- **Course:** CS 583 – Deep Learning  
- **Timeline:** Fall 2024  
- **Authors:** Sai Eshwar Gaddipati, Vedanth Sirimalla  
- **Dataset:** [NBA Player Stats (2023–2024)](https://www.kaggle.com/datasets/bryanchungweather/nba-player-stats-dataset-for-the-2023-2024)  
- **Target:** Predict Points Per Game (PPG)

---

## Dataset Details

- **Source:** Kaggle & Basketball Reference  
- **Shape after cleaning:** 1,989 rows × 33 features  
- **Key features:** FG%, 3P%, MP, TRB, AST, STL, BLK, TOV, PF, PTS, etc.  
- **New Features:**  
  - `PointsPerMinute = PTS / MP`  
  - `TotalRebounds = ORB + DRB`  
  - `ShootingEfficiency = avg(FG%, 3P%, FT%)`  
  - `ScoringCategory = Low / Average / High Scorer` based on PTS

---

## Data Preprocessing

- Filled missing values (e.g., FG%, 3P%) with column means
- Dropped duplicates and outliers using IQR method
- Normalized features using StandardScaler
- Split dataset into train, validation, and test sets

---

## Exploratory Data Analysis

- Distribution of players across NBA teams  
- Relationship between age and scoring performance  
- Role-wise averages (Guards, Forwards, Centers)  
- Correlation matrix of key stats  
- Trend analysis: Playing Time vs Points  
- Statistical validation using ANOVA and correlation tests  

---

## Models Explored

We explored six deep learning models and focused on the top three for final evaluation:

| Model       | Description                                                                |
|-------------|----------------------------------------------------------------------------|
|     LSTM    | Captures sequential dependencies; memory-driven architecture               |
|    BiLSTM   | Considers both past and future sequences for better context                |
| Transformer | Leverages self-attention for long-range dependency modeling                |

---

## Model Evaluation

| Model        | MSE       | RMSE     | R² Score |
|--------------|-----------|----------|----------|
|     LSTM     | 3.528     | 1.878    | 0.8927   |
|    BiLSTM    | 3.278     | 1.811    | 0.9003   |
| Transformer  | 5.138     | 2.266    | 0.8438   |

> **BiLSTM outperformed other models** with the highest R² and lowest RMSE/MSE.

---

## Case Study: Dante Exum

| Model       | Predicted PPG |
|-------------|----------------|
| Actual PPG  | 4.40           |
| BiLSTM      | 4.38           |
| LSTM        | 4.17           |
| Transformer | 3.47           |

> BiLSTM predicted Dante Exum’s scoring output with high precision.

---

## Limitations & Future Work

- Incorporate more advanced statistics (e.g., PER, BPM, Usage %)
- Extend predictions beyond PPG (e.g., rebounds, assists)
- Explore model ensembles (e.g., LSTM + Transformer)
- Apply transfer learning to other sports analytics domains

---

## References

- [Kaggle NBA Dataset](https://www.kaggle.com/datasets/bryanchungweather/nba-player-stats-dataset-for-the-2023-2024)
- [Basketball Reference](https://www.basketball-reference.com/leagues/NBA_2024_per_game.html)
- [Deep Learning Resources](https://arxiv.org/abs/2111.09695), [Simplilearn](https://www.simplilearn.com/tutorials/deep-learning-tutorial)

---

## Technologies Used

- Python, Pandas, NumPy  
- Scikit-learn for preprocessing  
- TensorFlow / Keras for model building  
- Matplotlib & Seaborn for visualization  
