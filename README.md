# 🧠 Startup Success Predictor (My First ML Project!)

Hi! I'm Priyanshu 👋  
This is a beginner-friendly machine learning project where I tried to predict whether a startup will succeed or fail using some basic data science skills.

 I built this from scratch using Python, and I really enjoyed the process. This README is just me trying to explain everything I did — so it's not fancy, just honest.

---

## 💡 What's the Idea?

I used a dataset of real startups (from [Kaggle](https://kaggle.com)) and tried to train a model to classify them into:

- ✅ Successful (if they got acquired or went IPO)
- ❌ Unsuccessful (if they shut down)

---

## 🧰 Tools I Used

- Python 3
- VS Code
- pandas, numpy
- scikit-learn
- (I might use Streamlit later)

---

## ✅ What I Did (Step-by-Step)

1. **Loaded the CSV file** and looked at the data
2. **Removed columns** that I felt were not useful (like ID, city, zip, etc.)
3. **Handled missing values** by filling them with median
4. **Converted the target column** ("status") into 1s and 0s (success/fail)
5. **One-hot encoded** the `state` and `category` columns
6. **Split the data** into training and testing sets
7. **Trained two models**:
   - Logistic Regression
   - Random Forest
8. **Compared the results**

---

## 📈 Results (Plain and Simple)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 80% ✅ |
| Random Forest | 78% ✅ |

So yeah... both models worked surprisingly well!

---

## 📝 What I Learned

- How to load and clean messy data
- How to train simple ML models
- That one-hot encoding can mess things up if you're not careful 😅
- Logistic Regression is a good baseline
- Random Forest is powerful but can overfit

---

## 🙋‍♂️ About Me

I'm Priyanshu Jaju, learning data science on my own.  
I'm putting in the work and slowly improving.  
Feel free to check out my [GitHub](https://github.com/JXP-ME) or connect with me on INSTA @jxp.mess

---

## 📌 Disclaimer

This is a learning project. The model is not production-ready.  
But it’s something I’m proud of as a beginner — and I hope it helps others who are just getting started too.

