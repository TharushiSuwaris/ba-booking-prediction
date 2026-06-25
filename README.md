# British Airways - Customer Booking Prediction

A machine learning project built as part of the **British Airways Data Science Virtual Experience Programme** (Forage). The goal is to predict whether a customer will complete a flight booking, enabling BA to proactively target high-intent customers before they book elsewhere.

---

## Background

Airlines can no longer rely on customers walking up to the gate to buy. Today's customer researches, compares, and decides long before they travel. This project uses historical booking data to build a predictive model that identifies customers likely to complete a booking - turning a reactive process into a proactive one.

---

## Project Structure

```
├── customer_booking.csv       # Raw dataset (50,000 records)
├── eda_and_prep.ipynb         # Exploratory data analysis + data preparation
├── modelling.ipynb            # Model training and evaluation
generator
├── processed_train.csv        # Processed training set (output of EDA notebook)
├── processed_test.csv         # Processed test set (output of EDA notebook)
```

---

## Dataset

- **50,000** customer booking records
- **14 features** covering booking behaviour, trip details, and ancillary preferences
- **Target**: `booking_complete` (binary — 1 if the customer completed the booking)
- **Class imbalance**: ~15% positive class

| Feature | Description |
|---|---|
| `num_passengers` | Number of passengers travelling |
| `sales_channel` | Internet or Mobile |
| `trip_type` | Round trip, One Way, or Circle Trip |
| `purchase_lead` | Days between booking date and travel date |
| `length_of_stay` | Days spent at destination |
| `flight_hour` | Hour of departure |
| `flight_day` | Day of week of departure |
| `route` | Origin–destination airport pair |
| `booking_origin` | Country the booking was made from |
| `wants_extra_baggage` | Whether the customer selected extra baggage |
| `wants_preferred_seat` | Whether the customer selected a preferred seat |
| `wants_in_flight_meals` | Whether the customer selected in-flight meals |
| `flight_duration` | Total flight duration (hours) |
| `booking_complete` | **Target** - did the customer complete the booking? |

---

## Methodology

### 1. Exploratory Data Analysis
- Class imbalance check (15% positive rate)
- Numerical feature distributions and outlier detection
- KDE plots per feature split by booking outcome
- Categorical completion rate analysis
- Pearson correlation heatmap
- **Mutual Information scores** for feature selection (captures non-linear relationships)
- Point-biserial correlation with statistical significance testing

### 2. Feature Engineering
- Extracted `route_origin` and `route_dest` from the 6-character route string
- Created `total_ancillaries` — sum of all three ancillary flags (funnel engagement signal)
- Created `is_weekend` and `is_solo` binary flags
- Log-transformed `purchase_lead` and `length_of_stay` to reduce skew

### 3. Data Preparation
- **Train/test split first** (80/20 stratified) — all transforms fitted on training data only to prevent data leakage
- Log transforms applied post-split (deterministic, no leakage risk)
- One-hot encoding for `sales_channel` and `trip_type` (nominal, low cardinality)
- Frequency encoding for `route` and `booking_origin` (high cardinality) — map fitted on train set only

### 4. Modelling
Two models trained and compared:

| Model | Imbalance Handling |
|---|---|
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` (neg/pos ratio) |

### 5. Evaluation
- **5-fold stratified cross-validation** (stratified to preserve 15% ratio per fold)
- Metrics: ROC-AUC, F1, Precision, Recall, Accuracy
- Confusion matrix, ROC curve, Precision-Recall curve
- Train vs CV vs Test comparison for overfitting diagnosis
- Feature importance ranking (Gini for RF, Gain for XGBoost)

> **Note**: Accuracy is not the primary metric here due to class imbalance - a model predicting "no booking" every time would score 85% accuracy. ROC-AUC and F1 are the meaningful measures.

---

## Key Findings

- **Purchase lead time** is the strongest predictor - how far in advance a customer books reflects their intent
- **Ancillary selections** (extra baggage, meals, preferred seat) signal deeper funnel engagement and higher completion likelihood
- **Route popularity** and **booking origin** capture regional behavioural differences

---


## Acknowledgements

Dataset and task provided by **British Airways** via the [Forage Virtual Experience Programme](https://www.theforage.com).
