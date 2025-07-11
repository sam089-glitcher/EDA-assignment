#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


df = pd.read_csv("netflix_customer_churn.csv")


# In[32]:


print("🔹 First 5 rows:")
display(df.head()) 


# In[33]:


missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_summary = (pd
                   .DataFrame({"Missing": missing, "Percent": missing_pct})
                   .sort_values("Missing", ascending=False))

print("\n🔹 Missing / Null Value Summary:")
display(missing_summary)


# In[34]:


cat_cols = ["gender",
            "subscription_type",
            "region",
            "device",
            "payment_method",
            "favorite_genre"]

print("\n🔹 Unique value counts:")
for col in cat_cols:
    uniques = df[col].unique()
    print(f"{col}: {len(uniques)} unique values → {uniques[:10]}{' ...' if len(uniques) > 10 else ''}")


# In[35]:


num_cols = ["age", "watch_hours", "monthly_fee"]

for col in num_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna(), bins=30)          # histogram
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# In[36]:


plt.figure(figsize=(6, 4))
churn_counts = df["churned"].value_counts().sort_index()
plt.bar(churn_counts.index.astype(str), churn_counts.values)
plt.title("Distribution of churned (0 = No, 1 = Yes)")
plt.xlabel("churned")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# In[37]:


cat_cols = [
    "subscription_type",
    "gender",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
]

for col in cat_cols:
    plt.figure(figsize=(10, 4))
    counts = df[col].value_counts()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# In[38]:


for col in ["subscription_type", "region", "device"]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=df, x=col, y="watch_hours", ax=axes[0], estimator=np.mean, hue=col, palette="Blues_d", legend=False)
    axes[0].set_title(f"Avg Watch Hours by {col}")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=df, x=col, y="monthly_fee", ax=axes[1], estimator=np.mean, hue=col, palette="Greens_d", legend=False)
    axes[1].set_title(f"Avg Monthly Fee by {col}")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# In[39]:


plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="favorite_genre",
    y="avg_watch_time_per_day",
    estimator=np.mean,
    hue="favorite_genre",
    palette="pastel",
    legend=False
)
plt.title("Average Watch Time Per Day by Favorite Genre")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[40]:


# Churn rate by categorical features (fixed)
for col in ["gender", "region", "subscription_type", "payment_method"]:
    plt.figure(figsize=(8, 4))
    churn = df.groupby(col)["churned"].mean().sort_values(ascending=False)
    sns.barplot(x=churn.index, y=churn.values, hue=churn.index, palette="Set2", legend=False)
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[41]:


corr = df[num_cols].corr()

# 4️⃣ Heat‑map
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f",
            cmap="coolwarm", center=0)
plt.title("Correlation Heatmap – Numerical Features")
plt.tight_layout()
plt.show()


# In[42]:


corr = df[["age", "watch_hours", "last_login_days", "monthly_fee", 
           "number_of_profiles", "avg_watch_time_per_day", "churned"]].corr()
churn_corr = corr["churned"].drop("churned").sort_values(key=abs, ascending=False)
print("🔍 Features most correlated with churn:\n")
print(churn_corr)


# In[43]:


def header(txt):
    print("\n" + txt)
    print("-" * len(txt))


# In[44]:


header("Churn rate by watch_hours quartile")
df["watch_q"] = pd.qcut(df["watch_hours"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
print(df.groupby("watch_q", observed=True)["churned"].mean().mul(100).round(1), "%")


# In[45]:


header("Churn rate by subscription_type")
print(df.groupby("subscription_type")["churned"].mean().mul(100).round(1), "%")


# In[46]:


header("Average watch_hours by region")
print(df.groupby("region")["watch_hours"].mean().round(2))


# In[47]:


header("Churn rate by monthly_fee bucket")
df["fee_bucket"] = pd.cut(df["monthly_fee"], bins=[0, 9, 14, 18, np.inf],
                          labels=["<= $9", "$9‑14", "$14‑18", "$18+"])
print(df.groupby("fee_bucket", observed=True)["churned"].mean().mul(100).round(1), "%")


# In[48]:


header("Average watch_hours & churn by device")
print(
    df.groupby("device").agg(
        avg_watch_hours=("watch_hours", "mean"),
        churn_rate=("churned", "mean")
    ).assign(churn_rate=lambda x: (x["churn_rate"] * 100).round(1))
        .round({"avg_watch_hours": 2})
)


# In[49]:


header("Average avg_watch_time_per_day by favorite_genre")
print(df.groupby("favorite_genre")["avg_watch_time_per_day"].mean().round(2).sort_values(ascending=False))


# In[50]:


df.drop(columns=["watch_q", "fee_bucket"], inplace=True)


# In[51]:


print("Customers in the lowest quartile of watch hours have the highest churn rate, while those in the top quartile show the lowest churn.")
print("✅ Watch hours are strongly negatively correlated with churn (≈ –0.48).")


# In[52]:


print("Basic plan users show a significantly higher churn rate compared to Premium plan users, indicating that Premium users are more satisfied or engaged.")
print("✅ Churn: Basic ≈ 24.6%, Premium ≈ 12.1%")


# In[58]:


print("Among all regions, the West shows the highest average watch hours, indicating strong user engagement.")
print("✅ Target this region for upsell opportunities.")


# In[59]:


print("Users with a higher number of days since last login are more likely to churn.")
print("✅ Last login days has a strong positive correlation with churn (≈ +0.47).")


# In[60]:


print("Users who primarily watch on TVs have higher average watch hours and lower churn rates compared to mobile users.")
print("✅ TV users may represent family/shared accounts.")


# In[61]:


print("Churn is slightly lower among users with higher monthly fees, possibly due to added features or multiple profiles being used.")
print("✅ $18+ plans show reduced churn compared to lower-tier plans.")


# In[62]:


print("Genres like Drama, Thriller, and Action have the highest average daily watch times.")
print("✅ These genres keep users hooked and may be valuable in retention strategies.")


# In[ ]:




