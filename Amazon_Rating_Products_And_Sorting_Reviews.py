#######################
# Yamac TAN - Data Science Bootcamp - Week 5 - Project 2
#######################

# %%
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("Odevler/HAFTA_05/PROJE_2/amazon_review.csv")
df = df_.copy()
df.isnull().sum()
df.describe().T
df.head()

# %%
###############################################
# Görev 1 - Adım 1
###############################################

df["overall"].mean()  # Dataset analizinde, ürünün ortalama puanı yaklaşık 4.58'dir.

# %%
###############################################
# Görev 1 - Adım 2
###############################################

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

current_date = df["reviewTime"].max()

df["days"] = (current_date - df["reviewTime"]).dt.days

df["days"].quantile([.25, .5, .75])
# 0.25: 280, 0.5: 430, 0.75: 600

df.loc[df["days"] <= 280, "overall"].mean() * (40 / 100) + \
df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean() * (30 / 100) + \
df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean() * (20 / 100) + \
df.loc[(df["days"] > 600), "overall"].mean() * (10 / 100)


# 4.62


# %%
###############################################
# Görev 1 - Adım 3
###############################################

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= df["day_diff"].quantile(q=0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > df["day_diff"].quantile(q=0.25)) &
                         (dataframe["days"] <= df["day_diff"].quantile(q=0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > df["day_diff"].quantile(q=0.50)) &
                         (dataframe["days"] <= df["day_diff"].quantile(q=0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > df["day_diff"].quantile(q=0.75)), "overall"].mean() * w4 / 100


time_based_weighted_average(df)  # 4.59

# %%
###############################################
# Görev 1 - Adım 4
###############################################

# Ceyrek 1: 4.69
# Ceyrek 2: 4.63
# Ceyrek 3: 4.57
# Ceyrek 4: 4.44

# Çeyreklik dilimlerdeki ortalamalara bakıldığı zaman, day_diff arttıkça ortalama değerlerin azaldığı görülmektedir.
# Buradan yola çıkarak, ürünün son zamanlardaki trendinin geçmişten daha yüksek olduğu yorumunu yapabiliriz.
# Temelde Time Based Average hesaplamasındaki amacımız, rating konseptindeki pozitif ve negatif trendlerin etkilerini
# yakalayabilmektir.

# %%
###############################################
# Görev 2 - Adım 1
###############################################

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# %%
###############################################
# Görev 2 - Adım 2
###############################################

def score_pos_neg_diff(pos, neg):
    return pos - neg


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


positives = df["helpful_yes"].sum()
negatives = df["helpful_no"].sum()

df["score_pos_neg_dif"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


# %%
###############################################
# Görev 2 - Adım 3
###############################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Wilson Lower Bound ile ilgili yorum yaparken yukarıda kullandığımız yöntemlerin hepsine değinmek daha sağlıklı
# olacaktır. İlk olarak, up-down farkıyla başladığımız zaman, veri setinde gözlemleyebileceğimiz üzere yorum sayısına
# olarak hesaplamanın içine bir bias dahil olmaktadır. Average rating hesaplamak istediğimiz zaman da veri bilimi
# çalışmalarındaki çoğu yerden alışık olduğumuz frequency yüksekliğine ait bir yanlılık ortaya çıkıyor. Bundan kaynaklı
# Wilson Lower Bound Score'u kullandık. Amacımız, değerlerimiz için bir güven aralığı hesaplamak ve bunun üzerinden
# hareket etmekti. Bu sıralamayla, overall'u çok düşük olan yorumların da ilk sıralarda yer alabildiğini, up-down
# difference'ı az olan yorumların da çok olanlarla aynı sıralamalara çıkabildiğini ve average rating hesaplamalarındaki
# frequency yanlılığının artık sıralamamıza etki etmediğini görebiliyoruz.



