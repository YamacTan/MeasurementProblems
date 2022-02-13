#######################
# Yamac TAN - Data Science Bootcamp - Week 5 - Project 1
#######################

# %%
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# %%
###############################################
# Görev 1 - Adım 1
###############################################

control = pd.read_excel("Odevler/HAFTA_05/PROJE_1/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("Odevler/HAFTA_05/PROJE_1/ab_testing.xlsx", sheet_name="Test Group")

# %%
###############################################
# Görev 1 - Adım 2
###############################################

control.describe().T
test.describe().T

control.head(5)
test.head(5)

control.isnull().sum()
test.isnull().sum()

# %%
###############################################
# Görev 1 - Adım 3
###############################################

# A/B testteki varsayım kontrollerinde gruplaşmayı kolaylaştırmak için group değişkenlerini ekleyelim.
control["bidding_method"] = "maximum"
test["bidding_method"] = "average"

df = pd.concat([control, test])
df.head()
df.describe().T

# %%
###############################################
# Görev 2 - Adım 1
###############################################

# H0: M1 = M2 :
# Kontrol grubu ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir farklılık yoktur.

# H1: M1 != M2 :
# Kontrol grubu ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir farklılık vardır.

# %%
###############################################
# Görev 2 - Adım 2
###############################################

control["Purchase"].mean()
test["Purchase"].mean()

# Buradan görebileceğimiz üzere ortalamalar arasında bir fark mevcuttur. Fakat bu istatistiksel olarak kanıtlanabilir
# bir farklılık mıdır, yoksa tamamiyle denk mi gelmiştir?

# %%
###############################################
# Görev 3 - Adım 1
###############################################

# Normallik Varsayımı:
# H0 : Normallik varsayımı sağlanmaktadır.
# H1 : Normallik varsayımı sağlanmamaktadır.

# Kontrol grubu normallik varsayımı:
test_stat, p_value = shapiro(df.loc[df["bidding_method"] == "maximum", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, p_value))
# P value = 0.5, 0.5 > 0.05, H0 REDDEDİLEMEZ!

# Test grubu normallik varsayımı:
test_stat, p_value = shapiro(df.loc[df["bidding_method"] == "average", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, p_value))
# P value = 0.15, 0.15 > 0.05, H0 REDDEDİLEMEZ!

# Varyans Homojenliği:
# H0 : Varyanslar homojendir.
# H1 : Varyanslar homojen degildir.

test_stat, pvalue = levene(df.loc[df["bidding_method"] == "maximum", "Purchase"],
                           df.loc[df["bidding_method"] == "average", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# P value = 0.10, 0.10 > 0.05, H0 REDDEDİLEMEZ!

# %%
###############################################
# Görev 3 - Adım 2
###############################################

test_stat, pvalue = ttest_ind(df.loc[df["bidding_method"] == "maximum", "Purchase"],
                              df.loc[df["bidding_method"] == "average", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# P value = 0.34, 0.34 > 0.05, H0 REDDEDİLEMEZ!

# %%
###############################################
# Görev 3 - Adım 3
###############################################

# Yapılan testin sonucunda p-value degeri 0.34 olarak elde edilmiştir. 0.34, 0.05 sınırının üstünde olduğundan H0
# reddedilemez. Bundan dolayı, Kontrol grubu ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı
# bir farklılık yoktur. Çalışmanın başındaki ortalamalarda gözlenen farklılık sadece bir tesadüftür.

# %%
###############################################
# Görev 4 - Adım 1
###############################################

# İlk olarak, iş problemindeki amaç iki ayrı durumun, iki ayrı uygulamanın (A ve B diyelim) karşılaştırmasını yapmayı
# gerektirmektedir. Karşılaştırma yapılmak istenen metrik, Purchase değişkeni ve bu değişkenin ortalamasıdır.
# Bununla birlikte, yapılan varsayım kontrollerinde normallik varsayımı ve varyans homojenliğinin
# sağlandığı görülmüştür. İki varsayım da geçerli olduğundan parametrik bir test uygulamak gereklidir.
# Burada Bağımsız İki Örneklem T Testi'nin kullanılması uygundur.

# %%
###############################################
# Görev 4 - Adım 2
###############################################

# Uygulanan bidding methodlarının getirileri (Purchase) arasında istatistiki olarak anlamlı bir farklılık yoktur.
# Bu noktada şirkete yapılabilecek ilk tavsiye, Purchase konusundaki değişiklik istekleri devam ediyorsa yeni yöntemlere
# yönelmeleri gerektiğidir. Bununla birlikte, yaptıkları değişikliklerin sonucunu kıyaslarken yanıltıcı olacak şekilde
# direkt olarak average gibi değerlere bakmak yanıltıcı olacağından, sürekli olarak A/B testleri yapmak ve istatistiksel
# dayanaklar üzerinden aksiyon almak gerekmektedir.
