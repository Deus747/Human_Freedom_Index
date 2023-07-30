from streamlit_extras.app_logo import add_logo
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
import numpy as np
import pandas as pd 
import streamlit as st 
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.discriminant_analysis import StandardScaler
from statsmodels.graphics.gofplots import qqplot
from factor_analyzer.factor_analyzer import  FactorAnalyzer
import matplotlib.pyplot as plt
add_logo(
    "Logo.png"
)
def count_terrorism(df_terrorism):
    counts_ = []
    for i in df_terrorism.groupby(['iyear', 'country_txt'])['eventid'].count():
        for j in range(i):
            counts_.append(i)
    return counts_

def get_region_mean_std(df):
    mean = {}
    std = {}
    #variance
    cv={}
    for region in df['region_txt'].unique():
        tmp = df[df['region_txt']==region]
        mean[region] = tmp['hf_score'].mean()
        std[region] = tmp['hf_score'].std()
        cv[region] = tmp['hf_score'].std()/tmp['hf_score'].mean()
    return cv,mean, std

def count(df_terrorism):
    counts_ = []
    for i in df_terrorism.groupby(['country_txt'])['eventid'].count():
        for j in range(i):
            counts_.append(i)
    return counts_

def get_locations(df_terror, df_freedom):
    dict_ = dict(zip(df_freedom['countries'].unique(), df_freedom['ISO_code'].unique()))
    codes = []
    for country in df_terror['country_txt']:
        if(country in dict_.keys()):
            codes.append(dict_[country])
        else:
           codes.append('NaN')
    return codes

def regions(test):
    regions = {}
    for region in test['region_txt'].unique():
        tmp = test[test['region_txt']==region]
        regions[region] = len(tmp['country_txt'].unique())
    return regions

def qq_plot(b, column):
    qqplot_data = qqplot(b[column], line='45').gca().lines
    return qqplot_data

def highlight_max_(s):
    return ['background-color: tomato' if v > 0.53 or v < -0.53 else '' for v in s]




st.markdown("# Analysis")


df_terrorism = pd.read_csv('globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1")
df_freedom_index = pd.read_csv('hfi_cc_2019.csv')

df_terrorism = df_terrorism.sort_values(['iyear', 'country_txt'], ascending=[True, True]) 
df_terrorism['attacks'] = count_terrorism(df_terrorism)

df_terrorism['attacks'] = df_terrorism['attacks'].astype(int)
df_terrorism['ISO_code'] = get_locations(df_terrorism, df_freedom_index)

analysis = df_terrorism.drop_duplicates(subset=['country_txt', 'attacks'], keep='first')
analysis = analysis[analysis['ISO_code']!='-']


success = df_terrorism[df_terrorism['success']==1].shape[0]
fail = df_terrorism[df_terrorism['success']==0].shape[0]

success_attacks = df_terrorism[df_terrorism['success']==1]
success_attacks = success_attacks.sort_values(['country_txt'], ascending=[True]) 

success_attacks = success_attacks[success_attacks['ISO_code']!='NaN']
success_attacks['success_attacks'] = count(success_attacks)

success_attacks = success_attacks.drop_duplicates(subset=['country_txt', 'success_attacks'], keep='first')


df_freedom_index = df_freedom_index[df_freedom_index['hf_score']!='-']
df_freedom_index['ISO_code'] = df_freedom_index['ISO_code'].astype(str)
df_freedom_index['hf_score'] = df_freedom_index['hf_score'].astype(float)


df_terrorism = df_terrorism.rename(columns={'iyear':'year'})
freedom_terrorism = pd.merge(df_terrorism, df_freedom_index, how='left', on=['ISO_code', 'year'])
freedom_terrorism.head()
freedom_terrorism = freedom_terrorism[freedom_terrorism['ISO_code']!='NaN']
test = freedom_terrorism[~freedom_terrorism['countries'].isna()]
test = test.drop_duplicates(subset=['country_txt', 'hf_score', 'year'], keep='first')
# setting up the datasets

# QQ plot
hf_frame = test[['hf_score','pf_ss_women','pf_ss', 'pf_movement_foreign', 'pf_religion', 'pf_association_political_establish', 'pf_expression_jailed','pf_expression_influence','pf_expression_control','pf_expression_newspapers','pf_expression_internet','pf_score','ef_government','ef_legal_judicial','ef_money_growth','ef_money_inflation','ef_money_currency', 'ef_money']]
hf_frame = hf_frame.replace('-', -1)
hf_frame = hf_frame.apply(pd.to_numeric)
#to make then in same range
scaler = StandardScaler().fit_transform(hf_frame)
hf_frame_std = pd.DataFrame(scaler, columns=hf_frame.columns)


qq_plot_data = qq_plot(hf_frame_std, 'hf_score')
st.header("QQ plot")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
# QQ plot


#Eigen Value
st.header("Screen Plot")
factoranalyzer = FactorAnalyzer()
factoranalyzer.fit(hf_frame_std)
ev, v = factoranalyzer.get_eigenvalues()
plt.scatter(range(1, hf_frame_std.shape[1]+1),ev)
plt.plot(range(1, hf_frame_std.shape[1]+1),ev)
plt.title('Screen Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
st.pyplot(plt)

#Eigen Value

#FactorAnalyzer
st.header("Factor Analysis")

col1, col2 = st.columns([4,1])

fa = FactorAnalyzer(n_factors=5)

fa.fit(hf_frame)
factors = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4','Factor5'])
factors = factors.set_index(hf_frame.columns)
#factors.style.apply(highlight_max_)
with col1:
    with st.container():
        st.dataframe(factors.style.apply(highlight_max_))
with col2:
    st.write('Factor 1   : Freedom of Expression')
    st.write('Factor 2   : Wealth')
    st.write('Factor 3   : Safety and Security')
    st.write('Factor 4   : Media')
    st.write('Factor 5   : Government')
#FactorAnalyzer


st.markdown("# Conclusion")
st.write("By understanding these underlying factors, we can now showcase how a country's economy, personal freedom, religious sentiments, media landscape, and the functioning of its government directly influence the overall level of human freedom within its borders. This insight can serve as a valuable resource for policymakers, researchers, and advocates seeking to promote and enhance freedom, peace, and prosperity globally."

"In summary, the results of our factor analysis offer a concise yet powerful framework for comprehending the multifaceted aspects that shape the Human Freedom Index. By addressing the dimensions of expression freedom, economic prosperity, personal safety, media freedom, and government, we can work towards creating a more inclusive, free, and equitable world for all.")


col1, col2, col3 = st.columns([1,5,1])

with col1 :
    next_page = st.button("Previous Page")
    if next_page : 
        switch_page("data visualization")