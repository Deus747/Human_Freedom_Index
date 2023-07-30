from streamlit_extras.app_logo import add_logo
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
import numpy as np
import pandas as pd 
import streamlit as st 
from tqdm import tqdm
import plotly.graph_objects as go
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



df_terrorism = pd.read_csv('globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1")
df_freedom_index = pd.read_csv('hfi_cc_2019.csv')

#Terrorism Events over years
df_terrorism = df_terrorism.sort_values(['iyear', 'country_txt'], ascending=[True, True]) 
df_terrorism['attacks'] = count_terrorism(df_terrorism)

df_terrorism['attacks'] = df_terrorism['attacks'].astype(int)
df_terrorism['ISO_code'] = get_locations(df_terrorism, df_freedom_index)

analysis = df_terrorism.drop_duplicates(subset=['country_txt', 'attacks'], keep='first')
analysis = analysis[analysis['ISO_code']!='-']
fig2 = px.scatter_geo(analysis, 
                     size="attacks", hover_name='country_txt',
                     locations='ISO_code',
                     projection="natural earth", 
                     animation_frame="iyear",
                     color='region_txt',
                     title='Terrorism Events over years')

fig2.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgrey", showcountries=True, countrycolor="Black")

#Terrorism Events over years
#Successful Attacks

success = df_terrorism[df_terrorism['success']==1].shape[0]
fail = df_terrorism[df_terrorism['success']==0].shape[0]

success_attacks = df_terrorism[df_terrorism['success']==1]
success_attacks = success_attacks.sort_values(['country_txt'], ascending=[True]) 

success_attacks = success_attacks[success_attacks['ISO_code']!='NaN']
success_attacks['success_attacks'] = count(success_attacks)

success_attacks = success_attacks.drop_duplicates(subset=['country_txt', 'success_attacks'], keep='first')

fig3 = px.scatter_geo(success_attacks, 
                     size="success_attacks", hover_name='country_txt',
                     locations='ISO_code',
                     hover_data='iyear',
                     projection="natural earth", 
                     color='region_txt',
                     title='Success Attacks')

fig3.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightGrey", showcountries=True, countrycolor="Black")

#Successful Attacks
# Human Freedom Index

df_freedom_index = df_freedom_index[df_freedom_index['hf_score']!='-']
df_freedom_index['ISO_code'] = df_freedom_index['ISO_code'].astype(str)
df_freedom_index['hf_score'] = df_freedom_index['hf_score'].astype(float)

fig1 = px.choropleth(df_freedom_index, 
                     locations='ISO_code',
                     color='hf_score',
                     hover_name='countries', 
                     projection="natural earth", 
                     animation_frame="year",
                     title='Human Freedom index over years(Heat Map)')
# Human Freedom Index

#merged database
df_terrorism = df_terrorism.rename(columns={'iyear':'year'})
freedom_terrorism = pd.merge(df_terrorism, df_freedom_index, how='left', on=['ISO_code', 'year'])
freedom_terrorism.head()
freedom_terrorism = freedom_terrorism[freedom_terrorism['ISO_code']!='NaN']
test = freedom_terrorism[~freedom_terrorism['countries'].isna()]
test = test.drop_duplicates(subset=['country_txt', 'hf_score', 'year'], keep='first')
#test.shape
#merged database



#initial data

st.markdown('Data Visualization :')

tab1, tab2, tab3 = st.tabs(["Human Freedom index over years", "Terrorism Events over years", "Successful Attacks"])


with tab1 :
    st.plotly_chart(fig1, use_container_width=True)
with tab2 :
    st.plotly_chart(fig2, use_container_width=True)
with tab3 :
    st.plotly_chart(fig3, use_container_width=True)
    
    
country_max = (test[test['hf_score']==test['hf_score'].max()]['country_txt'])
country_min = (test[test['hf_score']==test['hf_score'].min()]['country_txt'])

st.header('Country with the highest HF score :')
st.write(f"<h2>{str(country_max.iloc[0])}<h2>", unsafe_allow_html=True)
st.header('Country with the lowest HF score :')
st.write(f"<h2>{str(country_min.iloc[0])}<h2>", unsafe_allow_html=True)
#initial data

#countries per region
def regions(test):
    regions = {}
    for region in test['region_txt'].unique():
        tmp = test[test['region_txt']==region]
        regions[region] = len(tmp['country_txt'].unique())
    return regions

regions = regions(test)
fig1=px.bar(test,y=list(regions.values()), x=list(regions.keys()))
#countries per region

#standard deviation and mean
cv,mean, std = get_region_mean_std(test)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y = list(cv.values()), x=list(mean.keys()),
))
#standard deviation and mean
#scatterpot
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    y=list(std.values()),
    x=list(std.keys()),
    mode='markers',
    marker=dict(size= list(map(lambda x: x * 50, list(std.values()))),
               )
))
#scatterpot
tab1, tab2, tab3 = st.tabs(["No of countries in each region", "Standard deviation and mean of hf score in each region(Bar Graph)", "Standard deviation and mean of hf score in each region(Scatter Pot)"])


with tab1 :
    st.plotly_chart(fig1, use_container_width=True)
with tab2 :
    st.plotly_chart(fig2, use_container_width=True)
with tab3 :
    st.plotly_chart(fig3, use_container_width=True)

    
#known vs unknown    
df_terrorism_gname = df_terrorism.pivot_table(columns='gname', 
                                              aggfunc='size', fill_value=0)

unknown = df_terrorism[df_terrorism['gname']=='Unknown'].shape[0]
known = df_terrorism[df_terrorism['gname']!='Unknown'].shape[0]
fig1 = px.pie(values=[unknown, known], labels=['Unknown', 'Known'])
fig1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
#known vs unknown    

# Rank by size
known = df_terrorism[df_terrorism['gname']!='Unknown']
known = known.pivot_table(columns='gname', aggfunc='size', fill_value=0)
terror_gname = dict(zip(known.index, known[:]))
terror_gname = sorted(terror_gname.items(), key=lambda kv: kv[1], reverse=True)
terror_gname = dict(terror_gname)
terror_gname_100_keys = list(terror_gname.keys())
terror_gname_100_values = list(terror_gname.values())
terror_gname_100_values = terror_gname_100_values[:100]
terror_gname_100_keys =terror_gname_100_keys[0:100]
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y = terror_gname_100_values, x=terror_gname_100_keys
))
# Rank by size

# Weapon of choice
weapons = df_terrorism.pivot_table(columns='weaptype1_txt', aggfunc='size', fill_value=0)
weapons = dict(zip(weapons.index, weapons[:]))
weapons = sorted(weapons.items(), key=lambda kv: kv[1], reverse=True)
weapons_ = dict(weapons)
weapons_keys = list(weapons_.keys())
weapons_values = list(weapons_.values())
weapons_values = weapons_values[:100]
weapons_keys = weapons_keys[0:100]
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    y = weapons_values, x=weapons_keys,
))

# Weapon of choice
# Type of attack
attacks = df_terrorism.pivot_table(columns='attacktype1_txt', aggfunc='size', fill_value=0)
attacks = dict(zip(attacks.index, attacks[:]))
attacks = sorted(attacks.items(), key=lambda kv: kv[1], reverse=True)
attacks = dict(attacks)
attacks_keys = list(attacks.keys())
attacks_values = list(attacks.values())
attacks_values = attacks_values[:100]
attacks_keys = attacks_keys[0:100]
fig4 = go.Figure()

fig4.add_trace(go.Bar(
    x = attacks_keys, y=attacks_values,
))
# Type of attack

st.header("Terrorism Data Visualized")
tab1, tab2, tab3, tab4 = st.tabs(["Known vs Unknown organisation", "Terroist organization by size", "Weapon used in attacks", "Type of Attacks"])


with tab1 :
    st.plotly_chart(fig1, use_container_width=True)
    
with tab2 :
    st.plotly_chart(fig2, use_container_width=True)
with tab3 :
    st.plotly_chart(fig3, use_container_width=True)
with tab4 :
    st.plotly_chart(fig4, use_container_width=True)
    
    
col1, col2, col3 = st.columns([1,5,1])

with col3 :
    next_page = st.button("Next Page")
    if next_page : 
        switch_page("analysis")
with col1 :
    next_page = st.button("Previous Page")
    if next_page : 
        switch_page("home")