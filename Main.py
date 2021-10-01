import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pickle


EXPANDER_TEXT = """
    To change the current theme 🎈 
    In the app menu (☰ -> Settings -> Theme).
    """
    
READ_ME = """
This app is under development. 
To use the app, define the values of the predictors and 
the predicted responses of the pier will be updated under the section "Predicted responses of the pier".
In addition, the significance of main effect and interaction of the factors on each response variable is shown in the figures. 
🎈 
"""
# Primary accent for interactive elements
primaryColor = 'red'

# Background color for the main content area
backgroundColor = '#273346'

#st.markdown(html_temp, unsafe_allow_html=True)
alam = Image.open('ALAMS.png')
#st.image(alam, use_column_width=True)
#st.image(alam,width=450)



html_temp = """
<div style="background-color:black ;padding:10px">
<h2 style="color:cyan;text-align:center;"> 
Response prediction of rocking bridge piers using exPlainable ML model
(under development)
</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

#st.write("### By [Tadesse G. Wakjira](https://scholar.google.com/citations?user=Ka3iXSoAAAAJ)")
#import the dataset
df = pd.read_excel('data.xlsx', sheet_name = 'data')

""
with st.beta_expander("Read me"):
    st.write(READ_ME)
with st.beta_expander("To change Theme"):
    st.write(EXPANDER_TEXT)

""
""

# Header
st.sidebar.header('Define the values of input factors')
t=14
def user_defined_paremeters():
    dc = st.sidebar.slider('Diameter of the column, dc (mm)', float(df['dc'].min()), float(df['dc'].max()),
                               float(df['dc'][t]))
    dc_tc = st.sidebar.slider('Column diameter-to-thickness ratio, dc/tc', float(df['dc/tc'].min()), float(df['dc/tc'].max()),
                               float(df['dc/tc'][t]))
    hc_dc = st.sidebar.slider('Column height-to-diameter ratio, hc/dc', float(df['hc/dc'].min()), 
                            float(df['hc/dc'].max()),
                            float(df['hc/dc'][t]))
    Apt_Ac = st.sidebar.slider('Cross-sectional area of tendon to column ratio , Apt/Ac', float(df['Apt/Ac'].min()), 
                            float(df['Apt/Ac'].max()),
                            float(df['Apt/Ac'][t]))    
    n = st.sidebar.slider('dead load ratio , P/Ac fy,c', float(df['P/Ac fy,c'].min()), 
                            float(df['P/Ac fy,c'].max()),
                            float(df['P/Ac fy,c'][t]))   
    tbp = st.sidebar.slider('base plate thickness, tbp (mm)', float(df['tbp'].min()), 
                            float(df['tbp'].max()),
                            float(df['tbp'][t])) 
                              
    ebp = st.sidebar.slider('base plate extension, ebp (mm)', float(df['ebp'].min()), 
                            float(df['ebp'].max()),
                            float(df['ebp'][t]))
    fpt_fu = st.sidebar.slider('tendon initial post-tensioning ratio, fpt,0/fu', float(df['fpt,0/fu'].min()), 
                            float(df['fpt,0/fu'].max()),
                            float(df['fpt,0/fu'][t]))    
       
    data = {'dc': dc,
            'dc/tc': dc_tc, # conversion from percent
            'hc/dc': hc_dc,
            'Apt/Ac': Apt_Ac,
            'fpt,0/fu': fpt_fu,
            'P/Ac fy,c': n,            
            'tbp': tbp,
            'ebp': ebp,
            }

    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_defined_paremeters()

#st.header('User defined variables')
html_temp = """
<div style="background-color:gray ;padding:10px">
<h2 style="color:white;text-align:center;">User defined variables </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write('#### Geomtery of the column, axial load, and mechanical properties of material')

st.write(df1[['dc', 'dc/tc', 'hc/dc', 'Apt/Ac', 'fpt,0/fu', 'P/Ac fy,c', 'tbp', 'ebp']])

st.write('---')

# Normalize the user defined variables 
dfn=[]
for i in range(0,df1.shape[1]):
    a = (df1.iloc[:,i]-df.iloc[:,i].min())/(df.iloc[:,i].max()-df.iloc[:,i].min())
    dfn.append(a)
    
dfn = pd.DataFrame(np.array(dfn)).T.values

# dfn = pd.DataFrame(np.array(dfn)).T
# x=df.iloc[:,:-1]
# dfn.columns = x.columns

model1 = pickle.load(open('model_res_drift_tot.pkl', 'rb'))
model2 = pickle.load(open('model_DelY_hc.pkl', 'rb'))
model3 = pickle.load(open('model_k_deg_k_ini.pkl', 'rb'))
model4 = pickle.load(open('model_Vmax_Vup_rigid.pkl', 'rb'))
model5 = pickle.load(open('model_strength_r.pkl', 'rb'))

res_drift_tot =model1.predict(dfn)[0]
DelY_hc=model2.predict(dfn)[0]
k_deg_k_ini=model3.predict(dfn)[0]
Vmax_Vup_rigid=model4.predict(dfn)[0]
strength_r=model5.predict(dfn)[0]

# Inverse normalization
# observed responses
yy1 = df['res_drift_tot'].values 
yy2 = df['DelY/hc (tot)'].values 
yy3 = df['k_deg/k_ini'].values 
yy4 = df['Vmax/Vup_rigid'].values 
yy5 = df['strength_r'].values 

# predicted responses
y1=round(100*(yy1.min()+(yy1.max()-yy1.min()) * res_drift_tot),5)
y2=round(100*(yy2.min()+(yy2.max()-yy2.min()) * DelY_hc),5)
y3=round(100*(yy3.min()+(yy3.max()-yy3.min()) * k_deg_k_ini),5)
y4=round(yy4.min()+(yy4.max()-yy4.min()) * Vmax_Vup_rigid,5)
y5=round(100*(yy5.min()+(yy5.max()-yy5.min()) * strength_r),5)


html_temp = """
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Predicted responses of the pier </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write('### Residual drift (%) =', y1)
st.write('### Column cyclic shortening to height ratio (%) =', y2)
st.write('### Ratio of degraded stiffness to initial stiffness (%) =', y3)
st.write('### Maximum lateral strength to uplift force ratio =', y4)
st.write('### Lateral strength reduction (%) =', y5)

st.write('---')

# SHAP explanation
#st.header('Model explanation using SHAP approach and significance of the input factors')
html_temp = """
<div style="background-color:gray ;padding:10px">
<h2 style="color:white;text-align:center;">Model explanation using SHAP approach and significance of the input factors </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.markdown('## **SHAP summary plot**')
image1 = Image.open('Fig 1.PNG')
image2 = Image.open('Fig 2.PNG')
image3 = Image.open('Fig 3.PNG')
image4 = Image.open('Fig 4.PNG')
image5 = Image.open('Fig 5.PNG')

st.image(image1, use_column_width=True)
st.image(image2, use_column_width=True)
st.image(image3, use_column_width=True)
st.image(image4, use_column_width=True)
st.image(image5, use_column_width=True)

st.markdown('## **SHAP dependency plot for the first four significnat parameters**')
image6 = Image.open('Fig 6.PNG')
image7 = Image.open('Fig 7.PNG')
image8 = Image.open('Fig 8.PNG')
image9 = Image.open('Fig 9.PNG')
image10 = Image.open('Fig 10.PNG')

st.image(image6, use_column_width=True)
st.image(image7, use_column_width=True)
st.image(image8, use_column_width=True)
st.image(image9, use_column_width=True)
st.image(image10, use_column_width=True)

st.write('<style>h1{color: red;}</style>', unsafe_allow_html=True)
st.write('<style>h3{color: green;}</style>', unsafe_allow_html=True)

#st.write("### For any comment or furthermore assistance contact: tgwakjira@gmail.com [Tadesse G. Wakjira](https://scholar.google.com/citations?user=Ka3iXSoAAAAJ)")
st.write("### For any comments, please contact tgwakjira@gmail.com")
