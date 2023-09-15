import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
import scipy
import pickle
from sklearn.preprocessing import RobustScaler
from feature_engine.wrappers import SklearnTransformerWrapper
import base64
import warnings

warnings.filterwarnings("ignore")

separator_html = """
<div style="background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet); height: 3px;"></div>
"""


columns = ['transaction_id','order_dt','order_products','order_amount']
df_cdnow_raw = pd.read_csv('CDNOW_master.txt',names=columns,sep='\s+')

# chuyển cột date về kiểu datetime
df_cdnow_raw.order_dt = df_cdnow_raw.order_dt.astype(str)
df_cdnow_raw.order_dt= pd.to_datetime(df_cdnow_raw.order_dt, infer_datetime_format=True)

st.markdown('### Dữ liệu được cung cấp:')
st.dataframe(df_cdnow_raw.head())

st.markdown(separator_html, unsafe_allow_html=True)

st.write('### Một số thông tin cơ bản của dữ liệu:')

st.write("**Shape:**")
st.write(df_cdnow_raw.shape)
st.write("-------------------")
st.write("**Describe:**")
st.write(df_cdnow_raw.describe().T)
st.write("-------------------")
st.write('**Kiểm tra dữ liệu thiếu:**')
st.dataframe(pd.concat([df_cdnow_raw.isnull().sum(axis=0).sort_values(ascending=False),df_cdnow_raw.isnull().sum().sort_values(ascending=False)/len(df_cdnow_raw)*100], axis=1, ).rename(columns={0:'count',1:'percentage'}))
st.write('=> Không có dữ liệu thiếu')
st.write("-------------------")
st.write('**Kiểm tra dữ liệu trùng:**')
st.code('Dữ liệu thiếu ' + str(df_cdnow_raw.duplicated().sum()))
st.write('Có dữ liệu trùng => Tiến hành loại bỏ dữ liệu trùng')

st.write("- Số lượng dòng **trước** khi loại bỏ dữ liệu trùng: ", len(df_cdnow_raw))
# remove data duplicated
df_cdnow_raw.drop_duplicates(inplace=True)
st.write('- Số lượng dòng **sau** khi loại bỏ dữ liệu trùng: ',len(df_cdnow_raw))

st.markdown(separator_html, unsafe_allow_html=True)

st.write('### Khai phá dữ liệu:')

df_cdnow_raw['Year']= pd.DatetimeIndex(df_cdnow_raw['order_dt']).year
df_cdnow_raw['Year'] = df_cdnow_raw['Year'].astype(str)

st.write('#### Phân bố và outliers')


# st.image('images/1.png')
# st.image('images/2.png')
# st.image('images/3.png')

st.write("**Quantity**")
fig_1 = px.histogram(df_cdnow_raw, x="order_products", marginal="box", color='Year', color_discrete_sequence=['#F07B3F', '#609966'])
st.plotly_chart(fig_1)

st.write("**Price**")
fig_2 = px.histogram(df_cdnow_raw, x="order_amount", marginal="box", color='Year', color_discrete_sequence=['#903749', '#0D7377'] )
st.plotly_chart(fig_2)

st.write(''' **Nhận xét:**
* Biến quantity và price đều có outliers, số lượng không nhiều vì vậy có thể loại bỏ các outliers này mà không ảnh hưởng lớn đến dữ liệu, tuy nhiên, do đây là dữ liệu về lượt mua đĩa CD vì vậy các giá trị ngoại lai này cũng có thể hiểu là một hành vi bất thường không hiếm gặp của khách hàng mua sắm.
    
        => Đối với data này xem xét giữ lại outliers để tính toán

* Dữ liệu lệch phải nên áp dụng RobustScalers cho dữ liệu này.
''')


st.write('#### Doanh thu và lượt khách qua các năm')



# col1, col2 = st.columns(2)

st.write('**Doanh thu qua các năm**')
price_year = df_cdnow_raw.groupby('Year').agg(revenue=("order_amount", 'sum'))
price_year.reset_index(inplace=True)
st.dataframe(price_year)


price_year_plot = px.bar(price_year,x='Year', y='revenue', text_auto = '.s', color='Year', color_discrete_sequence=['#435B66', '#A76F6F'], template='seaborn')

price_year_plot.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)

st.plotly_chart(price_year_plot)

st.write('**Lượt mua qua các năm**')
customer_year = df_cdnow_raw.groupby('Year')['transaction_id'].count()
customer_year = customer_year.reset_index()
st.dataframe(customer_year)

customer_year_plot = px.bar(customer_year,x='Year', y='transaction_id', text_auto = '.s', color='Year', color_discrete_sequence=['#E06469', '#F2B6A0'], template='seaborn')
customer_year_plot.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)
st.plotly_chart(customer_year_plot)

st.write('''**Nhận xét:**
* Năm 1997 hơn năm 1998 cả về doanh thu và lượt khách mua CD.
''')

st.write('#### Doanh thu theo thời gian')

df1 = df_cdnow_raw.reset_index().set_index('order_dt')[['order_amount']].resample(rule="MS").sum()

fig = px.line(df1, y="order_amount", markers=True, color_discrete_sequence=['#B04759']) #text='order_amount'
fig.update_traces(textposition="top left")

st.plotly_chart(fig)


# st.line_chart(df1, color=['#EA5455'])

st.markdown(''' **Nhận xét:**
* Doanh thu có sự sụt giảm mạnh theo thời gian, bắt đầu từ tháng 2/1997
''')
