import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime, timedelta
import streamlit as st
from streamlit.components.v1 import iframe
# from streamlit import caching
import base64 as b64
from PIL import Image
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import re
import findspark
findspark.init()
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from utils import _initialize_spark
import sys

import plotly.express as px
import squarify
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler, MinMaxScaler, RobustScaler, Binarizer, Normalizer, scale
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif,f_classif, SelectFromModel, RFE
from sklearn.feature_extraction.text import CountVectorizer
# splitting the data set into training set and test set
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from scipy.stats.stats import pearsonr
from scipy.stats import shapiro, norm, zscore
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from datetime import datetime, timedelta
import pickle
import time

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

sns.set_style("whitegrid", {'axes.grid' : False})
## Sidebar: left
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            padding-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True
)
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: F0F3F4;
    }
    </style>
    """, unsafe_allow_html=True)
separator_html = """
<div style="background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet); height: 3px;"></div>
"""
with st.sidebar:
    st.image("images/Logo.png")
    st.image(f'Project_1/images/RFM.png', width=60)
### Functions: Chỉ cho hiện những hình nằm trong phạm vi cấu hình fr - to
def project_show_range_img(directory, fr=1, to=24):
    # Use os.listdir to get all files in the directory
    files = os.listdir(directory)

    def sort_key(file_name):
        # Extract the number from the file name (much number)
        number = int(file_name.split('.')[0])
        return number

    # Sort the list of files using the custom sorting function
    files = sorted(files, key=sort_key)
    print(files)
    # Filter the list to only include .PNG files with file names between fr and to
    images = [file for file in files if file.endswith('PNG') and fr <= int(file.split('.')[0]) <= to]
    return images
### Functions: Cho tùy chỉnh tùy biến danh sách các hình muốn hiển thị không cần theo quy luật nào.
def project_show_list_img(directory, list=[10, 5, 8]):
    files = os.listdir(directory)
    # Filter the list to only include .PNG files with file names in list
    images = [file for file in files if file.endswith('.PNG') and int(file.split('.')[0]) in list]
    # Arrange images based on their order in the list
    images = sorted(images, key=lambda x: list.index(int(x.split('.')[0])))
    return images
### markdown: right
from pathlib import Path
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = b64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path, width, height):
    img_html = "<img src='data:image/png;base64, {}' class='img-fluid' width='{}' height='{}'>".format(
        img_to_bytes(img_path), width, height
    )
    return img_html

st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('images/CapstoreProject.png', 700, 350)+"</p>", unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------------------------------------#

# Tạo cột bên trái cho menu
left_column = st.sidebar
# Chèn đoạn mã HTML để tùy chỉnh giao diện của chữ "Chọn dự án" : left_column.markdown('<span style="font-weight: bold; color: blue;">Chọn dự án</span>', unsafe_allow_html=True)
# Tạo danh sách các dự án
projects =  ['PROJECT 1: CUSTOMER SEGMENTATION','PROJECT 2: RECOMMENDATION SYSTEM', 'PROJECT 3: SENTIMENT ANALYSIS']

# Tạo menu dropdown list cho người dùng lựa chọn dự án
project = left_column.selectbox(":blue[**Select project:**]", projects, index=0)

# Lưu trữ chỉ số index của dự án được chọn
project_num = projects.index(project) + 1


def highlight_rows_even_odd_1(row):
    if row.name % 2 == 0:
        return ['background-color: lightgreen']*len(row)
    else:
        return ['background-color: white']*len(row)
    
def highlight_rows_even_odd_2(row):
    if row.name % 2 == 0:
        return ['background-color: lightcoral']*len(row)
    else:
        return ['background-color: white']*len(row)
# Chọn dự án
if project_num == 1:
    # Hiển thị tên của dự án 
    st.subheader("PROJECT 1: CUSTOMER SEGMENTATION")
    # Hiển thị thời gian bắt đầu Streamlit bắt đầu dự án, để kiểm tra chéo với thời gian kết thúc.
    first_step = datetime.now()
    st.markdown("----"*30)

### START: TRÁNH LÀM CHẬM HỆ THỐNG DO PHẢI XỬ LÝ LẠI MỖI KHI CHỌN CÁC CHỨC NĂNG ###
    # 1. Read data
    if 'df_cdnow_raw' not in st.session_state:
        # Load dataset
        columns = ['transaction_id','order_dt','order_products','order_amount']
        df_cdnow_raw = pd.read_table(f'Project_{project_num}/Data/CDNOW_master.txt',names=columns,sep='\s+')
        st.session_state['df_cdnow_raw'] = df_cdnow_raw
    else:
        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
        print("Dữ liệu dataframe \"df_cdnow_raw\" đã có sắn, chỉ việc lấy ra sử dụng")
        df_cdnow_raw = st.session_state['df_cdnow_raw']
    if 'df_cdnow' not in st.session_state:
        df_cdnow = pd.DataFrame(columns=["transaction_id", "order_dt", "order_products", "order_amount", "month"])
        # Load dataset
        df_cdnow = pd.read_table(f'Project_{project_num}/Data/cdnow_clean.txt',sep=' ', encoding='utf-8', names=df_cdnow.columns)
        df_cdnow['order_dt'] = pd.to_datetime(df_cdnow.order_dt,format="%Y-%m-%d")
        df_cdnow['order_dt'] = df_cdnow['order_dt'].astype('datetime64[ns]')
        df_cdnow['month'] = df_cdnow.order_dt.values.astype('datetime64[M]')
        st.session_state['df_cdnow'] = df_cdnow
    else:
        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
        print("Dữ liệu dataframe \"df_cdnow\" đã có sắn, chỉ việc lấy ra sử dụng")
        df_cdnow = st.session_state['df_cdnow']
    # Upload file
    uploaded_df_cdnow_clean = st.file_uploader("", type=['txt'])
    df_cdnow_new = pd.DataFrame(columns=["transaction_id", "order_dt", "order_products", "order_amount", "month"])
    if uploaded_df_cdnow_clean is not None:
        df_cdnow_new = pd.read_csv(uploaded_df_cdnow_clean, sep=' ', encoding='utf-8', names=df_cdnow_new.columns)
        if len(df_cdnow_new) > 1:
            df_cdnow_new.to_csv(f'Project_{project_num}/Data/Upload/cdnow_clean_new.txt', index=False,  header=True)
            df_cdnow_new['order_dt'] = pd.to_datetime(df_cdnow_new.order_dt,format="%Y-%m-%d")
            df_cdnow_new['order_dt'] = df_cdnow_new['order_dt'].astype('datetime64[ns]')
            df_cdnow_new['month'] = df_cdnow_new.order_dt.values.astype('datetime64[M]')
            st.session_state['df_cdnow'] = df_cdnow_new
            st.dataframe(df_cdnow_new)
    if 'rfm_df' not in st.session_state:
        # Convert string to date, get max date of dataframe
        max_date = df_cdnow['order_dt'].max().date()
        Recency = lambda x: (max_date - x.max().date()).days
        Frequency  = lambda x: x.count()
        Monetary = lambda x: round(sum(x), 2)
        rfm_df = df_cdnow.groupby('transaction_id').agg({'order_dt': Recency, 'order_products': Frequency, 'order_amount': Monetary})
        # Rename the columns of Dataframe
        rfm_df.columns = ['Recency', 'Frequency', 'Monetary']
        # Descending Sorting
        rfm_df = rfm_df.sort_values('Monetary', ascending=False)

        # Assume that you have a DataFrame `rfm_df` containing the RFM values for each customer
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])

        def join_rfm(x): return str(int(x['R_Score'])) + str(int(x['F_Score'])) + str(int(x['M_Score']))
        rfm_df['RFM_Segment'] = rfm_df.apply(join_rfm, axis= 1)
        # Combine the R_Score, F_Score and M_Score values to create a single RFM score for each customer
        rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

        # Define a dictionary to map each RFM score to a corresponding customer segment
        segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}
        # Map each RFM score to a corresponding customer segment using the dictionary
        rfm_df['Customer_Group'] = rfm_df['RFM_Score'].map(segment_dict)
        st.session_state['rfm_df'] = rfm_df
    else:
        rfm_df = st.session_state['rfm_df']
    if 'df_report' not in st.session_state:
        df_report = df_cdnow.copy()
        df_report= df_report.set_index('transaction_id')
        df_report= df_report.join(rfm_df)
        df_report= df_report[["order_dt", "order_products", "order_amount", "month", "RFM_Score", "Customer_Group"]]
        df_report= df_report.reset_index()
        df_report['order_dt'] = pd.to_datetime(df_report['order_dt'])
        df_report['Year'] = df_report['order_dt'].dt.year
        st.dataframe(df_report)
        st.session_state['df_report']  = df_report
    else:
        df_report = st.session_state['df_report']    
### END: TRÁNH LÀM CHẬM HỆ THỐNG DO PHẢI XỬ LÝ LẠI MỖI KHI CHỌN CÁC CHỨC NĂNG ###

    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Prediction'])
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':
        # Đường dẫn đến files png
        directory = f'Project_{project_num}/images/slides'
        images = project_show_range_img(directory, 2, 6)
        # Loop through the images and display them using st.image
        for image in images:
            img = Image.open(os.path.join(directory, image))
            print(img)
            st.image(img)
    elif step == 'Preprocessing + EDA':
        separator_html = """
        <div style="background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet); height: 3px;"></div>
        """

        # chuyển cột date về kiểu datetime
        df_cdnow_raw.order_dt = df_cdnow_raw.order_dt.astype(str)
        df_cdnow_raw.order_dt= pd.to_datetime(df_cdnow_raw.order_dt, infer_datetime_format=True)

        st.markdown('### Dữ liệu được cung cấp:')
        st.dataframe(df_cdnow_raw[["transaction_id", "order_dt", "order_products", "order_amount"]].head())

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

        st.markdown("**Quantity**")
        fig_1 = px.histogram(df_cdnow_raw, x="order_products", marginal="box", color='Year', color_discrete_sequence=['#F07B3F', '#609966'])
        st.plotly_chart(fig_1)

        st.markdown("**Price**")
        fig_2 = px.histogram(df_cdnow_raw, x="order_amount", marginal="box", color='Year', color_discrete_sequence=['#903749', '#0D7377'] )
        st.plotly_chart(fig_2)

        st.markdown('**Nhận xét:**')
        st.markdown('* Biến quantity và price đều có outliers, số lượng không nhiều vì vậy có thể loại bỏ các outliers này mà không ảnh hưởng lớn đến dữ liệu, tuy nhiên, do đây là dữ liệu về lượt mua đĩa CD vì vậy các giá trị ngoại lai này cũng có thể hiểu là một hành vi bất thường không hiếm gặp của khách hàng mua sắm.')
            
        st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=> &nbsp;Đối với data này xem xét giữ lại outliers để tính toán')

        st.markdown('* Dữ liệu lệch phải nên áp dụng RobustScalers cho dữ liệu này.')


        st.write('#### Doanh thu và lượt khách qua các năm')
        price_year = df_cdnow_raw.groupby('Year').agg(revenue=("order_amount", 'sum'))
        price_year.reset_index(inplace=True)
        price_year_plot = px.bar(price_year,x='Year', y='revenue', text_auto = '.s', color='Year', color_discrete_sequence=['#435B66', '#A76F6F'], template='seaborn')
        price_year_plot.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)

        customer_year = df_cdnow_raw.groupby('Year')['transaction_id'].count()
        customer_year = customer_year.reset_index()
        customer_year_plot = px.bar(customer_year,x='Year', y='transaction_id', text_auto = '.s', color='Year', color_discrete_sequence=['#E06469', '#F2B6A0'], template='seaborn')
        customer_year_plot.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)

        col1, col2 = st.columns(2)        
        # Vẽ biểu đồ 1 trong cột 1
        with col1:
            st.write('**Doanh thu qua các năm**')
            st.dataframe(price_year)
            # st.plotly_chart(price_year_plot)
            fig, axs = plt.subplots(1, 1, figsize=(10, 6))
            sns.barplot(x='Year', y='revenue', data=price_year, palette=['#435B66', '#A76F6F'])
            plt.title('')
            st.pyplot(fig)

        # Vẽ biểu đồ 2 trong cột 2
        with col2:
            st.write('**Lượt mua qua các năm**')
            st.dataframe(customer_year)
            # st.plotly_chart(customer_year_plot)
            fig, axs = plt.subplots(1, 1, figsize=(10, 6))
            sns.barplot(x='Year', y='transaction_id', data=customer_year, palette=['#E06469', '#F2B6A0'])
            plt.title('')
            st.pyplot(fig)

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
       
    elif step == 'Applicable models':
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["## **RFM**\n**\***", "## **RFM**\n**KMeans Sklearn**","## **RFM**\n**Hierarchical Scipy**","## **RFM**\n**Pyspark KMeans**","## **RFM**\n**MiniBatchKMeans**","## **RFM**\n**DBSCAN**","## **Evaluating the models**\n**Report**"])
        with tab1: # RFM
            # current_path = os.path.dirname(os.path.realpath(__file__))
            # st.write(current_path)
            # print(current_path)
            st.image(f'Project_{project_num}/images/slides/11.PNG')
            st.image(f'Project_{project_num}/images/slides/13.PNG')
            st.image(f'Project_{project_num}/images/slides/12.PNG')
            
            ### Virsulization
            st.markdown("#### **Virsulization for RFM**")
            # Create the figure and subplots
            fig, axs = plt.subplots(3, 1, figsize=(9,5))
            sns.distplot(rfm_df['Recency'], ax=axs[0]) # Plot distribution of R
            sns.distplot(rfm_df['Frequency'], ax=axs[1]) # Plot distribution of F
            sns.distplot(rfm_df['Monetary'], ax=axs[2]) # Plot distribution of M
            # Display the figure using st.pyplot
            st.pyplot(fig)
            fig, axs = plt.subplots(ncols=3, figsize=(9, 3))
            for i, column in enumerate(['Recency','Frequency','Monetary']):
                sns.boxplot(x=rfm_df[column], ax=axs[i])
                axs[i].set_title('Box plot of ' + column)
            st.dataframe(df_report)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Nhận xét**")
            st.markdown("- Recency lệch trái,  không có outliers")
            st.markdown("- Frequency, Monetary: lệch phải, có outliers")
            st.write(f"\n#### Chi tiết kết quả đánh giá cho phân nhóm RFM")
            mapping = {'Champion':12, 'Loyal Customers':11, 'Promising':10, 'New Customers':9, 'Abandoned Checkouts':8, 'Callback Requests':7, 'Warm_Leads':6, 'Cold Leads':5, 'Need Attention':4, 'Should not Lose':3, 'Sleepers':2, 'Lost':1}
            for k, v in mapping.items():
                st.write(k + ',')
                st.write(df_report[df_report['RFM_Score'] == v].drop(['transaction_id','RFM_Score','Year'], axis=1).describe().T)
                st.write()

            st.image(f'Project_{project_num}/images/slides/14.PNG')
            st.markdown("**Nhận xét**")
            st.markdown("- Lượng khách hàng mang lại dòng tiền cho công ty tốt là nhóm Champion ~ 10%, Loyal Customers ~ 8.11%")
            st.markdown("- Bên cạnh đó từ số liệu cho thấy cần hành động nhanh trước khi mất đi một lượng lớn khách hàng thuộc các nhóm Sleepers ~ 16.73%, Shouldn't Lose ~ 16.42%, Need Attention ~ 11.01%.")
            st.markdown("- Cũng như cần chạy thêm chiến dịch để có biến đổi những khách hàng đang không quan tâm trở lại thành khách hàng nhóm Cold Leads ~ 2.93%, Lost ~ 5.28%, Callback Requests ~ 5.92%")

            st.image(f'Project_{project_num}/images/slides/15.PNG')
            st.markdown("**Nhận xét**")
            st.markdown("- Rõ ràng bộ phận chăm sóc khách hàng, marketing, sales khá yếu dẫn đến số lượng chi tiêu bị sụt giảm rất lớn trên từng nhóm khách hàng")
            st.markdown("Thậm chí có những nhóm khách hàng Sleepers, Shouldn't Lose, Cold Leads, Warm_Leads trong 1 năm không được cải thiện, không phát sinh dòng tiền cho công ty => cần điều chỉnh các bộ phận")
            st.markdown("=> Có trục trặc lớn về sản phẩm or cơ cấu quản lý vận hành của công ty này khá yếu.")
            st.markdown("\n\n#### **Vậy cần cho ra những sản phẩm nào + giá thành khoảng bao nhiêu sẽ phù hợp ?**")

            st.image(f'Project_{project_num}/images/slides/16.PNG')
            st.markdown("**Nhận Xét**")
            st.markdown("- Nhóm khách hàng mang lại dòng tiền cho công ty có Recency ngắn ~ < 150 ngày và chi tiền nhiều")
            st.markdown("- Khoảng chi tiền nhiều nhất rơi vào ~ < 2000 => tập trung ra sản phầm tầm tổng giá trong phân khúc giá này.")
            st.markdown("- Trong phạm vi giới hạn thời gian nên tạm thời chúng ta tạm dừng phân tích sâu đến đây.")
            
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg = rfm_df.groupby('Customer_Group').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']
                }).round(0)

            rfm_agg.columns = rfm_agg.columns.droplevel()
            rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
            rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

            # Reset the index
            rfm_agg = rfm_agg.reset_index()
            st.dataframe(rfm_agg)
            #### Scatter Plot (RFM)
            st.markdown("\n#### **Scatter Plot (RFM)**")
            # Set the figure size before plotting
            fig, axs = plt.subplots(1, 1, figsize=(20,8))
            fig = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Customer_Group",
                    hover_name="Customer_Group", size_max=100)
            # plt.savefig('images/RFM Segments - Scatter Plot.png')
            st.plotly_chart(fig)
        with tab2: # RFM + KMeans
            st.image(f'Project_{project_num}/images/RFM_ML_KMeans.png', width=700   )
            # Đường dẫn đến files png
            directory = f'Project_{project_num}/images/slides'
            images = project_show_range_img(directory, 18, 20)
            # Loop through the images and display them using st.image
            for image in images:
                img = Image.open(os.path.join(directory, image))
                print(img)
                st.image(img)
        with tab3: # RFM + Hierarchical
            st.image(f'Project_{project_num}/images/RFM_ML_Hierarchical.png', width=700)
            # Đường dẫn đến files png
            directory = f'Project_{project_num}/images/slides'
            images = project_show_range_img(directory, 21, 23)
            # Loop through the images and display them using st.image
            for image in images:
                img = Image.open(os.path.join(directory, image))
                print(img)
                st.image(img)
        with tab4: # RFM + pyspark KMeans
            st.image(f'Project_{project_num}/images/RFM_pspark_KMeans.png', width=700)
            # Đường dẫn đến files png
            directory = f'Project_{project_num}/images/slides'
            images = project_show_range_img(directory, 24, 26)
            # Loop through the images and display them using st.image
            for image in images:
                img = Image.open(os.path.join(directory, image))
                print(img)
                st.image(img)

        with tab5: # RFM + MiniBatch
            print("tab")
            # Đường dẫn đến files png
            directory = f'Project_{project_num}/images/slides'
            images = project_show_range_img(directory, 30, 32)
            # Loop through the images and display them using st.image
            for image in images:
                img = Image.open(os.path.join(directory, image))
                print(img)
                st.image(img)
        with tab6: # RFM + DBSCAN
            print("tab")
            # Đường dẫn đến files png
            directory = f'Project_{project_num}/images/slides'
            images = project_show_range_img(directory, 27, 29)
            # Loop through the images and display them using st.image
            for image in images:
                img = Image.open(os.path.join(directory, image))
                print(img)
                st.image(img)
        with tab7: # Report
            # Đường dẫn đến files
            directory = f'Project_{project_num}/images/slides'
            images = project_show_list_img(directory, [37, 33, 34, 35])
            # Loop through the images and display them using st.image
            for image in images:
                img = Image.open(os.path.join(directory, image))
                print(img)
                st.image(img)




    elif step == 'Prediction':
    ## FUNCTIONS ##
        # Hàm tạo dữ liệu thay đổi không đáng kể so với dữ liệu gốc dựa trên ngưỡng cho phép thay đổi trên giao diện cấu hình bằng biến threshold
        def significantly_similar_data(df_original, threshold = 0.5, save_file= None):
            df_cdnow_predict = df_original.copy()
            # Chỉ điều chỉnh 2 cột  'order_products' and 'order_amount'
            for column in ['order_products', 'order_amount']:
                low, high = 0.1, 1
                while high - low > 1e-9:
                    multiplier = (low + high) / 2
                    df_cdnow_predict[column] = np.maximum(1, np.round(df_original[column] * multiplier))
                    max_diff = np.max(np.abs(df_cdnow_predict[column] - df_original[column]))
                    if max_diff <= threshold:
                        high = multiplier
                    else:
                        low = multiplier
            # Update this data is a new customer, Get the maximum 'transaction_id'
            max_id = df_cdnow_predict['transaction_id'].max()

            # Create a dictionary mapping the old transaction_ids to the new ones
            unique_transaction_ids = df_cdnow_predict['transaction_id'].unique()
            new_transaction_ids = range(max_id + 1, max_id + len(unique_transaction_ids) + 1)
            id_mapping = dict(zip(unique_transaction_ids, new_transaction_ids))

            # Update the DataFrame
            df_cdnow_predict['transaction_id'] = df_cdnow_predict['transaction_id'].map(id_mapping)

            if save_file != None:
                # Save file tracking
                df_cdnow_predict.to_csv(f'Project_{project_num}/Data/case_study/similar_data_{threshold}.txt', sep=' ', index=False, header=False)    
            return df_cdnow_predict
        
        # Hàm tạo dữ liệu thay đổi rất đáng kể so vối dữ liệu gốc dựa trên ngưỡng cho phép thay đổi trên giao diện cấu hình bằng biến threshold
        def significantly_dissimilar_data(df_original, threshold = 0.5, save_file= None):
            df_cdnow_predict = df_original.copy()
            
            # Chỉ điều chỉnh 2 cột  'order_products' and 'order_amount'
            for column in ['order_products', 'order_amount']:
                while True:
                    random_values = np.random.uniform(low=threshold, high=df_original[column].max(), size=len(df_original))
                    df_cdnow_predict[column] = df_original[column] + random_values
                    max_diff = np.max(np.abs(df_cdnow_predict[column] - df_original[column]))
                    if max_diff > threshold:
                        break
            # Update this data is a new customer, Get the maximum 'transaction_id'
            max_id = df_cdnow_predict['transaction_id'].max()

            # Create a dictionary mapping the old transaction_ids to the new ones
            unique_transaction_ids = df_cdnow_predict['transaction_id'].unique()
            new_transaction_ids = range(max_id + 1, max_id + len(unique_transaction_ids) + 1)
            id_mapping = dict(zip(unique_transaction_ids, new_transaction_ids))

            # Update the DataFrame
            df_cdnow_predict['transaction_id'] = df_cdnow_predict['transaction_id'].map(id_mapping)

            if save_file != None:
                # Save file tracking
                df_cdnow_predict.to_csv(f'Project_{project_num}/Data/case_study/dissimilar_data_{threshold}.txt', sep=' ', index=False, header=False)    
            return df_cdnow_predict
           
        # Đánh giá tự động dựa vào dữ liệu gốc và dữ liệu mới cần kiểm tra dựa trên ngưỡng cho phép thay đổi trên giao diện cấu hình bằng biến threshold
        # '''
        # Hàm compare_data là so sánh sự khác biệt giữa số lượng giá trị duy nhất của mỗi cột trong hai DataFrame df_original_sample và df_new. 
        # Nếu tất cả các giá trị khác biệt này đều nhỏ hơn hoặc bằng ngưỡng (threshold), hàm sẽ trả về “phương pháp 2”, nếu không, nó sẽ trả về “phương pháp 1”.

        # Đồng nghĩa mình đưa giá trị threshold càng thấp (ví dụ: 0.000001), thì khả năng cao rằng một số (hoặc tất cả) các giá trị khác biệt sẽ lớn hơn ngưỡng, và do đó, hàm compare_data sẽ trả về “phương pháp 1”.
        # Ngược lại, nếu mình đưa giá trị threshold càng cao((ví dụ: 100.0), thì khả năng cao rằng tất cả các giá trị khác biệt sẽ nhỏ hơn hoặc bằng ngưỡng, và do đó, hàm compare_data sẽ trả về “phương pháp 2”
        # '''
        def compare_data(df_original, df_new, numrecords_compare= 500, threshold= 0.5):

            # st.markdown(f"Final threshold: The current policy on threshold value is **{threshold}**")

            # Lấy ngẫu nhiên 500 dòng từ dữ liệu gốc
            df_original_sample = df_original.sample(n= numrecords_compare)
            
            # Tính số lượng khác nhau của mỗi cột
            diff_counts = {}
            for column in ["order_dt", "order_products", "order_amount", "month"]:
                diff_counts[column] = df_original_sample[column].value_counts().mean() - df_new[column].value_counts().mean()
            
            # Kiểm tra xem liệu dữ liệu mới có tương tự với dữ liệu gốc hay không
            is_similar = all(value <= threshold for value in diff_counts.values())
            
            # 1:Sẽ áp dụng phương pháp 1 => Tính toán lại RFM cho toàn bộ dữ liệu | 2:Sẽ áp dụng phương pháp 2 => Chỉ tính RFM cho dữ liệu mới, sau đó tận dụng lại model đã trained
            return 2 if is_similar else 1

        # Hàm so sánh và phân tích vẽ ra xu hướng và từng giai đoạn của dữ liệu gốc và dữ liệu mới dựa trên ngưỡng cho phép thay đổi trên giao diện cấu hình bằng biến threshold
        def analyze_trend_and_seasonality_pro(df1, df2, count_dataUpload, columnname, title1, title2, threshold):
            # Tính giá trị trung bình của order_amount theo tháng
            average_1 = df1.groupby('month')[columnname].mean() # original data
            average_2 = df2.groupby('month')[columnname].mean() # uploaded data

            # Vẽ biểu đồ
            fig, axs = plt.subplots(1, 2, figsize=(20,6))
            axs[0].plot(average_2.index.astype(str), average_2.values, color= '#CD5B45')
            axs[0].set_xlabel('Tháng')
            axs[0].set_ylabel(f'Giá trị trung bình của {columnname}')
            axs[0].set_title(title2.replace("#", columnname), fontsize=16)
            axs[0].set_xticklabels(average_2.index.astype(str), rotation=90)

            axs[1].plot(average_1.index.astype(str), average_1.values, color= 'green')
            axs[1].set_xlabel('Tháng')
            axs[1].set_ylabel(f'Giá trị trung bình của {columnname}')
            axs[1].set_title(title1.replace("#", columnname), fontsize=16)
            axs[1].set_xticklabels(average_1.index.astype(str), rotation=90)

            st.pyplot(fig)
            # return "\t\tKết quả: Kiểm tra liệu dữ liệu mới thay đổi không đáng kể dữ liệu gốc => Có thể áp dụng phương pháp 2 " if is_similar else "\t\tKết quả: Kiểm tra liệu dữ liệu mới thay đổi rất đáng kể dữ liệu gốc => Phải áp dụng phương pháp 1"
            solution = compare_data(df1, df2, count_dataUpload , threshold)
            
            return solution
        
        def apply_solution():
            as_dict = {12:'Cần tặng thưởng cho nhóm này và khuyến khích khách hàng viết đánh giá về chúng ta. Khi có sản phẩm mới, hãy nhanh chóng giới thiệu cho nhóm này trước khi giới thiệu cho những người khác.', \
                        11:'Cần bán các sản phẩm có giá trị cao hơn cho họ, mời khách hàng viết đánh giá, khuyến khích khách hàng mời bạn bè trở thành khách hàng và gửi quà tặng như thẻ mua sắm tăng giá trị hoặc ghi lưu ý cảm ơn.', \
                        10:'Mời đăng ký làm thành viên, tăng cường cá nhân hóa trong việc giới thiệu sản phẩm hoặc dịch vụ, khuyến khích viết đánh giá và gửi quà tặng như thẻ mua sắm hoặc ghi thiệp tay mang ý nghĩa.', \
                        9:'Cung cấp dịch vụ hậu mãi để khách hàng cảm thấy tự tin về việc lựa chọn. Cung cấp thẻ quà tặng có giá trị không quá cao và bắt đầu xây dựng mối quan hệ.', \
                        8:'Liên hệ để tháo dỡ những khó khan, vướng mắc trong quá trình mua hàng và thanh toán. Bắt đầu xây dựng mối quan hệ bằng cách tìm hiểu những gì khách hàng thích và ngăn khách hàng để lại sản phẩm trong giỏ hàng mà không mua.', \
                        7:'Gọi điện cho nhóm này ngay lập tức để hiểu xem những lo lắng hoặc không hài lòng về điều gì và cách thức tương tác.', \
                        6:'Cố gắng liên hệ với nhóm này càng nhiều càng tốt để ngăn khách hàng quên về thương hiệu. Hiểu rõ hơn về khách hàng và đảm bảo khách hàng quay lại mua hàng.', \
                        5:'Sử dụng tin nhắn SMS hoặc email để liên hệ với khách hàng dựa trên những thứ khách hàng quan tâm, sau đó đợi xem kết quả như thế nào.', \
                        4:'Tạo ra các ưu đãi giới hạn thời gian để kích thích mua hàng lặp lại, cung cấp ưu đãi cá nhân phù hợp với sở thích hoặc nhu cầu của họ.', \
                        3:'Hãy đưa khách hàng trở lại với các khuyến mãi mạnh mẽ. Nỗ lực để liên hệ và không để khách hàng chuyển sang đối thủ.', \
                        2:'Gửi email hoặc tin nhắn để đảm bảo khách hàng không quên thương hiệu và tìm giải pháp cho các vấn đề của họ.', \
                        1:'Chạy các chiến dịch tiếp thị trực tuyến để tìm khách hàng mới. Nếu không thành công, hãy chấp nhận để khách hàng ra đi.', \
                        0:'Chạy các chiến dịch tiếp thị trực tuyến để tìm khách hàng mới. Nếu không thành công, hãy chấp nhận để khách hàng ra đi.'\
                    }
            return as_dict
        
        def rfm(df):
            df['order_dt'] = pd.to_datetime(df['order_dt'])  # convert to datetime
            max_date = df['order_dt'].max().date()

            Recency = lambda x: (max_date - x.max().date()).days
            # Frequency = lambda x: len(x.unique())
            Frequency  = lambda x: x.count()
            Monetary = lambda x: round(sum(x), 2)

            rfm_df = df.groupby('transaction_id').agg({'order_dt': Recency, 'order_products': Frequency, 'order_amount': Monetary})
            # Rename the columns of Dataframe
            rfm_df.columns = ['Recency', 'Frequency', 'Monetary']

            # Descending Sorting
            rfm_df = rfm_df.sort_values('Monetary', ascending=False)

            # Assume that you have a DataFrame `rfm_df` containing the RFM values for each customer
            rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])

            def join_rfm(x): return str(int(x['R_Score'])) + str(int(x['F_Score'])) + str(int(x['M_Score']))
            rfm_df['RFM_Segment'] = rfm_df.apply(join_rfm, axis= 1)
            # Combine the R_Score, F_Score and M_Score values to create a single RFM score for each customer
            rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

            # Define a dictionary to map each RFM score to a corresponding customer segment

            ## Rule 1
            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}

            # Map each RFM score to a corresponding customer segment using the dictionary
            rfm_df['Customer_Group'] = rfm_df['RFM_Score'].map(segment_dict)
            rfm_df['Solution'] = rfm_df['RFM_Score'].map(apply_solution())

            return rfm_df
        
        def concat_rfm(df_now, df_new):
            df = pd.concat([df_now, df_new], axis = 0 , ignore_index=True)
            df['order_dt'] = pd.to_datetime(df['order_dt'])  # convert to datetime
            max_date = df['order_dt'].max().date()

            Recency = lambda x: (max_date - x.max().date()).days
            # Frequency = lambda x: len(x.unique())
            Frequency  = lambda x: x.count()
            Monetary = lambda x: round(sum(x), 2)

            rfm_df = df.groupby('transaction_id').agg({'order_dt': Recency, 'order_products': Frequency, 'order_amount': Monetary})
            # Rename the columns of Dataframe
            rfm_df.columns = ['Recency', 'Frequency', 'Monetary']

            # Descending Sorting
            rfm_df = rfm_df.sort_values('Monetary', ascending=False)

            # Assume that you have a DataFrame `rfm_df` containing the RFM values for each customer
            rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])

            def join_rfm(x): return str(int(x['R_Score'])) + str(int(x['F_Score'])) + str(int(x['M_Score']))
            rfm_df['RFM_Segment'] = rfm_df.apply(join_rfm, axis= 1)
            # Combine the R_Score, F_Score and M_Score values to create a single RFM score for each customer
            rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

            # Define a dictionary to map each RFM score to a corresponding customer segment

            ## Rule 1
            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}

            # Map each RFM score to a corresponding customer segment using the dictionary
            rfm_df['Customer_Group'] = rfm_df['RFM_Score'].map(segment_dict)
            rfm_df['Solution'] = rfm_df['RFM_Score'].map(apply_solution())

            return rfm_df
        
        def rfm_pyspark_kmeans(rfm_df_new, path_filename):
            # DO THỜI ĐIỂM LÀM VIỆC VỚI STREAMLIT SERVER DOWN LIÊN TỤC NÊN PHẢI VIẾT THÊM HÀM NÀY RETRY TRONG _initialize_spark
            if 'spark' not in globals():
                result = _initialize_spark()
                if result is not None:
                    spark, sc = result
                else:
                    # Handle the case where _initialize_spark() returns None
                    # For example, you could print an error message and exit the program
                    print("Error: _initialize_spark() returned None")
                    sys.exit(1)
            else:
                spark = globals()['spark']
                sc = spark.sparkContext
                # Kiểm tra xem liệu SparkContext đã tồn tại hay chưa
                if 'sc' not in globals():
                    sc = SparkContext.getOrCreate()
                else:
                    sc = globals()['sc']
            # Load the model
            model_kmeans_lds9 = KMeansModel.load(path_filename)
            # prediction
            spark_df = spark.createDataFrame(rfm_df_new)

            features = ['Recency', 'Frequency','Monetary']
            vec_assambler  = VectorAssembler(inputCols= features, outputCol="features")
            pre_data = vec_assambler.transform(spark_df).persist()
            predictions = model_kmeans_lds9.transform(pre_data)
            df_temp = predictions.toPandas()
            # df_temp = df_temp.set_index('transaction_id')
            rfm_df_new['cluster_kmeans_lds9'] = df_temp['prediction'].astype('int64')
            return rfm_df_new
        
        def analyze_to_find_silhouette_wssse_values(request_data):
            # DO THỜI ĐIỂM LÀM VIỆC VỚI STREAMLIT SERVER DOWN LIÊN TỤC NÊN PHẢI VIẾT THÊM HÀM NÀY RETRY TRONG _initialize_spark
            if 'spark' not in globals():
                result = _initialize_spark()
                if result is not None:
                    spark, sc = result
                else:
                    # Handle the case where _initialize_spark() returns None
                    st.write("Error: _initialize_spark() returned None")
                    sys.exit(1)
            else:
                spark = globals()['spark']
                sc = spark.sparkContext
                # Kiểm tra xem liệu SparkContext đã tồn tại hay chưa
                if 'sc' not in globals():
                    sc = SparkContext.getOrCreate()
                else:
                    sc = globals()['sc']
            MIN_K = 2
            MAX_K = 10
            K = range(MIN_K, MAX_K + 1)
            spark_data = spark.createDataFrame(request_data[['Recency','Frequency','Monetary']].reset_index())            
            ##### Chuyển đổi dữ liệu
            features = ['Recency', 'Frequency','Monetary']
            vec_assambler  = VectorAssembler(inputCols= features, outputCol="features")
            final_data = vec_assambler.transform(spark_data)

            #### Trains a k-means model
            k_list = []
            silhouette_list = []
            wssse_list = []
            for k in range(2, MAX_K +1):
                kmeans = KMeans(featuresCol="features", k= k)
                model = kmeans.fit(final_data)

                #wssse
                wssse = model.summary.trainingCost
                wssse_list.append(wssse)
                k_list.append(k)

                #silhoutte
                predictions = model.transform(final_data)
                # Evaluate clustering by computing Silhouette score
                evaluator = ClusteringEvaluator()
                silhouette = evaluator.evaluate(predictions)
                silhouette_list.append(silhouette)
                st.write("With k =", k, "Set Sum of Squared Errors=\t" + str(round(wssse,2)) + "\t|\t" + "With k =" + str(k) + " Silhouette =\t" + str(round(silhouette,6)))

            #Visualization
            fig, ax = plt.subplots(1,2,figsize=(12,4))
            plt.subplot(1, 2, 1)
            plt.plot(k_list, wssse_list, 'gs-')
            plt.title("inertia (sum of squared errors) vs. number of clusters")
            plt.xticks(np.arange(2, MAX_K + 1, 1.0))
            plt.title('Plot of Elbow Method')
            plt.xlabel("number of clusters K")
            plt.ylabel("Within-Cluster Sum of Squared Error");

            plt.subplot(1, 2, 2)
            plt.plot(k_list,silhouette_list,'bx-')
            plt.xticks(np.arange(2, MAX_K + 1 , 1.0))
            plt.title('Silhouette analysis For Optimal k')
            plt.xlabel('number of clusters K') 
            plt.ylabel('Silhouette score') 
            st.pyplot(fig)
            return final_data

        def build_save_model_pyspark_kmeans(final_data, k= 5):
            start_time = datetime.now()
            kmeans = KMeans(featuresCol="features", k= k)
            model_kmeans_lds9 = kmeans.fit(final_data)
            ##Save model
            model_kmeans_lds9.write().overwrite().save(f'Project_{project_num}/models/new_rfm_kmeans_lds9')

        def visulization_download(request_data, filter_transID= [0], label_key1='Download data as CSV', label_key2='Download data as Excel', filename= 'Data_models_clustering_report (upload file)'):
        ## Visualization
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg_kmeans_lds9 = request_data.groupby('cluster_kmeans_lds9').agg({
                'Recency':'mean',
                'Frequency':'mean',
                'Monetary':['mean', 'count']}
                ).round(0)
            rfm_agg_kmeans_lds9.columns = rfm_agg_kmeans_lds9.columns.droplevel()
            rfm_agg_kmeans_lds9.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
            rfm_agg_kmeans_lds9['Percent'] = np.round((rfm_agg_kmeans_lds9['Count']/rfm_agg_kmeans_lds9.Count.sum())*100, 2)
            # Reset the index
            rfm_agg_kmeans_lds9 = rfm_agg_kmeans_lds9.reset_index()
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,8))
            fig = px.scatter(rfm_agg_kmeans_lds9, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="cluster_kmeans_lds9",
                    hover_name="cluster_kmeans_lds9", size_max=100)
            plt.title("Customers Segments",fontsize=16,fontweight="bold")
            st.plotly_chart(fig)
        ## End - Visualization
            request_data = request_data.sort_values(by='transaction_id', ascending=True)
            request_data = request_data.reset_index()
            sum_request_data= request_data.copy()      # cho sheetname= 'Sum_&_segmentation_of_customers'
            if len(filter_transID) > 1: # Có truyền list transaction_id để lọc lại chỉ lấy dữ liệu đúng trong file upload lên phân tích
                request_data= request_data[~request_data['transaction_id'].isin(filter_transID)]
            st.dataframe(request_data.head(20).style.apply(highlight_rows_even_odd_2, axis=1))           

            col1, col2, col3, col4 = st.columns(4) 
            @st.cache_data
            def convert_df_csv(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv(header=True, index=False, encoding='utf-8')
            csv = convert_df_csv(request_data)
            # Button download
            col2.download_button(
                key= label_key1,
                label= 'Download data as CSV',
                data=csv,
                file_name= f'{filename}.csv',
                mime='text/csv',
            )

            # @st.cache_data
            def convert_df_excel():
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                with pd.ExcelWriter(f'Project_{project_num}/Export_Data/{filename}.xlsx') as writer:
                    sum_request_data.to_excel(writer, sheet_name='Sum_customer_segmentation',index=False)
                    rfm_agg_kmeans_lds9.to_excel(writer, sheet_name='Information_for_each_cluster', index=False)
                    if len(sum_request_data) > len(request_data): # Lấy đúng dữ liệu file upload lên ra riêng 1 sheet.
                        request_data.to_excel(writer, sheet_name='Uploaded_customer_segmentation', index=False)
                # Đọc tệp như một chuỗi byte
                with open(f'Project_{project_num}/Export_Data/{filename}.xlsx', 'rb') as f:
                    bytes_data = f.read()
                return bytes_data
            
            # Button download
            col3.download_button(
                key= label_key2,
                label= 'Download data as Excel',
                data= convert_df_excel(),
                file_name=f'{filename}.xlsx',
                mime='application/octet-stream',
                )
        def correctly_checkbox():
            # Kiểm tra và chỉnh lại giá trị cho checkbox
            if apply_solution1:
                st.session_state.apply_solution2 = False
            elif apply_solution2:
                st.session_state.apply_solution1 = False
                ## END - FUNCTIONS ##

        with st.expander('**APPROACH METHODS FOR 12 CUSTOMER GROUPS**'):
            st.image(f'Project_{project_num}/images/Phuong_phap_tiep_can_12_nhom_kh.png')     

        with st.expander('**CREATE TEST CASES**'):
            c1, c2, c3 = st.columns(3)
            # Hiển thị tên trường với một số căn chỉnh
            c1.markdown('##')
            c1.markdown('**Set the threshold value to:**')
            threshold = c2.slider('', min_value=0.01, max_value=1.0, value=0.5, format='%.1f')
            # st.write('Values:', threshold)
            c3.markdown('##')
            on_create_new_test_cases = c3.toggle(':orange[**Create data**]', value= False)
            if on_create_new_test_cases:
                # Tạo dữ liệu thay đổi không đáng kể và phân tích xu hướng và từng giai đoạn
                print("Dữ liệu thay đổi không đáng kể:")
                similar_data = significantly_similar_data(df_cdnow, threshold, save_file= 'yes')

                # Tạo dữ liệu thay đổi rất đáng kể và phân tích xu hướng và từng giai đoạn
                print("Dữ liệu thay đổi rất đáng kể:")
                dissimilar_data = significantly_dissimilar_data(df_cdnow, threshold, save_file= 'yes')

        # st.markdown('##### **Customer Segmentation area**')
        st.markdown('###')
        t1, t2 = st.tabs(["## **Quick analysis of customers having new transactions**\n**in the system**", "## **Upload the file, analyze the data, segment the customers,**\n**and propose a plan**"])
        with t1: # INPUT INFOR
            with st.expander('**The default parameters of the application** ***(adjust if any)***'):
                select_system = st.selectbox('**Select the system containing the transactions you want to analyze for rfm**', ['System_1', 'System_2'])
                col1, col2, col3 = st.columns(3)
                col1.markdown('####')
                col1.markdown('**Adjust the number of transactions for analysis**')
                num_rows = col2.number_input('', min_value=2, max_value= 50000, value=50)

            if 'clicked' not in st.session_state:
                st.session_state.clicked = False
            col1, col2, col3 = st.columns(3)
            def click_button():
                st.session_state.clicked = True
            col2.button('Start analyzing and evaluating', on_click=click_button)
            def reset_state():
                st.session_state.clicked = False
                st.session_state.pop('data_system', None)
                st.session_state.pop('tab1_rfm', None)
                st.session_state.pop('filter_trans_id', None)
            c3.markdown("#")
            c3.markdown("##")
            c3.markdown("###")
            choice = col3.checkbox('Reset status')
            if choice:
                reset_state()

            if st.session_state.clicked:

                def text_field(label, columns=None, **input_params):
                    c1, c2 = st.columns(columns or [1, 4])

                    # Hiển thị tên trường với một số căn chỉnh
                    c1.markdown("##")
                    c1.markdown(label)

                    # Đặt một tham số khóa mặc định để tránh lỗi khóa trùng lặp
                    input_params.setdefault("key", label)
                    # Chuyển tiếp các tham số đầu vào văn bản
                    return c2.text_input(label="", **input_params,label_visibility="hidden")

                # Kiểm tra xem customer_id có phải là một chuỗi rỗng hay không và thay thế nó bằng một giá trị mới
                if 'filter_trans_id' not in st.session_state or not st.session_state.filter_trans_id:
                    st.session_state.filter_trans_id = '0'
                elif st.session_state.filter_trans_id == '':
                    st.session_state.filter_trans_id = str(0)
                else:
                    # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
                    filter_trans_id = st.session_state['filter_trans_id']
                def callback():
                    st.text(st.session_state.filter_trans_id)
                    st.session_state.clicked = True
                
                # Gọi hàm text_field để tạo widget text_input
                filter_trans_id = text_field('**QUICK FILTER**', value=str(st.session_state.filter_trans_id).strip().replace(",",""), on_change=callback) #, key='key', autocomplete=None
                if filter_trans_id != '':
                    t0 = datetime.now()
                    ## THỜI GIAN CÓ HẠN NÊN TA SẼ HAND CODE ĐÂY: chỉ có 2 đối tác
                    if select_system == 'System_1':
                        data_system = pd.read_table(f'Project_{project_num}/Data/system_1/transactions_data.txt', names= ['transaction_id','order_dt','order_products','order_amount','month'], sep=' ')
                    else:
                        data_system = pd.read_table(f'Project_{project_num}/Data/system_2/transactions_data.txt', names= ['transaction_id','order_dt','order_products','order_amount','month'], sep=' ')
                    
                    # Lấy ra duy nhất transaction_ids
                    unique_ids = data_system['transaction_id'].unique()

                    # Lấy ngẫu nhiên num_rows transaction_id
                    random_ids = np.random.choice(unique_ids, size= num_rows)

                    # st.write(f'Giá trị filter transaction: {filter_trans_id}') # Debug
                    if 'data_system' not in st.session_state:
                        # Đảm bảo đọc đúng toàn bộ transaction unique theo đúng số lượng yêu cầu
                        data_system = data_system[data_system['transaction_id'].isin(random_ids)]
                        st.session_state['data_system'] = data_system
                    else:
                        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
                        data_system = st.session_state['data_system']

                    if 'tab1_rfm' not in st.session_state:
                        tab1_rfm = rfm(data_system)
                        st.session_state['tab1_rfm'] = tab1_rfm
                    else:
                        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
                        tab1_rfm = st.session_state['tab1_rfm']

                    if filter_trans_id == '0': # Lấy tất cả
                        data_system_filtered = data_system
                        tab1_rfm_filtered = tab1_rfm.reset_index()
                        
                        # Display filtered dataframes
                        # st.markdown("**Transaction**")
                        # st.dataframe(data_system_filtered)

                    else: # KH chọn lại giá trị trong selectbox
                        data_system_filtered = data_system.loc[data_system['transaction_id'] == int(filter_trans_id.strip().replace(",",""))]
                        tab1_rfm_filtered = tab1_rfm.reset_index()
                        tab1_rfm_filtered = tab1_rfm_filtered.loc[tab1_rfm_filtered['transaction_id'] == int(filter_trans_id.strip().replace(",",""))]
                        
                        # Display filtered dataframes
                        # st.markdown("**Transaction**")
                        # st.dataframe(data_system_filtered)

                    st.markdown("**RFM**")
                    st.dataframe(tab1_rfm_filtered.style.apply(highlight_rows_even_odd_2, axis=1))
                    col1, col2, col3 = st.columns(3)
                    if col2.button('Apply PySpark KMeans Export file', on_click=click_button):
                        t0_rfm_df_new = datetime.now()
                        rfm_df_new= rfm_pyspark_kmeans(tab1_rfm_filtered, f'Project_{project_num}/models/rfm_kmeans_lds9')
                        st.write("**RESULTS MODEL ANALYZED AND EVALUATED**", datetime.now() - t0_rfm_df_new)
                        visulization_download(rfm_df_new, [0],  label_key1='Download data as CSV', label_key2='Download data as Excel', filename= 'Data_models_clustering_report')

                    
        with t2: # UPLOAD INFOR
            # Upload file
            uploaded_file = st.file_uploader("Choose file", type=['txt'])
            data_upload = pd.DataFrame(columns=["transaction_id", "order_dt", "order_products", "order_amount", "month"])
            if uploaded_file is not None:
                data_upload = pd.read_csv(uploaded_file, sep=' ', encoding='utf-8', names=data_upload.columns)
                if len(data_upload) > 1:
                    # Khởi tạo giá trị ban đầu cho session state
                    if 'apply_solution1' not in st.session_state:
                        st.session_state.apply_solution1 = True
                    if 'apply_solution2' not in st.session_state:
                        st.session_state.apply_solution2 = False

                    data_upload.to_csv(f'Project_{project_num}/Data/Upload/data_upload.txt', index=False,  header=True)
                    data_upload['order_dt'] = pd.to_datetime(data_upload.order_dt,format="%Y-%m-%d")
                    data_upload['order_dt'] = data_upload['order_dt'].astype('datetime64[ns]')
                    data_upload['month'] = data_upload.order_dt.values.astype('datetime64[M]')
                    c1, c2 = st.columns(2)
                    c1.markdown('**Uploaded data**')
                    # Hiển thị DataFrame
                    c1.dataframe(data_upload.head(500).style.apply(highlight_rows_even_odd_2, axis=1))

                    c2.markdown('**Original data**')                    
                    c2.dataframe(df_cdnow.head(500).style.apply(highlight_rows_even_odd_1, axis=1))

                    r_solution= analyze_trend_and_seasonality_pro(df_cdnow, data_upload, len(data_upload), 'order_products', 'Xu hướng giá trị trung bình theo tháng (original data)', 'Xu hướng giá trị trung bình theo tháng (uploaded data)', threshold)

                    # st.markdown("#")
                    # Create 2 columns
                    # Kết quả trả về 1 hay 2 => thì tự động set checkbox phương pháp 1 và ngược lại
                    if r_solution == 1:

                        st.markdown(separator_html, unsafe_allow_html=True)
                        st.write(':blue[**HỆ THỐNG THẨM ĐỊNH: DỮ LIỆU MỚI THAY ĐỔI  :green[***KHÔNG ĐÁNG KỂ***]  SO VỚI DỮ LIỆU GỐC**]' if r_solution == 2 else ':blue[**HỆ THỐNG THẨM ĐỊNH: DỮ LIỆU MỚI THAY ĐỔI  :red[***RẤT ĐÁNG KỂ***]  SO VỚI DỮ LIỆU GỐC**]')
                        st.markdown(separator_html, unsafe_allow_html=True)
                        st.markdown("##")
                        c1, c2 = st.columns(2)
                        c1.image('images/tham_dinh.png', width= 70,caption='')
                        with c1.expander('Giải pháp'):
                            st.markdown("I.   Tổng hợp bao gồm dữ liệu cũ và mới đưa lên")
                            st.markdown("II.  Tính lại RFM cho dữ liệu tổng hợp bước I. mang lại kết quả chính xác hơn")
                            st.markdown("III. Huấn luyện lại model với bộ dữ liệu\nKết quả phân cụm có thể ổn định hơn \ndo sự biến động trong rfm_new \ncó thể được làm mịn bởi rfm ")
                        with c1.expander('Thỏa thuận dịch vụ'):
                            st.markdown("I.   Hệ thống sẽ....")
                            st.markdown("II.  Thời gian....")
                            st.markdown("III. Những lợi ích....")
                            st.markdown("IV. Những rủi ro....")
                            st.markdown("V. Những đồng thuận....")
                        st.session_state.apply_solution1 = True
                        st.session_state.apply_solution2 = False
                    else:
                        st.markdown(separator_html, unsafe_allow_html=True)
                        st.write(':blue[**HỆ THỐNG THẨM ĐỊNH: DỮ LIỆU MỚI THAY ĐỔI  :green[***KHÔNG ĐÁNG KỂ***]  SO VỚI DỮ LIỆU GỐC**]' if r_solution == 2 else ':blue[**HỆ THỐNG THẨM ĐỊNH: DỮ LIỆU MỚI THAY ĐỔI  :red[***RẤT ĐÁNG KỂ***]  SO VỚI DỮ LIỆU GỐC**]')
                        st.markdown(separator_html, unsafe_allow_html=True)
                        
                        st.markdown("##")
                        c1, c2 = st.columns(2)                        
                        c2.image('images/tham_dinh_2.png', width= 70,caption='')
                        
                        st.session_state.apply_solution1 = False
                        st.session_state.apply_solution2 = True
                        with c2.expander('Giải pháp'):
                            st.markdown("I.   Tổng hợp bao gồm dữ liệu cũ và mới đưa lên")
                            st.markdown("II.  Tính lại RFM cho dữ liệu tổng hợp bước I. mang lại kết quả chính xác hơn")
                            st.markdown("III. Dùng lại model đã huấn luyện với bộ dữ liệu II.")
                        with c2.expander('Thỏa thuận dịch vụ'):
                            st.markdown("I.   Hệ thống sẽ....")
                            st.markdown("II.  Thời gian....")
                            st.markdown("III. Những lợi ích....")
                            st.markdown("IV. Những rủi ro....")
                            st.markdown("V. Những đồng thuận....")
                    col1, col2 = st.columns(2)
                    apply_solution1 = col1.checkbox(label='FULL PROCESS METHOD',value= st.session_state.apply_solution1)
                    apply_solution2 = col2.checkbox(label='QUICK METHOD',value= st.session_state.apply_solution2)
                    c1, c2, c3 = st.columns(3)
                    # Hiển thị tên trường với một số căn chỉnh
                    c1.markdown('##')
                    c1.markdown('**Please choose how many clusters you want:**')
                    # c2.markdown('##')
                    select_k = c2.slider('', min_value=2, max_value=10, value=4, format='%.0f')
                    c3.markdown('##')                   
                    
                   
                    col1, col2, col3 = st.columns(3)
                    col2.markdown("###")
                    button_clicked = col2.button('**AGREE TO THE TERMS AGREE TO PROCEED**')
                    st.write("-------------------")

                    # st.write('FULL PROCESS METHOD',st.session_state.apply_solution1)  # Debug
                    # st.write('QUICK METHOD',st.session_state.apply_solution2)         # Debug

                    if  button_clicked and st.session_state.apply_solution1 == True:
                        tab2_concat_rfm = concat_rfm(df_cdnow, data_upload)
                        
                        t0_rfm_df_new = datetime.now()
                        ### START -DO CHỈ CÓ 15 PHÚT TRÌNH BÀY => CHỈ CHỤP HÌNH KẾT QUẢ CỦA DÒNG CODE NÀY, TẠM THỜI ĐÓNG LẠI 
                        
                        # final_data= analyze_to_find_silhouette_wssse_values(tab2_concat_rfm)
                        # build_save_model_pyspark_kmeans(final_data, select_k)
                        # rfm_df_new= rfm_pyspark_kmeans(tab2_concat_rfm, f'Project_{project_num}/models/new_rfm_kmeans_lds9')
                        # st.write("**BUILDED AND SAVED NEW MODEL**", datetime.now() - t0_rfm_df_new)
                        # visulization_download(rfm_df_new, df_cdnow['transaction_id'],  label_key1='Download data as CSV_3', label_key2='Download data as Excel_3', filename= 'Data_new_models_clustering_report (upload file)')
                        
                        ### END- DO CHỈ CÓ 15 PHÚT TRÌNH BÀY => CHỈ CHỤP HÌNH KẾT QUẢ CỦA DÒNG CODE NÀY, TẠM THỜI ĐÓNG LẠI
                        
                        # DO ĐÓNG DÒNG CODE TRÊN PHẢI CODE DƯ CHỖ NÀY LẤY NHANH final_data
                        # DO THỜI ĐIỂM LÀM VIỆC VỚI STREAMLIT SERVER DOWN LIÊN TỤC NÊN PHẢI VIẾT THÊM HÀM NÀY RETRY TRONG _initialize_spark
                        if 'spark' not in globals():
                            result = _initialize_spark()
                            if result is not None:
                                spark, sc = result
                            else:
                                # Handle the case where _initialize_spark() returns None
                                st.write("Error: _initialize_spark() returned None")
                                sys.exit(1)
                        else:
                            spark = globals()['spark']
                            sc = spark.sparkContext
                            # Kiểm tra xem liệu SparkContext đã tồn tại hay chưa
                            if 'sc' not in globals():
                                sc = SparkContext.getOrCreate()
                            else:
                                sc = globals()['sc']
                        spark_data = spark.createDataFrame(tab2_concat_rfm[['Recency','Frequency','Monetary']].reset_index())            
                        ##### Chuyển đổi dữ liệu
                        features = ['Recency', 'Frequency','Monetary']
                        vec_assambler  = VectorAssembler(inputCols= features, outputCol="features")
                        final_data = vec_assambler.transform(spark_data)
                        st.image(f'Project_{project_num}/models/analyze_to_find_silhouette_wssse_values.png',caption='')
                        # st.write("**Analyze to find silhouette, wssse values**", datetime.now() - t0_rfm_df_new)
                        build_save_model_pyspark_kmeans(final_data, select_k)

                        rfm_df_new= rfm_pyspark_kmeans(tab2_concat_rfm, f'Project_{project_num}/models/new_rfm_kmeans_lds9')
                        st.write(f"**BUILDED AND SAVED NEW MODEL WITH K = {select_k}**", datetime.now() - t0_rfm_df_new)
                        visulization_download(rfm_df_new, df_cdnow['transaction_id'],  label_key1='Download data as CSV_3', label_key2='Download data as Excel_3', filename= 'Data_new_models_clustering_report (upload file)')
                    
                        

                    elif button_clicked and st.session_state.apply_solution2 == True:
                        tab2_concat_rfm = concat_rfm(df_cdnow, data_upload)
                        
                        t0_rfm_df_new = datetime.now()
                        rfm_df_new= rfm_pyspark_kmeans(tab2_concat_rfm, f'Project_{project_num}/models/rfm_kmeans_lds9')
                        st.write("**RESULTS MODEL ANALYZED AND EVALUATED**", datetime.now() - t0_rfm_df_new)
                        visulization_download(rfm_df_new, df_cdnow['transaction_id'],  label_key1='Download data as CSV_2', label_key2='Download data as Excel_2', filename= 'Data_models_clustering_report (upload file)')


            # on_significantly_similar_dissimilar = st.toggle(':orange[**Interpret significantly similar data and significantly dissimilar data**]',value= False)
            # if on_significantly_similar_dissimilar:
            #     show_info='''
            #         FULL PROCESS METHOD: \nTính toán lại RFM cho toàn bộ dữ liệu (bao gồm dữ liệu cũ và mới) \nmang lại kết quả chính xác tốt nếu phân phối của dữ liệu thay đổi theo thời gian.\nĐồng nghĩa, điều này sẽ tốn kém về mặt tính toán với kích thước dữ liệu lớn hơn.
            #         \nQUICK METHOD: \nTận dụng lại mô hình đã được huấn luyện trước đó có thể tiết kiệm thời gian \nvà tài nguyên tính toán.
            #         '''
            #     st.code(show_info, language='python') 
        ## END - toggle on_significantly_similar_dissimilar
    else:
        print("Do something")
    # st.write(f'Total run time Project_{project_num} is: {str(datetime.now() - first_step)}')
elif project_num == 2:
    # Hiển thị tên của dự án 
    st.subheader("PROJECT 2: RECOMMENDATION SYSTEM")
    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Prediction'])
    
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':
        st.image(f'Project_{project_num}/images/a{project_num}.jpg')
    elif step == 'Preprocessing + EDA':
        st.image(f'Project_{project_num}/images/b{project_num}.jpg')
    elif step == 'Applicable models':
        st.image(f'Project_{project_num}/images/c{project_num}.jpg')
    elif step == 'Prediction':
        st.image(f'Project_{project_num}/images/d{project_num}.jpg')
else:
    # Hiển thị tên của dự án 
    st.subheader("PROJECT 3: SENTIMENT ANALYSIS")
    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Prediction'])
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':
        st.image(f'Project_{project_num}/images/a{project_num}.jpg')
    else:
        print("Do something")
