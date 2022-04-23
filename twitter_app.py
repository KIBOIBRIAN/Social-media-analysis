from ast import If
from cProfile import label
from logging.config import fileConfig
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
import io
import sqlite3
import seaborn.apionly as sns
import re
from wordcloud import WordCloud

# Upload CSV data
with st.sidebar.info('1. Upload your CSV data'):
    uploaded_file = st.file_uploader("Upload  CSV files:")
# project header
st.header(' Twitter exploratory tool on best time to post ')

# Load a csv and define it as df
if uploaded_file is not None:
    @st.cache(allow_output_mutation=True)
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    twitter=load_csv()
    # Cleaning
    twitter.columns=twitter.columns.str.replace(' ', '_')
    # convert posted columns to dates
    twitter['Date_&_Time']=pd.to_datetime(twitter['Date_&_Time'])
    # create a new columns with the days posted, spliting date and time
    twitter['Day_posted']=twitter['Date_&_Time'].dt.day_name()
    twitter['Date']=twitter['Date_&_Time'].dt.date
    twitter['Time']=twitter['Date_&_Time'].dt.time

    twitter['Time']=pd.to_datetime(twitter['Time'].astype(str))  #convert time
    twitter['Time_'] = twitter['Time'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

    days_map={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    twitter['Days']=twitter['Day_posted'].map(days_map) 

    bins = [0,2,4,6,8,10,12,14,16,18,20,22,24]
    # add custom labels if desired
    labels = ['00:00-01:59','02:00-03:59','04:00-05:59','06:00-07:59','08:00-09:59','10:00-11:59','12:00-13:59','14:00-15:59','16:00-17:59','18:00-19:59','20:00-21:59','22:00-24:00']
    # add the bins to the dataframe
    twitter['Time_Bin'] = pd.cut(twitter.Time_.dt.hour, bins, labels=labels, right=False)

    Time_map={'00:00-01:59':1,'02:00-03:59':2,'04:00-05:59':3,'06:00-07:59':4,'08:00-09:59':5,'10:00-11:59':6,'12:00-13:59':7,'14:00-15:59':8,'16:00-17:59':9,'18:00-19:59':10,'20:00-21:59':11,'22:00-24:00':12}
    twitter['Time_posted']=twitter['Time_Bin'].map(Time_map)
    st.write('Time is split into bins of two hours. Each bin is represented with a specific number. Check each time range representation below for ease of visualization translation.')
    st.info("'00:00-01:59':1,'02:00-03:59':2,'04:00-05:59':3,'06:00-07:59':4,'08:00-09:59':5,'10:00-11:59':6,'12:00-13:59':7,'14:00-15:59':8,'16:00-17:59':9,'18:00-19:59':10,'20:00-21:59':11,'22:00-24:00':12")
    st.write('Impressions')
    piv_ = pd.pivot_table(twitter, values="Impressions",index=["Time_posted"], columns=["Day_posted"], fill_value=0)
   
    #plot pivot table as heatmap using seaborn
    plt.figure(figsize=(15,8))
    ax = sns.heatmap(piv_, cmap="OrRd", square=False)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90 )
    plt.title('Impressions Distribution ', size=20)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig=plt) 
   

    st.write('Engagements')
    piv_ = pd.pivot_table(twitter, values="Engagements",index=["Time_posted"], columns=["Day_posted"], fill_value=0)
    #plot pivot table as heatmap using seaborn
    plt.figure(figsize=(15,8))
    ax = sns.heatmap(piv_, cmap="OrRd", square=False)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90 )
    plt.title('Engagements Distribution ', size=20)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig=plt)
   

    st.write('See the most and least engaging posts:')
    if st.button('Show Most engaging posts'):
        st.write(twitter.nlargest(n=5, columns=['Impressions','Engagements']))
    if st.button('show Least Engaging posts'):
        st.write(twitter.nsmallest(n=5, columns=['Impressions','Engagements']))
    # Hashtags
    # st.write('sed hastags')
    # if st.button('Show Hashtags'):
    #     m=twitter['Tweet_text'].to_list()
    #     hashtags = [re.findall('#\w+', i) for i in m]
    #     print(hashtags)
    #     #convert list to string and generate
    #     unique_string=(" ").join(m)
    #     wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
    #     fig,ax=plt.subplots(figsize=(12,8))
    #     # plt.figure(figsize=(15,8))
    #     ax.imshow(wordcloud)
    #     plt.axis("off")
    #     st.pyplot(fig)
    

    # modelling
    import numpy as np 
    features=np.array(twitter[['Days','Time_posted']],dtype='float32')
    targets=np.array(twitter['Impressions'],dtype='float32')
    maxValLikes=max(targets)
    print('Max value of target {}'.format(maxValLikes))
    targets=targets/maxValLikes
 
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest =train_test_split(features, targets, test_size=0.1, random_state=42)
    stdSc= StandardScaler()
    xTrain=stdSc.fit_transform(xTrain)
    xTest=stdSc.transform(xTest)
    from sklearn.ensemble import GradientBoostingRegressor
    gbr=GradientBoostingRegressor()
    gbr.fit(xTrain, yTrain)
    predictions=gbr.predict(xTest)
    plt.scatter(yTest, predictions)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('True values')
    plt.ylabel('predicted values')
    plt.title('GardientRegressor')
    plt.plot(np.arange(0,0.4,0.01),np.arange(0,0.4,0.01),color='green')
    plt.grid(True)
    def PredictionsWithDayPosted(model, daycount, scaller, maxVal):
        followers = daycount * np.ones(12)
        time_posted = np.arange(1,13)
    # defining vector 
        featureVector = np.zeros((12, 2))
        featureVector[:, 0] = followers
        featureVector [:, 1] = time_posted
    # doing scalling
        featureVector = scaller.transform(featureVector)
        predictions = model.predict(featureVector)
        predictions = (maxValLikes * predictions).astype('int')
        plt.figure(figsize= (10, 10))
        plt.plot(time_posted, predictions)
        plt.style.use('seaborn-whitegrid')
        plt.scatter(time_posted, predictions, color = 'g')
        plt.grid(True)
        plt.xlabel('Time posted', size=20)
        plt.ylabel('Impressions', size=20)
        plt.title('Impressions progression with the given day and time posted', size=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    st.title('Predict best time to post')
    # col1, col2 = st.columns([1,1])

# with col1:
    if st.button('Monday'):
        PredictionsWithDayPosted(gbr, 1, stdSc, maxValLikes)
        st.pyplot(fig=plt) 
# with col2:    
    if st.button('Tuesday'):
        PredictionsWithDayPosted(gbr, 2, stdSc, maxValLikes)
        st.pyplot(fig=plt)

# col3, col4 = st.columns([1,1])
# with col3:
    if st.button('Wednesday'):
        PredictionsWithDayPosted(gbr, 3, stdSc, maxValLikes)
        st.pyplot(fig=plt) 
# with col4:    
    if st.button('Thursday'):
        PredictionsWithDayPosted(gbr, 4, stdSc, maxValLikes)
        st.pyplot(fig=plt) 
    if st.button('Friday'):
        PredictionsWithDayPosted(gbr, 5, stdSc, maxValLikes)
        st.pyplot(fig=plt) 

else:
    st.write('1. View post impressions and engagements heat map. See how post post are perfoming with relation to day and time posted. The will help determine the best day to post on twitter')
    st.write('2. View the best and worst performing posts.')
    st.write('3. Predict the  best time to post on a given day.')
        
        
