import pandas as pd
import streamlit as st 

from sqlalchemy import create_engine
from urllib.parse import quote
import joblib, pickle

model1 = pickle.load(open('svc_rcv.pkl', 'rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')


def predict_Y(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    
    clean = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())
    
    clean[list(clean.iloc[:,2:].columns)] = winsor.transform(clean[list(clean.iloc[:,2:].columns)])
    prediction = pd.DataFrame(model1.predict(clean), columns = ['isFraud_pred'])
    
    final = pd.concat([prediction, data], axis = 1)
        
    final.to_sql('svm_test', con = engine, if_exists = 'replace', chunksize = 10000, index = False)

    return final

def main():

    st.title("AML Fraud Detection")
    st.sidebar.title("AML Fraud Detection")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Letter prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Upload a file" , type = ['csv','xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)

    else:
        st.sidebar.warning("Upload a CSV or Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here", type = 'password')
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_Y(data, user, pw, db)
                           
        import seaborn as sns
        cm = sns.light_palette("yellow", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm).set_precision(2))

                           
if __name__=='__main__':
    main()














































