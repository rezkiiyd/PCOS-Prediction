import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from dotenv import load_dotenv
import kagglehub
import google.generativeai as genai
import plost
import altair as alt

st.set_page_config(page_title="Aplikasi Prediksi PCOS", layout="wide")
st.markdown("<style>body {font-family: 'Segoe UI';}</style>", unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key="AIzaSyAmVmRlcQmpRNBYRocTIiNdS4ffOg7QGEI")

# Load model gemini
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

def ask_about_pcos(prompt):
    system_instruction = (
        "Kamu adalah asisten kesehatan khusus yang hanya menjawab tentang PCOS (Polycystic Ovary Syndrome). "
        "Jika ada pertanyaan di luar topik PCOS, tolong tolak secara sopan dan arahkan kembali ke topik PCOS.\n\n"
    )
    full_prompt = system_instruction + prompt
    chat = model_gemini.start_chat()
    response = chat.send_message(full_prompt)
    return response.text

# Load model ML asli
@st.cache_resource
def load_model():
    path = "trained_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file '{path}' tidak ditemukan!")
        return None

model_ml = load_model()

# Load dataset dari Kaggle
path = kagglehub.dataset_download("samikshadalvi/pcos-diagnosis-dataset")
dataset_path = path + "/pcos_dataset.csv"
data = pd.read_csv(dataset_path)

# Inisialisasi chat
if "messages" not in st.session_state:
    st.session_state.messages = []

#MENU V HOME
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

page_bg_img = '''
<style>

header[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0); /* transparan */
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

if st.session_state.page == "home":
    st.markdown(
        """
    <div style="text-align: center;">
        <img src="https://imgur.com/bLC32Ik.jpg" style="width:100%; height:auto; margin-bottom:30px;">
    </div>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
        """
    <div style="text-align: center;">
        <img src="https://imgur.com/NVXK7O8.jpg" style="width:100%; height:auto; margin-bottom:20px;">
    </div>
    """,
    unsafe_allow_html=True
    )
    st.markdown("#### Pilih Menu:")
# Tombol-tombol navigasi
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Explore the PCOS dataset", use_container_width=True, key="info_btn"):
            st.session_state.page = "info"
            st.rerun()
    with col2:
        if st.button("🔍 Prediksi", use_container_width=True, key="prediksi_btn"):
            st.session_state.page = "predict"
            st.rerun()
    with col3:
        if st.button("💬 Chatbot", use_container_width=True, key="chatbot_btn"):
            st.session_state.page = "chatbot"
            st.rerun()


# Logika pindah halaman (contoh)

# Halaman Info
elif st.session_state.page == "info":
  def metric_card(label, value, unit=""):
    return f"""
    <div style="text-align: center; background-color: #FFFFFF; border-left: 0.5rem solid #9AD8E1;
                border: 1px solid #CCCCCC; border-radius: 10px; padding: 1rem;
                box-shadow: 0 0.15rem 1.75rem rgba(58, 59, 69, 0.15);">
        <div style="color: #6c757d; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">
            {label}
        </div>
        <div style="font-size: 2.2rem; font-weight: bold; color: #212529;">
            {value}
        </div>
        <div style="font-size: 1rem; color: #664d00; margin-top: 0.25rem;">
            {unit}
        </div>
    </div>
    """
 # Hitung jumlah diagnosis
  diagnosis_count = data['PCOS_Diagnosis'].value_counts().reset_index()
  diagnosis_count.columns = ['Diagnosis', 'Jumlah']
  diagnosis_count['Diagnosis'] = diagnosis_count['Diagnosis'].map({1: 'PCOS', 0: 'Tidak PCOS'})
  # Chart diagnosis di col2
  chart = alt.Chart(diagnosis_count).mark_bar(color='#b30000').encode(
    x=alt.X('Diagnosis', title=None),
    y=alt.Y('Jumlah', title= 'Jumlah'),
    tooltip=['Diagnosis', 'Jumlah']
  ).properties(
    width=200,
    height=150,
    #title="PCOS Diagnosis"
  )
  chart = chart.configure_axisX(labelAngle=0)


# Row A
  st.markdown('### Overview')
  col1, col2, col3 = st.columns(3)
  col1.markdown(metric_card("Jumlah Data", "1000", "pasien"), unsafe_allow_html=True)
  col2.altair_chart(chart, use_container_width=True)
  col3.markdown(metric_card("Rata-rata Usia ter Diagnosis", "30", "tahun"), unsafe_allow_html=True)
  st.markdown('---')
# Row B
  c1, c2 = st.columns((6,4))
  st.markdown(
    """
    <style>
    .markdown-text-container h3 {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
  )
  with c1:

    st.markdown('### Histogram')
    #st.caption('Pilih fitur untuk melihat distribusi berdasarkan diagnosis PCOS.')

    fitur_kde = st.selectbox("Pilih fitur untuk melihat distribusi berdasarkan diagnosis PCOS.",['BMI', 'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count', 'Age'],key="kde")

    fig, ax = plt.subplots(figsize=(4, 2))
    sns.histplot(data=data, x=fitur_kde, hue='PCOS_Diagnosis', kde=True, palette=['#009900', '#b30000'], bins=30, element='step', ax=ax)
    ax.set_title(f'Distribusi {fitur_kde} Berdasarkan Diagnosis PCOS')
    ax.set_xlabel(fitur_kde)
    ax.set_ylabel('Jumlah')
    st.pyplot(fig)

  with c2:
    st.markdown('### Box Plot')

    fitur_box = st.selectbox(
        "Pilih fitur untuk Box Plot:",
        ['Antral_Follicle_Count', 'BMI', 'Testosterone_Level(ng/dL)'],
        key="box"
    )

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=data, x='PCOS_Diagnosis', y=fitur_box, palette=['#bfff80', '#b30000'], ax=ax2)
    ax2.set_title(f'{fitur_box} vs PCOS Diagnosis')
    ax2.set_xlabel('PCOS Diagnosis')
    ax2.set_ylabel(fitur_box)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Tidak PCOS', 'PCOS'])
    st.pyplot(fig2)
  st.markdown('---')
# Row C
  st.markdown("## Korelasi Antar Variabel")

  col1, col2 = st.columns((6, 4))

  with col1:
    st.markdown("### Heatmap Korelasi")

    # Ambil kolom numerik yang relevan
    numerik_df = data[['Age', 'BMI', 'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count', 'PCOS_Diagnosis']]
    corr_matrix = numerik_df.corr()

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt=".2f", linewidths=0.5, ax=ax3)
    ax3.set_title("Matriks Korelasi Variabel")
    st.pyplot(fig3)

  with col2:
    st.markdown("### Insight")

    st.markdown("""
    - **BMI (Body Mass Index)** memiliki korelasi positif paling tinggi dengan PCOS Diagnosis, yaitu sebesar 0.38. Hal ini menunjukkan bahwa semakin tinggi BMI seseorang, semakin besar kemungkinan ia didiagnosis PCOS.
    - Korelasi positif yang dimiliki **Testosterone Level** dan **Jumlah Antral Follicle** menunjukan fakta bahwa wanita dengan PCOS cenderung memiliki kadar hormon androgen (testosteron) yang lebih tinggi dari normal dan wanita dengan jumlah folikel antral yang tinggi lebih berisiko mengalami gangguan ovulasi, yang merupakan salah satu gejala utama PCOS.
    - **Age** memiliki korelasi negatif terhadap PCOS. Usia bukanlah faktor yang berpengaruh secara signifikan terhadap kemungkinan diagnosis PCOS.
    """)
  if st.button("Back"):
    go_to("home")

# Halaman Prediksi
elif st.session_state.page == "predict":
    st.title("🔍 Prediksi PCOS")
    st.write("Masukkan informasi berikut untuk memprediksi kemungkinan PCOS:")

    usia = st.number_input("Usia (tahun)", 15, 50, 25)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 22.0)
    siklus = st.selectbox("Siklus Menstruasi", ["Teratur", "Tidak Teratur"])
    testosteron = st.number_input("Kadar Testosteron (ng/dL)", 20.0, 200.0, 50.0)
    folikel = st.slider("Jumlah Folikel Antral", 0, 50, 10)

    # Encoding siklus menstruasi
    siklus_encoded = 0 if siklus == "Teratur" else 1

    # Validasi input
    valid = True
    warning_msgs = []
    if not (15 <= usia <= 120):
        valid = False
        warning_msgs.append("Usia harus antara 15 sampai 50 tahun.")
    if not (10.0 <= bmi <= 60.0):
        valid = False
        warning_msgs.append("BMI harus antara 10.0 sampai 50.0 kg/m².")
    if not (20.0 <= testosteron <= 200.0):
        valid = False
        warning_msgs.append("Kadar Testosteron harus antara 20.0 sampai 200.0 ng/dL.")
    if not (0 <= folikel <= 50):
        valid = False
        warning_msgs.append("Jumlah Folikel Antral harus antara 0 sampai 50.")

    if not valid:
        for msg in warning_msgs:
            st.warning(msg)
        st.warning("Harap cek data kembali.")

    if st.button("Prediksi Sekarang"):
        if model_ml is None:
            st.error("Model belum berhasil dimuat, prediksi tidak dapat dilakukan.")
        else:
            if not valid:
                st.error("Input tidak valid, prediksi dibatalkan.")
            else:
                input_data = [[usia, bmi, siklus_encoded, testosteron, folikel]]
                prediksi = model_ml.predict(input_data)[0]
                hasil = "Kemungkinan PCOS" if prediksi == 1 else "Tidak Terindikasi PCOS"
                st.success(f"Hasil Prediksi: **{hasil}**")
    if st.button("Back", key="back_btn_prediksi"):
        go_to("home")
        st.rerun()

# Halaman Chatbot
elif st.session_state.page == "chatbot":
    if st.button("Back", key="back_btn_chatbot"):
        go_to("home")
        st.rerun()
    st.title("💬 Chatbot PCOS Assistant")
    st.write("Tanyakan apapun seputar PCOS. Asisten AI hanya menjawab pertanyaan yang berhubungan dengan PCOS.")
    st.write("**Disclaimer**: Informasi yang diberikan oleh chatbot ini tidak dapat dijadikan sebagai rujukan medis yang 100% terpecaya, dan harus divalidasi oleh dokter ahli.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan sesuatu tentang PCOS:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = ask_about_pcos(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
