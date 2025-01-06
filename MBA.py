import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import io
import csv

st.title("Analisis Apriori dan Pola Antar Produk")

# Inisialisasi session state untuk menyimpan data dan hasil analisis
if "data" not in st.session_state:
    st.session_state.data = None
if "rules" not in st.session_state:
    st.session_state.rules = None
if "frequent_itemsets" not in st.session_state:
    st.session_state.frequent_itemsets = None
if "min_support" not in st.session_state:
    st.session_state.min_support = 0.5  # Default nilai minimum support
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = []
if "transaksi_col" not in st.session_state:
    st.session_state.transaksi_col = None
if "nama_barang_col" not in st.session_state:
    st.session_state.nama_barang_col = None

st.sidebar.title("Menu Utama")
st.sidebar.markdown(
    "**SELAMAT DATANG!**\n\n"
    "Silahkan Tentukan Pilihan Anda: **Analisis Apriori** dan **Pola Antar Produk**."
)
st.sidebar.markdown("---")

menu = st.sidebar.selectbox("Pilih Menu:", ["📊 Analisis Apriori", "🤖 Pola Antar Produk"])

# Tombol untuk menyelesaikan analisis dan menghapus data yang tersimpan
if st.sidebar.button("Selesai"):
    st.session_state.data = None
    st.session_state.rules = None
    st.session_state.frequent_itemsets = None
    st.session_state.min_support = 0.5
    st.session_state.selected_columns = []
    st.session_state.transaksi_col = None
    st.session_state.nama_barang_col = None
    st.success("Data dan hasil analisis telah dihapus.")

if menu == "📊 Analisis Apriori":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None and st.session_state.data is None:
        content = uploaded_file.getvalue().decode('utf-8')
        try:
            dialect = csv.Sniffer().sniff(content)
            df = pd.read_csv(uploaded_file, sep=dialect.delimiter, header=0)
        except csv.Error:
            st.error("Gagal mendeteksi delimiter. Pastikan file CSV Anda valid.")
            df = pd.read_csv(uploaded_file, header=0)  # baca dengan delimiter default (,)
        st.session_state.data = df
        st.write("Kolom yang tersedia:")
        columns = df.columns.tolist()
        st.write(columns)
    elif st.session_state.data is not None:
        st.write("Dataset yang saat ini digunakan:")
        st.write(st.session_state.data)
        columns = st.session_state.data.columns.tolist()

    if st.session_state.data is not None:
        # Tambahkan checkbox untuk memilih apakah preprocessing diperlukan
        perform_preprocessing = st.checkbox("Lakukan Preprocessing ( hapus duplikasi & nilai yang hilang )", value=False)

        if perform_preprocessing:
            # Preprocessing data terlebih dahulu
            df = st.session_state.data.drop_duplicates()  # Menghapus duplikasi
            df = df.dropna(subset=[col for col in columns])  # Menghapus missing values dari semua kolom
            st.write("Data setelah pembersihan:")
            st.write(df)
        else:
            df = st.session_state.data

        # Pemilihan kolom setelah pembersihan
        selected_columns = st.multiselect(
            "Pilih kolom yang ingin digunakan:",
            options=columns,
            default=st.session_state.selected_columns
        )
        if selected_columns:
            st.session_state.selected_columns = selected_columns

        if len(st.session_state.selected_columns) < 2:
            st.error("Pilih setidaknya dua kolom: satu kolom transaksi dan satu kolom nama barang.")
        else:
            st.session_state.transaksi_col = st.selectbox(
                "Pilih kolom transaksi:",
                options=st.session_state.selected_columns,
                index=st.session_state.selected_columns.index(st.session_state.transaksi_col)
                if st.session_state.transaksi_col in st.session_state.selected_columns else 0
            )
            st.session_state.nama_barang_col = st.selectbox(
                "Pilih kolom nama barang:",
                options=[col for col in st.session_state.selected_columns if col != st.session_state.transaksi_col],
                index=0
            )

            st.header("Atur Parameter Apriori")
            st.session_state.min_support = st.slider(
                "Minimum Support",
                0.01,
                1.0,
                st.session_state.min_support
            )

            if st.button("Jalankan Apriori"):
                # Transformasi Data
                transaksi = df.groupby(st.session_state.transaksi_col)[
                    st.session_state.nama_barang_col].apply(list).tolist()
                st.write("Data Transaksi:")
                st.write(transaksi)

                # Encoding Data dengan TransactionEncoder
                te = TransactionEncoder()
                te_ary = te.fit(transaksi).transform(transaksi)
                df_transaksi = pd.DataFrame(te_ary, columns=te.columns_)

                # Frequent itemsets
                frequent_itemsets = apriori(df_transaksi, min_support=st.session_state.min_support, use_colnames=True)
                st.session_state.frequent_itemsets = frequent_itemsets  # Simpan hasil frequent itemsets
                st.write("Frequent Itemset:")
                st.write(frequent_itemsets)

                # Visualisasi Frequent Itemsets
                st.header("Visualisasi Frequent Itemsets")
                plt.figure(figsize=(10, 6))
                sns.barplot(
                        x=st.session_state.frequent_itemsets['support'], 
                        y=st.session_state.frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
                    )
                plt.title('Frequent Itemsets and Their Support')
                plt.xlabel('Support')
                plt.ylabel('Itemsets')

                # Simpan gambar visualisasi ke dalam buffer untuk disimpan di session_state
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                img_buf.seek(0)
                st.session_state.frequent_itemsets_img = img_buf  # Simpan gambar

                st.pyplot(plt.gcf())

                # Association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                st.session_state.rules = rules  # Simpan hasil association rules
                st.write("Association Rules:")
                st.write(rules)

                # Visualisasi Association Rules (Confidence vs Lift)
                st.header("Visualisasi Association Rules(Confidence vs Lift)")
                if not st.session_state.rules.empty:
                    plt.figure(figsize=(10, 6))
                    scatter = sns.scatterplot(
                        data=st.session_state.rules,
                        x='confidence',
                        y='lift',
                        size='support',
                        sizes=(50, 500),
                        alpha=0.7,
                        legend=False
                        )

                    # Tambahkan anotasi untuk beberapa aturan
                    for i in range(len(st.session_state.rules)):
                        plt.text(
                            st.session_state.rules['confidence'].iloc[i],
                            st.session_state.rules['lift'].iloc[i],
                            f"A{i+1}",
                            fontsize=9,
                            alpha=0.7
                        )

                    # deskripsi aturan
                    deskripsi = "\n".join([f"{i+1}: {', '.join(list(rules['antecedents'].iloc[i]))} → {', '.join(list(rules['consequents'].iloc[i]))}" 
                                            for i in range(len(rules))])

                    # keterangan di bawah grafik
                    plt.figtext(0.1, -0.2, deskripsi, fontsize=10, ha='left', bbox=dict(facecolor='white', alpha=0.7))


                    plt.title('Association Rules (Confidence vs Lift)', fontsize=14)
                    plt.xlabel('Confidence', fontsize=12)
                    plt.ylabel('Lift', fontsize=12)
                    plt.xlim(0, 1)
                    plt.ylim(0, max(st.session_state.rules['lift']) * 1.1)


                    # Simpan gambar visualisasi ke dalam buffer untuk disimpan di session_state
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='png')
                    img_buf.seek(0)
                    st.session_state.association_rules_img = img_buf  # Simpan gambar


                    st.pyplot(plt.gcf())

                else:
                    st.warning("Tidak ada aturan asosiasi yang tersedia untuk divisualisasikan.")
                
            else :
                # Menampilkan hasil Apriori jika sudah tersedia
                if st.session_state.frequent_itemsets is not None:
                    st.header("Hasil Frequent Itemsets")
                    st.write(st.session_state.frequent_itemsets)
                    # Menampilkan gambar visualisasi Frequent Itemsets
                if 'frequent_itemsets_img' in st.session_state:
                    st.image(st.session_state.frequent_itemsets_img)

                if st.session_state.rules is not None:
                    st.header("Hasil Association Rules")
                    st.write(st.session_state.rules)
                    # Menampilkan gambar visualisasi Association Rules
                if 'association_rules_img' in st.session_state:
                    st.image(st.session_state.association_rules_img)


if menu == "🤖 Pola Antar Produk":
        st.header("Pola Antar Produk")
        if st.session_state.rules is None:
            st.error("Silakan jalankan analisis Apriori terlebih dahulu!")
        else:
            rules = st.session_state.rules
            st.write("Aturan asosiasi tersedia, silahkan lanjutkan.")

            # Memilih produk untuk rekomendasi
            available_items = sorted(set().union(*rules['antecedents'].apply(lambda x: list(x))))
            selected_items = st.multiselect("Pilih produk:", options=available_items)

            # Mencari rekomendasi berdasarkan aturan asosiasi
            if selected_items:
                rekomendasi = rules[rules['antecedents'].apply(lambda x: set(selected_items).issubset(x))]
                if not rekomendasi.empty:
                    # mengurutkan berdasarkan confidence
                    rekomendasi = rekomendasi.sort_values(by='confidence', ascending=False)

                    # Menyiapkan DataFrame untuk ditampilkan sebagai tabel
                    rekomendasi_tabel = pd.DataFrame({
                        'Produk yang Direkomendasikan': rekomendasi.apply(
                            lambda row: f"jika pelanggan membeli ( {', '.join(list(row['antecedents']))} ), maka kemungkinan {row['confidence'] * 100:.0f}% pelanggan membeli ( {', '.join(list(row['consequents']))} ).", axis=1),
                            'Confidence': rekomendasi['confidence'].round(2),
                        'Lift': rekomendasi['lift'].round(2),
                    })

                    # Menampilkan tabel pola antar produk
                    st.write("Produk yang direkomendasikan berdasarkan barang yang Anda pilih:")
                    st.table(rekomendasi_tabel)  # Menampilkan dalam bentuk tabel

                else:
                    st.write("Tidak ada rekomendasi produk untuk pilihan ini.")


            else:
                st.info("Pilih produk terlebih dahulu untuk melihat rekomendasi.")
