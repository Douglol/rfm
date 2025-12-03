import duckdb
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# CONFIGURAÇÕES ================================

RAW_PATH = "C:/Users/Wow-x/OneDrive/Desktop/projeto_fidelizacao_olist/data_raw/"
PROCESSED_PATH = "C:/Users/Wow-x/OneDrive/Desktop/projeto_fidelizacao_olist/data_processed/"
DB_PATH = PROCESSED_PATH + "olist.db"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# 1. ETL - DUCKDB ================================

def create_tables(db):
    db.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id VARCHAR,
        customer_unique_id VARCHAR,
        customer_zip_code_prefix INTEGER,
        customer_city VARCHAR,
        customer_state VARCHAR
    );

    CREATE TABLE IF NOT EXISTS geolocation (
        geolocation_zip_code_prefix INTEGER,
        geolocation_lat DOUBLE,
        geolocation_lng DOUBLE,
        geolocation_city VARCHAR,
        geolocation_state VARCHAR
    );

    CREATE TABLE IF NOT EXISTS order_items (
        order_id VARCHAR,
        order_item_id INTEGER,
        product_id VARCHAR,
        seller_id VARCHAR,
        shipping_limit_date TIMESTAMP,
        price DOUBLE,
        freight_value DOUBLE
    );

    CREATE TABLE IF NOT EXISTS order_payments (
        order_id VARCHAR,
        payment_sequential INTEGER,
        payment_type VARCHAR,
        payment_installments INTEGER,
        payments_value DOUBLE
    );

    CREATE TABLE IF NOT EXISTS order_reviews (
        review_id VARCHAR,
        order_id VARCHAR,
        review_score INTEGER,
        review_comment_title VARCHAR,
        review_comment_message VARCHAR,
        review_creation_date TIMESTAMP,
        review_answer_timestamp TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id VARCHAR,
        customer_id VARCHAR,
        order_status VARCHAR,
        order_purchase_timestamp TIMESTAMP,
        order_approved_at TIMESTAMP,
        order_delivered_carrier_date TIMESTAMP,
        order_delivered_customer_date TIMESTAMP,
        order_estimated_delivery_date TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS products (
        product_id VARCHAR,
        product_category_name VARCHAR,
        product_name_lenght INTEGER,
        product_description_lenght INTEGER,
        product_photos_qty INTEGER,
        product_weight_g DOUBLE,
        product_lenght_cm DOUBLE,
        product_height_cm DOUBLE,
        product_width_cm DOUBLE
    );

    CREATE TABLE IF NOT EXISTS sellers (
        seller_id VARCHAR,
        seller_zip_code_prefix INTEGER,
        seller_city VARCHAR,
        seller_state VARCHAR
    );

    CREATE TABLE IF NOT EXISTS product_category_name_translation (
        product_category_name VARCHAR,
        product_category_name_english VARCHAR
    );
    """)


def load_csvs(db):
    for name, file in [
        ("customers", "olist_customers_dataset.csv"),
        ("geolocation", "olist_geolocation_dataset.csv"),
        ("order_items", "olist_order_items_dataset.csv"),
        ("order_payments", "olist_order_payments_dataset.csv"),
        ("order_reviews", "olist_order_reviews_dataset.csv"),
        ("orders", "olist_orders_dataset.csv"),
        ("products", "olist_products_dataset.csv"),
        ("sellers", "olist_sellers_dataset.csv"),
        ("product_category_name_translation", "product_category_name_translation.csv")
    ]:
        db.execute(f"""
            COPY {name} FROM '{RAW_PATH}{file}' 
            (HEADER, DELIMITER ',');
        """)


def create_views(db):
    db.execute("""
    CREATE OR REPLACE VIEW vw_reference AS
    SELECT MAX(order_purchase_timestamp) AS max_purchase_date
    FROM orders
    WHERE order_status = 'delivered';

    CREATE OR REPLACE VIEW vw_customer_orders AS
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp,
        oi.price,
        oi.freight_value,
        p.product_category_name
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_status = 'delivered';
    """)


def compute_rfm(db):
    db.execute("""
    CREATE OR REPLACE VIEW vw_rfm_base AS
    SELECT
        customer_unique_id,
        COUNT(DISTINCT order_id) AS frequency,
        DATEDIFF('day', MAX(order_purchase_timestamp), 
                 (SELECT max_purchase_date FROM vw_reference)) AS recency,
        SUM(price) AS monetary
    FROM vw_customer_orders
    GROUP BY customer_unique_id;
    """)

    db.execute("""
    CREATE OR REPLACE VIEW vw_rfm_quantiles AS
    SELECT
        approx_quantile(recency, 0.25) AS r_q1,
        approx_quantile(recency, 0.50) AS r_q2,
        approx_quantile(recency, 0.75) AS r_q3,
        
        approx_quantile(frequency, 0.25) AS f_q1,
        approx_quantile(frequency, 0.50) AS f_q2,
        approx_quantile(frequency, 0.75) AS f_q3,

        approx_quantile(monetary, 0.25) AS m_q1,
        approx_quantile(monetary, 0.50) AS m_q2,
        approx_quantile(monetary, 0.75) AS m_q3
    FROM vw_rfm_base;
    """)

    db.execute("""
    CREATE OR REPLACE VIEW vw_rfm_scored AS
    SELECT
        b.customer_unique_id,
        b.recency,
        b.frequency,
        b.monetary,
        CASE 
            WHEN b.recency <= q.r_q1 THEN 4
            WHEN b.recency <= q.r_q2 THEN 3
            WHEN b.recency <= q.r_q3 THEN 2
            ELSE 1 END AS r_score,

        CASE 
            WHEN b.frequency <= q.f_q1 THEN 1
            WHEN b.frequency <= q.f_q2 THEN 2
            WHEN b.frequency <= q.f_q3 THEN 3
            ELSE 4 END AS f_score,

        CASE 
            WHEN b.monetary <= q.m_q1 THEN 1
            WHEN b.monetary <= q.m_q2 THEN 2
            WHEN b.monetary <= q.m_q3 THEN 3
            ELSE 4 END AS m_score

    FROM vw_rfm_base b
    CROSS JOIN vw_rfm_quantiles q;
    """)

    db.execute("""
    CREATE OR REPLACE VIEW vw_rfm_final AS
    SELECT *, (r_score + f_score + m_score) AS rfm_sum
    FROM vw_rfm_scored;
    """)


def export_views(db):
    db.execute(f"""
    COPY (SELECT * FROM vw_rfm_final)
    TO '{PROCESSED_PATH}rfm_final.csv' (HEADER, DELIMITER ',');
    """)

# 2. CLUSTERIZAÇÃO - PYTHON + SKLEARN ================================

def load_rfm_processed():
    df = pd.read_csv(PROCESSED_PATH + "rfm_final.csv")
    df = df.dropna()
    return df


def scale_features(df):
    X = df[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def choose_best_k(X_scaled, k_min=2, k_max=10):
    best_k, best_score = None, -1

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def clusterize(df, X_scaled, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = km.fit_predict(X_scaled)
    return df


def segment(df):
    summary = df.groupby("cluster").agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    })

    labels = []
    for idx, row in summary.iterrows():
        if row['frequency'] > summary['frequency'].quantile(0.75):
            labels.append("Alta Frequência")
        elif row['monetary'] > summary['monetary'].quantile(0.75):
            labels.append("Alta Receita")
        elif row['recency'] < summary['recency'].quantile(0.25):
            labels.append("Recente")
        else:
            labels.append("Baixo Valor")

    summary["segment"] = labels
    return summary


def export_cluster_results(df, summary):
    df.to_csv(PROCESSED_PATH + "clustered_customers.csv", index=False)
    summary.to_csv(PROCESSED_PATH + "cluster_summary.csv")


# RODA TUDO  ================================
if __name__ == "__main__":

    db = duckdb.connect(DB_PATH)

    create_tables(db)
    load_csvs(db)
    create_views(db)
    compute_rfm(db)
    export_views(db)

    df = load_rfm_processed()
    X_scaled, scaler = scale_features(df)

    best_k = 4

    df = clusterize(df, X_scaled, best_k)
    summary = segment(df)

    export_cluster_results(df, summary)

    print("Clusterização concluída.")
    print("Arquivos gerados:")
    print(" - clustered_customers.csv")
    print(" - cluster_summary.csv")
