from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar datos una sola vez
sales_df = pd.read_csv("sales_data_sample.csv", encoding='latin1')
sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'], errors='coerce')
sales_df.drop("ORDERDATE", axis=1, inplace=True)

scaler = StandardScaler()
sales_df_scaled = scaler.fit_transform(sales_df.select_dtypes(include='number'))

# Generar las gráficas una sola vez
graficas = []

def generar_graficas():
    global graficas
    graficas.clear()

    # Gráficas simples
    graficas_data = [
        ("Pedidos por País", 'COUNTRY', 'Muestra el número total de pedidos realizados desde cada país.'),
        ("Pedidos por Estado", 'STATUS', 'Cantidad de pedidos agrupados por su estado (Shipped, Cancelled, etc.).'),
        ("Pedidos por Línea de Producto", 'PRODUCTLINE', 'Visualiza qué líneas de productos tienen mayor cantidad de pedidos.'),
        ("Pedidos por Tamaño del Trato", 'DEALSIZE', 'Distribución de pedidos según el tamaño del trato comercial.'),
    ]

    for titulo, col, descripcion in graficas_data:
        vc = sales_df[col].value_counts()
        fig = px.bar(x=vc.index, y=vc.values, color=vc.index, title=titulo)
        graficas.append({
            'html': pio.to_html(fig, full_html=False),
            'titulo': titulo,
            'descripcion': descripcion
        })

    # Ventas por Mes
    sales_df_group = sales_df.groupby('MONTH_ID').sum(numeric_only=True)
    fig = px.line(x=sales_df_group.index, y=sales_df_group['SALES'], title='Ventas por Mes')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Ventas por Mes',
        'descripcion': 'Total de ventas agregadas por cada mes.'
    })

    # Heatmap de correlaciones
    corr_matrix = sales_df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cbar=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close('all')
    graficas.append({
        'html': f'<img src="data:image/png;base64,{img_base64}">',
        'titulo': 'Mapa de Calor de Correlaciones',
        'descripcion': 'Correlaciones entre variables numéricas del dataset.'
    })

    # Distribuciones
    for col in ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']:
        fig = ff.create_distplot([sales_df[col].astype(float)], [f"Distribución {col}"], show_hist=False)
        fig.update_layout(title_text=f'Distribución de {col}')
        graficas.append({
            'html': pio.to_html(fig, full_html=False),
            'titulo': f'Distribución de {col}',
            'descripcion': f'Distribución estadística de la variable {col}.'
        })

    # Método del Codo
    scores = []
    range_values = range(1, 6)  # menos clusters = menos cálculo
    for i in range_values:
        kmeans = KMeans(n_clusters=i, n_init=5, random_state=42)
        kmeans.fit(sales_df_scaled)
        scores.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range_values, scores, 'bx-')
    plt.title('Método del Codo para KMeans')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close('all')
    graficas.append({
        'html': f'<img src="data:image/png;base64,{img_base64}">',
        'titulo': 'Método del Codo',
        'descripcion': 'Método para determinar el número óptimo de clusters para KMeans.'
    })

    # PCA 2D
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    labels = kmeans.fit_predict(sales_df_scaled)
    pca = PCA(n_components=2)
    components = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(components, columns=['pca1', 'pca2'])
    pca_df['cluster'] = labels
    fig = px.scatter(pca_df, x='pca1', y='pca2', color=pca_df['cluster'].astype(str),
                     title='PCA 2D con Clustering')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'PCA 2D con Clustering',
        'descripcion': 'Visualización reducida a 2D con agrupación KMeans.'
    })

    # PCA 3D
    pca_3d = PCA(n_components=3)
    components_3d = pca_3d.fit_transform(sales_df_scaled)
    pca_df_3d = pd.DataFrame(components_3d, columns=['pca1', 'pca2', 'pca3'])
    fig_3d = px.scatter_3d(pca_df_3d, x='pca1', y='pca2', z='pca3',
                           color=pca_df_3d.index.astype(str),
                           title='PCA 3D')
    graficas.append({
        'html': pio.to_html(fig_3d, full_html=False),
        'titulo': 'PCA 3D',
        'descripcion': 'Visualización 3D de los datos reducidos mediante PCA.'
    })

# Generar al inicio del servidor
generar_graficas()

@app.route('/')
def index():
    return render_template("index.html", graficas=graficas)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
