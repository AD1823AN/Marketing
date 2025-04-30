from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.subplots as sp
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

sales_df = pd.read_csv("sales_data_sample.csv", encoding='latin1')
sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'], errors='coerce')
sales_df.drop("ORDERDATE", axis=1, inplace=True)
scaler = StandardScaler()
sales_df_scaled = scaler.fit_transform(sales_df.select_dtypes(include='number'))

@app.route('/')
def index():
    graficas = []

    # Pedidos por País
    fig = px.bar(x=sales_df['COUNTRY'].value_counts().index,
                 y=sales_df['COUNTRY'].value_counts(),
                 color=sales_df['COUNTRY'].value_counts().index,
                 title='Pedidos por País')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Pedidos por País',
        'descripcion': 'Muestra el número total de pedidos realizados desde cada país.'
    })

    # Pedidos por Estado
    fig = px.bar(x=sales_df['STATUS'].value_counts().index,
                 y=sales_df['STATUS'].value_counts(),
                 color=sales_df['STATUS'].value_counts().index,
                 title='Pedidos por Estado')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Pedidos por Estado',
        'descripcion': 'Cantidad de pedidos agrupados por su estado (Shipped, Cancelled, etc.).'
    })

    # Pedidos por Línea de Producto
    fig = px.bar(x=sales_df['PRODUCTLINE'].value_counts().index,
                 y=sales_df['PRODUCTLINE'].value_counts(),
                 color=sales_df['PRODUCTLINE'].value_counts().index,
                 title='Pedidos por Línea de Producto')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Pedidos por Línea de Producto',
        'descripcion': 'Visualiza qué líneas de productos tienen mayor cantidad de pedidos.'
    })

    # Pedidos por Tamaño del Trato
    fig = px.bar(x=sales_df['DEALSIZE'].value_counts().index,
                 y=sales_df['DEALSIZE'].value_counts(),
                 color=sales_df['DEALSIZE'].value_counts().index,
                 title='Pedidos por Tamaño del Trato')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Pedidos por Tamaño del Trato',
        'descripcion': 'Distribución de pedidos según el tamaño del trato comercial.'
    })

    # Ventas por Mes
    sales_df_group = sales_df.groupby('MONTH_ID').sum(numeric_only=True)
    fig = px.line(x=sales_df_group.index, y=sales_df_group['SALES'], title='Ventas por Mes')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Ventas por Mes',
        'descripcion': 'Total de ventas agregadas por cada mes.'
    })

    # Mapa de calor de correlaciones
    corr_matrix = sales_df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cbar=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    graficas.append({
        'html': f'<img src="data:image/png;base64,{img_base64}">',
        'titulo': 'Mapa de Calor de Correlaciones',
        'descripcion': 'Correlaciones entre variables numéricas del dataset.'
    })

    # Distribuciones de columnas
    for col in ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']:
        fig = ff.create_distplot([sales_df[col].astype(float)], [f"Distribución {col}"])
        fig.update_layout(title_text=f'Distribución de {col}')
        graficas.append({
            'html': pio.to_html(fig, full_html=False),
            'titulo': f'Distribución de {col}',
            'descripcion': f'Distribución estadística de la variable {col}.'
        })

    # Método del Codo
    scores = []
    range_values = range(1, 10)
    for i in range_values:
        kmeans = KMeans(n_clusters=i, n_init=10)
        kmeans.fit(sales_df_scaled)
        scores.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range_values, scores, 'bx-')
    plt.title('Método del Codo para KMeans')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    graficas.append({
        'html': f'<img src="data:image/png;base64,{img_base64}">',
        'titulo': 'Método del Codo',
        'descripcion': 'Método para determinar el número óptimo de clusters para KMeans.'
    })

    # PCA 2D con KMeans
    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(sales_df_scaled)
    pca = PCA(n_components=2)
    components = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(components, columns=['pca1', 'pca2'])
    pca_df['cluster'] = labels
    fig = px.scatter(pca_df, x='pca1', y='pca2', color=pca_df['cluster'].astype(str),
                     title='PCA 2D con Clustering')
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'PCA 2D con Clustering',
        'descripcion': 'Visualización reducida a 2D con agrupación KMeans.'
    })

    # PCA 3D
    pca_3d = PCA(n_components=3)
    components_3d = pca_3d.fit_transform(sales_df_scaled)
    pca_df_3d = pd.DataFrame(components_3d, columns=['pca1', 'pca2', 'pca3'])
    fig_3d = px.scatter_3d(pca_df_3d, x='pca1', y='pca2', z='pca3', color=pca_df_3d.index,
                           title='PCA 3D')
    graficas.append({
        'html': pio.to_html(fig_3d, full_html=False),
        'titulo': 'PCA 3D',
        'descripcion': 'Visualización 3D de los datos reducidos mediante PCA.'
    })

    return render_template("index.html", graficas=graficas)

if __name__ == '__main__':
    app.run(debug=True)
