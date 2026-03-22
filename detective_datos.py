"""
Hecho por : Rocha Cantu Nidia Wendoly  Fecha: 22  de Marzo 2026
Clase: Inteligencia artificial y su ética - Tema 4.5 Minería de Datos - Actividad 24
MIA - Intituto Tecnológico de Nuevo Laredo - Prof. Carlos Arturo Guerrero Crespo
Titulo: Detective de Datos: Descubriendo Patrones Ocultos
Descripción:
Analiza transacciones de un supermercado para encontrar patrones de compra,
segmentar clientes y detectar anomalías.
"""

# =========================
# LIBRERÍAS
# =========================
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# 1. SIMULACIÓN DE DATOS
# =========================
def generar_transacciones(n_transacciones=5000):
    np.random.seed(42)
    productos = ['Pan', 'Leche', 'Mantequilla', 'Cerveza', 'Papas',
                 'Dulces', 'Carne', 'Verduras', 'Salsa', 'Arroz', 'Frutas', 'Jugo']
    
    transacciones = []
    for _ in range(n_transacciones):
        n_items = np.random.randint(1, 5)
        transacciones.append(list(np.random.choice(productos, n_items, replace=False)))
    
    return transacciones

# =========================
# 2. ANÁLISIS DE CANASTA
# =========================
def analisis_canasta(transacciones):
    """Encuentra productos comprados juntos usando Apriori"""
    # Convertir a formato One-Hot
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transacciones).transform(transacciones)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Reglas frecuentes
    frecuentes = apriori(df, min_support=0.05, use_colnames=True)
    reglas = association_rules(frecuentes, metric="lift", min_threshold=1.2)

    print("\n=== Reglas de Asociación (Canasta de Compra) ===")
    for _, row in reglas.iterrows():
        print(f"{list(row['antecedents'])} → {list(row['consequents'])} | support: {row['support']:.2f}, lift: {row['lift']:.2f}")
    
    return reglas

# =========================
# 3. SEGMENTACIÓN DE CLIENTES
# =========================
def segmentacion_clientes(transacciones):
    """Clusterización de clientes según productos comprados"""
    # Vectorizar transacciones
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transacciones).transform(transacciones)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Escalado y KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    print("\n=== Segmentación de Clientes ===")
    for i in range(3):
        print(f"Cluster {i}: {np.sum(clusters==i)} clientes")
    
    return clusters

# =========================
# 4. DETECCIÓN DE ANOMALÍAS
# =========================
def deteccion_anomalias(transacciones):
    """Detecta transacciones sospechosas por rareza de productos"""
    from collections import Counter
    # Contar frecuencia de cada producto
    all_items = [item for t in transacciones for item in t]
    freq = Counter(all_items)

    # Calcular rareza por transacción
    scores = []
    for t in transacciones:
        score = sum(1/freq[i] for i in t)  # Más raro = mayor score
        scores.append(score)
    
    # Marcar top 1% como anomalías
    umbral = np.percentile(scores, 99)
    anomalías = [i for i, s in enumerate(scores) if s >= umbral]

    print(f"\n=== Detección de Anomalías ===")
    print(f"Se detectaron {len(anomalías)} transacciones sospechosas (top 1%)")
    
    return anomalías

# =========================
# 5. PROYECTO COMPLETO
# =========================
def proyecto_mineria_datos():
    print("Proyecto: Detective de Datos - Supermercado ABC")

    # Generar datos
    transacciones = generar_transacciones()

    # 1 Canasta de compra
    reglas = analisis_canasta(transacciones)

    # 2 Segmentación de clientes
    clusters = segmentacion_clientes(transacciones)

    # 3 Detección de anomalías
    anomalías = deteccion_anomalias(transacciones)

    print("\nProyecto completado")
    return reglas, clusters, anomalías

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    proyecto_mineria_datos()