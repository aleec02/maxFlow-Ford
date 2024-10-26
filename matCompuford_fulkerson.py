import streamlit as st
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
import string
import io
import pandas as pd

st.set_page_config(page_title="Algoritmo de Ford-Fulkerson", layout="wide")

st.markdown("""
    <style>
    .wiki-breadcrumb {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.9em;
        color: #666;
    }
    .wiki-content {
        padding: 20px;
        background-color: white;
        border-radius: 4px;
        border: 1px solid #eee;
    }
    .wiki-index {
        position: sticky;
        top: 3rem;
        padding: 10px;
        border-right: 1px solid #eee;
    }
    .wiki-section {
        margin: 20px 0;
        padding: 10px;
        border-bottom: 1px solid #f0f0f0;
    }
    .wiki-title {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .wiki-math {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        margin-top: 10px;
    }
    .matrix-input {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def generar_camino_principal(n):
    """
    Genera un camino aleatorio desde la fuente (0) hasta el sumidero (n-1)
    Retorna: lista de nodos que forman el camino
    """
    camino = [0] 
    current = 0
    while current != n-1:
        min_next = current + 1
        max_next = min(current + 3, n-1)
        next_node = random.randint(min_next, max_next)
        camino.append(next_node)
        current = next_node
        if current == n-1:
            break
    return camino

def verificar_conexiones_salientes(matriz):
    """
    Verifica que cada nodo (excepto el último) tenga al menos una conexión saliente
    Retorna: lista de nodos sin conexiones salientes
    """
    n = len(matriz)
    nodos_sin_conexion = []
    for i in range(n-1):
        if sum(matriz[i]) == 0:
            nodos_sin_conexion.append(i)
    return nodos_sin_conexion


import streamlit as st
import random

def crear_matriz(cantidad):
    """
    Genera una matriz de adyacencia con al menos una conexión saliente por fila, excepto la última.
    """
    matriz = [[0] * cantidad for _ in range(cantidad)]
    
    # Generar el camino principal
    camino_principal = generar_camino_principal(cantidad)
    for i in range(len(camino_principal)-1):
        nodo_actual = camino_principal[i]
        nodo_siguiente = camino_principal[i+1]
        matriz[nodo_actual][nodo_siguiente] = random.randint(10, 20)

    # Generar un camino alternativo
    if random.random() < 0.7:
        camino_alt = generar_camino_principal(cantidad)
        for i in range(len(camino_alt)-1):
            nodo_actual = camino_alt[i]
            nodo_siguiente = camino_alt[i+1]
            if matriz[nodo_actual][nodo_siguiente] == 0:
                matriz[nodo_actual][nodo_siguiente] = random.randint(5, 15)

    aristas_adicionales = random.randint(cantidad // 2, cantidad)
    for _ in range(aristas_adicionales):
        i = random.randint(0, cantidad - 2)
        j = random.randint(i + 1, cantidad - 1)
        if matriz[i][j] == 0:
            matriz[i][j] = random.randint(1, 10)


    for i in range(cantidad - 1):
        if sum(matriz[i]) == 0:
            posibles_destinos = [j for j in range(cantidad) if j != i]
            destino = random.choice(posibles_destinos)
            matriz[i][destino] = random.randint(1, 10)
    
    return matriz




def validar_matriz_manual(entrada):
    try:
        filas = entrada.strip().split(';')
        matriz = []
        for fila in filas:
            valores = list(map(int, fila.split()))
            matriz.append(valores)
        
        filas_ingresadas = len(matriz)
        columnas_ingresadas = len(matriz[0]) if matriz else 0
        
        if filas_ingresadas != columnas_ingresadas:
            return None, "Error: La matriz debe ser cuadrada."
        
        if filas_ingresadas < 8 or filas_ingresadas > 16:
            return None, f"Error: El tamaño de la matriz debe estar entre 8 y 16. La matriz ingresada es de {filas_ingresadas}x{columnas_ingresadas}."
        
        if not all(len(fila) == columnas_ingresadas for fila in matriz):
            return None, "Error: Todas las filas deben tener la misma cantidad de columnas."
        
        return matriz, ""
    except ValueError:
        return None, "Error: Ingrese solo números separados por espacios y punto y coma."
    

def dfs(grafo, fuente, sumidero, parent, visitado):
    visitado[fuente] = True
    if fuente == sumidero:
        return True
    for v, capacidad in enumerate(grafo[fuente]):
        if not visitado[v] and capacidad > 0:
            parent[v] = fuente
            if dfs(grafo, v, sumidero, parent, visitado):
                return True
    return False


# Implementación del algoritmo de Ford-Fulkerson
def ford_fulkerson(grafo, fuente, sumidero):
    n = len(grafo)
    flujo_maximo = 0
    parent = [-1] * n
    grafo_residual = np.copy(grafo)
    
    while True:
        visitado = [False] * n
        if not dfs(grafo_residual, fuente, sumidero, parent, visitado):
            break

        flujo_camino = float('Inf')
        v = sumidero
        while v != fuente:
            u = parent[v]
            flujo_camino = min(flujo_camino, grafo_residual[u][v])
            v = u

        v = sumidero
        while v != fuente:
            u = parent[v]
            grafo_residual[u][v] -= flujo_camino
            grafo_residual[v][u] += flujo_camino
            v = u

        flujo_maximo += flujo_camino
    
    return flujo_maximo, grafo_residual

# Función para dibujar el grafo
def crear_grafo_imagen(matriz, etiquetas, flujo=False):
    G = nx.DiGraph()
    
    for i in range(len(etiquetas)):
        G.add_node(etiquetas[i])
    
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            if matriz[i][j] > 0:
                G.add_edge(etiquetas[i], etiquetas[j], capacity=matriz[i][j])

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=10, font_weight='bold', ax=ax)

    labels = nx.get_edge_attributes(G, 'capacity')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Red de flujos" if not flujo else "Flujo máximo")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def create_breadcrumb(current_section):
    sections = {
        "inicio": "Inicio",
        "teoria": "Fundamento Teórico",
        "programa": "Ejecutar Programa",
        "equipo": "Equipo"
    }
    breadcrumb = f"Programa > {sections.get(current_section, current_section)}"
    st.markdown(f'<div class="wiki-breadcrumb">{breadcrumb}</div>', unsafe_allow_html=True)

def show_matrix_input_section(n):
    st.markdown('<div class="matrix-input">', unsafe_allow_html=True)
    st.markdown("""
    **Formato de entrada:**
    - Valores separados por espacios
    - Filas separadas por punto y coma (;)
    - Ejemplo: `0 10 5 15 0 0 0 0; 0 0 4 0 9 15 0 0; ...`
    """)
    
    matriz_input = st.text_area(
        "Ingrese la matriz de adyacencia:",
        value=st.session_state.get('matriz_input', ''),
        height=200,
        key="matriz_input_area"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    return matriz_input

def show_section(title, content):
    st.markdown(f'<div class="wiki-section"><h2 class="wiki-title">{title}</h2>{content}</div>', 
                unsafe_allow_html=True)


def main():
    st.sidebar.markdown('<div class="wiki-index">', unsafe_allow_html=True)
    st.sidebar.title("Menú")
    
    section = st.sidebar.radio("", 
        ["📚 Inicio", 
         "📖 Fundamento Teórico",
         "⚡ Ejecutar Programa",
         "👥 Equipo"],
        index=0,
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
    if 'matriz' not in st.session_state:
        st.session_state.matriz = None

    if "📚 Inicio" in section:
        create_breadcrumb("inicio")
        st.title("Algoritmo de Ford-Fulkerson")
        
        st.markdown("""
        ### Enunciado del Problema

        Dado $n \in [8, 16]$ ingresado por el usuario, el programa debe generar aleatoriamente una matriz $n × n$
        (con elementos positivos) o solicitar el ingreso de cada elemento de la matriz (según decisión del usuario). Además, debe mostrar la red de flujos óptima asociada a esta matriz y calcular el flujo máximo que existe entre dos vértices seleccionados por el usuario. Todo el proceso, desde la generación de la matriz hasta el cálculo del flujo máximo, se debe mostrar paso a paso, proporcionando una visualización clara y detallada del funcionamiento interno del algoritmo.
        """)


        
        show_section("Descripción General", """
        El algoritmo de Ford-Fulkerson es un método para encontrar el flujo máximo en una red de flujo.
        Este proyecto implementa una versión interactiva del algoritmo con visualización de grafos, cumpliendo con los requerimientos del enunciado del problema.
        """)
        
    elif "📖 Fundamento Teórico" in section:
        create_breadcrumb("teoria")
        
        with st.expander("1. Conceptos Básicos", expanded=True):
            st.markdown("""
            - **Red de Flujo:** Grafo dirigido con capacidades
            - **Capacidad ($c$):** Es una propiedad de cada arista del grafo que define el límite máximo de flujo que puede pasar por esa arista.
            - **Flujo:** Cantidad que se mueve a través de una red o grafo.  Asigna valores a las aristas del grafo. Representa la cantidad de fujo que pasa por una arista en un tiempo dado.
            - **Nodos:** Incluye al menos un nodo fuente $(s)$ y un nodo sumidero $(t)$.
            - **Flujo máximo($|f|$):** Se refiere a la cantidad máxima total de flujo que puede ser transportada desde el nodo fuente hasta el nodo sumidero.
            """)
            st.latex(r"0 \leq f(u,v) \leq c(u,v)")

        with st.expander("2. Condiciones", expanded=True):
            st.markdown("### Condiciones del Flujo")
            st.latex(r"""
            \begin{aligned}
            & \text{Capacidad: } & 0 \leq f(u,v) \leq c(u,v) \\
            & \text{Conservación: } & \sum_{w \in V} f(v,w) = \sum_{u \in V} f(u,v)
            \end{aligned}
            """)
            st.markdown("**Capacidad:** El flujo en la arista $(u,v)$ debe ser mayor o igual que creo y el flujo en la arisa (u,v) debe ser menos o igual que la capacidad.")
            st.markdown("**Conservación:** El flujo que entra a un nodo debe ser igual al flujo que sale de él, salvo en el caso de los nodos fuente y sumidero.")


        with st.expander("4. Representación", expanded=True):
            st.markdown("### Flujo máximo")

            st.latex(r"""
            \text{Flujo Máximo: } \quad f_{\text{max}} = \sum_{v \in S} f(s,v)
            """)
            st.markdown("donde $S$ es el conjunto de nodos alcanzables desde la fuente en el flujo final.")

        with st.expander("3. Algoritmo Ford-Fulkerson", expanded=True):
            st.markdown("### Proceso Iterativo; pseudocodigo")
            st.code("""
            while existe_camino_aumentante(grafo, fuente, sumidero):
                encontrar_camino_minimo()
                actualizar_flujos()
            """)

    elif "⚡ Ejecutar Programa" in section:
            create_breadcrumb("programa")
            st.title("Ejecutar programa")
            
            with st.sidebar:
                st.markdown("### Configuración")
                
                if 'matriz' not in st.session_state:
                    st.session_state.matriz = None
                if 'matriz_input' not in st.session_state:
                    st.session_state.matriz_input = ''
                if 'tamano_matriz' not in st.session_state:
                    st.session_state.tamano_matriz = 8

                n = st.slider(
                    "Tamaño de la matriz",
                    8, 16, 
                    st.session_state.tamano_matriz
                )

                metodo = st.selectbox(
                    "Método de generación",
                    ["Aleatorio", "Manual"]
                )


            if metodo == "Aleatorio":
                if st.sidebar.button("Generar nueva matriz aleatoria"):
                    for key in ['matriz', 'matriz_input']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    nueva_matriz = crear_matriz(n)
                    
                    zero_rows = [i for i in range(n-1) if sum(nueva_matriz[i]) == 0]
                    if zero_rows:
                        st.error(f"Error: Filas {zero_rows} tienen todos ceros")
                    else:
                        st.session_state.matriz = nueva_matriz
                        st.session_state.tamano_matriz = n
                        st.rerun()

            else:
                matriz_input = show_matrix_input_section(n)
                
                if st.button("Validar y cargar matriz"):
                    matriz, error = validar_matriz_manual(matriz_input)
                    if matriz is not None:
                        st.session_state.matriz = matriz
                        st.session_state.matriz_input = matriz_input
                        st.session_state.tamano_matriz = len(matriz)
                        st.success("Matriz cargada correctamente")
                        st.rerun()
                    else:
                        st.error(error)

            # Display matrix if it exists
            if st.session_state.matriz is not None:
                n = len(st.session_state.matriz)
                etiquetas = list(string.ascii_uppercase[:n])
                
                st.subheader("Matriz de adyacencia")
                df = pd.DataFrame(
                    st.session_state.matriz,
                    columns=etiquetas,
                    index=etiquetas
                )
                st.dataframe(df)

                col1, col2 = st.columns(2)
                
                with col1:
                    fuente = st.selectbox(
                        "Seleccione el vértice fuente",
                        etiquetas
                    )

                with col2:
                    sumidero = st.selectbox(
                        "Seleccione el vértice sumidero",
                        etiquetas,
                        index=len(etiquetas)-1
                    )

                if st.button("Calcular flujo máximo"):
                    fuente_idx = etiquetas.index(fuente)
                    sumidero_idx = etiquetas.index(sumidero)

                    flujo_max, grafo_residual = ford_fulkerson(
                        np.copy(st.session_state.matriz),
                        fuente_idx,
                        sumidero_idx
                    )

                    st.success(f"El flujo máximo entre {fuente} y {sumidero} es: {flujo_max}")

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Red de flujos original")
                        buf = crear_grafo_imagen(st.session_state.matriz, etiquetas, False)
                        st.image(buf, use_column_width=True)
                    
                    with col2:
                        st.subheader("Red con flujo máximo")
                        buf = crear_grafo_imagen(grafo_residual, etiquetas, True)
                        st.image(buf, use_column_width=True)

    else:
        create_breadcrumb("equipo")
        st.title("Equipo de Desarrollo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Desarrollador 1
            - Estudiante: Alexia Conza
            - Rol: Frontend e Implementación del algoritmos.
            """)
        
        with col2:
            st.markdown("""
            ### Desarrollador 2
            - Estudiante: Andrés Coca
            - Rol: Implementación del algoritmos.
            """)

if __name__ == "__main__":
    main()

