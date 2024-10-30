import streamlit as st
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
import string
import io
import pandas as pd
from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Reporte de Flujo Máximo - Algoritmo Ford-Fulkerson', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'{self.page_no()}/{{nb}}', 0, 0, 'C')


def generar_reporte(matriz, etiquetas, fuente, sumidero, flujo_max, rutas, grafo_original, grafo_residual):
    pdf = PDF('L')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_margins(20, 20, 20)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Fecha y hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Parámetros del Problema:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f'* Tamaño de la matriz: {len(matriz)}x{len(matriz)}', 0, 1)
    pdf.cell(0, 10, f'* Nodo fuente: {fuente}', 0, 1)
    pdf.cell(0, 10, f'* Nodo sumidero: {sumidero}', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Matriz de Adyacencia:', 0, 1)
    pdf.set_font('Arial', '', 8)
    
    col_width = min(15, (pdf.w - 40) / (len(matriz) + 1))
    row_height = 7
    
    pdf.cell(col_width, row_height, '', 1)
    for etiqueta in etiquetas:
        pdf.cell(col_width, row_height, etiqueta, 1)
    pdf.ln()
    
    for i, fila in enumerate(matriz):
        pdf.cell(col_width, row_height, etiquetas[i], 1)
        for valor in fila:
            pdf.cell(col_width, row_height, str(valor), 1)
        pdf.ln()
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Resultados:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f'Flujo máximo: {flujo_max}', 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Rutas tomadas:', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    for i, ruta in enumerate(rutas, 1):
        camino_str = " -> ".join(ruta["camino"])
        pdf.cell(0, 10, f'Ruta {i}:', 0, 1)
        pdf.cell(20, 10, '', 0, 0)
        pdf.cell(0, 10, f'Camino: {camino_str}', 0, 1)
        pdf.cell(20, 10, '', 0, 0)
        pdf.cell(0, 10, f'Flujo: {ruta["flujo"]}', 0, 1)
    pdf.ln(5)
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Visualización de Grafos:', 0, 1)
    
    pdf.cell(0, 10, 'Red de flujos original:', 0, 1)
    grafo_original.seek(0)
    pdf.image(grafo_original, x=20, w=250)
    
    pdf.add_page()
    pdf.cell(0, 10, 'Red con flujo máximo:', 0, 1)
    grafo_residual.seek(0)
    pdf.image(grafo_residual, x=20, w=250)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"FordFulkerson_Resultados_{timestamp}.pdf"
    
    pdf.output(filename)
    return filename

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


def encontrar_camino(grafo_residual, fuente, sumidero, parent):
    """
    Reconstruye el camino desde la fuente al sumidero usando el array parent
    """
    camino = []
    v = sumidero
    while v != fuente:
        u = parent[v]
        camino.append(v)
        v = u
    camino.append(fuente)
    return list(reversed(camino))

def ford_fulkerson(grafo, fuente, sumidero, etiquetas):
    n = len(grafo)
    flujo_maximo = 0
    grafo_residual = np.copy(grafo)

    rutas = []
    
    while True:
        parent = [-1] * n
        visitado = [False] * n
        if not dfs(grafo_residual, fuente, sumidero, parent, visitado):
            break

        flujo_camino = float('Inf')
        v = sumidero
        while v != fuente:
            u = parent[v]
            flujo_camino = min(flujo_camino, grafo_residual[u][v])
            v = u

        camino = encontrar_camino(grafo_residual, fuente, sumidero, parent)
        
        camino_letras = [etiquetas[i] for i in camino]
        
        rutas.append({
            "camino": camino_letras,
            "flujo": flujo_camino
        })

        v = sumidero
        while v != fuente:
            u = parent[v]
            grafo_residual[u][v] -= flujo_camino
            grafo_residual[v][u] += flujo_camino
            v = u

        flujo_maximo += flujo_camino
    
    return flujo_maximo, grafo_residual, rutas


# Función para dibujar el grafo
def crear_grafo_imagen(matriz, etiquetas, flujo=False, rutas=None, grafo_residual=None):
    G = nx.DiGraph()
    
    for i in range(len(etiquetas)):
        G.add_node(etiquetas[i])
    
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            if matriz[i][j] > 0:
                if grafo_residual is not None:
                    flow = matriz[i][j] - grafo_residual[i][j]
                    G.add_edge(etiquetas[i], etiquetas[j], capacity=matriz[i][j], flow=flow)
                else:
                    G.add_edge(etiquetas[i], etiquetas[j], capacity=matriz[i][j], flow=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    color_map = ['#2ECC71' if i == 0 else '#E74C3C' if i == len(etiquetas)-1 
                 else '#3498DB' for i in range(len(etiquetas))]
    
    nx.draw_networkx_nodes(G, pos,
                          node_color=color_map,
                          node_size=1500)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    path_colors = ['#FF69B4', '#FFD700', '#32CD32', '#00c5cd', '#FF7F50']

    edge_paths = {}
    if rutas:
        for idx, ruta in enumerate(rutas):
            path = ruta["camino"]
            path_edges = list(zip(path[:-1], path[1:]))
            
            for edge in path_edges:
                if edge not in edge_paths:
                    edge_paths[edge] = []
                edge_paths[edge].append({
                    'color': path_colors[idx % len(path_colors)],
                    'flow': ruta["flujo"],
                    'path_num': idx + 1
                })

    base_edge_width = 2.0
    edge_shift = 0.03

    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.3)

    if edge_paths:
        for edge, paths in edge_paths.items():
            num_paths = len(paths)
            
            if num_paths > 1:
                shifts = np.linspace(-edge_shift * (num_paths-1)/2, 
                                   edge_shift * (num_paths-1)/2, 
                                   num_paths)
            else:
                shifts = [0]

            for path_idx, path_info in enumerate(paths):
                node1, node2 = edge
                pos1, pos2 = pos[node1], pos[node2]
                
                shift = shifts[path_idx]
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                angle = np.arctan2(dy, dx) + np.pi/2
                
                shift_x = shift * np.cos(angle)
                shift_y = shift * np.sin(angle)
                
                new_pos1 = (pos1[0] + shift_x, pos1[1] + shift_y)
                new_pos2 = (pos2[0] + shift_x, pos2[1] + shift_y)
                
                plt.arrow(new_pos1[0], new_pos1[1],
                         new_pos2[0] - new_pos1[0],
                         new_pos2[1] - new_pos1[1],
                         color=path_info['color'],
                         width=0.002,
                         head_width=0.03,
                         length_includes_head=True,
                         alpha=0.7)

    edge_labels = {}
    for (u, v, d) in G.edges(data=True):
        if (u, v) in edge_paths:
            # lista de etiqueta de varias líneas que muestre todos los flujos a través de una arista
            flows_text = []
            total_flow = 0
            for path_info in edge_paths[(u, v)]:
                flows_text.append(f"R{path_info['path_num']}: {path_info['flow']}")
                total_flow += path_info['flow']
            flows_str = '\n'.join(flows_text)
            edge_labels[(u, v)] = f"{total_flow}/{d['capacity']}\n{flows_str}"
        else:
            edge_labels[(u, v)] = f"0/{d['capacity']}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
                  markersize=15, label='Nodo Fuente'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
                  markersize=15, label='Nodo Sumidero'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
                  markersize=15, label='Nodos Intermedios')
    ]

    if rutas:
        for idx, ruta in enumerate(rutas):
            legend_elements.append(
                plt.Line2D([0], [0], color=path_colors[idx % len(path_colors)],
                          label=f'Ruta {idx+1} (Flujo: {ruta["flujo"]})',
                          linewidth=2)
            )

    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1, 0.5))

    plt.title("Red de flujos" if not flujo else "Flujo máximo")
    plt.tight_layout()
    
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



def show_editable_matrix(matriz, etiquetas):

    df = pd.DataFrame(
        matriz,
        columns=etiquetas,
        index=etiquetas
    )
    
    edited_df = st.data_editor(
        df,
        key="matrix_editor",
        use_container_width=True,
        hide_index=False,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                min_value=0,
                max_value=999,
                step=1,
                format="%d"
            ) for col in etiquetas
        },
        disabled=False
    )
    
    edited_matrix = edited_df.to_numpy()
    
    if not np.array_equal(matriz, edited_matrix):
        return edited_matrix
    return matriz



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

                if st.session_state.matriz is not None:
                    n = len(st.session_state.matriz)
                    etiquetas = list(string.ascii_uppercase[:n])
                    
                    st.subheader("Matriz de adyacencia")
                    st.markdown("""
                    💡 **Tip**: Puedes editar los valores de la matriz directamente haciendo clic en las celdas.
                    Los cambios se guardarán automáticamente.
                    """)
                    
                    edited_matrix = show_editable_matrix(st.session_state.matriz, etiquetas)
                    
                    if not np.array_equal(st.session_state.matriz, edited_matrix):
                        st.session_state.matriz = edited_matrix
                        st.success("Matriz actualizada correctamente")



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

                    flujo_max, grafo_residual, rutas = ford_fulkerson(
                        np.copy(st.session_state.matriz),
                        fuente_idx,
                        sumidero_idx,
                        etiquetas
                    )
                    st.session_state.last_results = {
                        'flujo_max': flujo_max,
                        'rutas': rutas,
                        'grafo_residual': grafo_residual,
                        'fuente': fuente,
                        'sumidero': sumidero
                    }

                if 'last_results' in st.session_state:
                    st.success(f"Flujo máximo de {st.session_state.last_results['fuente']} a {st.session_state.last_results['sumidero']}: {st.session_state.last_results['flujo_max']}")
                    
                    st.markdown("### Rutas tomadas:")
                    for i, ruta in enumerate(st.session_state.last_results['rutas'], 1):
                        camino_str = " → ".join(ruta["camino"])
                        st.markdown(f"- **Ruta {i}:** {camino_str} (Flujo: **{ruta['flujo']}**)")

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Red de flujos original")
                        buf = crear_grafo_imagen(st.session_state.matriz, etiquetas, False)
                        st.image(buf, use_column_width=True)
                    
                    with col2:
                        st.subheader("Red con flujo máximo")
                        buf = crear_grafo_imagen(st.session_state.matriz, etiquetas, True, 
                                               rutas=st.session_state.last_results['rutas'], 
                                               grafo_residual=st.session_state.last_results['grafo_residual'])
                        st.image(buf, use_column_width=True)

                    if st.button("Exportar Resultados"):
                        filename = generar_reporte(
                            st.session_state.matriz,
                            etiquetas,
                            st.session_state.last_results['fuente'],
                            st.session_state.last_results['sumidero'],
                            st.session_state.last_results['flujo_max'],
                            st.session_state.last_results['rutas'],
                            crear_grafo_imagen(st.session_state.matriz, etiquetas, False),
                            crear_grafo_imagen(st.session_state.matriz, etiquetas, True, 
                                            rutas=st.session_state.last_results['rutas'], 
                                            grafo_residual=st.session_state.last_results['grafo_residual'])
                        )
                        with open(filename, 'rb') as pdf_file:
                            PDFbyte = pdf_file.read()
                        
                        st.download_button(
                            label="Descargar Reporte PDF",
                            data=PDFbyte,
                            file_name=filename,
                            mime='application/pdf'
                        )


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

