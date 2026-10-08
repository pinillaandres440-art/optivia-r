import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

Node = Tuple[int, int]

@dataclass
class Weights:
    distance: float
    slope: float
    environment: float
    risk: float


def normalize_weights(w: Weights) -> Weights:
    total = w.distance + w.slope + w.environment + w.risk
    if total <= 0:
        return Weights(0.25, 0.25, 0.25, 0.25)
    return Weights(w.distance/total, w.slope/total, w.environment/total, w.risk/total)


def generate_terrain(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.4, 2.4, n)
    y = np.linspace(-2.4, 2.4, n)
    X, Y = np.meshgrid(x, y)
    elevation = (
        34 + 16*np.exp(-((X+0.8)**2 + (Y-0.1)**2)/0.75)
        + 12*np.exp(-((X-1.0)**2 + (Y+0.9)**2)/0.55)
        + 3*np.sin(1.7*X)*np.cos(1.3*Y)
        + rng.normal(0, 0.7, (n, n))
    )
    environment = np.clip(
        0.10 + 0.90*np.exp(-((X-0.2)**2 + (Y-0.5)**2)/0.35), 0, 1
    )
    risk = np.clip(
        0.08 + 0.75*np.exp(-((X+0.9)**2 + (Y+1.0)**2)/0.45), 0, 1
    )
    return elevation, environment, risk


def neighbors(node: Node, n: int):
    r, c = node
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc


def heuristic(a: Node, b: Node) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def edge_components(a: Node, b: Node, elevation, environment, risk):
    horizontal = heuristic(a, b)
    dz = abs(float(elevation[b]) - float(elevation[a]))
    slope = dz / max(horizontal, 1e-9)
    env = (float(environment[a]) + float(environment[b])) / 2
    hazard = (float(risk[a]) + float(risk[b])) / 2
    return horizontal, slope, env, hazard


def astar(elevation, environment, risk, start: Node, goal: Node, weights: Weights):
    n = elevation.shape[0]
    w = normalize_weights(weights)
    open_heap = [(0.0, start)]
    came_from: Dict[Node, Node] = {}
    g = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path)), g[goal]

        for nxt in neighbors(current, n):
            d, s, e, r = edge_components(current, nxt, elevation, environment, risk)
            step_cost = w.distance*d + w.slope*s + w.environment*e*5 + w.risk*r*5
            tentative = g[current] + step_cost
            if tentative < g.get(nxt, float('inf')):
                came_from[nxt] = current
                g[nxt] = tentative
                priority = tentative + w.distance*heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))
    raise ValueError('No se encontró una ruta válida.')


def path_metrics(path: List[Node], elevation, environment, risk):
    distances, slopes, envs, hazards = [], [], [], []
    for a, b in zip(path[:-1], path[1:]):
        d, s, e, r = edge_components(a, b, elevation, environment, risk)
        distances.append(d); slopes.append(s); envs.append(e); hazards.append(r)
    return {
        'Longitud de cuadrícula': float(np.sum(distances)),
        'Pendiente media': float(np.mean(slopes) if slopes else 0),
        'Pendiente máxima': float(np.max(slopes) if slopes else 0),
        'Afectación ambiental media': float(np.mean(envs) if envs else 0),
        'Riesgo medio': float(np.mean(hazards) if hazards else 0),
    }


def monte_carlo_robustness(elevation, environment, risk, start, goal, weights, base_cost, runs=80, seed=99):
    rng = np.random.default_rng(seed)
    satisfactory = 0
    ratios = []
    for _ in range(runs):
        e2 = elevation + rng.normal(0, 0.8, elevation.shape)
        env2 = np.clip(environment * rng.normal(1, 0.10, environment.shape), 0, 1)
        risk2 = np.clip(risk * rng.normal(1, 0.18, risk.shape), 0, 1)
        try:
            _, cost = astar(e2, env2, risk2, start, goal, weights)
            ratio = cost / base_cost
            ratios.append(ratio)
            if ratio <= 1.15:
                satisfactory += 1
        except ValueError:
            ratios.append(np.nan)
    valid = np.array([x for x in ratios if np.isfinite(x)])
    robustness = satisfactory / runs
    return robustness, valid


def plot_map(layer, path, start, goal, title, cmap='terrain'):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(layer, cmap=cmap, origin='upper')
    if path:
        yy = [p[0] for p in path]; xx = [p[1] for p in path]
        ax.plot(xx, yy, color='#00F5A0', linewidth=3, label='Ruta OPTIVÍA R')
    ax.scatter([start[1]], [start[0]], c='#0B5FFF', s=90, marker='o', label='Inicio')
    ax.scatter([goal[1]], [goal[0]], c='#FF3B30', s=110, marker='X', label='Destino')
    ax.set_title(title); ax.set_xlabel('Columna'); ax.set_ylabel('Fila')
    ax.legend(loc='upper right'); fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout(); return fig


def main():
    st.set_page_config(page_title='OPTIVÍA R', layout='wide')
    st.title('OPTIVÍA R: rutas inteligentes bajo incertidumbre')
    st.caption('Prototipo educativo de A* multicriterio + simulación Monte Carlo')

    with st.sidebar:
        st.header('Configuración')
        n = st.slider('Tamaño de la cuadrícula', 20, 60, 36, 2)
        seed = st.number_input('Semilla del terreno', 1, 999, 7)
        st.subheader('Pesos')
        wd = st.slider('Distancia', 0.0, 1.0, 0.30, 0.05)
        wp = st.slider('Pendiente', 0.0, 1.0, 0.30, 0.05)
        we = st.slider('Ambiente', 0.0, 1.0, 0.25, 0.05)
        wr = st.slider('Riesgo', 0.0, 1.0, 0.15, 0.05)
        scenario = st.selectbox('Escenario', ['Base', 'Lluvia intensa', 'Protección ambiental'])
        runs = st.slider('Simulaciones Monte Carlo', 20, 200, 80, 20)

    elevation, environment, risk = generate_terrain(n, int(seed))
    if scenario == 'Lluvia intensa':
        risk = np.clip(risk*1.45 + 0.10, 0, 1)
    elif scenario == 'Protección ambiental':
        environment = np.clip(environment*1.35 + 0.08, 0, 1)

    start, goal = (n-3, 2), (2, n-3)
    weights = Weights(wd, wp, we, wr)
    path, total_cost = astar(elevation, environment, risk, start, goal, weights)
    metrics = path_metrics(path, elevation, environment, risk)
    robustness, ratios = monte_carlo_robustness(
        elevation, environment, risk, start, goal, weights, total_cost, runs, int(seed)+100
    )

    c1, c2, c3 = st.columns(3)
    c1.metric('Costo multicriterio', f'{total_cost:.2f}')
    c2.metric('Robustez', f'{robustness*100:.1f}%')
    c3.metric('Nodos de la ruta', len(path))

    left, right = st.columns([1.25, 1])
    with left:
        st.pyplot(plot_map(elevation, path, start, goal, 'Terreno y ruta optimizada'))
    with right:
        st.subheader('Indicadores de la ruta')
        df = pd.DataFrame({'Indicador': list(metrics.keys()), 'Valor': list(metrics.values())})
        st.dataframe(df.style.format({'Valor':'{:.3f}'}), use_container_width=True, hide_index=True)
        st.info('Robustez = proporción de escenarios cuyo costo no supera en más de 15% el costo base.')
        if len(ratios):
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.hist(ratios, bins=14, color='#1D6A73', edgecolor='white')
            ax.axvline(1.15, color='#D97706', linestyle='--', label='Umbral 1,15')
            ax.set_xlabel('Costo del escenario / costo base'); ax.set_ylabel('Frecuencia')
            ax.legend(); fig.tight_layout(); st.pyplot(fig)

    st.subheader('Perfil longitudinal simplificado')
    profile = [elevation[p] for p in path]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(profile, color='#123047', linewidth=2)
    ax.fill_between(range(len(profile)), profile, alpha=0.18, color='#1D6A73')
    ax.set_xlabel('Punto consecutivo de la ruta'); ax.set_ylabel('Elevación simulada')
    ax.grid(alpha=0.2); fig.tight_layout(); st.pyplot(fig)

    st.warning('Uso académico: el terreno y los resultados son simulados. No sustituyen estudios de ingeniería vial.')


if __name__ == '__main__':
    main()
