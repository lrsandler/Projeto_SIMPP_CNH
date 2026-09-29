from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os 
import umap
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from analise_vibracao import  preparar_diretorio_saida

def rotular_fase(tempo_centro_s: np.ndarray, aquecimento_fim_s: float | None):
    """aquecimento_fim_s = None -> ponto de medicao sem aquecimento (tudo regime estacionario)"""
    if aquecimento_fim_s is None:
        return np.full(len(tempo_centro_s), "regime_estacionario", dtype=object)
    return np.where(tempo_centro_s < aquecimento_fim_s, "aquecimento", "regime_estacionario")

def paleta_deslocada(paleta):
    return ListedColormap(paleta(np.linspace(0.4, 1.0, 256)))

def cores_por_fase(tempo_centro_s: np.ndarray, aquecimento_fim_s: float | None, paletas: dict):
    fase = rotular_fase(tempo_centro_s, aquecimento_fim_s)
    faltando = set(fase) - set(paletas)
    if faltando:
        raise KeyError(f"paleta nao definida no config para: {sorted(faltando)}")
    cores = np.empty((len(tempo_centro_s), 4))
    info_fases = {}

    for nome_fase in ("aquecimento", "regime_estacionario"):
        mascara = fase == nome_fase
        if not mascara.any():
            continue
        tempo_fase = tempo_centro_s[mascara]
        normalizador = plt.Normalize(vmin=tempo_fase.min(), vmax=tempo_fase.max())
        cmap = paleta_deslocada(plt.get_cmap(paletas[nome_fase]))
        cores[mascara] = cmap(normalizador(tempo_fase))
        info_fases[nome_fase] = (cmap, normalizador)

    return cores, fase, info_fases

def calcular_embeddings(X: np.ndarray, config: dict):
    cfg = config["embeddings"]
    emb_tsne = TSNE(n_components=2, perplexity=cfg["tsne_perplexity"], random_state=cfg["random_state"]).fit_transform(X)
    emb_umap = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"], random_state=cfg["random_state"]).fit_transform(X)
    return emb_tsne, emb_umap


def plotar_embeddings(df: pd.DataFrame, caminho_csv: Path, config: dict, dir_saida: Path, caminho_saida: Path = None, eixo: str = None):
    colunas_features = [c for c in df.columns if c not in ("indice_janela", "tempo_centro_s")]

    X = StandardScaler().fit_transform(df[colunas_features].values)
    paletas = config["dados"]["paletas"]
    cores, _, info_fases = cores_por_fase(df["tempo_centro_s"].values, config["dados"]["aquecimento_fim_s"], paletas)

    emb_tsne, emb_umap = calcular_embeddings(X, config)

    legenda = [
        Line2D([0], [0], marker="o", color="w", label=nome_fase, markerfacecolor=plt.get_cmap(paletas[nome_fase])(0.8), markersize=10)
        for nome_fase in info_fases]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=8, color=cores)
    axs[1].scatter(emb_umap[:, 0], emb_umap[:, 1], s=8, color=cores)

    axs[0].set_title("t-SNE")
    axs[1].set_title("UMAP")

    for ax in axs:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.legend(handles=legenda)

    for nome_fase, (cmap, normalizador) in info_fases.items():
        mappable = plt.cm.ScalarMappable(norm=normalizador, cmap=cmap)
        fig.colorbar(mappable, ax=axs, orientation="horizontal", fraction=0.04, pad=0.1)

    titulo = f"{config['dados']['nome']} - {caminho_csv.stem}"
    if eixo is not None:
        titulo += f" - {eixo}"
    fig.suptitle(titulo, fontsize=16, fontweight="bold")

    if caminho_saida is None:
        caminho_saida = dir_saida / f"{caminho_csv.stem}_embeddings.png"

    fig.savefig(caminho_saida, dpi=config["saida"]["dpi"], bbox_inches="tight")
    plt.close(fig)

def plot_embeddings_por_eixo(caminho_csv: Path, config: dict, dir_saida: Path):

    for eixo in config["eixos"]:
        colunas_features = [c for c in pd.read_csv(caminho_csv, nrows=0).columns if c.startswith(f"{eixo}_")]
        df_eixo = pd.read_csv(caminho_csv, usecols=["indice_janela", "tempo_centro_s"] + colunas_features)
        plotar_embeddings(df_eixo, caminho_csv, config, dir_saida, caminho_saida=dir_saida / f"{caminho_csv.stem}_{eixo}_embeddings.png", eixo=eixo)

def carregar_ponto_medicao(ponto: dict, arquivo_csv: str, base_dir: Path):
    caminho_csv = (base_dir / ponto["dir_csv"] / arquivo_csv).resolve()
    df = pd.read_csv(caminho_csv)
    df["ponto_medicao"] = ponto["nome"]
    return df

def plot_embeddings_pontos_medicao_combinados(pontos: list, arquivo_csv: str, config: dict, dir_saida: Path, base_dir: Path):
    """plota embeddings combinados de varios pontos de medicao no mesmo grafico.
    Cada ponto tem sua propria familia de cores e dentro de cada ponto o gradiente segue o tempo
    em cada fase. Pontos com aquecimento_fim_s = null so tem regime estacionario."""

    dfs = [carregar_ponto_medicao(ponto, arquivo_csv, base_dir) for ponto in pontos]

    colunas_features = [c for c in dfs[0].columns if c not in ("indice_janela", "tempo_centro_s", "ponto_medicao")]
    for ponto, df in zip(pontos, dfs):
        faltando = set(colunas_features) - set(df.columns)
        if faltando:
            raise ValueError(f"{ponto['nome']} sem as colunas: {sorted(faltando)}")

    # escalonamento unico sobre todos os pontos
    #um StandardScaler por ponto centralizaria cada um em zero e os sobreporia artificialmente

    X = StandardScaler().fit_transform(np.vstack([df[colunas_features].values for df in dfs]))
    emb_tsne, emb_umap = calcular_embeddings(X, config)

    fig, axs = plt.subplots(1, 2, figsize=(16, 10), layout="constrained")
    legenda = []
    barras_cor = []
    inicio = 0
    for ponto, df in zip(pontos, dfs):
        fim = inicio + len(df)
        paletas = ponto["paletas"]
        cores, _, info_fases = cores_por_fase(df["tempo_centro_s"].values, ponto.get("aquecimento_fim_s"), paletas)

        for ax, emb in zip(axs, (emb_tsne, emb_umap)):
            ax.scatter(emb[inicio:fim, 0], emb[inicio:fim, 1], s=8, color=cores, marker="o")

        for nome_fase, (cmap, normalizador) in info_fases.items():
            rotulo = f"{ponto['nome']} - {nome_fase}"
            legenda.append(Line2D([0], [0], marker="o", color="w", label=rotulo,
                                  markerfacecolor=plt.get_cmap(paletas[nome_fase])(0.8), markersize=10))
            barras_cor.append((rotulo, cmap, normalizador))
        inicio = fim

    axs[0].set_title("t-SNE")
    axs[1].set_title("UMAP")
    for ax in axs:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.legend(handles=legenda)

    for rotulo, cmap, normalizador in barras_cor:
        barra = fig.colorbar(plt.cm.ScalarMappable(norm=normalizador, cmap=cmap), ax=axs,
                             orientation="horizontal", fraction=0.04, pad=0.1)
        barra.set_label(f"{rotulo} - tempo (s)")

    nomes = " x ".join(ponto["nome"] for ponto in pontos)
    fig.suptitle(f"{nomes} - {Path(arquivo_csv).stem}", fontsize=16, fontweight="bold")

    nome_arquivo = "_".join(ponto["nome"] for ponto in pontos) + f"_{Path(arquivo_csv).stem}_embeddings.png"
    fig.savefig(dir_saida / nome_arquivo, dpi=config["saida"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = Path(__file__).parent
    with open(os.path.join(base_dir, "config.json")) as f:
        config = json.load(f)
        
    dir_saida = preparar_diretorio_saida(config, base_dir)
    dir_tsne_umap = dir_saida / "tsne_umap"
    dir_tsne_umap.mkdir(parents=True, exist_ok=True)

    dir_csv = dir_saida / "csv"

    for caminho_csv in sorted(dir_csv.glob("metricas_*.csv")):
        df = pd.read_csv(caminho_csv)
        plotar_embeddings(df,caminho_csv, config, dir_tsne_umap)
        plot_embeddings_por_eixo(caminho_csv, config, dir_tsne_umap)

    #plot combinado de pontos de medicao diferentes (cada ponto com seus proprios parametros no config)
    cfg_comb = config["combinado"]
    save_dir = (base_dir / cfg_comb["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    #plot_embeddings_pontos_medicao_combinados(cfg_comb["pontos"], cfg_comb["arquivo_plot_combinado"], config, save_dir, base_dir)

if __name__ == "__main__":
    main()
