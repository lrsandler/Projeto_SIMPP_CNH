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

paleta_aquecimento = plt.get_cmap("Reds")
paleta_regime = plt.get_cmap("Blues")

def rotular_fase(tempo_centro_s: np.ndarray, aquecimento_fim_s: float):
    return np.where(tempo_centro_s < aquecimento_fim_s, "aquecimento", "regime_estacionario")

def paleta_deslocada(paleta):
    return ListedColormap(paleta(np.linspace(0.4, 1.0, 256)))

def cores_por_fase(tempo_centro_s: np.ndarray, aquecimento_fim_s: float):
    fase = rotular_fase(tempo_centro_s, aquecimento_fim_s)
    cores = np.empty((len(tempo_centro_s), 4))
    info_fases = {}

    for nome_fase, paleta in (("aquecimento", paleta_aquecimento), ("regime_estacionario", paleta_regime)):
        mascara = fase == nome_fase
        if not mascara.any():
            continue
        tempo_fase = tempo_centro_s[mascara]
        normalizador = plt.Normalize(vmin=tempo_fase.min(), vmax=tempo_fase.max())
        cmap = paleta_deslocada(paleta)
        cores[mascara] = cmap(normalizador(tempo_fase))
        info_fases[nome_fase] = (cmap, normalizador)

    return cores, fase, info_fases



def plotar_embeddings(df: pd.DataFrame, caminho_csv: Path, config: dict, dir_saida: Path, caminho_saida: Path = None, eixo: str = None):
    colunas_features = [c for c in df.columns if c not in ("indice_janela", "tempo_centro_s")]

    X = StandardScaler().fit_transform(df[colunas_features].values)
    cores, _, info_fases = cores_por_fase(df["tempo_centro_s"].values, config["dados"]["aquecimento_fim_s"])

    cfg = config["embeddings"]
    emb_tsne = TSNE(n_components=2, perplexity=cfg["tsne_perplexity"], random_state=cfg["random_state"]).fit_transform(X)
    emb_umap = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"], random_state=cfg["random_state"]).fit_transform(X)

    legenda = [
        Line2D([0], [0], marker="o", color="w", label="aquecimento", markerfacecolor=paleta_aquecimento(0.8), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="regime_estacionario", markerfacecolor=paleta_regime(0.8), markersize=10),]

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


if __name__ == "__main__":
    main()
