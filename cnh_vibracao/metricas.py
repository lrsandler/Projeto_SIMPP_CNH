from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew

from analise_vibracao import (
    carregar_dados, recortar_dados, preparar_diretorio_saida,
    gerar_combinacoes_janela, criar_janelas_deslizantes, cores_por_eixo, finalizar_figura,)

def metrica_mean(janelas: np.ndarray):
    return np.mean(janelas, axis=1)

def metrica_rms(janelas: np.ndarray):
    return np.sqrt(np.mean(np.square(janelas), axis=1))

def metrica_std(janelas: np.ndarray):
    return np.std(janelas, axis=1)

def metrica_variance(janelas: np.ndarray):
    return np.var(janelas, axis=1)

def metrica_peak(janelas: np.ndarray):
    return np.max(np.abs(janelas), axis=1)

def metrica_peak_to_peak(janelas: np.ndarray):
    return np.max(janelas, axis=1) - np.min(janelas, axis=1)

def metrica_sra(janelas: np.ndarray):
    return np.mean(np.sqrt(np.abs(janelas)), axis=1) ** 2

def metrica_skewness(janelas: np.ndarray):
    return skew(janelas, axis=1)

def metrica_kurtosis(janelas: np.ndarray):
    return kurtosis(janelas, axis=1)

def metrica_crest_factor(janelas: np.ndarray):
    rms = metrica_rms(janelas)
    peak = metrica_peak(janelas)
    return np.divide(peak, rms, out=np.zeros_like(peak), where=rms != 0)

def metrica_shape_factor(janelas: np.ndarray):
    media_abs = np.mean(np.abs(janelas), axis=1)
    rms = metrica_rms(janelas)
    return np.divide(rms, media_abs, out=np.zeros_like(rms), where=media_abs != 0)

def metrica_impulse_factor(janelas: np.ndarray):
    media_abs = np.mean(np.abs(janelas), axis=1)
    peak = metrica_peak(janelas)
    return np.divide(peak, media_abs, out=np.zeros_like(peak), where=media_abs != 0)

def metrica_clearance_factor(janelas: np.ndarray):
    sra = metrica_sra(janelas)
    peak = metrica_peak(janelas)
    return np.divide(peak, sra, out=np.zeros_like(peak), where=sra != 0)

def metrica_entropia(janelas: np.ndarray):
    """Entropia de Shannon da distribuicao de amplitudes de cada janela: H = -sum P(x) log P(x).

    O que mede: o quanto os valores do sinal se espalham entre as faixas de amplitude.
    0 = todas as amostras na mesma faixa (sinal quase constante); 1 = amostras igualmente
    distribuidas em todas as faixas. Descreve o formato da distribuicao de amplitudes, nao o seu tamanho.

    Como calcula:
      1. divide a faixa [min, max] de cada janela em n_bins intervalos iguais (bins); por usar o
         min/max da propria janela, a entropia nao depende da escala do sinal
      2. conta quantas amostras caem em cada bin e divide por N -> probabilidade P(x) de cada bin
      3. soma -P log P (bins vazios nao contribuem) e divide por log(n_bins), o maximo possivel -> 0 a 1

    n_bins pela regra de Rice, n_bins = ceil(2 * N^(1/3)), com N = amostras por janela
    (= taxa de amostragem x duracao da janela): 20 bins para 1 s, 35 para 5 s e 44 para 10 s a 1000 Hz.
    Depende so do tamanho da janela, entao todos os pontos com a mesma janela usam o mesmo n_bins
    e suas entropias sao comparaveis. Entropias de tamanhos de janela diferentes NAO sao comparaveis.

    janelas: (n_janelas, N) -> retorna (n_janelas,)
    """
    n_janelas, n_amostras = janelas.shape
    n_bins = int(np.ceil(2 * n_amostras ** (1 / 3)))

    # posicao relativa de cada amostra dentro da faixa da janela (0 = min, 1 = max)
    minimo = janelas.min(axis=1, keepdims=True)
    amplitude = janelas.max(axis=1, keepdims=True) - minimo
    posicao = np.divide(janelas - minimo, amplitude, out=np.zeros_like(janelas), where=amplitude != 0)

    indice_bin = np.minimum((posicao * n_bins).astype(int), n_bins - 1)

    deslocado = indice_bin + n_bins * np.arange(n_janelas)[:, None]
    contagens = np.bincount(deslocado.ravel(), minlength=n_bins * n_janelas).reshape(n_janelas, n_bins)
    prob = contagens / n_amostras
    return -np.sum(prob * np.log(prob, out=np.zeros_like(prob), where=prob > 0), axis=1) / np.log(n_bins)

# mostra quanta potencia do sinal esta em cada frequencia (periodograma); remove a media antes do calculo
def calcular_densidade_espectral_potencia(janelas: np.ndarray, fs: float):
    return periodogram(janelas, fs=fs, axis=1, scaling="spectrum")

#metricas dominio da frequencia 

# espectro de magnitude unilateral em g
def calcular_espectro_amplitude(janelas: np.ndarray, fs: float):
    freqs, psd = calcular_densidade_espectral_potencia(janelas, fs)
    amplitude = np.sqrt(2 * psd)
    amplitude[:, 0] = np.sqrt(psd[:, 0])
    if janelas.shape[1] % 2 == 0:
        amplitude[:, -1] = np.sqrt(psd[:, -1])
    return freqs, amplitude

# as metricas espectrais recebem o espectro de amplitude (n_janelas x n_frequencias) ja calculado
# assim a FFT e feita uma vez so por eixo

def metrica_media_espectral(espectro: np.ndarray):
    return np.mean(espectro, axis=1)

def metrica_variancia_espectral(espectro: np.ndarray):
    return np.var(espectro, axis=1)

nomes_metricas_tempo = ["mean", "rms", "std", "variance", "peak", "peak_to_peak", "sra",
                   "skewness", "kurtosis", "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
                   "entropia"]

nomes_metricas_frequencia = ["media_espectral", "variancia_espectral"]

def calcular_grade(n_itens: int, max_colunas: int = 4):
    n_colunas = min(max_colunas, n_itens)
    #calcula n de linhas necessarias p todos itens 
    n_linhas = -(-n_itens // n_colunas)  
    return n_linhas, n_colunas

#pode variar a quantidade de metricas a serem plotadas (passadas em metricas_boxplot)
def plotar_boxplots_metricas(colunas_features: dict, metricas_boxplot: list, eixos: list, config: dict, dir_saida: Path,
                              frame_size_s: float, overlap_pct: float):
    cores = cores_por_eixo(eixos)
    n_linhas, n_colunas = calcular_grade(len(metricas_boxplot), max_colunas=5)

    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(4 * n_colunas, 3.3 * n_linhas), squeeze=False)
    eixos_grafico = list(axs.flat)

    for ax, metrica in zip(eixos_grafico, metricas_boxplot):
        dados = [colunas_features[f"{eixo}_{metrica}"] for eixo in eixos]
        caixas = ax.boxplot(dados, tick_labels=["x", "y", "z"], patch_artist=True)
        for patch, eixo in zip(caixas["boxes"], eixos):
            patch.set_facecolor(cores[eixo])
        for mediana in caixas["medians"]:
            mediana.set_color("black")
        ax.set_title(metrica, fontsize=16, fontweight="bold")

        ax.tick_params(axis="x", labelsize=16 )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.grid(True, axis="y", linestyle="-", alpha=0.3)

    for ax_sobrando in eixos_grafico[len(metricas_boxplot):]:
        ax_sobrando.axis("off")

    rotulo = f"janela {f"{frame_size_s:g}s"} / overlap {overlap_pct:g}%"
    fig.suptitle(f"Boxplots das Metricas - {config['dados']['nome']} - {rotulo}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    dir_saida_boxplot = dir_saida / "boxplots_metricas"
    dir_saida_boxplot.mkdir(parents=True, exist_ok=True)
    finalizar_figura(fig, dir_saida_boxplot, f"boxplot_metricas_frame{f"{frame_size_s:g}s"}_overlap{overlap_pct:g}pct", config)


def calcular_metricas(df, config: dict, dir_saida_csv: Path, dir_saida_boxplot: Path):
    eixos = config["eixos"]
    fs = config["dados"]["taxa_amostragem_hz"]
    coluna_tempo = config["dados"]["colunas"][0]
    janela_cfg = config["janelamento"]
    combinacoes = gerar_combinacoes_janela(janela_cfg["tamanhos_janela_s"], janela_cfg["overlaps_percentuais"])
    tempo = df[coluna_tempo].values
    nomes_metricas = nomes_metricas_tempo + nomes_metricas_frequencia

    for frame_size_s, hop_s, overlap_pct in combinacoes:
        colunas_features = {}
        tempos_centro_ref = None

        for eixo in eixos:
            janelas, tempos_centro = criar_janelas_deslizantes(df[eixo].values, tempo, fs, frame_size_s, hop_s)
            if janelas is None:
                tempos_centro_ref = None
                break
            tempos_centro_ref = tempos_centro

            _, espectro = calcular_espectro_amplitude(janelas, fs)

            valores = {
                "mean": metrica_mean(janelas),
                "rms": metrica_rms(janelas),
                "std": metrica_std(janelas),
                "variance": metrica_variance(janelas),
                "peak": metrica_peak(janelas),
                "peak_to_peak": metrica_peak_to_peak(janelas),
                "sra": metrica_sra(janelas),
                "skewness": metrica_skewness(janelas),
                "kurtosis": metrica_kurtosis(janelas),
                "crest_factor": metrica_crest_factor(janelas),
                "shape_factor": metrica_shape_factor(janelas),
                "impulse_factor": metrica_impulse_factor(janelas),
                "clearance_factor": metrica_clearance_factor(janelas),
                "entropia": metrica_entropia(janelas),
                "media_espectral": metrica_media_espectral(espectro),
                "variancia_espectral": metrica_variancia_espectral(espectro),
            }
            for nome in nomes_metricas:
                colunas_features[f"{eixo}_{nome}"] = valores[nome]

        if tempos_centro_ref is None:
            continue

        tabela = {"indice_janela": np.arange(len(tempos_centro_ref)), "tempo_centro_s": tempos_centro_ref}
        tabela.update(colunas_features)

        caminho = dir_saida_csv / f"metricas_frame{f"{frame_size_s:g}s"}_overlap{overlap_pct:g}pct.csv"
        pd.DataFrame(tabela).to_csv(caminho, index=False)

        plotar_boxplots_metricas(colunas_features, nomes_metricas, eixos, config, dir_saida_boxplot, frame_size_s, overlap_pct)


def main():
    base_dir = Path(__file__).parent
    with open(os.path.join(base_dir, "config.json")) as f:
        config = json.load(f)
    df = carregar_dados(config, base_dir)

    coluna_tempo = config["dados"]["colunas"][0]
    inicio_s, fim_s = config["dados"]["intervalo_tempo_s"]
    df = recortar_dados(df, coluna_tempo, inicio_s, fim_s)

    dir_saida = preparar_diretorio_saida(config, base_dir)
    dir_saida_csv = dir_saida / "csv"
    dir_saida_csv.mkdir(parents=True, exist_ok=True)
    calcular_metricas(df, config, dir_saida_csv, dir_saida)


if __name__ == "__main__":
    main()
