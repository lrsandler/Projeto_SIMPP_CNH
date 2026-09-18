import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolver_caminho(base_dir: Path, caminho: str):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else (base_dir / caminho)

def carregar_dados(config: dict, base_dir: Path):
    dados_cfg = config["dados"]
    arquivo = resolver_caminho(base_dir, dados_cfg["arquivo"])
    return pd.read_csv(arquivo, sep=r"\s+", header=None, names=dados_cfg["colunas"])

def preparar_diretorio_saida(config: dict, base_dir: Path):
    dir_saida = resolver_caminho(base_dir, config["saida"]["diretorio"])
    dir_saida.mkdir(parents=True, exist_ok=True)
    return dir_saida

def finalizar_figura(fig, dir_saida: Path, nome_arquivo: str, config: dict):
    saida_cfg = config["saida"]
    caminho = dir_saida / f"{nome_arquivo}.png"
    fig.savefig(caminho, dpi=saida_cfg["dpi"], bbox_inches="tight")
    if saida_cfg.get("mostrar_graficos", False):
        plt.show()
    plt.close(fig)


paleta = plt.get_cmap("tab10").colors
def cores_por_eixo(eixos: list):
    return {eixo: paleta[i % len(paleta)] for i, eixo in enumerate(eixos)}


def gerar_combinacoes_janela(tamanhos_s: list, overlaps_percentuais: list):
    combinacoes = []
    for frame_size_s in tamanhos_s:
        for overlap_pct in overlaps_percentuais:
            if overlap_pct >= 100:
                print(f"overlap de {overlap_pct}% invalido")
                continue
            #calcula os saltos a partir do percentual de overlap
            hop_s = frame_size_s * (1 - overlap_pct / 100)
            combinacoes.append((frame_size_s, hop_s, overlap_pct))
    return combinacoes

def recortar_dados(df: pd.DataFrame, coluna_tempo: str, inicio_s: float, fim_s: float):
    return df[(df[coluna_tempo] >= inicio_s) & (df[coluna_tempo] <= fim_s)].reset_index(drop=True)


def plotar_corte(df: pd.DataFrame, config: dict, dir_saida: Path, inicio_s: float, fim_s: float):
    coluna_tempo = config["dados"]["colunas"][0]
    eixos = config["eixos"]
    cores = cores_por_eixo(eixos)
    aquecimento_fim_s = config["dados"]["aquecimento_fim_s"] #teste com aquecimento 300s

    fig, axs = plt.subplots(len(eixos), 1, figsize=(12, 3 * len(eixos)), sharex=True)
    if len(eixos) == 1:
        axs = [axs]

    for ax, eixo in zip(axs, eixos):
        ax.plot(df[coluna_tempo], df[eixo], linewidth=0.5, color=cores[eixo])
        ax.axvspan(inicio_s, aquecimento_fim_s, color="red", alpha=0.08)
        ax.axvspan(aquecimento_fim_s, fim_s, color="blue", alpha=0.08)
        ax.axvline(inicio_s, color="black", linestyle="--", linewidth=1)
        ax.axvline(aquecimento_fim_s, color="black", linestyle=":", linewidth=1)
        ax.axvline(fim_s, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(eixo, fontsize=16, fontweight="bold")
        ax.grid(True, linestyle="-", alpha=0.3)

        ax.tick_params(axis="x", labelsize=16 )
        ax.tick_params(axis="y", labelsize=16 )
        

    axs[-1].set_xlabel(coluna_tempo)
    fig.suptitle(f"Recorte da Série (aquecimento {inicio_s:g}s-{aquecimento_fim_s:g}s + regime estacionario {aquecimento_fim_s:g}s-{fim_s:g}s) - {config['dados']['nome']}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    finalizar_figura(fig, dir_saida, "corte_serie", config)


def plotar_dominio_tempo(df: pd.DataFrame, config: dict, dir_saida: Path):
    coluna_tempo = config["dados"]["colunas"][0]
    eixos = config["eixos"]
    fator = 1 # fator de downsample se o sinal futuramente for muito grande

    tempo = df[coluna_tempo].values[::fator]
    cores = cores_por_eixo(eixos)
    fig, axs = plt.subplots(len(eixos), 1, figsize=(12, 3 * len(eixos)), sharex=True)
    if len(eixos) == 1:
        axs = [axs]

    for ax, eixo in zip(axs, eixos):
        ax.plot(tempo, df[eixo].values[::fator], linewidth=0.5, color=cores[eixo])
        ax.set_ylabel(eixo)
        ax.grid(True, linestyle="-", alpha=0.3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.tick_params(axis="x", labelsize=16 )
        ax.tick_params(axis="y", labelsize=16 )

    axs[-1].set_xlabel(coluna_tempo)
    fig.suptitle(f"Serie Temporal - {config['dados']['nome']}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    finalizar_figura(fig, dir_saida, "serie_temporal", config)

def calcular_espectro_fft(sinal: np.ndarray, tempo: np.ndarray, fs: float, frame_size_s: float, hop_s: float):
    janelas, _ = criar_janelas_deslizantes(sinal, tempo, fs, frame_size_s, hop_s)
    if janelas is None:
        return None, None

    frame_len = janelas.shape[1]

    freqs = np.fft.fftfreq(frame_len, d=1 / fs)[:frame_len // 2]
    amplitudes = np.empty((janelas.shape[0], frame_len // 2))
    for i, sinal_janela in enumerate(janelas):
        espectro = np.fft.fft(sinal_janela)
        amplitudes[i] = (2.0 / frame_len) * np.abs(espectro[:frame_len // 2])

    espectro_medio = amplitudes.mean(axis=0)
    return freqs, espectro_medio


def plotar_dominio_frequencia(df: pd.DataFrame, config: dict, dir_saida: Path):
    janela_cfg = config["janelamento"]
    fs = config["dados"]["taxa_amostragem_hz"]
    coluna_tempo = config["dados"]["colunas"][0]
    eixos = config["eixos"]
    cores = cores_por_eixo(eixos)
    combinacoes = gerar_combinacoes_janela(janela_cfg["tamanhos_janela_s"], janela_cfg["overlaps_percentuais"])
    tempo = df[coluna_tempo].values

    for frame_size_s, hop_s, overlap_pct in combinacoes:
        rotulo = f"janela {f"{frame_size_s:g}s"} / overlap {overlap_pct:g}%"

        fig, axs = plt.subplots(len(eixos), 1, figsize=(12, 3 * len(eixos)), sharex=True)
        if len(eixos) == 1:
            axs = [axs]

        algum_valido = False
        for ax, eixo in zip(axs, eixos):
            freqs, amplitude = calcular_espectro_fft(df[eixo].values, tempo, fs, frame_size_s, hop_s)
            if freqs is None:
                ax.text(0.5, 0.5, "janela maior que o sinal disponivel", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.tick_params(axis="x", labelsize=16 )
            ax.tick_params(axis="y", labelsize=16 )

            algum_valido = True
            ax.plot(freqs, amplitude, linewidth=1, color=cores[eixo])
            ax.set_ylabel(eixo, fontsize=16, fontweight="bold")
            ax.grid(True, linestyle="-", alpha=0.3)

        axs[-1].set_xlabel("Frequencia [Hz]", fontsize=16, fontweight="bold")
        axs[-1].set_xlim(0, fs / 2)
        axs[0].set_title(f"Espectro de Frequencia - FFT (Amplitude [g]) - {config['dados']['nome']} - {rotulo}", fontsize=16, fontweight="bold")
        fig.tight_layout()

        if algum_valido:
            finalizar_figura(fig, dir_saida, f"espectro_freq_frame{f"{frame_size_s:g}s"}_overlap{overlap_pct:g}pct", config)
        else:
            plt.close(fig)


def calcular_espectrograma(sinal: np.ndarray, tempo: np.ndarray, fs: float, frame_size_s: float, hop_s: float):
    janelas, tempos_centro = criar_janelas_deslizantes(sinal, tempo, fs, frame_size_s, hop_s)

    frame_len = janelas.shape[1]
    peso_janela = np.hanning(frame_len)
    fator_normalizacao = peso_janela.sum()

    freqs = np.fft.fftfreq(frame_len, d=1 / fs)[:frame_len // 2]
    amplitudes = np.empty((janelas.shape[0], frame_len // 2))
    for i, sinal_janela in enumerate(janelas):
        espectro = np.fft.fft(sinal_janela * peso_janela)
        amplitudes[i] = (2.0 / fator_normalizacao) * np.abs(espectro[:frame_len // 2])

    return tempos_centro, freqs, amplitudes


def plotar_espectrograma(df: pd.DataFrame, config: dict, dir_saida: Path):
    janela_cfg = config["janelamento"]
    fs = config["dados"]["taxa_amostragem_hz"]
    coluna_tempo = config["dados"]["colunas"][0]
    eixos = config["eixos"]
    combinacoes = gerar_combinacoes_janela(janela_cfg["tamanhos_janela_s"], janela_cfg["overlaps_percentuais"])
    tempo = df[coluna_tempo].values

    for frame_size_s, hop_s, overlap_pct in combinacoes:
        rotulo = f"janela {f"{frame_size_s:g}s"} / overlap {overlap_pct:g}%"

        fig, axs = plt.subplots(len(eixos), 1, figsize=(12, 3 * len(eixos)), sharex=True)
        if len(eixos) == 1:
            axs = [axs]

        algum_valido = False
        for ax, eixo in zip(axs, eixos):
            tempos_centro, freqs, amplitudes = calcular_espectrograma(df[eixo].values, tempo, fs, frame_size_s, hop_s)
            if tempos_centro is None:
                ax.text(0.5, 0.5, "janela maior que o sinal", ha="center", va="center", transform=ax.transAxes)
                continue

            algum_valido = True
            amplitudes_db = 20 * np.log10(np.maximum(amplitudes, 1e-12))
            malha = ax.pcolormesh(tempos_centro, freqs, amplitudes_db.T, shading="auto", cmap="plasma")
            fig.colorbar(malha, ax=ax, label="Amplitude [dB]")
            ax.set_ylabel(f"{eixo}\nFrequencia [Hz]", fontsize=16, fontweight="bold")
            ax.set_ylim(0, fs / 2)
            ax.set_title(eixo, fontsize=16, fontweight="bold")
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

        axs[-1].set_xlabel(coluna_tempo, fontsize=16, fontweight="bold")
        fig.suptitle(f"Espectrograma - {config['dados']['nome']} - {rotulo}", fontsize=16, fontweight="bold")
        fig.tight_layout()

        if algum_valido:
            finalizar_figura(fig, dir_saida, f"espectrograma_frame{f"{frame_size_s:g}s"}_overlap{overlap_pct:g}pct", config)
        else:
            plt.close(fig)


def criar_janelas_deslizantes(sinal: np.ndarray, tempo: np.ndarray, fs: float, frame_size_s: float, hop_s: float):
    frame_len = int(round(frame_size_s * fs))
    hop_len = int(round(hop_s * fs))

    if frame_len < 1 or hop_len < 1 or frame_len > len(sinal):
        return None, None

    indices_inicio = range(0, len(sinal) - frame_len + 1, hop_len)
    janelas = np.empty((len(indices_inicio), frame_len))
    tempos_centro = np.empty(len(indices_inicio))
    for i, inicio in enumerate(indices_inicio):
        janelas[i] = sinal[inicio:inicio + frame_len]
        tempos_centro[i] = tempo[inicio + frame_len // 2]

    return janelas, tempos_centro


def main():

    base_dir = Path(__file__).parent
    with open(os.path.join(os.path.join(base_dir, "config.json")), "r", encoding="utf-8") as f:
        config = json.load(f)

    df = carregar_dados(config, base_dir)
    dir_saida = preparar_diretorio_saida(config, base_dir)

    coluna_tempo = config["dados"]["colunas"][0]
    inicio_s, fim_s = config["dados"]["intervalo_tempo_s"]
    plotar_corte(df, config, dir_saida, inicio_s, fim_s)

    #dataframe recortado para as demais analises!
    df = recortar_dados(df, coluna_tempo, inicio_s, fim_s)

    plotar_dominio_tempo(df, config, dir_saida)
    plotar_dominio_frequencia(df, config, dir_saida)
    plotar_espectrograma(df, config, dir_saida)

if __name__ == "__main__":
    main()
