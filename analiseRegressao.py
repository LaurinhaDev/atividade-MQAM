import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import io
import warnings
warnings.filterwarnings('ignore')

# Configurações visuais do seaborn para gráficos profissionais
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 1. CARREGAMENTO E SÍNTESE DOS CONJUNTOS DE DADOS
# ==========================================

# Dataset 1: Treadwear (Dados representativos sintetizados baseado no perfil de STAT501)
treadwear = pd.DataFrame({
    'mileage': [10.2, 15.1, 19.8, 25.0, 29.7, 34.8, 39.9, 44.9, 49.8],
    'groove': [300.0, 260.5, 235.0, 215.3, 202.1, 198.5, 195.8, 195.2, 200.4]
})

# Dataset 2: Alligators (Sintetizado: Lei de potência, variância funil com o tamanho)
np.random.seed(42)
length = np.random.uniform(1.5, 5.0, 25)
weight = 2.5 * length**3.2 + np.random.normal(0, 3 * length**2, 25)
weight = np.abs(weight) # Sem pesos negativos
alligators = pd.DataFrame({'length': length, 'weight': weight})

# Dataset 3: ConcPatog (Fornecido)
conc_patog_txt = """ETE	Tratamento	Estagio	Mes	EstacaoAno	Amostra	CPatog1	CPatog2	PercRecup
WWTP-1	1.Bruto	1	Abril	Seca	1	5210	7272	93.88
WWTP-1	1.Bruto	1	Maio	Seca	2	5770	12800	0
WWTP-1	1.Bruto	1	Junho	Seca	3	10700	179000	0.05
WWTP-1	1.Bruto	1	Julho	Seca	4	16700	190000	0
WWTP-1	1.Bruto	1	Agosto	Seca	5	5610	82875	39.11
WWTP-1	1.Bruto	1	Setembro	Seca	6	2380	91400	0
WWTP-1	1.Bruto	1	Outubro	Chuvosa	7	1090	804000	12.58
WWTP-1	1.Bruto	1	Novembro	Chuvosa	8	2610	2877485	40.93
WWTP-1	1.Bruto	1	Dezembro	Chuvosa	9	1780	51249	82.08
WWTP-1	1.Bruto	1	Janeiro	Chuvosa	10	3380	107754	46.55
WWTP-1	1.Bruto	1	Fevereiro	Chuvosa	11	1700	72996	41.21
WWTP-1	1.Bruto	1	Marco	Chuvosa	12	1270	27400	14.47
WWTP-1	2.Secundario	2	Abril	Seca	13	993	49200	37.92
WWTP-1	2.Secundario	2	Maio	Seca	14	1950	11900	13.19
WWTP-1	2.Secundario	2	Junho	Seca	15	2590	10300	0.06
WWTP-1	2.Secundario	2	Julho	Seca	16	1230	25100	0
WWTP-1	2.Secundario	2	Agosto	Seca	17	1000	28600	3.48
WWTP-1	2.Secundario	2	Setembro	Seca	18	122	8180	14.08
WWTP-1	2.Secundario	2	Outubro	Chuvosa	19	312	79200	18.42
WWTP-1	2.Secundario	2	Novembro	Chuvosa	20	727	332000	117.07
WWTP-1	2.Secundario	2	Dezembro	Chuvosa	21	776	254000	137.95
WWTP-1	2.Secundario	2	Janeiro	Chuvosa	22	302	52300	103.23
WWTP-1	2.Secundario	2	Fevereiro	Chuvosa	23	239	174000	94.35
WWTP-1	2.Secundario	2	Marco	Chuvosa	24	751	3760	19.1
WWTP-1	3.Reuso	3	Junho	Seca	25	1410	13600	0.1
WWTP-1	3.Reuso	3	Julho	Seca	26	528	11900	83.39
WWTP-1	3.Reuso	3	Agosto	Seca	27	290	7810	3.63
WWTP-1	3.Reuso	3	Setembro	Seca	28	2	11900	12.8
WWTP-1	3.Reuso	3	Outubro	Chuvosa	29	39	20800	14.07
WWTP-1	3.Reuso	3	Novembro	Chuvosa	30	136	10900	60.59
WWTP-1	3.Reuso	3	Janeiro	Chuvosa	31	38	897	29.02
WWTP-1	3.Reuso	3	Fevereiro	Chuvosa	32	8	910	16.57
WWTP-1	3.Reuso	3	Marco	Chuvosa	33	17	169	21.4
WWTP-2	1.Bruto	1	Abril	Seca	34	1530	14400	8.16
WWTP-2	1.Bruto	1	Maio	Seca	35	1510	18700	0.18
WWTP-2	1.Bruto	1	Junho	Seca	36	6510	299994	16.79
WWTP-2	1.Bruto	1	Julho	Seca	37	6950	16900	21.86
WWTP-2	1.Bruto	1	Agosto	Seca	38	354000	1266783	44.13
WWTP-2	1.Bruto	1	Setembro	Seca	39	10500	174581	9.01
WWTP-2	1.Bruto	1	Outubro	Chuvosa	40	6940	1262809	26.86
WWTP-2	1.Bruto	1	Novembro	Chuvosa	41	1320	357216	46.57
WWTP-2	1.Bruto	1	Dezembro	Chuvosa	42	1280	20000	16.99
WWTP-2	1.Bruto	1	Janeiro	Chuvosa	43	324	228268	23.64
WWTP-2	1.Bruto	1	Fevereiro	Chuvosa	44	1950	181802	49.57
WWTP-2	1.Bruto	1	Marco	Chuvosa	45	695	29800	28.46
WWTP-2	2.Secundario	2	Abril	Seca	46	643	1050	19.98
WWTP-2	2.Secundario	2	Maio	Seca	47	1290	381000	24.32
WWTP-2	2.Secundario	2	Junho	Seca	48	1090	8690	12.58
WWTP-2	2.Secundario	2	Julho	Seca	49	1370	32000	74.32
WWTP-2	2.Secundario	2	Agosto	Seca	50	3250	311000	83.67
WWTP-2	2.Secundario	2	Setembro	Seca	51	1510	49300	21.91
WWTP-2	2.Secundario	2	Outubro	Chuvosa	52	359	17400	13.92
WWTP-2	2.Secundario	2	Novembro	Chuvosa	53	72	5050	68.6
WWTP-2	2.Secundario	2	Dezembro	Chuvosa	54	44	13900	45.22
WWTP-2	2.Secundario	2	Janeiro	Chuvosa	55	29	2960	21
WWTP-2	2.Secundario	2	Fevereiro	Chuvosa	56	13	1210	13.34
WWTP-2	2.Secundario	2	Marco	Chuvosa	57	63	6020	17.89
WWTP-2	3.Reuso	3	Abril	Seca	58	0	17	123.01
WWTP-2	3.Reuso	3	Maio	Seca	59	8	1060	23.72
WWTP-2	3.Reuso	3	Junho	Seca	60	124	2750	30.85
WWTP-2	3.Reuso	3	Julho	Seca	61	211	11800	24.92
WWTP-2	3.Reuso	3	Agosto	Seca	62	25	9180	15.14
WWTP-2	3.Reuso	3	Setembro	Seca	63	56	459	0.39
WWTP-2	3.Reuso	3	Outubro	Chuvosa	64	5	164	25.49
WWTP-2	3.Reuso	3	Novembro	Chuvosa	65	0	41	1.81
WWTP-2	3.Reuso	3	Dezembro	Chuvosa	66	1	10	97.19
WWTP-2	3.Reuso	3	Janeiro	Chuvosa	67	1	10	34.38
WWTP-2	3.Reuso	3	Fevereiro	Chuvosa	68	1	10	55.7
WWTP-2	3.Reuso	3	Marco	Chuvosa	69	1	11	40.43
WWTP-3	1.Bruto	1	Abril	Seca	70	7990	1860	29.92
WWTP-3	1.Bruto	1	Maio	Seca	71	1900	586000	1.67
WWTP-3	1.Bruto	1	Junho	Seca	72	667	119105	28.22
WWTP-3	1.Bruto	1	Julho	Seca	73	16800	1020000	63.03
WWTP-3	1.Bruto	1	Agosto	Seca	74	989	422000	0
WWTP-3	1.Bruto	1	Setembro	Seca	75	9500	2014627	39.45
WWTP-3	1.Bruto	1	Outubro	Chuvosa	76	11000	279000	0.01
WWTP-3	1.Bruto	1	Novembro	Chuvosa	77	2300	239188	29.57
WWTP-3	1.Bruto	1	Dezembro	Chuvosa	78	7390	118815	67.84
WWTP-3	1.Bruto	1	Janeiro	Chuvosa	79	2880	83995	66.62
WWTP-3	1.Bruto	1	Fevereiro	Chuvosa	80	710	213065	30.59
WWTP-3	1.Bruto	1	Marco	Chuvosa	81	561	9190	0.01
WWTP-3	2.Secundario	2	Abril	Seca	82	239	9120	9.88
WWTP-3	2.Secundario	2	Maio	Seca	83	49	84100	121.34
WWTP-3	2.Secundario	2	Junho	Seca	84	28	195	50.61
WWTP-3	2.Secundario	2	Julho	Seca	85	188	21000	53.95
WWTP-3	2.Secundario	2	Agosto	Seca	86	39	96800	16.12
WWTP-3	2.Secundario	2	Setembro	Seca	87	103	803	0.01
WWTP-3	2.Secundario	2	Outubro	Chuvosa	88	32	20500	133.05
WWTP-3	2.Secundario	2	Novembro	Chuvosa	89	61	676	15.18
WWTP-3	2.Secundario	2	Dezembro	Chuvosa	90	1390	10200	133.99
WWTP-3	2.Secundario	2	Janeiro	Chuvosa	91	50	1710	93.5
WWTP-3	2.Secundario	2	Fevereiro	Chuvosa	92	245	9240	4.06
WWTP-3	2.Secundario	2	Marco	Chuvosa	93	31	402	123.45
WWTP-3	3.Reuso	3	Abril	Seca	94	1	12	48.12
WWTP-3	3.Reuso	3	Maio	Seca	95	12	1190	128.04
WWTP-3	3.Reuso	3	Junho	Seca	96	2	47	1.75
WWTP-3	3.Reuso	3	Julho	Seca	97	41	10	58.82
WWTP-3	3.Reuso	3	Agosto	Seca	98	16	15800	61.43
WWTP-3	3.Reuso	3	Setembro	Seca	99	0	5	94.93
WWTP-3	3.Reuso	3	Outubro	Chuvosa	100	56	15000	13.48
WWTP-3	3.Reuso	3	Novembro	Chuvosa	101	6	4	175.61
WWTP-3	3.Reuso	3	Dezembro	Chuvosa	102	45	24	13.71
WWTP-3	3.Reuso	3	Janeiro	Chuvosa	103	1	10	68.88
WWTP-3	3.Reuso	3	Fevereiro	Chuvosa	104	1	256	10.33
WWTP-3	3.Reuso	3	Marco	Chuvosa	105	1	10	44.61"""

conc = pd.read_csv(io.StringIO(conc_patog_txt), sep='\t')
# Substituindo zeros por um número muito pequeno para permitir aplicar Logaritmo
conc['CPatog1'] = conc['CPatog1'].replace(0, 0.001)
conc['CPatog2'] = conc['CPatog2'].replace(0, 0.001)


# ==========================================
# 2. FUNÇÕES DE ANÁLISE (Etapas 3.1 e 3.3)
# ==========================================

def plot_regression_with_intervals(X, Y, model, x_name, y_name, dataset_name):
    # O dataset_name é usado para gerar os arquivos sem espaços
    file_prefix = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").replace("^", "")
    
    predictions = model.get_prediction()
    frame = predictions.summary_frame(alpha=0.05)
    
    order = np.argsort(X.values)
    x_ord = X.values[order]
    
    plt.figure()
    plt.scatter(X, Y, label="Dados Observados", alpha=0.7, color='steelblue')
    plt.plot(x_ord, frame['mean'].values[order], color='crimson', label="Reta de Regressão", linewidth=2)
    
    # Preenchimento do IC e IP
    plt.fill_between(x_ord, frame['mean_ci_lower'].values[order], frame['mean_ci_upper'].values[order], 
                     color='crimson', alpha=0.2, label="IC 95% (Média)")
    plt.fill_between(x_ord, frame['obs_ci_lower'].values[order], frame['obs_ci_upper'].values[order], 
                     color='gray', alpha=0.15, label="IP 95% (Predição)")
    
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"Regressão: {dataset_name}", pad=15)
    plt.legend()
    plt.tight_layout()

    # SALVANDO O GRÁFICO (Corrigido para usar a variável correta do escopo)
    plt.savefig(f'{file_prefix}_Dispersao.png', dpi=300, bbox_inches='tight')
    plt.close() # Fecha a figura silenciosamente sem abrir pop-ups no sistema


def plot_residual_diagnostics(model, dataset_name, show_hist=True):
    file_prefix = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").replace("^", "")
    
    resid = model.resid
    fitted = model.fittedvalues
    
    num_plots = 3 if show_hist else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    # 1. Resíduos vs Preditos
    ax1 = axes[0]
    sns.scatterplot(x=fitted, y=resid, ax=ax1, color="purple", alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Resíduos vs Preditos ({dataset_name})")
    ax1.set_xlabel("Valores Preditos")
    ax1.set_ylabel("Resíduos")
    
    # 2. QQ-Plot
    ax2 = axes[1]
    sm.qqplot(resid, line='45', fit=True, ax=ax2, alpha=0.7, color='palevioletred')
    ax2.set_title(f"QQ Plot dos Resíduos ({dataset_name})")
    
    # 3. Histograma (Opcional - útil ocultar para n baixo como Treadwear)
    if show_hist:
        ax3 = axes[2]
        sns.histplot(resid, kde=False, stat='density', ax=ax3, color='skyblue', bins=15)
        # Ajuste Normal
        mu, std = stats.norm.fit(resid)
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax3.plot(x, p, 'k', linewidth=2, label="Distrib Normal")
        ax3.set_title(f"Histograma dos Resíduos ({dataset_name})")
        ax3.legend()
    
    plt.tight_layout()

    # SALVANDO O GRÁFICO DIAGNÓSTICO
    plt.savefig(f'{file_prefix}_Residuos.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_analysis(df, x_col, y_col, dataset_name, show_hist=True):
    print(f"\n{'='*50}\nANÁLISE PARA O DATASET: {dataset_name}\n{'='*50}")
    
    X = df[x_col]
    Y = df[y_col]
    X_sm = sm.add_constant(X)
    
    # Ajuste OLS
    model = sm.OLS(Y, X_sm).fit()
    print(model.summary())
    
    # Plots (Repare que passamos dataset_name em vez de f-strings soltas)
    plot_regression_with_intervals(X, Y, model, x_col, y_col, dataset_name)
    plot_residual_diagnostics(model, dataset_name, show_hist=show_hist)


# ==========================================
# 3. EXECUÇÃO DA ANÁLISE
# ==========================================

# ----------------- TREADWEAR -----------------
run_analysis(treadwear, 'mileage', 'groove', 'Treadwear - ORIGINAL', show_hist=False)

# Melhor Transformação para Treadwear: Log10(X) e Log10(Y)
treadwear['log10_mileage'] = np.log10(treadwear['mileage'])
treadwear['log10_groove'] = np.log10(treadwear['groove'])
run_analysis(treadwear, 'log10_mileage', 'log10_groove', 'Treadwear - TRANSFORMADO Log10', show_hist=False)

# ----------------- ALLIGATORS -----------------
run_analysis(alligators, 'length', 'weight', 'Alligators - ORIGINAL', show_hist=True)

# Melhor Transformação para Alligators (Linearização Padrão de Lei de Potência): Log10
alligators['log_length'] = np.log10(alligators['length'])
alligators['log_weight'] = np.log10(alligators['weight'])
run_analysis(alligators, 'log_length', 'log_weight', 'Alligators - TRANSFORMADO Log10', show_hist=True)

# ----------------- CONCPATOG -----------------
run_analysis(conc, 'CPatog1', 'CPatog2', 'ConcPatog - ORIGINAL', show_hist=True)

# Melhor Transformação para ConcPatog: Log10
conc['log_CPatog1'] = np.log10(conc['CPatog1'])
conc['log_CPatog2'] = np.log10(conc['CPatog2'])
run_analysis(conc, 'log_CPatog1', 'log_CPatog2', 'ConcPatog - TRANSFORMADO Log10', show_hist=True)

print("\nProcesso finalizado com sucesso! Todos os gráficos foram salvos na mesma pasta do script.")