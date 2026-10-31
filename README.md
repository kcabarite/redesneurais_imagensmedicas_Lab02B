# Classificação de pixels com redes neurais MLP

Atividades práticas de segmentação de imagens em tons de cinza com uma rede
neural MLP (*Multilayer Perceptron*), desenvolvidas em Python com Keras e
TensorFlow. Cada pixel é classificado como fundo ou objeto, usando uma máscara
de referência (*gold standard*) como alvo do treinamento.

## Atividades

| Arquivo | Atributos de entrada | Dados utilizados |
| --- | --- | --- |
| `Atividade_IntroPixClass01-Parte01.py` | Intensidade do pixel repetida em duas colunas | `IV00.pgm` e `GS00.pgm` |
| `Atividade_IntroPixClass01-Parte02.py` | Intensidade do pixel e distância normalizada ao centro da imagem | `IVNoise_00.pgm` a `IVNoise_02.pgm` e `GS00.pgm` a `GS02.pgm` |

Na **Parte 01**, os mesmos pixels são usados no treinamento e na avaliação.
A atividade permite observar a classificação da imagem, mas sua acurácia não
mede a generalização para imagens novas. O treinamento usa pesos de classe
`{0: 1.0, 1: 5.0}` e o limiar de classificação é `0.7`.

Na **Parte 02**, as imagens de índices `01` e `02` são usadas para treinamento,
e a imagem de índice `00` é usada para validação durante o treinamento e para
avaliação final. O limiar de classificação é `0.5`. O código espera imagens de
**400 × 400 pixels**.

As duas atividades usam uma camada de entrada com 2 unidades e ativação linear,
uma camada oculta com 12 unidades e ativação sigmoid e uma saída sigmoid para
a classificação binária. O treinamento utiliza Adam, perda
`binary_crossentropy`, lotes de 16.000 pixels e 1.000 épocas.

## Arquivos de imagens

- `IV*.pgm`: imagens de entrada em tons de cinza, incluindo as versões
  introdutórias e as versões com ruído (`IVNoise_*.pgm`).
- `GS*.pgm`: máscaras de referência correspondentes. Nos scripts, valores
  normalizados maiores que `0.5` representam a classe positiva.
- `requirements.txt`: dependências Python do projeto.

Os arquivos `IVIntro01.pgm`, `IVIntro02.pgm`, `GSIntro01.pgm` e
`GSIntro02.pgm` são alternativas comentadas no código da Parte 01.

## Preparar o ambiente

Em um terminal PowerShell aberto na pasta do projeto, com Python instalado:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

As dependências incluem NumPy, Matplotlib, OpenCV, pandas, seaborn,
scikit-learn, scikit-image, SciPy, TensorFlow, Keras, openpyxl e ipykernel.

## Executar no VS Code

1. Instale as extensões **Python** e **Jupyter**, da Microsoft.
2. Abra a pasta do projeto no VS Code.
3. Use **Python: Select Interpreter** para selecionar
   `.venv\Scripts\python.exe`.
4. Abra o script da atividade e clique em **Run Cell** acima das células
   marcadas com `#%%`, executando-as na ordem do arquivo.
5. Na janela **Python Interactive**, selecione o mesmo ambiente como kernel.
   Reinicie o kernel se tiver alterado o ambiente.

Execute as atividades com a pasta do projeto como diretório de trabalho,
pois as imagens são carregadas por caminhos relativos.

Os gráficos exibem as imagens, as máscaras de referência e a segmentação
prevista. A acurácia fica na variável `precisao` e a matriz de confusão em
`matriz_de_confusao`; os scripts não imprimem esses valores explicitamente.
Para consultá-los, execute em uma célula após a avaliação:

```python
print("Acurácia:", precisao)
print("Matriz de confusão:\n", matriz_de_confusao)
```

Também é possível executar pelo terminal:

```powershell
.\.venv\Scripts\python.exe .\Atividade_IntroPixClass01-Parte01.py
.\.venv\Scripts\python.exe .\Atividade_IntroPixClass01-Parte02.py
```

Para visualizar os gráficos ao executar pelo terminal, adicione `plt.show()`
ao final do script. Na janela interativa, os gráficos aparecem durante a
execução das células.

## Problemas comuns

- **Módulo não encontrado:** instale `requirements.txt` no ambiente usado
  para executar o código. No VS Code, confira também o kernel da janela
  interativa, além do interpretador do editor.
- **Falha ao carregar uma imagem:** confira o nome do arquivo e execute o
  código a partir da pasta do projeto.
- **Treinamento demorado:** ambos os scripts estão configurados para 1.000
  épocas. Para experimentar o fluxo, reduza `epochs` na chamada `model.fit`.
- **Kernel encerrado por falta de memória:** feche outros kernels e
  aplicativos para liberar memória antes de tentar novamente.
