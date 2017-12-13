# Perceptron Multicamadas

Implementação sequencial da rede neural artificial Perceptron
multicamadas com treinamento utilizando o algoritmo Backpropagation
para realizar a classificação das amostras desenvolvido para meu TCC,
onde o mesmo tem o objetivo de comparar APIs de programação paralela
que utilizam o adaptador gráfico (essa implementação é sequencial, com
o objetivo de servir apenas como referência).  Recomendo olhar o
arquivo de cabeçalho "perceptron_multicamadas.h" para maior detalhes
de como alocar as estruturas do Perceptron Multicamadas na memória,
carregar arquivos com padrões de treinamento (formato do arquivo será
detalhado adiante) e etc.

Lembrando que as funções principais são a "PerceptronMulticamadas_inicializar",
que aloca toda a estrutura do Perceptron Multicamadas na memória de acordo com
os parâmetros informados a mesma (inclusive a mesma já gera valores iniciais
aleatórios para os pesos das camadas dentro dos limites informados pelas macros 
_RAND_LIM_MIN_ e _RAND_LIM_MAX_) e "PerceptronMulticamadas_backpropagation" 
que como o próprio nome já diz, realiza o treinamento da rede através do 
algoritmo Backpropagation.

```c
/**
 * Método que aloca o Perceptron Multicamadas na memória.
 *
 * @param qtdNeuroniosEntrada Quantidade de neurônios da camada de
 *                              entrada.
 *
 * @param qtdCamadas Quantidade de camadas (em que há processamento).
 *
 * @param qtdNeuroniosCamada Vetor com a quantidade de neurônios para cada
 *                           camada.
 *
 * @param funcaoAtivacaoRede Função de ativação da rede, ou seja, todas as
 *                           camadas da rede estarão atribuidas para serem
 *                           ativadas com tal função (usar a enumeração
 *                           "FuncoesAtivacaoEnum"). Lembrando que, caso se
 *                           deseje que as camadas tenham funções de ativação
 *                           diferente, estabelecer manualmente estes valores
 *                           nas respectivas camadas.
 *
 * @return Referência para a estrutura alocada.
 */
PerceptronMulticamadas *
PerceptronMulticamadas_inicializar(int qtdNeuroniosEntrada,
                                   int qtdCamadas,
                                   int * qtdNeuroniosCamada,
                                   int funcaoAtivacaoRede);
```

```c
/**
 * Método que realiza o "backpropagation" da rede através dos padrões de
 * treinamento até que o erro da rede seja menor ou igual ao erro desejado
 * OU o treinamento atinga a quantidade máxima de epocas (QTD_MAX_EPOCAS).
 *
 * @param pm Perceptron.
 *
 * @param padroes Padrões para treinamento.
 *
 * @param qtdPadroesTreinamento Quantidade de padrões de treinamento.
 *
 * @param taxaAprendizagem Taxa de aprendizagem.
 *
 * @param erroDesejado Condição de parada para o treinamento
 *                     da rede.
 *
 * @param gerarHistorico Se será necessário gerar o histórico ou não.
 *
 * @return Histórico do treinamento. 
 */
HistoricoTreinamento *
PerceptronMulticamadas_backpropagation(PerceptronMulticamadas * pm,
				       PadraoTreinamento * padroes,
				       int qtdPadroesTreinamento,
				       float taxaAprendizagem,
				       float erroDesejado,
				       bool gerarHistorico);
```

## Formato dos arquivos com os padrões de treinamento (não muito inteligente)

Os padrões de treinamento devem ser divididos em dois arquivos, um arquivo
possúindo as amostras, onde cada linha do arquivo é uma amostra com os valores
separados por um ponto e vírgula ";", e outro arquivo com a saída desejada para
a amostra, com o valor de saída desejado para cada neurônio da camada de saída
da rede separados por um ponto e vírgula ";". Por exemplo:

__arquivo-amostras-xor.csv__
```csv
1;0;
1;1;
1;0;
0;1;
```

__arquivo-objetivos-xor.csv__
```csv
1;
0;
1;
1;
```

É importante ressaltar que a própria função que carrega os padrões de treinamento
para o arquivo realiza a normalização das amostras, só sendo necessário informar
à mesma o menor e maior valor presente na matriz de amostras para que seja 
realizada a normalização.

```c
/**
 * Método que carrega os padrões de treinamento de dois arquivos, um com as
 * amostras (que são normalizadas pela função), onde cada linha do mesmo
 * representa uma amostra (com os valores separados por ponto e vírgula ";"), e
 * outro com o vetor de objetivos para cada amostra respectivamente
 * (com os valores separados por ponto e vírgula também).
 *
 * @param nomeArquivoAmostras Nome do arquivo (com extensão) com as amostras
 *                            dos padrões de treinamento ou de teste.
 *
 * @param nomeArquivoObjetivos Nome do arquivo (com extensão) com os
 *                             objetivos dos padrões de treinamento ou
 *                             de teste.
 *
 * @param menorValAmostra Menor valor presente nas amostras (para normalização).
 *
 * @param maiorValAmostra Maior valor presente nas amostras (para normalização).
 *
 * @param qtdItensAmostra Quantidade de itens por amostra.
 *
 * @param qtdItensVetorObjetivo Quantidade de itens por vetor de objetivo.
 *
 * @param qtdPadroes Quantidade de padrões nos arquivos.
 *
 * @return Vetor com os padrões de treinamento ou de teste carregados ou
 *         NULO caso não seja possível abrir os arquivos para leitura.
 */
PadraoTreinamento *
PadraoTreinamento_carregarPadroesArquivo(char * nomeArquivoAmostras,
                                         char * nomeArquivoObjetivos,
                                         float menorValAmostra,
                                         float maiorValAmostra,
                                         int qtdItensAmostra,
                                         int qtdItensVetorObjetivo,
                                         int qtdPadroes);
```
