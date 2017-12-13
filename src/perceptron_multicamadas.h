/****************************************************************************
 * Projeto Perceptron Multicamadas (sequencial).                            *
 *                                                                          *
 * Implementação da rede Perceptron Multicamadas com treinamento utilizando *
 * o método "backpropagation" para classificação das amostras.              *
 *                                                                          *
 * @author Gilberto Augusto de Oliveira Bastos.                             *
 * @copyright BSD-2-Clause                                                  *
 ****************************************************************************/

#ifndef PERCEPTRON_MULTICAMADAS_H
#define PERCEPTRON_MULTICAMADAS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>
#include <complex.h>
#include "uniform.h" /* Biblioteca para gerar números aleatórios uniformemente
                        distribuídos. */

/* Quantidade máxima de épocas de treinamento. */
#define QTD_MAX_EPOCAS 1000

/* Intervalos para geração dos números randômicos
para os pesos. */
#define RAND_LIM_MIN -1
#define RAND_LIM_MAX  1

/* Valor inicial para o BIAS... */
#define BIAS 1.0

/* Para mostra informações estatísticas. */
#define INFO_ESTATISTICAS 1

/**
 * Enumerações para as funções de ativação da rede
 */
enum FuncoesAtivacaoEnum
{
  Identidade,
  Degrau,
  Sigmoide,
  TangHiperbolica,
};

/***********************************************
 * Estruturas do Perceptron Multicamadas e etc *
 ***********************************************/

/**
 * Estrutura que irá representar uma camada da rede neural, os dados da
 * mesma serão organizados em vetores e tipos primitivos para facilitar
 * o envio dos mesmos para o dispositivo acelerador através do OpenACC.
 */
typedef struct
{
  /** Vetor que irá armazenar os pesos desta camada na conveção
  "row-major". */
  float * W;

  /** Vetor que irá armazenar o grau de ativação dos neurônios
  desta camada. */
  float * neuronioAtivacao;

  /** Vetor que irá armazenar a derivada da função de ativação dos neurônios
  desta camada. */
  float * neuronioDerivada;

  /** Vetor que irá armazenar o erro retropropagado calculado para cada
  neurônio desta camada. */
  float * neuronioErroRprop;

  /** Vetor que irá armazenar os bias para cada neurônio da camada. */
  float * bias;

  /** Variável que irá armazenar a quantidade de neurônios desta camada. */
  int qtdNeuronios;

  /** Variável que irá armazenar a função de ativação para esta camada
  (usar a enumeração "FuncoesAtivacaoEnum"). */
  int funcaoAtivacao;

} Camada;

/**
 * Estrutura que irá armazenar as camadas do Perceptron.
 */
typedef struct
{
  /** Vetor de camadas. */
  const Camada ** camadas;

  /** Quantidade de camadas. */
  int qtdCamadas;

  /** Tamanho da entrada (quantidade de "neurônios"). */
  int qtdNeuroniosEntrada;

} PerceptronMulticamadas;

/**
 * Estrutura que irá representar um padrão de treinamento
 * para ser apresentado à rede.
 */

typedef struct
{
  /** A vetor com a amostra (entrada da rede). */
  const float * amostra;

  /** Vetor com os valores desejados para saída da rede (objetivo). */
  const float * alvo;

} PadraoTreinamento;

/***********************************************************
 * Estruturas que irão armazenar as informações referentes *
 * ao treinamento da rede.                                 *
 ***********************************************************/

/**
 * Estrutura que irá armazenar as informações de 
 * uma época de treinamento da rede neural.
 */
typedef struct 
{
  /** Duração para o treinamento da época. */
  float duracaoSegs;

  /** Erro global para época após o treinamento
  da mesma. */
  float erroGlobal;
  
} InfoEpocaTreinamento;

/** Nó da lista... */
struct ListaNo
{
  InfoEpocaTreinamento dado;
  struct ListaNo * proxNo;
};

/**
 * Estrutura que irá armazenar informações
 * como a configuração da rede neural, taxa de 
 * aprendizagem e as informações sobre as 
 * épocas de treinamento.
 */
typedef struct
{
  /** Tamanho da entrada (camada de entrada). */
  int qtdNeuroniosEntrada;

  /** Quantidade de camadas da rede. */
  int qtdCamadas;

  /** Variável que irá armazenar a configuração
    * da rede neural em uma "string."
    *
    * Ex: 800-700-600-500
    */
  char strConfigRedeNeural[240];

  /** Taxa de aprendizagem da rede neural. */
  float taxaAprendizagem;

  /** Erro desejado. */
  float erroDesejado;

  /** Épocas de treinamento (linked list). */
  struct ListaNo * listaEpocas;
  
} HistoricoTreinamento;


/*********************************************************************
 * Funções da rede neural (inicialização do Perceptron Multicamadas, *
 * backpropagation, feedfoward e etc)...                             *
 *********************************************************************/

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

/**
 * Método que aloca uma camada e retorna a referência para a mesma.
 *
 * @param qtdNeuronios Quantidade de neurônios que a camada irá possuir.
 *
 * @param qtdPesosNeuronio Quantidade de pesos que cada
 * neurônio irá possuir.
 *
 * @param funcaoAtivacao Função de ativação desta camada
 *                       (usar a enumeração "FuncoesAtivacaoEnum").
 *
 * @return Referência para a camada alocada.
 */
Camada * __alocarCamada(int qtdNeuronios, int qtdPesosNeuronio,
                        int funcaoAtivacao);

/**
 * Método que aloca um vetor de pesos, gera os números aleatórios
 * (intervalo de RAND_LIM_MIN, RAND_LIM_MAX) e retorna a referência para o mesmo.
 *
 * @param qtdPesos Quantidade de pesos do a serem alocados.
 *
 * @return Referência para o vetor de pesos alocado na memória.
 */
float * __alocarVetorPesosRandomicos(int qtdPesos);

/**
 * Método que tem o objetivo de calcular a ativação dos neurônios da primeira
 * camada.
 *
 * @param camada Primeira camada.
 *
 * @param amostra Amostra sendo apresentada à rede.
 *
 * @param qtdNeuroniosEntrada Tamanho da entrada (quantidade de itens da
 *                            amostra).
 */
void Camada_calcularAtivacaoNeuroniosPrimeiraCamada(const Camada * camada,
                                                    const float * amostra,
                                                    int qtdNeuroniosEntrada);

/**
 * Método que tem o objetivo de calcular a ativação dos neurônios de qualquer
 * camada salvo a primeira.
 *
 * @param camadaAnterior Camada anterior à camada que se deseja calcular a
 *                       ativação dos neurônios.
 *
 * @param camada Camada da qual se deseja calcular a ativação dos neurônios.
 */
void Camada_calcularAtivacaoNeuroniosCamada(const Camada * camadaAnterior,
                                            const Camada * camada);

/**
 * Método que realizar o cálculo do erro retropropagado dos neurônios de uma
 * camada da rede salvo a última.
 *
 * @param camada Camada da qual se deseja calcular o erro retropropagado dos
 *               neurônios.
 *
 * @param camadaPosterior Camada posterior à camada que se deseja calcular o
 *                        o erro retropropagado dos neurônios.
 */ 
void Camada_calcularErroRpropNeuroniosCamada(const Camada * camada,
                                             const Camada * camadaPosterior);

/**
 * Método que realiza o cálculo do erro retropropagado dos neurônios da última
 * camada.
 *
 * @param camada Última camada da rede.
 *
 * @param alvo Vetor com os valores desejados para os neurônios desta camada
 *             (vetor de objetivo).
 *
 * @return Erro para o padrão apresentado à rede.
 */
float Camada_calcularErroRpropNeuroniosUltimaCamada(const Camada * camada,
                                                    const float * alvo);

/**
 * Método que atualiza os pesos dos neurônios da primeira camada.
 *
 * @param camada Primeira camada.
 *
 * @param amostra Amostra.
 *
 * @param qtdNeuroniosEntrada Tamanho da entrada (quantidade de itens da
 *                            amostra).
 *
 * @param taxaAprendizagem Taxa de aprendizagem.
 */
void Camada_atualizarPesosNeuroniosPrimeiraCamada(const Camada * camada,
                                                  const float * amostra,
                                                  int qtdNeuroniosEntrada,
                                                  float taxaAprendizagem);

/**
 * Método que atualiza os pesos dos neurônios de uma camada da rede salvo a
 * a primeira.
 *
 * @param camadaAnterior Camada anterior à camada da qual se deseja atualizar
 *                       os pesos dos neurônios.
 *
 * @param camada Camada da qual se deseja atualizar os pesos dos neurônios.
 *
 * @param taxaAprendizagem Taxa de aprendizagem.
 */
void Camada_atualizarPesosNeuroniosCamada(const Camada * camadaAnterior,
                                          const Camada * camada,
                                          float taxaAprendizagem);

/**
 * Método que realiza alimentação da rede (feedfoward) com amostra.
 *
 * @param pm Referência para Perceptron Multicamadas.
 *
 * @param amostra Vetor da amostra.
 */
void PerceptronMulticamadas_feedfoward(PerceptronMulticamadas * pm,
                                       const float * amostra);

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

/**
 * Método que realiza a normalização de um vetor através do método "min-max".
 *
 * @param v Vetor a ser normalizado.
 *
 * @param n Quantidade de itens do vetor.
 *
 * @param min Menor valor presente no vetor.
 *
 * @param max Maior valor presente no vetor.
 */
void normalizacaoMinMax(float * v, int n, float min, float max);

/***********************
 * Funções de ativação *
 ***********************/

/**
 * Método que realiza o cálculo da função degrau.
 *
 * @param z Parâmetro para o cálculo da função degrau.
 *
 * @return Valor do cálculo da função degrau.
 */
float funcaoDegrau(float z);

/**
 * Método que realiza o cálculo da derivada da função degrau.
 *
 * @param valDegrau Valor da função degrau da qual se deseja calcular
 *                  a derivada.
 *
 * @return Valor do cálculo da derivada da função degrau.
 */
float derivadaFuncaoDegrau(float valDegrau);

/**
 * Método que realiza o cálculo da função sigmóide.
 *
 * @param z Parâmetro para o cálculo da função sigmóide.
 *
 * @return Valor do cálculo da função sigmóide.
 */
float funcaoSigmoide(float z);

/**
 * Método que realizar o cálculo da derivada da função sigmóide.
 *
 * @param valSigmoide Valor da função sigmóide da qual se deseja calcular
 *                    a derivada.
 *
 * @return Valor do cálculo da derivada da função sigmóide.
 */
float derivadaFuncaoSigmoide(float valSigmoide);

/**
 * Método que realiza o cálculo da função tangente hiperbólica.
 *
 * @param z Parâmetro para o cálculo da função tangente hiperbólica.
 *
 * @return Valor do cálculo da função tangente hiperbólica.
 */
float funcaoTangHiperbolica(float z);

/**
 * Método que realiza o cálculo da derivada da função tangente hiperbólica.
 *
 * @param valTangHiperbolica Valor da função tangente hiperbólica da qual se
 *                           deseja calcular a derivada.
 *
 * @return Valor do cálculo da derivada da função tangente hiperbólica.
 */
float derivadaFuncaoTangHiperbolica(float valTangHiperbolica);

/****************************************************************************
 * Funções para carregar os padrões de treinamento de arquivos, calcular    *
 * a corretude de um treinamento anteriormente realizado utilizando padrões *
 * de treinamento e etc...                                                  *
 ****************************************************************************/

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

/**
 * Método que calcula a taxa de acerto de uma rede Perceptron Multicamdas já
 * treinada utilizando os padrões de teste.
 *
 * @param pm Referência para o Perceptron Multicamadas.
 *
 * @param padroesTeste Padrões de teste (devem utilizar a
 *                     a mesma estrutura dos padrões de treinamento).
 *
 * @param qtdPadroesTeste Quantidade de padrões de teste.
 *
 * @return Taxa de acerto da rede (erro MSE).
 *
 */
float PerceptronMulticamadas_calcularTaxaAcerto(PerceptronMulticamadas * pm,
                                                PadraoTreinamento * padroesTeste,
                                                int qtdPadroesTeste);

#endif
