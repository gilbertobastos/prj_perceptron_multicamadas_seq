/****************************************************************************
 * Projeto Perceptron Multicamadas.                                         *
 *                                                                          *
 * Implementação da rede Perceptron Multicamadas com treinamento utilizando *
 * o método "backpropagation" para classificação das amostras.              *
 *                                                                          *
 * @author Gilberto Augusto de Oliveira Bastos.                             *
 * @copyright BSD-2-Clause                                                  *
 ****************************************************************************/

#ifndef HISTORICO_TREINAMENTO_H
#define HISTORICO_TREINAMENTO_H

#include <stdio.h>
#include <stdlib.h>
#include "perceptron_multicamadas.h"

#define DELIMITADOR_CSV ','

/***************************************************************
 * OBS: As estruturas utilizadas pelas funções abaixo estão no *
 *      no arquivo "perceptron_multicamadas.h" para evitar     *
 *      a dependência ciclica.            :\		       *
 ***************************************************************/

/**
 * Método que tem o objetivo de inicializar a estrutura do
 * histórico de treinamento. 
 *
 * @param pm Referência para o Perceptron Multicamadas.
 *
 * @param taxaAprendizagem Taxa de aprendizagem da rede.
 *
 * @param erroDesejado Erro desejado para que seja encerrado o
 *                     treinamento.
 *
 * @return Referência para a estrutura alocada.
 */
HistoricoTreinamento *
HistoricoTreinamento_inicializar(PerceptronMulticamadas * pm,
				 float taxaAprendizagem,
				 float erroDesejado);

/**
 * Método que tem o objetivo de adicionar as informações
 * de uma época de treinamento de uma rede neural no 
 * histórico.
 *
 * @param historicoTreinamento Estrutura do histórico de treinamento.
 *
 * @param duracaoSegs Duração para o treinamento da 
 *                    época em segundos.
 *
 * @param erroGlobal Erro global para época após o treinamento
 *                   da mesma.
 */
void HistoricoTreinamento_adicionarInfoEpoca(HistoricoTreinamento * 
					     historicoTreinamento,
					     float duracaoSegs,
					     float erroGlobal);

/**
 * Método que cria um arquivo *.csv com todo histórico das 
 * épocas.
 *
 * A primeira linha do arquivo irá conter a arquitetura da rede,
 * a quantidade de padrões de treinamento, taxa de aprendizagem e o
 * erro desejado respectivamente.
 * As demais linhas irão conter o número da época, duração da época
 * em segundos e por fim o erro global após o treinamento da época.
 * 
 * @param historicoTreinamento Estrutura do histórico de treinamento.
 *
 * @param nomeArquivo Nome do arquivo *.csv a ser gerado.
 */
void HistoricoTreinamento_gerarArquivoCSV(HistoricoTreinamento * 
					  historicoTreinamento,
					  char * nomeArquivo);

#endif
