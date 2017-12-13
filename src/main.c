#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "perceptron_multicamadas.h"
#include "historico_treinamento.h"

int main()
{
  PadraoTreinamento * padroesXOR =
    PadraoTreinamento_carregarPadroesArquivo
    ("./arquivo-amostras-xor.csv",
     "./arquivo-objetivos-xor.csv", 0, 1, 2, 1, 4);

  /* Criando a rede... */
  PerceptronMulticamadas * pm;
  int qtdNeuroniosCamada[] = {2, 1};
  pm = PerceptronMulticamadas_inicializar(2, 2, qtdNeuroniosCamada, Sigmoide);

  /* Treinando a rede e criando um histórico com as
     informações do treinamento. */
  HistoricoTreinamento * historicoTreinamento;
  historicoTreinamento =
    PerceptronMulticamadas_backpropagation(pm, padroesXOR,
					   4, 0.001,
					   0.0010, true);

  /* Salvando histórico de treinamento em um arquivo. */
  char nomeArquivoHistorico[] = "historico_treinamento_XOR_seq.csv";
  HistoricoTreinamento_gerarArquivoCSV(historicoTreinamento, nomeArquivoHistorico);

  return 0;
}
