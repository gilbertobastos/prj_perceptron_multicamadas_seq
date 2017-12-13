#include "historico_treinamento.h"

HistoricoTreinamento *
HistoricoTreinamento_inicializar(PerceptronMulticamadas * pm,
				 float taxaAprendizagem,
				 float erroDesejado)
{
  /* Alocando a estrutura do histórico na memória. */
  HistoricoTreinamento * historicoTreinamento;
  historicoTreinamento =
    (HistoricoTreinamento *) malloc(sizeof(HistoricoTreinamento));


  /* Populando a variável "strConfigRedeNeural". */
  historicoTreinamento->strConfigRedeNeural[0] = '\0';
  char strQtdNeuroniosCamada[10];
  sprintf(&strQtdNeuroniosCamada, "%d", pm->qtdNeuroniosEntrada);
  strcat(historicoTreinamento->strConfigRedeNeural, &strQtdNeuroniosCamada);
  
  for (int i = 0; i < pm->qtdCamadas; i++)
  {
    sprintf(&strQtdNeuroniosCamada, "-%d", pm->camadas[i]->qtdNeuronios);
    strcat(historicoTreinamento->strConfigRedeNeural, &strQtdNeuroniosCamada);
  }

  /* Preenchendo os atributos. */
  historicoTreinamento->qtdNeuroniosEntrada = pm->qtdNeuroniosEntrada;
  historicoTreinamento->qtdCamadas = pm->qtdCamadas;
  historicoTreinamento->taxaAprendizagem = taxaAprendizagem;
  historicoTreinamento->erroDesejado = erroDesejado;
  historicoTreinamento->listaEpocas = NULL;

  /* Retornando a estrutura criada. */
  return historicoTreinamento;
}

void HistoricoTreinamento_adicionarInfoEpoca(HistoricoTreinamento *
					     historicoTreinamento,
					     float duracaoSegs,
					     float erroGlobal)
{
  /* Alocando o nó para ser inserido na lista. */
  struct ListaNo * no = (struct ListaNo *) malloc(sizeof(struct ListaNo));
  no->dado.duracaoSegs = duracaoSegs;
  no->dado.erroGlobal = erroGlobal;
  no->proxNo = NULL;
  
  /* Verificando se não há nenhuma época inserida ainda
  na lista. */
  if (historicoTreinamento->listaEpocas == NULL)
  {
    historicoTreinamento->listaEpocas = no;
  }
  else
  {
    /* Localizando o último nó da lista. */
    struct ListaNo * noAtual;
    noAtual = historicoTreinamento->listaEpocas;
    while (noAtual->proxNo != NULL)
    {
      noAtual = noAtual->proxNo;
    }
      
    /* Por fim inserindo o último nó na "última posição" 
    da lista. */
    noAtual->proxNo = no;
  }
}

void HistoricoTreinamento_gerarArquivoCSV(HistoricoTreinamento * 
					  historicoTreinamento,
					  char * nomeArquivo)
{
  /* Tentando criar o arquivo para gravação. */
  FILE * arquivoCSV;
  arquivoCSV = fopen(nomeArquivo, "a");

  /* Verificando se o arquivo foi aberto com sucesso. */
  if (arquivoCSV == NULL)
    return; // Não conseguiu abrir arquivo.

  /* Gravando a primeira linha do arquivo que irá conter a
  a arquitetura da rede, a quantidade de padrões de treinamento,
  taxa de aprendizagem e o erro desejado respectivamente. */

  /* Imprimindo a arquitetura da rede. */
  fprintf(arquivoCSV, "%s%c", historicoTreinamento->strConfigRedeNeural,
	  DELIMITADOR_CSV);

  /* Imprimindo a taxa de aprendizagem e o erro desejado. */
  fprintf(arquivoCSV, "%f%c", historicoTreinamento->taxaAprendizagem, DELIMITADOR_CSV);
  fprintf(arquivoCSV, "%f%c\n", historicoTreinamento->erroDesejado, DELIMITADOR_CSV);

  /* Imprimindo as informações sobre as épocas. */
  struct ListaNo * noAtual = historicoTreinamento->listaEpocas;
  int numEpoca = 0;
  while (noAtual != NULL)
  {
    /* Imprimindo o número da época, duração da época em segundos
    e por fim o erro global após o treinamento da mesma. */
    fprintf(arquivoCSV, "%d%c", ++numEpoca, DELIMITADOR_CSV);
    fprintf(arquivoCSV, "%f%c", noAtual->dado.duracaoSegs, DELIMITADOR_CSV);
    fprintf(arquivoCSV, "%f%c\n", noAtual->dado.erroGlobal, DELIMITADOR_CSV);

    /* Indo para a próxima época. */
    noAtual = noAtual->proxNo;
  }

  /* Fechando o arquivo... */
  fclose(arquivoCSV);
}
