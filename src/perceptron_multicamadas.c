#include "perceptron_multicamadas.h"
#include "historico_treinamento.h"

PerceptronMulticamadas *
PerceptronMulticamadas_inicializar(int qtdNeuroniosEntrada,
                                   int qtdCamadas,
                                   int * qtdNeuroniosCamada,
                                   int funcaoAtivacaoRede)
{
  /* Criando o vetor de referências que irá armazenar as camadas. */
  const Camada ** camadas = malloc(sizeof(Camada *) * qtdCamadas);

  /* Alocando as camadas. */
  camadas[0] = __alocarCamada(qtdNeuroniosCamada[0], qtdNeuroniosEntrada,
                              funcaoAtivacaoRede);

  for (int i = 1; i < qtdCamadas; i++)
  {
    camadas[i] = __alocarCamada(qtdNeuroniosCamada[i],qtdNeuroniosCamada[i - 1],
                                funcaoAtivacaoRede);
  }

  /* Alocando a estrutura que irá abrigar as camadas. */
  PerceptronMulticamadas * pm = malloc(sizeof(PerceptronMulticamadas));

  /* Preenchendo os atributos do Perceptron. */
  pm->camadas = camadas;
  pm->qtdCamadas = qtdCamadas;
  pm->qtdNeuroniosEntrada = qtdNeuroniosEntrada;

  /* Retornando a estrutura alocada. */
  return pm;
}

Camada * __alocarCamada(int qtdNeuronios, int qtdPesosNeuronio,
                        int funcaoAtivacao)
{
  /* Alocando a camada. */
  Camada * camada = malloc(sizeof(Camada));

  /* Alocando o vetor de pesos dos neurônios. */
  camada->W = __alocarVetorPesosRandomicos(qtdPesosNeuronio * qtdNeuronios);

  /* Alocando o vetor que irá armazenar a ativação dos neurônios. */
  camada->neuronioAtivacao = malloc(sizeof(float) * qtdNeuronios);

  /* Alocando o vetor que irá armazenar as derivadas dos neurônios. */
  camada->neuronioDerivada = malloc(sizeof(float) * qtdNeuronios);

  /* Alocando o vetor que irá armazenar o erro retropropagado calculado para
  cada neurônio. */
  camada->neuronioErroRprop = malloc(sizeof(float) * qtdNeuronios);

  /* Alocando vetor de bias. */
  camada->bias = malloc(sizeof(float) * qtdNeuronios);

  /* Inicializando os valores do vetor de bias através da macro
  definida no arquivo de cabeçalho (BIAS). */
  for (int i = 0; i < qtdNeuronios; i++)
  {
    camada->bias[i] = BIAS;
  }

  /* Preenchendo os demais atributos. */
  camada->qtdNeuronios = qtdNeuronios;
  camada->funcaoAtivacao = funcaoAtivacao;

  /* Retornando a referência para a camada alocada. */
  return camada;
}

float * __alocarVetorPesosRandomicos(int qtdPesos)
{
  /* Alocando o vetor de pesos. */
  float * vetorPesos;
  vetorPesos = malloc(sizeof(float) * qtdPesos);

  /* Gerando a semente (número entre 0 a 99 999)... */
  srand(time(NULL));
  int semente = rand() % 100000;

  for (int i = 0; i < qtdPesos; i++)
  {
    /* Gerando o peso no intervalo de RAND_LIM_MIN..RAND_LIM_MAX. */
    vetorPesos[i] = r4_uniform_ab(RAND_LIM_MIN, RAND_LIM_MAX, &semente);
  }

  /* Retornando a referência para o vetor alocado. */
  return vetorPesos;
}

void Camada_calcularAtivacaoNeuroniosPrimeiraCamada(const Camada * camada,
                                                    const float * amostra,
                                                    int qtdNeuroniosEntrada)
{
  /* Percorrendo todos os neurônios da camada. */
  for (int n = 0; n < camada->qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada->W[qtdNeuroniosEntrada * n];

    /* Calculando o valor da função de integração o neurônio. */
    float valFuncIntegracao = 0.0;

    for (int i = 0; i < qtdNeuroniosEntrada; i++)
    {
      /* Somando o item "i-ésimo" da amostra pelo peso "i-ésimo" do
      neurônio "n-ésimo". */
      valFuncIntegracao += w[i] * amostra[i];
    }

    /* Por fim calculando a ativação do neurônio (usando o bias) junto com
    sua derivada. */
    float ativacaoNeuronio;

    switch (camada->funcaoAtivacao)
    {
    case Identidade:
       camada->neuronioAtivacao[n] = valFuncIntegracao + camada->bias[n];
       camada->neuronioDerivada[n] = 1;
       break;
    case Degrau:
      ativacaoNeuronio = funcaoDegrau(valFuncIntegracao +
                                      camada->bias[n]);
      camada->neuronioAtivacao[n] = ativacaoNeuronio;
      camada->neuronioDerivada[n] = derivadaFuncaoDegrau(ativacaoNeuronio);
      break;
    case Sigmoide:
      ativacaoNeuronio = funcaoSigmoide(valFuncIntegracao +
                                        camada->bias[n]);
      camada->neuronioAtivacao[n] = ativacaoNeuronio;
      camada->neuronioDerivada[n] = derivadaFuncaoSigmoide(ativacaoNeuronio);
      break;
    case TangHiperbolica:
      ativacaoNeuronio = funcaoTangHiperbolica(valFuncIntegracao +
                                               camada->bias[n]);
      camada->neuronioAtivacao[n] = ativacaoNeuronio;
      camada->neuronioDerivada[n] = derivadaFuncaoTangHiperbolica(ativacaoNeuronio);
    }
  }
}

void Camada_calcularAtivacaoNeuroniosCamada(const Camada * camadaAnterior,
                                            const Camada * camada)
{
  /* Percorrendo todos os neurônios da camada. */
  for (int n = 0; n < camada->qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada->W[camadaAnterior->qtdNeuronios * n];

    /* Calculando o valor da função de integração o neurônio. */
    float valFuncIntegracao = 0.0;

    for (int i = 0; i < camadaAnterior->qtdNeuronios; i++)
    {
      /* Somando a ativação do neurônio "i-ésimo" pelo peso "i-ésimo" do
      neurônio "n-ésimo". */
      valFuncIntegracao += w[i] * camadaAnterior->neuronioAtivacao[i];
    }

    /* Por fim calculando a ativação do neurônio (usando o bias) junto com
    sua derivada. */
    float ativacaoNeuronio;

    switch (camada->funcaoAtivacao)
    {
    case Identidade:
       camada->neuronioAtivacao[n] = valFuncIntegracao + camada->bias[n];
       camada->neuronioDerivada[n] = 1;
       break;
    case Degrau:
      ativacaoNeuronio = funcaoDegrau(valFuncIntegracao +
                                      camada->bias[n]);
      camada->neuronioAtivacao[n] = ativacaoNeuronio;
      camada->neuronioDerivada[n] = derivadaFuncaoDegrau(ativacaoNeuronio);
      break;
    case Sigmoide:
      ativacaoNeuronio = funcaoSigmoide(valFuncIntegracao +
                                        camada->bias[n]);
      camada->neuronioAtivacao[n] = ativacaoNeuronio;
      camada->neuronioDerivada[n] = derivadaFuncaoSigmoide(ativacaoNeuronio);
      break;
    case TangHiperbolica:
      ativacaoNeuronio = funcaoTangHiperbolica(valFuncIntegracao +
                                               camada->bias[n]);
      camada->neuronioAtivacao[n] = ativacaoNeuronio;
      camada->neuronioDerivada[n] = derivadaFuncaoTangHiperbolica(ativacaoNeuronio);
    }
  }
}

void Camada_calcularErroRpropNeuroniosCamada(const Camada * camada,
                                             const Camada * camadaPosterior)
{
  /* Percorrendo todos os neurônios da camada. */
  for (int n = 0; n < camada->qtdNeuronios; n++)
  {
    /* Calculando a soma dos erros da camada posterior multiplicados
    pelo respectivos pesos. */
    float somaErroCamadaPosterior = 0.0;

    for (int i = 0; i < camadaPosterior->qtdNeuronios; i++)
    {
      /* Coletando o peso do neurônio "i-ésimo" da camada posterior
      que se conecta ao respectivo neurônio "n-ésimo" que está tendo seu
      erro calculado. */
      float w = camadaPosterior->W[camada->qtdNeuronios * i + n];

      /* Calculando o erro do neurônio "i-ésimo" da camada posterior
      multiplicado pelo respectivo peso da camada posterior que se
      conecta ao respectivo neurônio "n-ésimo" que está tendo seu erro
      calculado, e somando... */
      somaErroCamadaPosterior += w * camadaPosterior->neuronioErroRprop[i];
    }

    /* Por fim, calculando o erro retropropagado do neurônio. */
    camada->neuronioErroRprop[n] = camada->neuronioDerivada[n] *
                                   somaErroCamadaPosterior;
  }
}

float Camada_calcularErroRpropNeuroniosUltimaCamada(const Camada * camada,
                                                    const float * alvo)
{
  /* Variável que irá armazenar o erro para o padrão apresentado à rede. */
  float erroPadrao = 0.0;

  /* Percorrendo todos os neurônios da camada. */
  for (int n = 0; n < camada->qtdNeuronios; n++)
  {
    /* Calculando o erro da saída do neurônio "n-ésimo". */
    float erroSaidaNeuronio = camada->neuronioAtivacao[n] - alvo[n];

    /* Calculando o erro retropropagado. */
    camada->neuronioErroRprop[n] = erroSaidaNeuronio *
                                   camada->neuronioDerivada[n];

    /* Calculando o erro para o padrão... */
    erroPadrao += 0.5 * powf(erroSaidaNeuronio, 2);
  }

  /* Retornando o erro do padrão apresentado à rede. */
  return erroPadrao;
}

void Camada_atualizarPesosNeuroniosPrimeiraCamada(const Camada * camada,
                                                  const float * amostra,
                                                  int qtdNeuroniosEntrada,
                                                  float taxaAprendizagem)
{
  /* Percorrendo todos os neurônios da camada. */
  for (int n = 0; n < camada->qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada->W[qtdNeuroniosEntrada * n];

    /* Percorrendo todos os pesos do neurônio. */
    for (int i = 0; i < qtdNeuroniosEntrada; i++)
    {
      /* Atualizando o peso "i-ésimo" do neurônio "n-ésimo". */
      w[i] += -taxaAprendizagem * amostra[i] * camada->neuronioErroRprop[n];
    }

    /* Atualizando o bias do neurônio "n-ésimo"... */
    camada->bias[n] += -taxaAprendizagem * camada->neuronioErroRprop[n];
  }
}

void Camada_atualizarPesosNeuroniosCamada(const Camada * camadaAnterior,
                                          const Camada * camada,
                                          float taxaAprendizagem)
{
  /* Percorrendo todos os neurônios da camada. */
  for (int n = 0; n < camada->qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada->W[camadaAnterior->qtdNeuronios * n];

    /* Percorrendo todos os pesos do neurônio. */
    for (int i = 0; i < camadaAnterior->qtdNeuronios; i++)
    {
      /* Atualizando o peso "i-ésimo" do neurônio "n-ésimo". */
      w[i] += -taxaAprendizagem * camadaAnterior->neuronioAtivacao[i] *
              camada->neuronioErroRprop[n];
    }

    /* Atualizando o bias do neurônio "n-ésimo"... */
    camada->bias[n] += -taxaAprendizagem * camada->neuronioErroRprop[n];
  }
}

void PerceptronMulticamadas_feedfoward(PerceptronMulticamadas * pm,
                                       const float * amostra)
{
  /* Calculando a ativação dos neurônios da primeira camada. */
  Camada_calcularAtivacaoNeuroniosPrimeiraCamada(pm->camadas[0], amostra,
                                                 pm->qtdNeuroniosEntrada);

  /* Calculando a ativação dos neurônios das demais camadas. */
  for (int c = 1; c < pm->qtdCamadas; c++)
  {
    Camada_calcularAtivacaoNeuroniosCamada(pm->camadas[c - 1], pm->camadas[c]);
  }
}

HistoricoTreinamento *
PerceptronMulticamadas_backpropagation(PerceptronMulticamadas * pm,
				       PadraoTreinamento * padroes,
				       int qtdPadroesTreinamento,
				       float taxaAprendizagem,
				       float erroDesejado,
				       bool gerarHistorico)
{
  /* Inicializando a estrutura. */
  HistoricoTreinamento * historicoTreinamento; 
  if (gerarHistorico)
  {
    historicoTreinamento = HistoricoTreinamento_inicializar(pm,
							    taxaAprendizagem,
							    erroDesejado);
  }
  
  /* Variável que irá armazenar o erro global da rede após
  a apresentação dos padrões. */
  float erroGlobal;

  /* O treinamento irá ocorrer enquanto o erro da rede estiver acima
  do desejado OU a quantidade de épocas não tenha atingido o limite. */
  int epocas = 0;

  do
  {
    /* Inicializando com 0 para para a época atual. */
    erroGlobal = 0.0;

    /* Variáveis que serão utilizadas para armazenar a hora que foi iniciada
    e finalizada o treinamento da rede para a época atual. */
    struct timeval horaAntesTreinamento;
    struct timeval horaDepoisTreinamento;

    /* Coletando a hora antes do treinamento. */
    gettimeofday(&horaAntesTreinamento, NULL);

    /* Apresentando os padrões de treinamento para rede e realizando o
    treinamento da mesma. */
    for (int i = 0; i < qtdPadroesTreinamento; i++)
    {
      /* Alimentando a rede com o padrão "i-ésimo". */
      PerceptronMulticamadas_feedfoward(pm, padroes[i].amostra);

      /* Realizando a retropropagação do erro para a última camada e já
      somando o erro calculado para o padrão no erro global. */
      erroGlobal += Camada_calcularErroRpropNeuroniosUltimaCamada
	            (pm->camadas[pm->qtdCamadas - 1], padroes[i].alvo);

      /* Realizando a retropropagação do erro para as demais camadas. */
      for (int c = pm->qtdCamadas - 2; c >= 0; c--)
      {
        Camada_calcularErroRpropNeuroniosCamada(pm->camadas[c],
                                                pm->camadas[c + 1]);
      }

      /* Atualizando os pesos dos neurônios da primeira camada. */
      Camada_atualizarPesosNeuroniosPrimeiraCamada(pm->camadas[0],
                                                   padroes[i].amostra,
                                                   pm->qtdNeuroniosEntrada,
                                                   taxaAprendizagem);

      /* Atualizando os pesos dos neurônios das demais camadas. */
      for (int c = 1; c < pm->qtdCamadas; c++)
      {
        Camada_atualizarPesosNeuroniosCamada(pm->camadas[c - 1],
                                             pm->camadas[c],
                                             taxaAprendizagem);
      }
    }

    /* Coletando a hora depois do treinamento. */
    gettimeofday(&horaDepoisTreinamento, NULL);

    /* Realizando o cálculo do MSE. */
    erroGlobal = erroGlobal/ qtdPadroesTreinamento;

    /* Atualizando a quantidade de épocas. */
    epocas++;

    /* Calculando o tempo de treinamento. */
    float segs =  (horaDepoisTreinamento.tv_sec +
		   horaDepoisTreinamento.tv_usec / 1000000.0) -
                  (horaAntesTreinamento.tv_sec +
                   horaAntesTreinamento.tv_usec / 1000000.0);

    /* Adicionando as informações desta época no histórico de
    treinamento. */
    if (gerarHistorico)
    {
      HistoricoTreinamento_adicionarInfoEpoca(historicoTreinamento,
					      segs,
					      erroGlobal);
    }
    
    /* Imprimindo as informações de estatística (caso seja necessário). */
    if (INFO_ESTATISTICAS)
    {
      printf("Época: %d\nErro MSE: %.4f\n", epocas, erroGlobal);
      printf("Tempo total de execução da época: %.2f segundo(s)\n\n", segs);
    }
    
  } while (erroGlobal > erroDesejado && epocas < QTD_MAX_EPOCAS);
  
  return historicoTreinamento;
}

void normalizacaoMinMax(float * v, int n, float min, float max)
{
  /* Percorrendo todos os itens do vetor e realizando a normalização
  dos mesmos. */
  for (int i = 0; i < n; i++)
  {
    v[i] = (v[i] - min) / (max - min);
  }
}

inline float funcaoDegrau(float z)
{
  return (z >= 0) ? 1 : 0;
}

float derivadaFuncaoDegrau(float valDegrau)
{
  return 1.0;
}

inline float funcaoSigmoide(float z)
{
  return 1.0 / ((1.0) + expf(-z));
}

inline float derivadaFuncaoSigmoide(float valSigmoide)
{
  return valSigmoide * (1.0 - valSigmoide);
}

inline float funcaoTangHiperbolica(float z)
{
  return tanhf(z);
}

inline float derivadaFuncaoTangHiperbolica(float valTangHiperbolica)
{
  return 1 - (valTangHiperbolica * valTangHiperbolica);
}

PadraoTreinamento *
PadraoTreinamento_carregarPadroesArquivo(char * nomeArquivoAmostras,
                                         char * nomeArquivoObjetivos,
                                         float menorValAmostra,
                                         float maiorValAmostra,
                                         int qtdItensAmostra,
                                         int qtdItensVetorObjetivo,
                                         int qtdPadroes)
{
  /* Tentando abrir os arquivos para leitura. */
  FILE * arqAmostras;
  FILE * arqVetorObjetivos;

  arqAmostras = fopen(nomeArquivoAmostras, "r");
  arqVetorObjetivos = fopen(nomeArquivoObjetivos, "r");

  /* Verificando se os arquivos foram abertos com sucesso. */
  if (arqAmostras == NULL || arqVetorObjetivos == NULL)
  {
    /* Retornando NULL. */
    return NULL;
  }

  /* Alocando o vetor que irá armazenar os padrões. */
  PadraoTreinamento * padroes;
  padroes = malloc(sizeof(PadraoTreinamento) * qtdPadroes);

  /* Primeiramente lendo as amostras e inserindo as mesmas nos respectivos
  padrões. */
  for (int i = 0; i < qtdPadroes; i++)
  {
    /* Coletando a linha com a amostra do arquivo. */
    char linhaAmostra[4096]; // 4 Kbytes
    fscanf(arqAmostras, "%s", linhaAmostra);

    /* Alocando o vetor para armazenar a amostra "i-ésima". */
    float * amostra = (float *) malloc(sizeof(float) * qtdItensAmostra);

    /* Extraindo o primeiro item da amostra. */
    amostra[0] = atof(strtok(linhaAmostra, ";\n\0"));

    /* Extraindo os demais itens da amostra. */
    for (int j = 1; j < qtdItensAmostra; j++)
    {
      /* Extraindo o item "j-ésimo" da amostra. */
      amostra[j] = atof(strtok(NULL, ";\n\0"));
    }

    /* Normalizando a amostra coletada... */
    normalizacaoMinMax(amostra, qtdItensAmostra, menorValAmostra,
                       maiorValAmostra);

    /* Por fim, colocando no padrão a amostra acima extraida
    do arquivo. */
    padroes[i].amostra = amostra;
  }

  /* Lendo os vetores de objetivo e inserindo os mesmos nos respectivos
  padrões. */
  for (int i = 0; i < qtdPadroes; i++)
  {
    /* Coletando a linha com o vetor de objetivo do arquivo. */
    char linhaVetorObjetivo[4096]; // 4 Kbytes
    fscanf(arqVetorObjetivos, "%s", linhaVetorObjetivo);

    /* Alocando o vetor para armazenar o vetor de objetivo "i-ésimo". */
    float * vetorObjetivo = (float *) malloc(sizeof(float) * qtdItensVetorObjetivo);

    /* Extraindo o primeiro item do vetor de objetivo. */
    vetorObjetivo[0] = atof(strtok(linhaVetorObjetivo, ";\n\0"));

    /* Extraindo os demais itens do vetor de objetivo. */
    for (int j = 1; j < qtdItensVetorObjetivo; j++)
    {
      /* Extraindo o item "j-ésimo" do vetor de objetivo. */
      vetorObjetivo[j] = atof(strtok(NULL, ";\n\0"));
    }

     /* Por fim, colocando no padrão o vetor de objetivo extraido do
     arquivo. */
     padroes[i].alvo = vetorObjetivo;
  }

  fclose(arqAmostras);
  fclose(arqVetorObjetivos);

  return padroes;
}

float PerceptronMulticamadas_calcularTaxaAcerto(PerceptronMulticamadas * pm,
                                                PadraoTreinamento * padroesTeste,
                                                int qtdPadroesTeste)
{
   /* Variável que irá armazenar a soma dos erros de todos os
  padrões. */
  float somaErroPadroes = 0.0;

  /* Percorrendo os padrões de teste. */
  for (int i = 0; i < qtdPadroesTeste; i++)
  {
    /* Alimentando a rede com o padrão de teste "i-ésimo". */
    PerceptronMulticamadas_feedfoward(pm, padroesTeste[i].amostra);


    /* Somando à soma dos erros de todos os padrões. */
    somaErroPadroes += Camada_calcularErroRpropNeuroniosUltimaCamada
                       (pm->camadas[pm->qtdCamadas - 1],
			padroesTeste[i].alvo);
  }

  /* Retornando o erro MSE calculado. */
  return somaErroPadroes / qtdPadroesTeste;
}

