Author: George Nunes de Moura Filho - Computer Engineering student
Brazil - Federal University of Paraiba - 2017

These 4 scripts contains the whole classification pipeline in order to build a binary classifier to detect fraud attempts on credit card shopping (Using python and scikit learn lib) on a specific database.

Pre-processing/Analisis algorithms: hierarchical clustering, correlation matriz and grid-search. 
Classification algorithms: KNN, SVM, Random Forest.

Cross-Validation used.

-----------------------------------------------------------------------------

Os objetivos dos 4 códigos contidos nesta pasta são os seguintes:

- saver.py(Não executar):

Esse é um arquivo exemplo de como gerar modelos externos a partir de classificadores treinados, e também de como carrega-los. Além disso, como carregar
um dataframe a partir de um arquivo .csv, para que não seja necessário executar uma query toda vez. 

- data_view.py

Este código gera e exibe o Dendograma(Clusterização hieráquica) e a Matriz de Correlação dos dados da base. Ambos algorítmos de pré-processamento
se suma importância na análise dos dados.

- grid_search.py

Este código contém o processo de busca por parâmetros ótimos dos 3 algoritmos para uma dada base. Consiste em um processamento pesado,
pois além dos testes de todas as combinações do conjunto de parâmetros para cada um dos 3 algorítmos, é feito um Cross-validation(5 splits) para
validar o resultado.

- cross_validation_train_and_test.py

Este código contém todo o treinamento, desde a query no banco até a classificação, cross-validation(5 splits) e exibição de resultados.
Os algoritmos de pré-processamento, como Grid-Search, Clusterização Hierarquica e Matriz de Correlação, devem ser executados antes,
em detrimento à escolha de parâmetros e de atributos deste código. Muito embora, os parâmetros e atributos do código já passaram
por todo o pipeline de pré-processamento necessário, ou seja, configuração ótima para a análise feita. O código já salva automaticamente
os modelos treinados, que podem ser posteriormente carregados em qualquer aplicação.
