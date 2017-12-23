Authors: George Nunes de Moura Filho
         Gabriel Goncalves Barreto dos Santos - Github: github.com/gabrielgoncalves95

Brazil - Federal University of Paraiba - 2017
e-mail: georgenmoura@gmail.com

These 4 scripts contains the whole classification pipeline in order to build a binary classifier to detect fraud attempts on credit card shopping (Using python and scikit learn lib) on a specific database.

Pre-processing/Analisis algorithms: hierarchical clustering, correlation matriz and grid-search. 
Classification algorithms: KNN, SVM, Random Forest.

Cross-Validation used.

-----------------------------------------------------------------------------

Os objetivos dos 4 códigos contidos nesta pasta são os seguintes:

- saver.py(Não executar):

Esse é um arquivo exemplo de como gerar modelos externos a partir de classificadores treinados, e também de como carrega-los. Além disso, como carregar um dataframe a partir de um arquivo .csv, para que não seja necessário executar uma query toda vez. 

- data_view.py

Este código gera e exibe o dendograma(Clusterização hieráquica) e a matriz de correlação dos dados da base.

- grid_search.py

Este código contém o processo de busca por parâmetros ótimos dos 3 algoritmos para uma dada base. Consiste em um processamento pesado.

- cross_validation_train_and_test.py

Este código contém todo o treinamento, desde a query no banco até a classificação, cross-validation(5 splits) e exibição de resultados. Os algoritmos de pré-processamento, como Grid-Search, Clusterização Hierarquica e Matriz de Correlação, devem ser executados antes, em detrimento à escolha de parâmetros e de atributos deste código.
