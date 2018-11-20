/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package medeclassificador;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
import javax.swing.JOptionPane;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author raul, Gabriel, Priscila
 */
public class MedeClassificador {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        //chamando método principal de avaliação cruzada
        AvaliacaoCruzada();
        
    }
    
    public static void AvaliacaoCruzada() throws Exception {
        
        //criando formatação de apenas duas casas decimais
        DecimalFormat df2 = new DecimalFormat(".##");
        
        //String de retorno para escolha da base de dados weka
        String retorno = escolheBase(MostraTela());
        System.out.println(retorno);
        //puxando valores de base de dados escolhida
        if (!retorno.equals(""))
        {
            DataSource arff = new DataSource("weka-3-8-3/data/"+retorno+".arff"); //string com nome de base
            Instances base = arff.getDataSet();
            base.setClassIndex(DelimitarClasseBase(retorno));
            base = base.resample(new Random());

            Instances baseTreino = base.trainCV(3, 0);
            Instances baseTeste = base.testCV(3, 0);

            IBk vizinho = new IBk(); // vizinho mais próximo
            IBk knn3 = new IBk(3); // k = 3
            J48 arvore = new J48(); // arvore de decisão

            vizinho.buildClassifier(baseTreino);
            knn3.buildClassifier(baseTreino);
            arvore.buildClassifier(baseTreino);


            System.out.println("real;knn;knn(3);arvore");
            for (int i = 0; i < baseTeste.numInstances(); i++) {
                Instance teste = baseTeste.instance(i);

                double real = teste.classValue();


                double classe = vizinho.classifyInstance(teste);


                double classe3 = knn3.classifyInstance(teste);


                double classeArvore = arvore.classifyInstance(teste);


                String result = real + ";" + classe + ";" + classe3 + ";" + classeArvore;

                System.out.println(result);

            }

            //avaliando precisão, revocação e acurácia do vizinho mais próximo
            Evaluation avaliaVizinho = new Evaluation(base);
            avaliaVizinho.crossValidateModel(vizinho, baseTeste, 10, new Random(1));

            JOptionPane.showMessageDialog(null, "Matriz de confusão do classificador vizinho mais próximo: \n"
            + avaliaVizinho.toMatrixString()+"\n"
            + "acurácia do classificador vizinho mais próximo: "+df2.format(avaliaVizinho.pctCorrect())+"%;\n"
            + "Precisão do classificador vizinho mais próximo: "+df2.format(avaliaVizinho.precision(1) * 100) + "%;\n"
            + "Revocação do classificador vizinho mais próximo: "+ df2.format(avaliaVizinho.recall(1) * 100) + "%;\n");

            //avaliando precisão, revocação e acurácia do knn3
            Evaluation avaliaKnn3 = new Evaluation(base);
            avaliaKnn3.crossValidateModel(knn3, baseTeste, 10, new Random(1));

            JOptionPane.showMessageDialog(null, "Matriz de confusão do classificador knn3: \n"
            + avaliaKnn3.toMatrixString()+"\n"
            + "acurácia do classificador knn3: "+df2.format(avaliaKnn3.pctCorrect())+"%;\n"
            + "Precisão do classificador knn3: "+df2.format(avaliaKnn3.precision(1) * 100) + "%;\n"
            + "Revocação do classificador knn3: "+ df2.format(avaliaKnn3.recall(1) * 100) + "%;\n");

            //avaliando precisão, revocação e acurácia da árvore de decisão
            Evaluation avaliaArvore = new Evaluation(base);
            avaliaArvore.crossValidateModel(arvore, baseTeste, 10, new Random(1));

            JOptionPane.showMessageDialog(null, "Matriz de confusão do classificador da arvore de decisão: \n"
            + avaliaArvore.toMatrixString()+"\n"
            + "acurácia do classificador da arvore de decisão: "+df2.format(avaliaArvore.pctCorrect())+"%;\n"
            + "Precisão do classificador da arvore de decisão: "+df2.format(avaliaArvore.precision(1) * 100) + "%;\n"
            + "Revocação do classificador da arvore de decisão: "+ df2.format(avaliaArvore.recall(1) * 100) + "%;\n");


            //ver qual classificador é melhor dependendo da base
            //média vizinho
            double acuraciaVizinho = avaliaVizinho.pctCorrect();
            double precisaoVizinho = avaliaVizinho.precision(1) * 100;
            double revocacaoVizinho = avaliaVizinho.recall(1) * 100;
            double mediaVizinho = (acuraciaVizinho + precisaoVizinho + revocacaoVizinho) / 3; 

            //media knn3
            double acuraciaKnn3 = avaliaKnn3.pctCorrect();
            double precisaoKnn3 = avaliaKnn3.precision(1) * 100;
            double revocacaoKnn3 = avaliaKnn3.recall(1) * 100;
            double mediaKnn3 = (acuraciaKnn3 + precisaoKnn3 + revocacaoKnn3) / 3;

            //média árvore
            double acuraciaArvore = avaliaArvore.pctCorrect();
            double precisaoArvore = avaliaArvore.precision(1) * 100;
            double revocacaoArvore = avaliaArvore.recall(1) * 100;
            double mediaArvore = (acuraciaArvore + precisaoArvore + revocacaoArvore) / 3;

            //exibindo qual melhor classificador
            if (mediaVizinho > mediaKnn3 && mediaVizinho > mediaArvore) //se a média do vizinho for a maior 
            {
                JOptionPane.showMessageDialog(null, "O classificador Vizinho Mais próximo"
                        + " foi a melhor opção para a base "+ retorno + ".arff");
            }
            else if (mediaKnn3 > mediaVizinho && mediaKnn3 > mediaArvore) //se o knn3 for a melhor escolha
            {
                JOptionPane.showMessageDialog(null, "O classificador Knn de nível 3"
                        + " foi a melhor opção para a base "+ retorno + ".arff");
            }
            else if (mediaArvore > mediaVizinho && mediaArvore > mediaKnn3) //se a árvore for a melhor escolha
            {
                JOptionPane.showMessageDialog(null, "O classificador de árvore de decisão"
                        + " foi a melhor opção para a base "+ retorno + ".arff");
            }
            else
            {
                JOptionPane.showMessageDialog(null, "Para o resultado da base "+ retorno +" em questão, todos"
                        + " ou quase todos os classificadores têm desempenho semelhante");
            }
        }
    }

    

    
    public static String escolheBase(int variavelCase) throws Exception{
        
        //criando variável para devolver texto
        String base = "";
                //iniciando case para determinar valor de índice da classe e base de dados
                switch(variavelCase)
                {
                    case 1: 

                        base = "breast-cancer";

                        break;
                    case 2: 

                        base = "contact-lenses";

                    break;
                    case 3:

                        base = "diabetes";

                    break;
                    case 4:

                        base = "iris";

                    break;
                    case 5:

                        base = "labor";

                    break;
                    case 6:

                        base = "wheater.nominal";

                    break;
                    default:
                    JOptionPane.showMessageDialog(null, "Índice inexistente");
                    break;
                }
                
        return base;
        
    }
    
    public static int MostraTela() throws Exception{
        
        //inicializando variável para switch case
        int variavelCase = 0;
        
        //Iniciando JOption Pane para escolha de base pelo usuário
        String entrada = JOptionPane.showInputDialog("Escolha a base para ser treinada: \n"
                + "1 - breast-cancer.arff\n"
                + "2 - contact-lenses.arff\n"
                + "3 - diabetes.arff\n"
                + "4 - iris.arff\n"
                + "5 - labor.arff\n"
                + "6 - wheather.nominal.arff\n");
        
        //colocando valor digitado em variável do switch
        variavelCase = Integer.parseInt(entrada);
        
        return variavelCase;
        
    }
    
    public static int DelimitarClasseBase(String MostraTela) throws Exception{
        
        int IndiceClasse = 0;
        
            switch(MostraTela)
            {
                case "breast-cancer": 

                    IndiceClasse = 9;

                    break;
                case "contact-lenses": 

                    IndiceClasse = 4;

                break;
                case "diabetes":

                    IndiceClasse = 8;

                break;
                case "iris":

                    IndiceClasse = 4;

                break;
                case "labor":

                    IndiceClasse = 16;

                break;
                case "wheater.nominal":

                    IndiceClasse = 4;

                break;

            }
        
        return IndiceClasse;
        
    }
    
    
}

