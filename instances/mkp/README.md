# Instâncias para o Problema da Mochila Multidimensional
## Créditos para [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/info.html)

## Formato do Arquivo

O formato dos arquivos de dados é:
* número de problemas de teste (K)
* para cada problema de teste k (k=1,...,K) em sequência:
    * número de variáveis (n), número de restrições (m), valor da solução ótima (zero se indisponível)
    * os coeficientes p(j); j=1,...,n
    * para cada restrição i (i=1,...,m): os coeficientes r(i,j); j=1,...,n 
    * os lados direitos das restrições b(i); i=1,...,m

## Arquivos de Instâncias

Este diretório contém os seguintes arquivos:

* **mknapcb1.txt** - 30 problemas de teste com 5 restrições e 100 variáveis cada
* **mknapcb2.txt** - 30 problemas de teste com 10 restrições e 100 variáveis cada  
* **mknapcb3.txt** - 30 problemas de teste com 30 restrições e 100 variáveis cada
* **mkcbres.txt** - Melhores soluções viáveis conhecidas e valores de relaxação LP para todos os problemas

## Convenção de Nomenclatura dos Problemas

Os problemas são nomeados usando o formato: `m.n-instância`, onde:
* `m` = número de restrições (5, 10 ou 30)
* `n` = número de variáveis (100, 250 ou 500)
* `instância` = número da instância (00-29)

Por exemplo: `5.100-00` refere-se à primeira instância com 5 restrições e 100 variáveis.

## Referências

As instâncias são de:
* [P.C. Chu and J.E. Beasley "A genetic algorithm for the multidimensional knapsack problem", Journal of Heuristics, vol. 4, 1998, pp63-86](https://www.scirp.org/reference/referencespapers?referenceid=1638824)