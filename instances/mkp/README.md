# Instances for the Multidimensional Knapsackproblem
## Credits to [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/info.html)

* The format of this data file is:
    * number of test problems (K)
    * for each test problem k (k=1,...,K) in turn:
        * number of variables (n), number of constraints (m), optimal solution value (zero if unavailable)
        * the coefficients p(j); j=1,...,n
        * for each constraint i (i=1,...,m): the coefficients r(i,j); j=1,...,n 
            * the constraint right-hand sides b(i); i=1,...,m

* For now the instances named mknapX are from [C.C.Petersen "Computational experience with variants of the Balas algorithm applied to the selection of R&D projects" Management Science 13(9) (1967) 736-750.](https://www.sciencedirect.com/science/article/pii/0898122190901449)
* And the instances named mknapcbX are from [P.C.Chu and J.E.Beasley "A genetic algorithm for the multidimensional knapsack problem", Journal of Heuristics, vol. 4, 1998, pp63-86](https://www.scirp.org/reference/referencespapers?referenceid=1638824)