from typing import List

def parse_dimacs_cnf(filepath:str):
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    problem_definition = lines[0].strip().split()
    num_literals = problem_definition[2]
    num_clauses = problem_definition[3]
    
    clauses = []
    for i in range(1, len(lines)):
        line = lines[i].strip().split()[:-1]
        new_clause = Clause(line)
        clauses.append(new_clause)
    
    return CNF(clauses)

class Clause:
    """
    Simple class representing an AND clause; i.e. a line in a CNF problem.
    """
    def __init__(self, line:List[str]):
        self.variables = {int(var) for var in line}

class CNF:
    def __init__(self, clauses:List[Clause]):
        self.clauses = clauses
        variables = set()
        for clause in self.clauses:
            base_vars = {abs(var) for var in clause.variables}
            variables = variables.union(base_vars)
        variables = list(variables)
        self.variables = sorted(variables)

if __name__ == "__main__":
    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic Graph Representation\neurosat\dimacs\test\sr5\grp1\sr_n=0006_pk2=0.30_pg=0.40_t=9_sat=0.dimacs"
    cnf = parse_dimacs_cnf(test_path)