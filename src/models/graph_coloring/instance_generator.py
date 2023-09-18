# Code adapted from https://github.com/machine-reasoning-ufrgs/GNN-GCP

import random
import numpy as np
import networkx as nx
import sys, os, json, argparse, itertools
import grinpy as gp
from ortools.sat.python import cp_model


def solve_csp(M, n_colors, nmin=40):
    model = cp_model.CpModel()
    N = len(M)
    variables = []

    variables = [model.NewIntVar(0, n_colors-1, '{i}'.format(i=i)) for i in range(N)]

    for i in range(N):
        for j in range(i+1, N):
            if M[i][j] == 1:
                model.Add(variables[i] != variables[j])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = int(((10.0 / nmin) * N))
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        solution = dict()
        for k in range(N):
            solution[k] = solver.Value(variables[k])
        return solution
    elif status == cp_model.INFEASIBLE:
        return None
    else:
        raise Exception("CSP is unsure about the problem")

def is_cn(Ma, cn_i):
    if solve_csp(Ma, cn_i-1) == None:
        return True
    else:
        return False


def find_diff_edge(Ma, CN, not_edges):
    for k, (i, j) in enumerate(not_edges):
        Ma[i, j] = Ma[j, i] = 1

        sol = solve_csp(Ma, CN)
        if sol is None:  # diff_edge found
            diff_edge = (i, j)
            Ma[i, j] = Ma[j, i] = 0  # backtrack
            return diff_edge
    
    return None

def create_dataset(nmin, nmax, path, samples):
    if samples > 1 and not os.path.exists(path):
        os.makedirs(path)
    
    z = 0
    er = 0
    # probability intervals for each CN, given a nmax size, we calculated it outside
    prob_constraints = {3: (0.01, 0.1), 4: (0.1, 0.2), 5: (0.2, 0.3), 6: (0.2, 0.3), 7: (0.3, 0.4), 8: (0.4, 0.5)}

    while z in range(samples):
        N = np.random.randint(nmin, nmax+1)
        Ma = np.zeros((N, N))

        Cn = np.random.randint(3, 8)
        lim_inf, lim_sup = prob_constraints[Cn][0], prob_constraints[Cn][1]

        p_connected = random.uniform(lim_inf, lim_sup)
        Ma = gen_matrix(N, p_connected)

        try:
            init_sol = solve_csp(Ma, Cn)
            if init_sol is not None and is_cn(Ma, Cn):
                deg_rank = degree_ranking(Ma)  # we sort edges by their current degrees to increase the chances of finding the diff edge
                for w in deg_rank:
                    np.fill_diagonal(Ma, 1)
                    not_edges = [(w, j) for j in range(N) if Ma[w, j] == 0]
                    random.shuffle(not_edges)
                    np.fill_diagonal(Ma, 0)
                    diff_edge = find_diff_edge(Ma, Cn, not_edges)
                    if diff_edge is not None:
                        if samples == 1:
                            return Ma, Cn, diff_edge
                        # Write graph to file
                        write_graph(Ma, Ma, diff_edge, "{}/m{}.graph".format(path, z), False, cn=Cn)
                        z += 1
                        if (z-1) % (samples//10) == 0:
                            print('{}% Complete'.format(np.round(100*z/samples)), flush=True)
                        break         
                    else:
                        # print("Cant find diff_edge")
                        er += 1
                        
            elif init_sol is None:
                # remove edges to find a derived instance which satisfies the current cn
                edges = [(i, j) for i in range(N) for j in range(i+1, N) if Ma[i, j] == 1]
                random.shuffle(edges)
                diff_edge = None
                for k, (i, j) in enumerate(edges):
                    Ma[i, j] = Ma[j, i] = 0
                    sol = solve_csp(Ma, Cn)
                    if sol is not None and is_cn(Ma, Cn):
                        diff_edge = (i, j)
                        break                
                
                if diff_edge is not None:
                    if samples == 1:
                        return Ma, Cn, diff_edge
                    # Write graph to file
                    write_graph(Ma, Ma, diff_edge, "{}/m{}.graph".format(path, z), False, cn=Cn)
                    z += 1
                    if (z-1) % (samples//10) == 0:
                        print('{}% Complete'.format(np.round(100*z/samples)), flush=True)
                else:
                    # print("Cant find diff_edge")
                    er += 1
                
        except Exception as error:
            print(repr(error))
            er += 1
    
    print('Could not solve n-color for {} random generated graphs'.format(er))

def gen_matrix(N, prob):
    Ma = np.zeros((N, N))
    Ma = np.random.choice([0, 1], size=(N, N), p=[1-prob, prob])
    i_lower = np.tril_indices(N, -1)
    Ma[i_lower] = Ma.T[i_lower]  # make the matrix symmetric
    np.fill_diagonal(Ma, 0)

    return Ma

def degree_ranking(Ma):
    G = nx.from_numpy_array(Ma)
    deg = np.asarray(gp.degree_sequence(G))
    deg = (np.amax(deg)+1) - deg  # higher degree comes first
    deg_rank = np.argsort(deg)

    return deg_rank


def write_graph(Ma, Mw, diff_edge, filepath, int_weights=False, cn=0):
    with open(filepath, "w") as out:
        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        out.write('TYPE : Graph Coloring\n')
        out.write('DIMENSION: {n}\n'.format(n=n))
        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')

        # List edges in the (generally not complete) graph
        out.write('EDGE_DATA_SECTION\n')
        for (i, j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {}\n".format(i, j))
        
        out.write('-1\n')

        # Write edge weights as a complete matrix
        out.write('EDGE_WEIGHT_SECTION\n')
        for i in range(n):
            if int_weights:
                out.write('\t'.join([str(int(Mw[i, j])) for j in range(n)]))
            else:
                out.write('\t'.join([str(float(Mw[i, j])) for j in range(n)]))
            out.write('\n')

        # Write diff edge
        out.write('DIFF_EDGE\n')
        out.write('{}\n'.format(' '.join(map(str, diff_edge))))
        if cn > 0:
            # Write chromatic number
            out.write('CHROM_NUMBER\n')
            out.write('{}\n'.format(cn))

        out.write('EOF\n')

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-samples', default=2**6, type=int, help='How many samples?')
    parser.add_argument('-path', default=r'C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\graph_coloring\data\raw', type=str, help='Save path')
    parser.add_argument('-nmin', default=20, type=int, help='Min. number of vertices')
    parser.add_argument('-nmax', default=28, type=int, help='Max. number of vertices')
    parser.add_argument('--train', action='store_true', help='To define the seed')

    # Parse arguments from command line
    args = parser.parse_args()
    random_seed = 1327 if vars(args)['train'] else 3712
    random.seed(random_seed)
    np.random.seed(random_seed)

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    import cProfile
    import pstats
    profile = cProfile.Profile()
    profile.run("create_dataset(vars(args)['nmin'], vars(args)['nmax'],samples=vars(args)['samples'],path=vars(args)['path'])")
    stats = pstats.Stats(profile)
    stats.sort_stats('tottime')
    stats.print_stats()
    