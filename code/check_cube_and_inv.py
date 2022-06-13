'''
Check the subset and unsat ratio of cube (s) and invariant clause
'''

import z3

file_path_prefix = "/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/"
#file_suffix = "cmu.dme1.B"
file_suffix = "eijk.S208o.S"
inv_cnf = file_path_prefix + file_suffix + "/inv.cnf"
with open(inv_cnf, 'r') as f:
    lines = f.readlines()
    f.close()
inv_lines = [(line.strip()).split() for line in lines]
print("print the clauses in inv.cnf:")
print(inv_lines[:3])

#q_list_cnf = ['110', '141', '192', '206', '211', '231']
#q_list_cnf =  ['114', '118', '126', '133', '134', '137', '141', '142', '144', '211', '231']

cube_file_path = "./IC3ref/cube_before_generalization_without_unsat.txt"
with open(cube_file_path, 'r') as f:
    lines = f.readlines()
    f.close()
cube_lines = [(line.strip()).split(',') for line in lines]
print("print the clauses in json")
print(cube_lines[:3])

# Record the sucess rate of finding the inductive clauses in inv.cnf
subset_success = 0
subset_fail = 0

# Record the unsat pass ratio
unsat_success = 0
unsat_fail = 0
unsat_fail_normal = 0
unsat_fail_wrost = 0
for cube_line in cube_lines:
    this_fail = 0
    for clause in inv_lines[1:]: #scan every clause in inv.cnf
        # Test if the s is subset of the invariant clauses 
        if(all(x in cube_line for x in clause)):
            #print("clause: ", clause)
            subset_success += 1
        else:
            subset_fail += 1
        # Test if the s.cube & clauses from inv is unsat
        s_clause = z3.Solver()
        s_cube = z3.Solver()
        s = z3.Solver()
        for lt in clause:
            if int(lt) % 2 == 1:
                lt_bool = z3.Bool(lt)
                s_clause.add(lt_bool==False)
            else:
                lt_bool = z3.Bool(lt)
                s_clause.add(lt_bool==True)
        for lt in cube_line:
            if int(lt) % 2 == 1:
                lt_bool = z3.Bool(lt)
                s_cube.add(lt_bool==False)
            else:
                lt_bool = z3.Bool(lt)
                s_cube.add(lt_bool==True)

        clauses_lst = list(s_clause.assertions())
        cube_lst = list(s_cube.assertions())
        s.add(z3.Not(z3.simplify(z3.And(clauses_lst))))
        s.add(z3.simplify(z3.And(cube_lst)))
        if s.check()==z3.unsat:
            unsat_success += 1
        elif s.check()==z3.sat:
            this_fail += 1
        else:
            AssertionError
    if this_fail == len(inv_lines) - 1:
        unsat_fail += 1
    elif this_fail == len(inv_lines) - 2:
        unsat_fail_wrost += 1
    elif this_fail < len(inv_lines)*0.9:
        unsat_fail_normal += 1
        
print("subset success ratio:", subset_success/(subset_success + subset_fail) * 100, "%")
print("unsat success ratio:", unsat_success/(unsat_success + unsat_fail) * 100, "%")
print("wrost unsat success ratio:", unsat_fail_wrost/len(cube_lines) * 100, "%")
print("normal unsat success ratio:", unsat_fail_normal/len(cube_lines) * 100, "%")
print("unsat success in all combination:", unsat_success/(len(cube_lines)*(len(inv_cnf)-1)) * 100, "%")