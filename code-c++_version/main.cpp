/*********************************************************************
Copyright (c) 2013, Aaron Bradley

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*********************************************************************/

#include <iostream>
#include <string>
#include <time.h>
#include <z3++.h>
#include "aig_simple_parser.h"

extern "C" {
#include "aiger.h"
}
#include "IC3.h"
#include "Model.h"
#include <fstream>
#include <map>
//#include "json.hpp"
#include <vector>
#include "rapidcsv.h"

// #include <mlpack/core.hpp>
// #include <mlpack/methods/ann/ffn.hpp>
// #include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
// #include <mlpack/methods/ann/layer/layer.hpp>
// #include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
// #include <mlpack/methods/reinforcement_learning/q_learning.hpp>
// #include <mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp>
// #include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
// #include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
// #include <mlpack/methods/reinforcement_learning/training_config.hpp>
// #include <ensmallen.hpp>
// using namespace mlpack;
// using namespace mlpack::ann;
// using namespace ens;
// using namespace mlpack::rl;
using namespace z3;

//using json = nlohmann::json;

// void test_mlpack_rl() {
//   //test mlpack rl
//   const clock_t begin_time = clock();
//   //SimpleDQN<> mymodel(4, 64, 32, 2);
//   FFN<MeanSquaredError<>,GaussianInitialization> network(MeanSquaredError<>() ,GaussianInitialization(0, 0.001));
// 	network.Add<Linear<>>(4, 64); // 4 is the observation space . 
// 	network.Add<ReLULayer<>>();
// 	network.Add<Linear<>>(64, 32); 
// 	network.Add<ReLULayer<>>();
// 	network.Add<Linear<>>(32, 2);

//   SimpleDQN<> mymodel(network);

//   GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
//   int batch_size = 20;
// 	int replaySize = 10000;
// 	RandomReplay<CartPole> replayMethod(batch_size, replaySize);
  
//   TrainingConfig config;
// 	config.StepSize() = 0.01;
// 	config.Discount() = 0.99;
// 	config.TargetNetworkSyncInterval() = 100;
// 	config.ExplorationSteps() = 100;
// 	config.DoubleQLearning() = false;
// 	config.StepLimit() = 400;
  
//   QLearning<CartPole , decltype(mymodel) , AdamUpdate , decltype(policy)>  agent(config, mymodel, policy, replayMethod);
//   arma::running_stat<double> averageReturn;
// 	size_t episode = 0;
// 	size_t maxiter = 1000;
// 	size_t requirement = 50; // This variable checks if the game is converging or not .
// 	// References for armadillo running_stat : http://arma.sourceforge.net/docs.html#running_stat
// 	int i = 0;
// 	while ( episode <= maxiter)
// 	{
// 		double epi_return = agent.Episode();
// 		averageReturn(epi_return);
// 		episode = episode + 1;
// 	    std::cout << "Average return: " << averageReturn.mean()<< " Episode return: " << epi_return<< std::endl;
// 	    if (averageReturn.mean() > requirement)
// 	    {
// 	    	agent.Deterministic() = true;
// 	    	arma::running_stat<double> testReturn; // check the stats for test run to take place
	    	
// 	    	for (size_t i = 0; i < 20; ++i)// 20 test runs
// 		        testReturn(agent.Episode()); // variable defined above

// 		    std::cout << endl <<"Converged with return " <<  testReturn.mean()  << " with number of " << episode << " iterations"<<endl;
// 		    break;
// 	    }	
// 	    // check converged or not?
// 	}
// 	if (episode >= maxiter)
// 	{
// 		std::cout << "Cart Pole with DQN failed to converge in " << maxiter << " iterations." <<std::endl;
// 	}
// 	std::cout << "Time take is  " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
// }

void walk(int tab, expr e, vector<expr> & bfs_queue)
{
    string blanks(tab, ' ');

    if(e.is_const())
    {
        cout << blanks << "ARGUMENT"<<"(id:"<<e.id()<<")"<<": "<< e << endl;
        bfs_queue.push_back(e);
    }
    else
    {
        cout << blanks << "APP: " <<"(id:"<<e.id()<<")"<<": "<< e.decl().name() << endl;
        bfs_queue.push_back(e);
        for(int i = 0; i < e.num_args(); i++)
        {
            walk(tab + 5, e.arg(i),bfs_queue);
        }
    }
}

void visit(expr const & e) {
    if (e.is_app()) {
        unsigned num = e.num_args();
        for (unsigned i = 0; i < num; i++) {
            visit(e.arg(i));
        }
        // do something
        // Example: print the visited expression
        func_decl f = e.decl();
        std::cout << "application of " << f.name() << ": " << e << "\n";
    }
    else if (e.is_quantifier()) {
        visit(e.body());
        // do something
    }
    else { 
        assert(e.is_var());
        // do something
    }
}

struct cmpByZ3ExprID {
    bool operator()(const z3::expr& a, const z3::expr& b) const {
        return a.id() < b.id();
    }
};

int get_nid(expr const & e, map<expr, int,cmpByZ3ExprID> & nid_map) {
    if (e.is_app()) {
      // determine the node is in nid_map or not
        if(nid_map.count(e) == 0) {
            int nid = nid_map.size();
            nid_map.insert(make_pair(e, nid));
            //cout<<"new node: "<<e<<" "<<nid<<endl;
            return nid;
        }
        else if(nid_map.find(e) != nid_map.end()){
            //cout<<"find the node in nid_map, id:"<<e.id()<<endl;
            return nid_map[e];
        }
    }
    else {
        assert(false);
    }
}


int main(int argc, char ** argv) {
  //test jsoncpp
  // json j;			// create json object
	// ifstream jfile("test.json");
	// jfile >> j;		// read json as file stream
	// float pi = j.at("pi");
	// bool happy = j.at("happy");
  // cout<<"pi: "<<pi<<endl;
  // cout<<"happy: "<<happy<<endl;

  //test mlpack
  //cout << mlpack::util::GetVersion() << endl;
  //test_mlpack_rl();

  //test rapidcsv
  // rapidcsv::Document doc("test.csv");
  // std::vector<float> col = doc.GetColumn<float>("Close");
  // std::cout << "Read " << col.size() << " values." << std::endl;

  //test z3
  /**
   Demonstration of how Z3 can be used to prove validity of
   De Morgan's Duality Law: {e not(x and y) <-> (not x) or ( not y) }
  */
  // std::cout << "de-Morgan example\n";
  // context c;
  // expr x = c.bool_const("x");
  // expr y = c.bool_const("y");
  // expr z = c.bool_const("z");
  // expr e = (x || !z) && (y || !z) && (!x || !y || z);

  // expr conjecture = (!(x && y)) == (!x || !y);
  // solver s(c);
  // // adding the negation of the conjecture as a constraint.
  // s.add(!conjecture);
  // std::cout << s << "\n";
  // std::cout << s.to_smt2() << "\n";
  // switch (s.check()) {
  // case unsat:   std::cout << "de-Morgan is valid\n"; break;
  // case sat:     std::cout << "de-Morgan is not valid\n"; break;
  // case unknown: std::cout << "unknown\n"; break;
  // }
  //walk(0,e);

  z3::context ctx;
  auto&& opt = z3::optimize(ctx);
  Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "nusmv.syncarb5^2.B_0.smt2", 0, 0, 0, 0, 0, 0);
  
  // Get all the constriants
  // Z3_ast* args = new Z3_ast[Z3_ast_vector_size(ctx, b)];
  // for (unsigned i = 0; i < Z3_ast_vector_size(ctx, b); ++i) { //execute from 0 to size of b
  //     args[i] = Z3_ast_vector_get(ctx, b, i);
  // }
  // z3::ast result(ctx, Z3_mk_and(ctx, Z3_ast_vector_size(ctx, b), args));

  // Get only the last constriant
  // Z3_ast* args = new Z3_ast[1];
  // unsigned i = Z3_ast_vector_size(ctx, b)-1;
  // cout<<"i: "<<int(i)<<endl;
  // args[0] = Z3_ast_vector_get(ctx, b, i);
  // z3::ast result(ctx, Z3_mk_and(ctx, 1, args));

  // fetch the last constriant - one line method
  Z3_ast result = Z3_ast_vector_get(ctx, b, Z3_ast_vector_size(ctx, b)-1);
  
  ctx.check_error();
  //walk(0,ctx);
  //z3.toExpr(result);
  expr k(ctx, result); 
  opt.add(k);

  auto&& res = opt.check();
  switch (res) {
      case z3::sat: std::cout << "Sat" << std::endl;break;
      case z3::unsat: std::cout << "Unsat" << std::endl;break;
      case z3::unknown: std::cout << "Unknown" << std::endl;break;
  }
  vector<expr> bfs_queue;
  walk(0,k, bfs_queue);
  //visit(k);
  cout<<"bfs_queue size: "<<bfs_queue.size()<<endl;
  map<expr,int,cmpByZ3ExprID> map_expr;
  //map_expr.insert(make_pair(bfs_queue[0],0));
  set<pair<int, int>> set_expr;
  for(int i = 0; i < bfs_queue.size(); i++)
  {
    int node_id = get_nid(bfs_queue[i], map_expr);
    //cout<<bfs_queue[i].decl().name()<<endl;
    for(int j = 0; j < bfs_queue[i].num_args(); j++)
    {
      int children_nid = get_nid(bfs_queue[i].arg(j),map_expr);
      //self.edges.add((node_id, children_nid))
      set_expr.insert(make_pair(node_id, children_nid));
    }
  }
  cout<<"map_expr size: "<<map_expr.size()<<endl;
  cout<<"set_expr size: "<<set_expr.size()<<endl;
  //print all edge in the set set_expr
  cout<<"ready to print all edge in the set set_expr"<<endl;
  set<pair<int, int>>::iterator it;
  for(it=set_expr.begin();it!=set_expr.end();it++)
  {
      printf("%d %d\n",it->first,it->second);
  }

  // z3::context ctx;
  // Z3_ast_vector v = Z3_parse_smtlib2_file(ctx, "nusmv.syncarb5^2.B_0.smt2", 0, 0, 0, 0, 0, 0);
  // Z3_ast_vector_inc_ref(ctx, v);
  // unsigned sz = Z3_ast_vector_size(ctx, v);
  // Z3_ast* vv = malloc(sz);
  // for (unsigned I = 0; I < sz; ++I) vv[I] = Z3_ast_vector_get(ctx, v, I);
  // Z3_ast* result = Z3_mk_and(ctx, sz, vv);
  // Z3_inc_ref(ctx, result);
  // free(vv);
  // Z3_ast_vector_dec_ref(ctx, v);
  
  // //parse .smt2 file into z3
  // z3::context ctx;
  // auto&& opt = z3::optimize(ctx);
  // z3::set_param("timeout", 1000);
  // Z3_ast a = Z3_parse_smtlib2_file(ctx, "nusmv.syncarb5^2.B_0.smt2", 0, 0, 0, 0, 0, 0);
  // z3::expr e(ctx, a);
  // opt.add(e);

  // auto&& res = opt.check();
  // switch (res) {
  //     case z3::sat: std::cout << "Sat" << std::endl;break;
  //     case z3::unsat: std::cout << "Unsat" << std::endl;break;
  //     case z3::unknown: std::cout << "Unknown" << std::endl;break;
  // }

  // test aiger parser
  //AigParser p("palu.aag");
  cout<<"begin to parse the aiger file"<<endl;
  AigParser p("nusmv.syncarb5^2.B.aag");

  // real main function

  unsigned int propertyIndex = 0;
  bool basic = false, random = false;
  int verbose = 0;
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "-v")
      // option: verbosity
      verbose = 2;
    else if (string(argv[i]) == "-s")
      // option: print statistics
      verbose = max(1, verbose);
    else if (string(argv[i]) == "-r") {
      // option: randomize the run, which is useful in performance
      // testing; default behavior is deterministic
      srand(time(NULL));
      random = true;
    }
    else if (string(argv[i]) == "-b")
      // option: use basic generalization
      basic = true;
    else
      // optional argument: set property index
      propertyIndex = (unsigned) atoi(argv[i]);
  }

  // read AIGER model
  aiger * aig = aiger_init();
  //freopen("./eijk.S208o.S.aag", "r", stdin);
  //freopen("./nusmv.reactor^4.C.aag", "r", stdin);
  //freopen("./cmu.dme1.B.aag", "r", stdin);
  freopen("./nusmv.syncarb5^2.B.aag", "r", stdin);
  //fp = fopen("./eijk.S208o.S.aag", "r");
  const char * msg = aiger_read_from_file(aig, stdin);
  if (msg) {
    cout << msg << endl;
    return 0;
  }

  //parse AIGER model by z3
  //AigParser p("./eijk.S208o.S.aag");

  // create the Model from the obtained aig
  Model * model = modelFromAiger(aig, propertyIndex);
  aiger_reset(aig);
  if (!model) return 0;

  // model check it
  //verbose = 2; //open verbose mode
  bool rv = IC3::check(*model, verbose, basic, random);
  // print 0/1 according to AIGER standard
  cout << !rv << endl;

  delete model;

  return 1;
}
