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

extern "C" {
#include "aiger.h"
}
#include "IC3.h"
#include "Model.h"
#include <fstream>
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

void walk(int tab, expr e)
{
    string blanks(tab, ' ');

    if(e.is_const())
    {
        cout << blanks << "ARGUMENT: " << e << endl;
    }
    else
    {
        cout << blanks << "APP: " << e.decl().name() << endl;
        for(int i = 0; i < e.num_args(); i++)
        {
            walk(tab + 5, e.arg(i));
        }
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
  std::cout << "de-Morgan example\n";
  context c;
  expr x = c.bool_const("x");
  expr y = c.bool_const("y");
  expr z = c.bool_const("z");
  expr e = (x || !z) && (y || !z) && (!x || !y || z);
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
  walk(0,e);
  

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
  freopen("./nusmv.reactor^4.C.aag", "r", stdin);
  //freopen("./cmu.dme1.B.aag", "r", stdin);
  //freopen("./nusmv.syncarb5^2.B.aag", "r", stdin);
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
