#!/bin/bash
n=25
python plot_utils.py \
	--output rule_"$n".pdf \
	--input_info eval_results_"$n"_mem0/reward_rule_agent.txt,red,p\(STM\-\>LTM\)\=5%\;eval_results_"$n"_mem25/reward_rule_agent.txt,blue,p\(STM\-\>LTM\)\=25%\;eval_results_"$n"_mem100/reward_rule_agent.txt,green,p\(STM\-\>LTM\)\=100%\;eval_results_"$n"/acc_reward_model_agent.txt,black,Causal_Modeling
