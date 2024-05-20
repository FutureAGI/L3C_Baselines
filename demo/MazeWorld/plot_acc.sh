#!/bin/bash
n=35
python plot_utils.py \
	--output acc_"$n".pdf \
	--y_axis 0.0,10 \
	--input_info eval_results_"$n"_random/acc_reward_rule_random.txt,black,Random\;eval_results_"$n"_mem0/acc_reward_rule_agent.txt,cornflowerblue,p\(STM\-\>LTM\)\=5%\;eval_results_"$n"_mem25/acc_reward_rule_agent.txt,blue,p\(STM\-\>LTM\)\=25%\;eval_results_"$n"_mem100/acc_reward_rule_agent.txt,darkblue,p\(STM\-\>LTM\)\=100%\;eval_results_"$n"/acc_reward_model_agent.txt,red,Causal_Modeling
