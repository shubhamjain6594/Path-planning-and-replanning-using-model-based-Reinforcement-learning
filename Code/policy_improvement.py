import numpy as np

def policyEval(policy, Grid, value,BlockState, discount = 1.0, theta = 0.1):
    for BlockInd, BlockVal in enumerate(BlockState):
        BlockVal = int(BlockVal)
        value[BlockVal] = -100.0

    while True:
        
        delta = 0.0
        for s in range(Grid.nS):
            if value[s] == -100.0:
                continue
            v = 0.0

            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in Grid.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount * value[next_state])
                    
            delta = max(delta, np.abs(v - value[s]))
            value[s] = v
        
        if delta < theta:
            break
    
    return np.array(value)
        
def policy_improvement_fun(env,value,BlockState, policy = np.ones([900, 4]) / 4, policy_eval_fn=policyEval, discount_factor=1.0, count = 0, theta = 0.1):

    def one_step_lookahead(state, V):
        
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
        
    for i in BlockState:
        i = int(i)
        policy[i] = [0.,0.,0.,0.]
    
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, value, BlockState, theta = theta)
        
        count += 1
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            if V[s] == -100.0:
                continue
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V, count
