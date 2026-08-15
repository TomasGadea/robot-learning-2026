Initialize the network $Q_\theta(s, a)$ with random network parameters $\theta$

Initialize the target network $Q_{\bar{\theta}}$ by copying the same parameters $\bar{\theta} \leftarrow \theta$

Initialize the replay buffer $\mathcal{R}$

for episode $e = 1 \rightarrow E$ do:

    Get the environment's initial state $s_1$
    
    for timestep $t = 1 \rightarrow T$ do:
    
        Select action $a_t$ based on the current network $Q_\theta(s, a)$ using the $\epsilon$-greedy strategy
        
        Execute action $a_t$, obtain reward $r_t$, and the environment state transitions to $s_{t+1}$
        
        Store $(s_t, a_t, r_t, s_{t+1})$ into the replay buffer $\mathcal{R}$
        
        If there is enough data in $\mathcal{R}$, sample $N$ transitions $\{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ from $\mathcal{R}$
        
        For each transition, calculate $y_i = r_i + \gamma \max_{a'} Q_{\bar{\theta}}(s'_i, a')$ using the target network
        
        Update the current network $Q_\theta$ by minimizing the target loss $\mathcal{L} = \frac{1}{N} \sum_i (y_i - Q_\theta(s_i, a_i))^2
        
        Update the target network
        
    end for
    
end for
