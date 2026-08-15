Initialize $\pi(s)$ and $V(s)$ arbitrarily

while $\Delta > \theta$ do: (Policy Evaluation Loop)

    $\Delta \leftarrow 0$
    
    for each state $s \in \mathcal{S}$:
        
        $v \leftarrow V(s)$
        
        $V(s) \leftarrow r(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s))V(s')$
        
        $\Delta \leftarrow \max(\Delta, |v - V(s)|)$
        
end while

$\pi_{old} \leftarrow \pi$

for each state $s \in \mathcal{S}$:

    $\pi(s) \leftarrow \arg\max_a \left[ r(s, a) + \gamma \sum_{s'} P(s'|s, a)V(s') \right]$
    
If $\pi_{old} = \pi$, then stop and return $V$ and $\pi$; otherwise go back to Policy Evaluation Loop
