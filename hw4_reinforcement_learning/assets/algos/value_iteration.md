Initialize $V(s)$ arbitrarily

while $\Delta > \theta$ do:
    
    $\Delta \leftarrow 0$
    
    for each state $s \in \mathcal{S}$:
        
        $v \leftarrow V(s)$
        
        $V(s) \leftarrow \max_a \left[ r(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s') \right]$
        
        $\Delta \leftarrow \max(\Delta, |v - V(s)|)$
        
end while

Return a deterministic policy $\pi(s) = \arg\max_a \left\{ r(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s') \right\}$
